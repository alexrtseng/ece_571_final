"""
Orchestrate the full pipeline for one or more model configurations:
  preprocess → train → sample (DDPM + DDIM) → evaluate → backtest

Usage:
    python runner.py                          # run all configs
    python runner.py --configs default,big    # run named subset
    python runner.py --skip_train            # sample/eval/backtest only (need best.pt)

Flags mirror the underlying scripts and are passed through verbatim.
"""

import argparse
import os
import subprocess
import sys

# ── Config registry ───────────────────────────────────────────────────────────

ALL_CONFIGS = [
    {
        "name":     "default",
        "config":   "configs/default.yaml",
        "data_dir": "./data",
        "out_dir":  "./runs/default",
        "preprocess_config": "configs/default.yaml",
    },
    {
        "name":     "calendar",
        "config":   "configs/calendar.yaml",
        "data_dir": "./data",
        "out_dir":  "./runs/calendar",
        "preprocess_config": "configs/default.yaml",   # shares ./data with default
    },
    {
        "name":     "big",
        "config":   "configs/big.yaml",
        "data_dir": "./data/big",
        "out_dir":  "./runs/big",
        "preprocess_config": "configs/big.yaml",
    },
    {
        "name":     "big_calendar",
        "config":   "configs/big_calendar.yaml",
        "data_dir": "./data/big",
        "out_dir":  "./runs/big_calendar",
        "preprocess_config": "configs/big.yaml",       # shares ./data/big
    },
    {
        "name":     "multi_node",
        "config":   "configs/multi_node.yaml",
        "data_dir": "./data/multi_node",
        "out_dir":  "./runs/multi_node",
        "preprocess_config": "configs/multi_node.yaml",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(msg: str):
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  {msg}")
    print(f"{bar}", flush=True)


def run(cmd: list[str], **kwargs):
    """Run a command, streaming output, and raise on non-zero exit."""
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_preprocess(cfg: dict):
    banner(f"PREPROCESS  [{cfg['name']}]  →  {cfg['data_dir']}")
    run([sys.executable, "data/preprocess.py", "--config", cfg["preprocess_config"]])


def step_train(cfg: dict):
    banner(f"TRAIN  [{cfg['name']}]")
    run([sys.executable, "train.py", "--config", cfg["config"]])


def step_sample(cfg: dict, n_samples: int):
    banner(f"SAMPLE  [{cfg['name']}]")
    checkpoint = os.path.join(cfg["out_dir"], "best.pt")
    base = [
        sys.executable, "sample.py",
        "--config",     cfg["config"],
        "--checkpoint", checkpoint,
        "--n_samples",  str(n_samples),
        "--all_test",
    ]
    run(base)                       # DDPM
    run(base + ["--ddim"])          # DDIM


def step_evaluate(cfg: dict, skip_battery: bool):
    banner(f"EVALUATE  [{cfg['name']}]")
    out = cfg["out_dir"]
    cmd = [
        sys.executable, "evaluate.py",
        "--config",       cfg["config"],
        "--ddpm_samples", os.path.join(out, "samples_ddpm.npz"),
        "--ddim_samples", os.path.join(out, "samples_ddim.npz"),
        "--real_npz",     os.path.join(cfg["data_dir"], "test.npz"),
        "--out_dir",      os.path.join(out, "eval"),
    ]
    if skip_battery:
        cmd.append("--skip_battery")
    run(cmd)


def step_backtest(cfg: dict, start: str, end: str, n_samples: int, skip_battery: bool):
    banner(f"BACKTEST  [{cfg['name']}]  {start} → {end}")
    cmd = [
        sys.executable, "backtest.py",
        "--config",      cfg["config"],
        "--checkpoint",  os.path.join(cfg["out_dir"], "best.pt"),
        "--start_date",  start,
        "--end_date",    end,
        "--n_samples",   str(n_samples),
        "--out_dir",     os.path.join(cfg["out_dir"], "eval"),
    ]
    if skip_battery:
        cmd.append("--skip_battery")
    run(cmd)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full pipeline runner.")
    parser.add_argument(
        "--configs", default=None,
        help="Comma-separated config names to run (default: all). "
             f"Choices: {', '.join(c['name'] for c in ALL_CONFIGS)}",
    )
    parser.add_argument("--n_samples",       type=int, default=10,
                        help="Samples per test-set day for sampling + backtest (default: 10)")
    parser.add_argument("--backtest_start",  default="2025-01-01",
                        help="Backtest start date YYYY-MM-DD (default: 2025-01-01)")
    parser.add_argument("--backtest_end",    default="2025-10-31",
                        help="Backtest end date YYYY-MM-DD (default: 2025-10-31)")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_train",      action="store_true")
    parser.add_argument("--skip_sample",     action="store_true")
    parser.add_argument("--skip_eval",       action="store_true")
    parser.add_argument("--skip_backtest",   action="store_true")
    parser.add_argument("--skip_battery",    action="store_true",
                        help="Pass --skip_battery to evaluate.py and backtest.py")
    args = parser.parse_args()

    # Select configs
    if args.configs:
        names = [n.strip() for n in args.configs.split(",")]
        by_name = {c["name"]: c for c in ALL_CONFIGS}
        unknown = [n for n in names if n not in by_name]
        if unknown:
            parser.error(f"Unknown config(s): {unknown}. "
                         f"Valid: {list(by_name)}")
        selected = [by_name[n] for n in names]
    else:
        selected = ALL_CONFIGS

    print(f"\nConfigs to run: {[c['name'] for c in selected]}")
    print(f"n_samples={args.n_samples}  "
          f"backtest={args.backtest_start}→{args.backtest_end}  "
          f"skip_battery={args.skip_battery}")

    # Track which data_dirs have been preprocessed so we don't redo them.
    preprocessed: set[str] = set()

    for cfg in selected:
        banner(f"CONFIG: {cfg['name'].upper()}")

        if not args.skip_preprocess:
            if cfg["data_dir"] not in preprocessed:
                step_preprocess(cfg)
                preprocessed.add(cfg["data_dir"])
            else:
                print(f"[preprocess] Skipping — {cfg['data_dir']} already processed.")

        if not args.skip_train:
            step_train(cfg)

        if not args.skip_sample:
            step_sample(cfg, args.n_samples)

        if not args.skip_eval:
            step_evaluate(cfg, args.skip_battery)

        if not args.skip_backtest:
            step_backtest(cfg, args.backtest_start, args.backtest_end,
                          args.n_samples, args.skip_battery)

    banner("ALL RUNS COMPLETE")
    print("Results:")
    for cfg in selected:
        print(f"  {cfg['name']:15s}  {cfg['out_dir']}/eval/")


if __name__ == "__main__":
    main()
