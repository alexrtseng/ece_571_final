"""
End-to-end backtest: load real prices for a date range, generate DDPM/DDIM
forecasts, run battery LP, and compare revenues.

Usage:
    python backtest.py \\
        --config configs/default.yaml \\
        --checkpoint runs/small/best.pt \\
        --start_date 2025-01-01 \\
        --end_date 2025-03-31 \\
        --n_samples 10
    # outputs → runs/small/eval/
"""

import argparse
import datetime
import os
import sys

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from sample import load_scaler, denormalize, get_device
from models.unet import UNet1D
from models.schedule import NoiseSchedule
from models.diffusion import sample_ddpm, sample_ddim
from battery import batch_revenue


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)

    def flush(self):
        for f in self.files:
            f.flush()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_date_range(data_dir: str, start_date: str, end_date: str,
                    pnode_id: int | None = None) -> dict:
    """
    Return normalized DA/RT/calendar arrays for all days in [start_date, end_date].
    Searches train, val, and test splits and merges chronologically.
    If pnode_id is given, only rows for that node are returned.
    """
    d0 = datetime.date.fromisoformat(start_date)
    d1 = datetime.date.fromisoformat(end_date)

    collected = {k: [] for k in ("dates", "da", "rt", "calendar", "node_ids")}
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{split}.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True)
        dates = data["dates"]
        node_ids = data["node_ids"] if "node_ids" in data.files else None

        mask = np.array([d0 <= d <= d1 for d in dates])
        if pnode_id is not None and node_ids is not None:
            mask &= (node_ids == pnode_id)
        if not mask.any():
            continue
        collected["dates"].append(dates[mask])
        collected["da"].append(data["da"][mask])
        collected["rt"].append(data["rt"][mask])
        collected["calendar"].append(
            data["calendar"][mask] if "calendar" in data.files
            else np.zeros((mask.sum(), 4), dtype=np.int8)
        )
        collected["node_ids"].append(
            node_ids[mask] if node_ids is not None
            else np.zeros(mask.sum(), dtype=np.int64)
        )

    if not collected["dates"]:
        node_str = f" for pnode {pnode_id}" if pnode_id is not None else ""
        raise ValueError(f"No data found for range {start_date} → {end_date}{node_str}")

    result = {
        "dates":    np.concatenate(collected["dates"]),
        "da":       np.concatenate(collected["da"],       axis=0),
        "rt":       np.concatenate(collected["rt"],       axis=0),
        "calendar": np.concatenate(collected["calendar"], axis=0),
        "node_ids": np.concatenate(collected["node_ids"], axis=0),
    }
    order = np.argsort(result["dates"])
    return {k: v[order] for k, v in result.items()}


# ── Model loading ──────────────────────────────────────────────────────────────

def build_model(cfg: dict, checkpoint_path: str, device: torch.device):
    model = UNet1D(
        base_channels=cfg["base_channels"],
        num_res_blocks=cfg["num_res_blocks"],
        attn_heads=cfg["attn_heads"],
        use_calendar=cfg.get("use_calendar", False),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    schedule = NoiseSchedule(T=cfg["T"], schedule=cfg["schedule"]).to(device)
    return model, schedule


# ── Per-day sampling ───────────────────────────────────────────────────────────

def sample_day(
    model, schedule, da_norm_24: np.ndarray, cal_row: np.ndarray,
    use_calendar: bool, device: torch.device,
    n_samples: int, ddim_steps: int,
):
    """
    da_norm_24 : (24,) normalized float32
    cal_row    : (4,)  int
    Returns normalized arrays (not yet denormalized):
        ddpm_norm : (n_samples, 288)
        ddim_norm : (1, 288)
    """
    da_288 = np.repeat(da_norm_24, 12).astype(np.float32)
    da_cond = torch.from_numpy(da_288).unsqueeze(0).unsqueeze(0).to(device)

    cal_tensor = None
    if use_calendar:
        cal_tensor = torch.from_numpy(cal_row.astype(np.int64)).unsqueeze(0).to(device)

    ddpm_norm = sample_ddpm(model, schedule, da_cond, n_samples=n_samples, cal=cal_tensor)
    ddpm_norm = ddpm_norm.squeeze(1).cpu().numpy()   # (n_samples, 288)

    ddim_norm = sample_ddim(model, schedule, da_cond, n_samples=1,
                            S=ddim_steps, cal=cal_tensor)
    ddim_norm = ddim_norm.squeeze(1).cpu().numpy()   # (1, 288)

    return ddpm_norm, ddim_norm


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_revenue_summary(
    real_total: float, ddim_total: float, ddpm_paths: np.ndarray,
    real_daily: np.ndarray, ddim_daily: np.ndarray, M: int, N: int,
):
    sep = "─" * 65
    print(f"\n{sep}")
    print(f"  Backtest Revenue Summary  ({M} days, 1 MW / 4 MWh battery)")
    print(f"{sep}")
    print(f"  {'Scenario':28s}  {'Total ($)':>12}  {'Mean $/day':>10}")
    print(f"  {'Real RT':28s}  {real_total:>12.2f}  {real_daily.mean():>10.2f}")
    print(f"  {'DDIM (1 path)':28s}  {ddim_total:>12.2f}  {ddim_daily.mean():>10.2f}")
    print(f"{sep}")
    print(f"  DDPM ({N} sample paths, total revenue over range):")
    print(f"    mean  = {ddpm_paths.mean():>10.2f}")
    print(f"    std   = {ddpm_paths.std():>10.2f}")
    print(f"    min   = {ddpm_paths.min():>10.2f}")
    print(f"    max   = {ddpm_paths.max():>10.2f}")
    print(f"{sep}")
    print(f"  DDPM mean vs real:  {ddpm_paths.mean() - real_total:>+.2f}  "
          f"({(ddpm_paths.mean() - real_total) / max(abs(real_total), 1e-9):.1%})")
    print(f"  DDIM      vs real:  {ddim_total - real_total:>+.2f}  "
          f"({(ddim_total - real_total) / max(abs(real_total), 1e-9):.1%})")


def plot_revenue_histogram(
    ddpm_paths: np.ndarray, real_total: float, ddim_total: float,
    out_dir: str, node_tag: str = "",
):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ddpm_paths, bins=max(10, len(ddpm_paths) // 2),
            color="steelblue", alpha=0.75, label=f"DDPM paths (N={len(ddpm_paths)})")
    ax.axvline(real_total, color="black",  lw=2, ls="-",
               label=f"Real RT  (${real_total:.0f})")
    ax.axvline(ddim_total, color="tomato", lw=2, ls="--",
               label=f"DDIM     (${ddim_total:.0f})")
    ax.set_xlabel("Total Revenue over Date Range ($)")
    ax.set_ylabel("Count")
    ax.set_title("DDPM Path Revenue Distribution vs Real & DDIM")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"backtest_revenue{node_tag}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved {fname}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backtest battery dispatch revenue over a date range."
    )
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--start_date",  required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date",    required=True, help="YYYY-MM-DD")
    parser.add_argument("--n_samples",   type=int, default=10,
                        help="DDPM samples per day (default: 10)")
    parser.add_argument("--pnode_id",    type=int, default=None,
                        help="Filter backtest to a specific pnode (auto-detected for single-node runs)")
    parser.add_argument("--out_dir",     default=None,
                        help="Output dir; defaults to <checkpoint_dir>/eval/")
    parser.add_argument("--skip_battery", action="store_true",
                        help="Skip LP optimization (for quick testing)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.checkpoint)), "eval"
        )
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve pnode_id before opening log so it can go in the filename.
    pnode_id = args.pnode_id
    sc_data = np.load(os.path.join(cfg["data_dir"], "scaler.npz"))
    if pnode_id is None and "pnode_ids" in sc_data.files:
        trained = sc_data["pnode_ids"].tolist()
        if len(trained) == 1:
            pnode_id = int(trained[0])

    node_tag = f"_pnode{pnode_id}" if pnode_id is not None else ""

    log_path = os.path.join(
        args.out_dir,
        f"backtest{node_tag}_{args.start_date}_{args.end_date}.txt",
    )
    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    if pnode_id is not None and args.pnode_id is None:
        print(f"Auto-detected single-node model: pnode_id={pnode_id}")

    device = get_device()
    mean, std = load_scaler(cfg["data_dir"])
    ddim_steps = cfg.get("ddim_steps", 50)
    use_calendar = cfg.get("use_calendar", False)
    N = args.n_samples

    print(f"Backtest: {args.start_date} → {args.end_date}"
          + (f"  pnode={pnode_id}" if pnode_id is not None else "  (all nodes)"))
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}  |  DDPM samples/day: {N}  |  DDIM steps: {ddim_steps}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_date_range(cfg["data_dir"], args.start_date, args.end_date,
                           pnode_id=pnode_id)
    M = len(data["dates"])
    print(f"\nFound {M} days in [{args.start_date}, {args.end_date}]")
    if M == 0:
        print("No days found — exiting.")
        sys.stdout = sys.__stdout__
        log_file.close()
        return

    # ── Build model ───────────────────────────────────────────────────────────
    model, schedule = build_model(cfg, args.checkpoint, device)

    # ── Sampling loop ─────────────────────────────────────────────────────────
    ddpm_norm_list = []
    ddim_norm_list = []

    print(f"\nSampling {N} DDPM + 1 DDIM trace(s) for each of {M} days...")
    for i in range(M):
        ddpm_n, ddim_n = sample_day(
            model, schedule,
            data["da"][i], data["calendar"][i],
            use_calendar, device, N, ddim_steps,
        )
        ddpm_norm_list.append(ddpm_n)
        ddim_norm_list.append(ddim_n)
        if (i + 1) % 10 == 0 or i == M - 1:
            print(f"  {i+1}/{M} days sampled", flush=True)

    # ── Bulk denormalize ──────────────────────────────────────────────────────
    rt_real  = denormalize(data["rt"], mean, std)                                # (M, 288)
    ddpm_all = denormalize(np.concatenate(ddpm_norm_list, axis=0), mean, std)   # (M*N, 288)
    ddpm_all = ddpm_all.reshape(M, N, 288)
    ddim_all = denormalize(np.concatenate(ddim_norm_list, axis=0), mean, std)   # (M, 288)

    # ── Battery LP ────────────────────────────────────────────────────────────
    if args.skip_battery:
        print("\n[skip_battery] Skipping LP — using zero revenues.")
        real_daily = np.zeros(M)
        ddim_daily = np.zeros(M)
        ddpm_daily = np.zeros((M, N))
    else:
        print("\nRunning battery arbitrage LP...")
        real_daily  = batch_revenue(rt_real,                      desc="real RT")  # (M,)
        ddim_daily  = batch_revenue(ddim_all,                     desc="DDIM   ")  # (M,)
        ddpm_rev    = batch_revenue(ddpm_all.reshape(M * N, 288), desc="DDPM   ")  # (M*N,)
        ddpm_daily  = ddpm_rev.reshape(M, N)                                        # (M, N)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    real_total = float(real_daily.sum())
    ddim_total = float(ddim_daily.sum())
    ddpm_paths = ddpm_daily.sum(axis=0)   # (N,) — total revenue per sample path

    print_revenue_summary(real_total, ddim_total, ddpm_paths,
                          real_daily, ddim_daily, M, N)

    # ── Save ──────────────────────────────────────────────────────────────────
    npz_path = os.path.join(args.out_dir, f"backtest_revenues{node_tag}.npz")
    np.savez(npz_path,
             real_daily=real_daily,
             ddim_daily=ddim_daily,
             ddpm_daily=ddpm_daily,
             dates=data["dates"])
    print(f"\nSaved revenues → {npz_path}")

    if not args.skip_battery:
        plot_revenue_histogram(ddpm_paths, real_total, ddim_total, args.out_dir,
                               node_tag=node_tag)

    print(f"\nAll outputs saved to {args.out_dir}/")

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Stats log saved to {log_path}")


if __name__ == "__main__":
    main()
