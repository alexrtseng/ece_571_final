"""
Evaluate generated samples vs real RT prices.

Usage:
    python evaluate.py \\
        --config configs/default.yaml \\
        --ddpm_samples runs/samples_ddpm.npz \\
        --ddim_samples runs/samples_ddim.npz \\
        --out_dir runs/eval/
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from battery import batch_revenue


class _Tee:
    """Write to multiple file-like objects simultaneously."""
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for f in self.files:
            f.write(s)

    def flush(self):
        for f in self.files:
            f.flush()


# ── Stats helpers ──────────────────────────────────────────────────────────────

def spike_freq(prices: np.ndarray, threshold: float) -> float:
    """Fraction of 5-min intervals above threshold ($/MWh)."""
    return float((prices > threshold).mean())


def hourly_vol(prices: np.ndarray) -> np.ndarray:
    """Mean intra-hour std by hour of day. prices: (N, 288)"""
    reshaped = prices.reshape(prices.shape[0], 24, 12)  # (N, 24, 12)
    return reshaped.std(axis=2).mean(axis=0)              # (24,)


def cumulative_movement(prices: np.ndarray) -> tuple[float, float]:
    """Mean cumulative upward and downward movement per day. prices: (N, 288)"""
    diffs = np.diff(prices, axis=1)  # (N, 287)
    up   = float(np.where(diffs > 0, diffs, 0).sum(axis=1).mean())
    down = float(np.where(diffs < 0, -diffs, 0).sum(axis=1).mean())
    return up, down


def volatility_stats(prices: np.ndarray, label: str) -> dict:
    flat = prices.ravel()
    rows = prices if prices.ndim == 2 else prices[None, :]
    cum_up, cum_down = cumulative_movement(rows)
    return {
        "label": label,
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "spike_100": spike_freq(flat, 100),
        "spike_500": spike_freq(flat, 500),
        "spike_neg": float((flat < 0).mean()),
        "cum_up": cum_up,
        "cum_down": cum_down,
    }


def print_stats(stats_dict: dict):
    print(f"\n{'─'*60}")
    print(f"  {stats_dict['label']}")
    print(f"{'─'*60}")
    print(f"  Mean:          {stats_dict['mean']:>10.2f} $/MWh")
    print(f"  Std:           {stats_dict['std']:>10.2f}")
    print(f"  Median:        {stats_dict['p50']:>10.2f}")
    print(f"  95th pct:      {stats_dict['p95']:>10.2f}")
    print(f"  99th pct:      {stats_dict['p99']:>10.2f}")
    print(f"  Spike >$100:   {stats_dict['spike_100']:>10.3%}")
    print(f"  Spike >$500:   {stats_dict['spike_500']:>10.3%}")
    print(f"  Negative:      {stats_dict['spike_neg']:>10.3%}")
    print(f"  Cum up/day:    {stats_dict['cum_up']:>10.2f} $/MWh")
    print(f"  Cum down/day:  {stats_dict['cum_down']:>10.2f} $/MWh")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_traces(real_rt, ddpm_samples, ddim_samples, out_dir, n_show=10):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    x = np.arange(288) * 5 / 60  # hours

    for ax, (samples, title) in zip(axes, [
        (real_rt[:n_show], "Real RT"),
        (ddpm_samples[:n_show], "DDPM Generated"),
        (ddim_samples[:n_show], "DDIM Generated"),
    ]):
        for trace in samples:
            ax.plot(x, trace, alpha=0.5, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Price ($/MWh)")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traces.png"), dpi=150)
    plt.close()
    print("  Saved traces.png")


def plot_distribution(real_rt, ddpm_samples, ddim_samples, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(real_rt.min(), ddpm_samples.min(), ddim_samples.min()),
        np.percentile(
            np.concatenate([real_rt.ravel(), ddpm_samples.ravel(), ddim_samples.ravel()]), 99.5
        ),
        100,
    )
    for arr, label, color in [
        (real_rt, "Real RT", "black"),
        (ddpm_samples, "DDPM", "steelblue"),
        (ddim_samples, "DDIM", "tomato"),
    ]:
        ax.hist(arr.ravel(), bins=bins, density=True, histtype="step", label=label,
                color=color, linewidth=1.5)

    ax.set_xlabel("Price ($/MWh)")
    ax.set_ylabel("Density")
    ax.set_title("Price Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distribution.png"), dpi=150)
    plt.close()
    print("  Saved distribution.png")


def plot_hourly_vol(real_rt, ddpm_samples, ddim_samples, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    hours = np.arange(24)
    for arr, label, color in [
        (real_rt, "Real RT", "black"),
        (ddpm_samples, "DDPM", "steelblue"),
        (ddim_samples, "DDIM", "tomato"),
    ]:
        ax.plot(hours, hourly_vol(arr), label=label, color=color, linewidth=1.5, marker="o", ms=4)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Intra-hour Std ($/MWh)")
    ax.set_title("Hourly Volatility Profile")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hourly_vol.png"), dpi=150)
    plt.close()
    print("  Saved hourly_vol.png")


def plot_acf(real_rt, ddpm_samples, ddim_samples, out_dir, max_lag=48):
    def acf(x, max_lag):
        x = x.ravel()
        x = x - x.mean()
        full = np.correlate(x, x, mode="full")
        full = full[full.size // 2:]
        return full[:max_lag] / full[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    lags = np.arange(max_lag) * 5  # minutes
    for arr, label, color in [
        (real_rt, "Real RT", "black"),
        (ddpm_samples, "DDPM", "steelblue"),
        (ddim_samples, "DDIM", "tomato"),
    ]:
        # Average ACF across samples
        acfs = np.stack([acf(arr[i], max_lag) for i in range(len(arr))])
        ax.plot(lags, acfs.mean(axis=0), label=label, color=color, linewidth=1.5)
        ax.fill_between(lags, acfs.min(axis=0), acfs.max(axis=0), alpha=0.1, color=color)

    ax.set_xlabel("Lag (minutes)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation Function")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acf.png"), dpi=150)
    plt.close()
    print("  Saved acf.png")


def wasserstein(a, b):
    return float(stats.wasserstein_distance(a.ravel(), b.ravel()))


# ── Battery evaluation ─────────────────────────────────────────────────────────

def battery_eval(real_rt, ddpm_samples, ddim_samples, out_dir):
    """
    For each day in real_rt and each generated sample, solve the battery LP
    and compare revenues.

    When there are multiple generated samples per real day (N_gen > N_real),
    we assume they are ordered as [day0_s0, day0_s1, ..., day1_s0, ...] if
    N_gen is a multiple of N_real, otherwise treat all samples independently.
    """
    print("\nRunning battery arbitrage optimization...")
    real_rev  = batch_revenue(real_rt,      desc="real RT   ")
    ddpm_rev  = batch_revenue(ddpm_samples, desc="DDPM samples")
    ddim_rev  = batch_revenue(ddim_samples, desc="DDIM samples")

    # If generated arrays are multiples of real (e.g. 3 samples per day),
    # average revenues over samples for each corresponding real day.
    n_real = len(real_rev)
    def aggregate(gen_rev):
        if len(gen_rev) % n_real == 0:
            k = len(gen_rev) // n_real
            return gen_rev.reshape(n_real, k).mean(axis=1)
        return gen_rev

    ddpm_rev_agg = aggregate(ddpm_rev)
    ddim_rev_agg = aggregate(ddim_rev)
    n_compare = min(n_real, len(ddpm_rev_agg), len(ddim_rev_agg))

    real_c  = real_rev[:n_compare]
    ddpm_c  = ddpm_rev_agg[:n_compare]
    ddim_c  = ddim_rev_agg[:n_compare]
    delta_ddpm = ddpm_c - real_c
    delta_ddim = ddim_c - real_c

    # Print summary
    print(f"\n{'─'*65}")
    print(f"  Battery Arbitrage Revenue  (1 MW / 4 MWh, $/day)")
    print(f"{'─'*65}")
    print(f"  {'':20s} {'Mean':>9} {'Std':>9} {'Median':>9} {'p95':>9}")
    for label, arr in [("Real RT", real_c), ("DDPM generated", ddpm_c), ("DDIM generated", ddim_c)]:
        print(f"  {label:20s} {arr.mean():>9.2f} {arr.std():>9.2f} "
              f"{np.median(arr):>9.2f} {np.percentile(arr, 95):>9.2f}")
    print(f"{'─'*65}")
    print(f"  {'DDPM delta':20s} {delta_ddpm.mean():>+9.2f} {delta_ddpm.std():>9.2f}  "
          f"(RMSE {np.sqrt((delta_ddpm**2).mean()):.2f})")
    print(f"  {'DDIM delta':20s} {delta_ddim.mean():>+9.2f} {delta_ddim.std():>9.2f}  "
          f"(RMSE {np.sqrt((delta_ddim**2).mean()):.2f})")

    # Save raw revenues
    np.savez(
        os.path.join(out_dir, "battery_revenues.npz"),
        real=real_c, ddpm=ddpm_c, ddim=ddim_c,
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Distribution of daily revenues
    ax = axes[0]
    bins = np.linspace(
        min(real_c.min(), ddpm_c.min(), ddim_c.min()),
        max(real_c.max(), ddpm_c.max(), ddim_c.max()),
        40,
    )
    for arr, label, color in [
        (real_c, "Real RT", "black"),
        (ddpm_c, "DDPM",    "steelblue"),
        (ddim_c, "DDIM",    "tomato"),
    ]:
        ax.hist(arr, bins=bins, density=True, histtype="step",
                label=f"{label} (μ={arr.mean():.0f})", color=color, linewidth=1.8)
    ax.set_xlabel("Revenue ($/day)"); ax.set_ylabel("Density")
    ax.set_title("Battery Revenue Distribution")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Scatter: real vs generated revenue, per day
    ax = axes[1]
    lims = [min(real_c.min(), ddpm_c.min(), ddim_c.min()),
            max(real_c.max(), ddpm_c.max(), ddim_c.max())]
    ax.plot(lims, lims, "k--", lw=1, label="perfect match")
    ax.scatter(real_c, ddpm_c, s=15, alpha=0.5, color="steelblue", label="DDPM")
    ax.scatter(real_c, ddim_c, s=15, alpha=0.5, color="tomato",    label="DDIM")
    ax.set_xlabel("Real RT revenue ($/day)"); ax.set_ylabel("Generated revenue ($/day)")
    ax.set_title("Revenue: Real vs Generated (per day)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Delta histogram (bias view)
    ax = axes[2]
    for arr, label, color in [
        (delta_ddpm, "DDPM − Real", "steelblue"),
        (delta_ddim, "DDIM − Real", "tomato"),
    ]:
        ax.hist(arr, bins=30, density=True, histtype="step",
                label=f"{label} (μ={arr.mean():+.0f})", color=color, linewidth=1.8)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Revenue delta ($/day)"); ax.set_ylabel("Density")
    ax.set_title("Revenue Error Distribution")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle("Battery Arbitrage Evaluation (1 MW / 4 MWh)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "battery.png"), dpi=150)
    plt.close()
    print("  Saved battery.png")

    return real_c, ddpm_c, ddim_c


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ddpm_samples", required=True)
    parser.add_argument("--ddim_samples", required=True)
    parser.add_argument("--real_npz", default=None,
                        help="Path to test.npz; if omitted, uses real_rt from samples file")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory; defaults to <run_dir>/eval/ next to the samples file")
    parser.add_argument("--skip_battery", action="store_true",
                        help="Skip battery LP (faster; use if Gurobi unavailable)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.out_dir is None:
        run_dir = os.path.dirname(os.path.abspath(args.ddpm_samples))
        args.out_dir = os.path.join(run_dir, "eval")

    os.makedirs(args.out_dir, exist_ok=True)

    log_path = os.path.join(args.out_dir, "stats.txt")
    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    # Load samples
    ddpm = np.load(args.ddpm_samples)
    ddim = np.load(args.ddim_samples)
    ddpm_samples = ddpm["samples"]   # (n_samples, 288)
    ddim_samples = ddim["samples"]

    # Real RT: prefer test.npz so we have many days
    if args.real_npz:
        from data.preprocess import arcsinh_inverse
        test = np.load(args.real_npz, allow_pickle=True)
        sc = np.load(os.path.join(cfg["data_dir"], "scaler.npz"))
        mean, std = float(sc["mean"]), float(sc["std"])
        real_rt = arcsinh_inverse(test["rt"] * std + mean)  # (N_test, 288)
    else:
        real_rt = ddpm["real_rt"]
        if real_rt.ndim == 1:
            real_rt = real_rt[None, :]

    print(f"\nReal RT:      {real_rt.shape}")
    print(f"DDPM samples: {ddpm_samples.shape}")
    print(f"DDIM samples: {ddim_samples.shape}")

    # Price stats
    for arr, label in [
        (real_rt, "Real RT"),
        (ddpm_samples, "DDPM Generated"),
        (ddim_samples, "DDIM Generated"),
    ]:
        print_stats(volatility_stats(arr, label))

    w_ddpm = wasserstein(real_rt, ddpm_samples)
    w_ddim = wasserstein(real_rt, ddim_samples)
    print(f"\nWasserstein distance  DDPM: {w_ddpm:.4f}  DDIM: {w_ddim:.4f}")

    # Price plots
    print("\nGenerating price plots...")
    plot_traces(real_rt, ddpm_samples, ddim_samples, args.out_dir)
    plot_distribution(real_rt, ddpm_samples, ddim_samples, args.out_dir)
    plot_hourly_vol(real_rt, ddpm_samples, ddim_samples, args.out_dir)
    plot_acf(real_rt, ddpm_samples, ddim_samples, args.out_dir)

    # Battery evaluation
    if not args.skip_battery:
        battery_eval(real_rt, ddpm_samples, ddim_samples, args.out_dir)

    print(f"\nAll outputs saved to {args.out_dir}/")

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"Stats log saved to {log_path}")


if __name__ == "__main__":
    main()
