"""
Generate RT price samples from a trained checkpoint.

Usage:
    # Sample from the test set, DDPM:
    python sample.py --config configs/default.yaml --checkpoint runs/best.pt --n_samples 10

    # Same but with DDIM (faster, deterministic):
    python sample.py --config configs/default.yaml --checkpoint runs/best.pt --n_samples 10 --ddim

    # Condition on custom DA prices from a CSV:
    python sample.py --config configs/default.yaml --checkpoint runs/best.pt --da_csv my_da.csv
"""

import argparse
import os
import numpy as np
import torch
import yaml

from data.preprocess import arcsinh_transform, arcsinh_inverse
from models import UNet1D, NoiseSchedule, sample_ddpm, sample_ddim


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_scaler(data_dir):
    sc = np.load(os.path.join(data_dir, "scaler.npz"))
    return float(sc["mean"]), float(sc["std"])


def normalize(x, mean, std):
    return (arcsinh_transform(x) - mean) / std


def denormalize(x, mean, std):
    return arcsinh_inverse(x * std + mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--ddim", action="store_true", help="Use DDIM sampler")
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="Override DDIM steps from config")
    parser.add_argument("--da_csv", default=None,
                        help="CSV with columns [hour (0-23), da_price] for custom conditioning")
    parser.add_argument("--day_index", type=int, default=0,
                        help="Which test-set day to condition on (if --da_csv not given)")
    parser.add_argument("--all_test", action="store_true",
                        help="Generate n_samples for every test-set day; saves (N_days*n_samples, 288)")
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD date for calendar features when using --da_csv")
    parser.add_argument("--out", default=None, help="Output .npz path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    scaler_mean, scaler_std = load_scaler(cfg["data_dir"])

    use_calendar = cfg.get("use_calendar", False)

    # Build model
    model = UNet1D(
        base_channels=cfg["base_channels"],
        num_res_blocks=cfg["num_res_blocks"],
        attn_heads=cfg["attn_heads"],
        use_calendar=use_calendar,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    schedule = NoiseSchedule(T=cfg["T"], schedule=cfg["schedule"]).to(device)

    # ── All-test-day batch mode ────────────────────────────────────────────────
    if args.all_test:
        test_data = np.load(os.path.join(cfg["data_dir"], "test.npz"), allow_pickle=True)
        sampler_name = "ddim" if args.ddim else "ddpm"
        S = args.ddim_steps or cfg.get("ddim_steps", 50)

        # Filter test rows to only the node(s) this model was trained on.
        sc_data = np.load(os.path.join(cfg["data_dir"], "scaler.npz"))
        trained_pnodes = sc_data["pnode_ids"].tolist() if "pnode_ids" in sc_data.files else None
        test_node_ids = test_data["node_ids"] if "node_ids" in test_data.files else None

        if test_node_ids is not None and trained_pnodes is not None:
            mask = np.isin(test_node_ids, trained_pnodes)
            da_arr    = test_data["da"][mask]
            rt_arr    = test_data["rt"][mask]
            cal_arr   = test_data["calendar"][mask] if "calendar" in test_data.files else np.zeros((mask.sum(), 4), dtype=np.int64)
            dates_arr = test_data["dates"][mask]
            nid_arr   = test_node_ids[mask]
            if len(trained_pnodes) == 1:
                print(f"Single-node model (pnode {trained_pnodes[0]}): "
                      f"using {mask.sum()} matching test rows.")
            else:
                print(f"Multi-node model ({len(trained_pnodes)} nodes): "
                      f"using {mask.sum()} test rows.")
        else:
            da_arr    = test_data["da"]
            rt_arr    = test_data["rt"]
            cal_arr   = test_data["calendar"] if "calendar" in test_data.files else np.zeros((len(test_data["da"]), 4), dtype=np.int64)
            dates_arr = test_data["dates"]
            nid_arr   = test_node_ids if test_node_ids is not None else np.zeros(len(da_arr), dtype=np.int64)

        n_days = len(da_arr)
        all_samples, all_real = [], []

        print(f"Sampling {args.n_samples} trace(s) for each of {n_days} test days "
              f"with {sampler_name.upper()}...")

        for i in range(n_days):
            da_288 = np.repeat(da_arr[i], 12).astype(np.float32)
            da_c = torch.from_numpy(da_288).unsqueeze(0).unsqueeze(0).to(device)
            cal_t = None
            if use_calendar:
                cal_feat = cal_arr[i].astype(np.int64)
                cal_t = torch.from_numpy(cal_feat).unsqueeze(0).to(device)
            if args.ddim:
                s = sample_ddim(model, schedule, da_c, n_samples=args.n_samples, S=S, cal=cal_t)
            else:
                s = sample_ddpm(model, schedule, da_c, n_samples=args.n_samples, cal=cal_t)
            all_samples.append(s.squeeze(1).cpu().numpy())
            all_real.append(denormalize(rt_arr[i], scaler_mean, scaler_std))

        samples_norm_all = np.concatenate(all_samples, axis=0)           # (N_days*n_samples, 288)
        samples_all = denormalize(samples_norm_all, scaler_mean, scaler_std)
        real_all    = np.stack(all_real, axis=0)                          # (N_days, 288)

        out_path = args.out or os.path.join(cfg["out_dir"], f"samples_{sampler_name}.npz")
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        np.savez(out_path, samples=samples_all, real_rt=real_all,
                 dates=dates_arr, node_ids=nid_arr)
        print(f"Saved {samples_all.shape} samples ({n_days} days × {args.n_samples}) to {out_path}")
        return

    # ── Single-day mode ────────────────────────────────────────────────────────
    # Build DA conditioning and calendar features
    cal_tensor = None
    if args.da_csv:
        import pandas as pd
        da_raw = pd.read_csv(args.da_csv)["da_price"].values.astype(np.float32)
        assert len(da_raw) == 24, "DA CSV must have exactly 24 hourly prices"
        da_288 = np.repeat(normalize(da_raw, scaler_mean, scaler_std), 12).astype(np.float32)
        da_cond = torch.from_numpy(da_288).unsqueeze(0).unsqueeze(0).to(device)
        real_rt = None
        if use_calendar and args.date:
            from data.preprocess import build_calendar_features
            import datetime
            d = datetime.date.fromisoformat(args.date)
            cal_feat = build_calendar_features(np.array([d]))[0]  # (4,)
            cal_tensor = torch.from_numpy(cal_feat).unsqueeze(0).to(device)  # (1, 4)
    else:
        test_data = np.load(os.path.join(cfg["data_dir"], "test.npz"), allow_pickle=True)
        da_norm_24 = test_data["da"][args.day_index]  # (24,) already normalized
        da_288 = np.repeat(da_norm_24, 12).astype(np.float32)  # (288,)
        da_cond = torch.from_numpy(da_288).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 288)
        rt_norm = test_data["rt"][args.day_index]
        real_rt = denormalize(rt_norm, scaler_mean, scaler_std)
        dates = test_data["dates"]
        print(f"Conditioning on test day: {dates[args.day_index]}")
        if use_calendar and "calendar" in test_data:
            cal_feat = test_data["calendar"][args.day_index].astype(np.int64)  # (4,)
            cal_tensor = torch.from_numpy(cal_feat).unsqueeze(0).to(device)    # (1, 4)

    # Sample
    sampler_name = "ddim" if args.ddim else "ddpm"
    print(f"Sampling {args.n_samples} traces with {sampler_name.upper()}...")

    if args.ddim:
        S = args.ddim_steps or cfg.get("ddim_steps", 50)
        samples_norm = sample_ddim(model, schedule, da_cond, n_samples=args.n_samples, S=S, cal=cal_tensor)
    else:
        samples_norm = sample_ddpm(model, schedule, da_cond, n_samples=args.n_samples, cal=cal_tensor)

    samples_norm = samples_norm.squeeze(1).cpu().numpy()  # (n_samples, 288)
    samples = denormalize(samples_norm, scaler_mean, scaler_std)

    out_path = args.out or os.path.join(cfg["out_dir"], f"samples_{sampler_name}.npz")
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    np.savez(out_path, samples=samples, real_rt=real_rt if real_rt is not None else np.array([]))
    print(f"Saved {samples.shape} samples to {out_path}")
    print(f"  Sample mean: {samples.mean():.2f}  std: {samples.std():.2f}  "
          f"max: {samples.max():.2f}  min: {samples.min():.2f}")


if __name__ == "__main__":
    main()
