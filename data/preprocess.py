"""
Build aligned (da_24, rt_288) daily arrays for a single pnode.
Saves data/train.npz, data/val.npz, data/test.npz, data/scaler.npz.
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import yaml
import holidays as hol_lib


# ── Scaler ────────────────────────────────────────────────────────────────────

def arcsinh_transform(x: np.ndarray) -> np.ndarray:
    return np.arcsinh(x / 100.0)


def arcsinh_inverse(x: np.ndarray) -> np.ndarray:
    return 100.0 * np.sinh(x)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_da(raw_dir: str, pnode_id: int) -> pd.DataFrame:
    """Load all DA monthly CSVs, return hourly DataFrame for one pnode."""
    files = sorted(glob.glob(os.path.join(raw_dir, "da_hrl_lmps_*.csv")))
    if not files:
        raise FileNotFoundError(f"No DA CSVs found in {raw_dir}")
    chunks = []
    for f in files:
        df = pd.read_csv(f, usecols=["datetime_beginning_utc", "pnode_id", "total_lmp_da"])
        df = df[df["pnode_id"] == pnode_id]
        chunks.append(df)
    da = pd.concat(chunks, ignore_index=True)
    da["datetime"] = pd.to_datetime(da["datetime_beginning_utc"], utc=True)
    da = da.sort_values("datetime").drop(columns=["datetime_beginning_utc", "pnode_id"])
    return da


def load_rt(raw_dir: str, pnode_id: int) -> pd.DataFrame:
    """Load all RT monthly CSVs, return 5-min DataFrame for one pnode."""
    files = sorted(glob.glob(os.path.join(raw_dir, "rt_fivemin_hrl_lmps_*.csv")))
    if not files:
        raise FileNotFoundError(f"No RT CSVs found in {raw_dir}")
    chunks = []
    for f in files:
        df = pd.read_csv(f, usecols=["datetime_beginning_utc", "pnode_id", "total_lmp_rt"])
        df = df[df["pnode_id"] == pnode_id]
        chunks.append(df)
    rt = pd.concat(chunks, ignore_index=True)
    rt["datetime"] = pd.to_datetime(rt["datetime_beginning_utc"], utc=True)
    rt = rt.sort_values("datetime").drop(columns=["datetime_beginning_utc", "pnode_id"])
    return rt


# ── Alignment ─────────────────────────────────────────────────────────────────

def build_daily_arrays(da: pd.DataFrame, rt: pd.DataFrame):
    """
    Return (dates, da_arr, rt_arr) where:
      da_arr: (N, 24) float32 — hourly DA prices
      rt_arr: (N, 288) float32 — 5-min RT prices
    Only days with exactly 24 DA hours and 288 RT intervals are kept.
    """
    da["date"] = da["datetime"].dt.date
    rt["date"] = rt["datetime"].dt.date

    da_grouped = da.groupby("date")["total_lmp_da"].apply(list)
    rt_grouped = rt.groupby("date")["total_lmp_rt"].apply(list)

    common_dates = sorted(set(da_grouped.index) & set(rt_grouped.index))

    dates, da_rows, rt_rows = [], [], []
    for d in common_dates:
        da_vals = da_grouped[d]
        rt_vals = rt_grouped[d]
        if len(da_vals) != 24 or len(rt_vals) != 288:
            continue
        dates.append(d)
        da_rows.append(da_vals)
        rt_rows.append(rt_vals)

    return (
        np.array(dates),
        np.array(da_rows, dtype=np.float32),
        np.array(rt_rows, dtype=np.float32),
    )


# ── Calendar features ─────────────────────────────────────────────────────────

def build_calendar_features(dates: np.ndarray) -> np.ndarray:
    """
    Return (N, 4) int8 array: [day_of_week (0-6), month (0-11), is_weekend, is_holiday].
    Uses US federal holidays (relevant for PJM market).
    """
    years = {d.year for d in dates}
    us_holidays = hol_lib.US(years=years)

    rows = []
    for d in dates:
        dow      = d.weekday()           # 0=Mon … 6=Sun
        month    = d.month - 1           # 0-indexed
        weekend  = int(dow >= 5)
        holiday  = int(d in us_holidays)
        rows.append([dow, month, weekend, holiday])
    return np.array(rows, dtype=np.int8)


# ── Split ─────────────────────────────────────────────────────────────────────

def split_by_date(dates, da_arr, rt_arr, cal_arr, train_end: str, val_end: str):
    import datetime
    te = datetime.date(int(train_end[:4]), int(train_end[5:7]), 28)  # last ~day of month
    ve = datetime.date(int(val_end[:4]), int(val_end[5:7]), 28)

    train_mask = dates <= te
    val_mask = (dates > te) & (dates <= ve)
    test_mask = dates > ve

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        splits[name] = (dates[mask], da_arr[mask], rt_arr[mask], cal_arr[mask])
    return splits


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Support both pnode_id (single int) and pnode_ids (list) in config
    if "pnode_ids" in cfg:
        pnode_ids = [int(p) for p in cfg["pnode_ids"]]
    else:
        pnode_ids = [int(cfg["pnode_id"])]

    data_dir = cfg["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    all_dates, all_da, all_rt, all_node_ids = [], [], [], []

    for pnode_id in pnode_ids:
        print(f"Loading pnode {pnode_id}...")
        da = load_da(cfg["raw_da_dir"], pnode_id)
        rt = load_rt(cfg["raw_rt_dir"], pnode_id)
        print(f"  {len(da)} DA rows, {len(rt)} RT rows")
        dates, da_arr, rt_arr = build_daily_arrays(da, rt)
        print(f"  {len(dates)} complete days")
        all_dates.append(dates)
        all_da.append(da_arr)
        all_rt.append(rt_arr)
        all_node_ids.append(np.full(len(dates), pnode_id, dtype=np.int64))

    dates   = np.concatenate(all_dates)
    da_arr  = np.concatenate(all_da,  axis=0)
    rt_arr  = np.concatenate(all_rt,  axis=0)
    nid_arr = np.concatenate(all_node_ids)

    # Sort chronologically so same-date rows from different nodes are adjacent
    order  = np.argsort(dates, kind="stable")
    dates  = dates[order];  da_arr = da_arr[order]
    rt_arr = rt_arr[order]; nid_arr = nid_arr[order]

    print(f"\nTotal: {len(dates)} days from {len(pnode_ids)} node(s) ({dates[0]} → {dates[-1]})")

    print("Computing calendar features...")
    cal_arr = build_calendar_features(dates)
    hol_count  = int(cal_arr[:, 3].sum())
    wknd_count = int(cal_arr[:, 2].sum())
    print(f"  {hol_count} holidays, {wknd_count} weekend days")

    print("Splitting train / val / test...")
    splits = split_by_date(dates, da_arr, rt_arr, cal_arr, cfg["train_end"], cfg["val_end"])
    for name, (d, da_, rt_, cal_) in splits.items():
        print(f"  {name}: {len(d)} days")

    # Fit arcsinh + z-score scaler on training data only (across all nodes)
    train_da = splits["train"][1]
    train_rt = splits["train"][2]
    all_train = np.concatenate([
        arcsinh_transform(train_da).ravel(),
        arcsinh_transform(train_rt).ravel(),
    ])
    scaler_mean = float(all_train.mean())
    scaler_std  = float(all_train.std())
    print(f"Scaler: mean={scaler_mean:.4f}, std={scaler_std:.4f}")

    np.savez(
        os.path.join(data_dir, "scaler.npz"),
        mean=scaler_mean,
        std=scaler_std,
        pnode_ids=np.array(pnode_ids, dtype=np.int64),
    )

    def normalize(x):
        return (arcsinh_transform(x) - scaler_mean) / scaler_std

    # Split node_ids using the same date masks as split_by_date
    import datetime
    te = datetime.date(int(cfg["train_end"][:4]), int(cfg["train_end"][5:7]), 28)
    ve = datetime.date(int(cfg["val_end"][:4]),   int(cfg["val_end"][5:7]),   28)
    nid_splits = {
        "train": nid_arr[dates <= te],
        "val":   nid_arr[(dates > te) & (dates <= ve)],
        "test":  nid_arr[dates > ve],
    }

    for name, (d, da_, rt_, cal_) in splits.items():
        path = os.path.join(data_dir, f"{name}.npz")
        np.savez(path, dates=d, da=normalize(da_), rt=normalize(rt_),
                 calendar=cal_, node_ids=nid_splits[name])
        print(f"  Saved {path}  da={da_.shape}  rt={rt_.shape}  calendar={cal_.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
