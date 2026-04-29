import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PriceDataset(Dataset):
    """
    Each sample: (da_cond, rt_target, cal_feat)
      da_cond:   (1, 288) float32 — DA price repeated 12× per hour, normalized
      rt_target: (1, 288) float32 — RT 5-min prices, normalized
      cal_feat:  (4,)     int64   — [day_of_week, month, is_weekend, is_holiday]
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.da = data["da"].astype(np.float32)   # (N, 24)
        self.rt = data["rt"].astype(np.float32)   # (N, 288)
        # Backward compat: old npz files without calendar key
        if "calendar" in data:
            self.calendar = data["calendar"].astype(np.int64)  # (N, 4)
        else:
            self.calendar = np.zeros((len(self.da), 4), dtype=np.int64)

    def __len__(self):
        return len(self.da)

    def __getitem__(self, idx):
        da_24 = self.da[idx]               # (24,)
        rt_288 = self.rt[idx]              # (288,)

        # Upsample DA to 288 by repeating each hourly value 12 times.
        # This preserves the step-function structure of the DA clearing.
        da_288 = np.repeat(da_24, 12)      # (288,)

        da_cond   = torch.from_numpy(da_288).unsqueeze(0)              # (1, 288)
        rt_target = torch.from_numpy(rt_288).unsqueeze(0)              # (1, 288)
        cal_feat  = torch.from_numpy(self.calendar[idx])               # (4,)
        return da_cond, rt_target, cal_feat


def make_loaders(data_dir: str, batch_size: int, num_workers: int = 4):
    import os
    loaders = {}
    for split in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{split}.npz")
        ds = PriceDataset(path)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
