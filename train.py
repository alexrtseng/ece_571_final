"""
Train the DDPM diffusion model.

Usage:
    python train.py --config configs/default.yaml
"""

import argparse
import os
import math
import time
import yaml
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data.dataset import make_loaders
from models import UNet1D, NoiseSchedule, DDPMTrainer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Device: {device}")

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Data
    loaders = make_loaders(cfg["data_dir"], cfg["batch_size"])
    print(f"Train batches: {len(loaders['train'])}  Val batches: {len(loaders['val'])}")

    use_calendar = cfg.get("use_calendar", False)

    # Model & schedule
    model = UNet1D(
        base_channels=cfg["base_channels"],
        num_res_blocks=cfg["num_res_blocks"],
        attn_heads=cfg["attn_heads"],
        use_calendar=use_calendar,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    schedule = NoiseSchedule(T=cfg["T"], schedule=cfg["schedule"]).to(device)
    trainer = DDPMTrainer(model, schedule).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    use_amp = cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    total_steps = len(loaders["train"]) * cfg["epochs"]
    warmup_steps = cfg.get("lr_warmup_steps", 500)
    patience = cfg.get("patience", None)
    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epoch_times = []
    train_start = time.perf_counter()

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.perf_counter()
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for da_cond, rt, cal_feat in tqdm(loaders["train"], desc=f"Epoch {epoch}", leave=False):
            da_cond = da_cond.to(device)
            rt = rt.to(device)
            cal = cal_feat.to(device) if use_calendar else None

            # LR schedule
            lr = cosine_lr(global_step, warmup_steps, total_steps, cfg["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss = trainer.loss(rt, da_cond, cal=cal)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            global_step += 1

        train_loss /= len(loaders["train"])

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for da_cond, rt, cal_feat in loaders["val"]:
                da_cond = da_cond.to(device)
                rt = rt.to(device)
                cal = cal_feat.to(device) if use_calendar else None
                with autocast(enabled=use_amp):
                    val_loss += trainer.loss(rt, da_cond, cal=cal).item()
        val_loss /= len(loaders["val"])

        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_times.append(epoch_elapsed)
        print(f"Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f} | lr={lr:.2e} | {epoch_elapsed:.1f}s")

        # ── Checkpoints ───────────────────────────────────────────────────
        if epoch % cfg.get("save_every", 10) == 0:
            ckpt = os.path.join(out_dir, f"ckpt_epoch{epoch:04d}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss}, ckpt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss},
                       os.path.join(out_dir, "best.pt"))
        else:
            epochs_without_improvement += 1

        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    total_elapsed = time.perf_counter() - train_start
    n_epochs_run = len(epoch_times)
    avg_epoch = sum(epoch_times) / n_epochs_run if n_epochs_run else 0.0
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_elapsed:.1f}s  |  "
          f"Avg per epoch: {avg_epoch:.1f}s  |  Epochs run: {n_epochs_run}")

    stats_path = os.path.join(out_dir, "training_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        f.write(f"epochs_run: {n_epochs_run}\n")
        f.write(f"total_training_time_s: {total_elapsed:.2f}\n")
        f.write(f"avg_epoch_time_s: {avg_epoch:.2f}\n")
        for i, t in enumerate(epoch_times, 1):
            f.write(f"  epoch_{i:04d}_time_s: {t:.2f}\n")
    print(f"Training stats saved to {stats_path}")


if __name__ == "__main__":
    main()
