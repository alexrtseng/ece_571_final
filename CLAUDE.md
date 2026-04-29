# ECE 573 Final Project — RT Price Diffusion Model

## Project Goal
Conditional stochastic super-resolution of electricity prices: given 24 hourly day-ahead (DA) clearing prices, generate realistic 288 five-minute real-time (RT) prices for the same day using diffusion models (DDPM and DDIM).

**Purpose**: Not point-forecast accuracy. The goal is statistical realism — generated RT traces should exhibit realistic volatility, spikes, and autocorrelation structure, enabling long-horizon asset valuation (e.g., storage dispatch) from capacity expansion model outputs.

## Data
- `pjm_lmps_da/` — hourly DA LMPs, columns: `datetime_beginning_utc, pnode_id, total_lmp_da, ...`
- `pjm_lmps/` — 5-min RT LMPs, columns: `datetime_beginning_utc, pnode_id, total_lmp_rt, ...`
- 38 pnodes, data from 2020-10 through 2025-10
- `inferred_pnodes.csv` — maps pnode_id to human-readable bus names

## Architecture
- **Model**: 1D U-Net, pure PyTorch (no diffusion libraries)
- **Input**: `(B, 2, 288)` — channel 0 is noisy RT, channel 1 is DA upsampled by repeating each hourly value 12 times
- **Output**: `(B, 1, 288)` — predicted noise ε
- **Samplers**: DDPM (T=1000 steps) and DDIM (S=50 steps, deterministic, same weights)

## Normalization
Two-stage pipeline applied to all prices (DA and RT):
1. `arcsinh(x / 100.0)` — compresses spikes, handles negatives
2. Z-score using mean/std fit on training data only
Scaler params saved to `data/scaler.npz`.

## Data Split (chronological)
- Train: 2020-10 → 2023-12
- Val: 2024-01 → 2024-12
- Test: 2025-01 → 2025-10

## Repo Structure
```
configs/default.yaml   — all hyperparameters
data/preprocess.py     — build train/val/test .npz files
data/dataset.py        — PyTorch Dataset
models/schedule.py     — cosine noise schedule
models/unet.py         — 1D U-Net backbone
models/diffusion.py    — DDPM trainer, DDPM/DDIM samplers
train.py               — training loop
sample.py              — generate samples from checkpoint
evaluate.py            — volatility/spike stats + plots
requirements.txt
```

## Key Commands
```bash
python data/preprocess.py --config configs/default.yaml
python train.py --config configs/default.yaml
python sample.py --config configs/default.yaml --checkpoint runs/best.pt --n_samples 10
python sample.py --config configs/default.yaml --checkpoint runs/best.pt --n_samples 10 --ddim
python evaluate.py --config configs/default.yaml --samples_path runs/samples_ddpm.npz
```

## Config Keys
`pnode_id`, `data_dir`, `out_dir`, `T` (diffusion steps), `epochs`, `batch_size`, `lr`, `base_channels`, `ddim_steps`

## Evaluation
Primary downstream metric: battery dispatch profit delta (real vs generated prices). Also tracked: volatility stats, spike frequency (>$100, >$500/MWh), ACF comparison, distribution plots.
