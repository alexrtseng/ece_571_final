#!/bin/bash -l
#SBATCH --job-name=rt-diffusion
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL

set -euo pipefail

echo "======================================================"
echo "Node:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Started: $(date)"
echo "======================================================"

mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load proxy/default || { echo "WARNING: proxy/default not available"; }

# Load Gurobi — check what version is available with: module avail gurobi
# Update the version string below to match what Princeton Della has.
module load gurobi/11.0.1 || module load gurobi/10.0.3 || \
    { echo "WARNING: No compatible Gurobi module found. Battery eval will fail."; }

# Activate the project virtual environment.
# Create it on Della first with:
#   module load python/3.11  (or whichever Python Della provides)
#   python -m venv .venv
#   source .venv/bin/activate && pip install -r requirements.txt
source .venv/bin/activate

# Thread throttles
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TORCH_MATMUL_PRECISION=tf32

PYBIN="$(command -v python)"

# Quick diagnostics
nvidia-smi || true
$PYBIN -c "import sys, torch; print(sys.executable); \
    print('Torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), \
    '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

# ── Helper: sample + evaluate one run ─────────────────────────────────────────
# Usage: run_eval <config> <out_dir> <data_dir>
run_eval() {
    local cfg=$1
    local out=$2
    local data=$3

    echo ""
    echo ">>> Sampling: $cfg"
    srun -u "$PYBIN" sample.py \
        --config "$cfg" \
        --checkpoint "$out/best.pt" \
        --n_samples 10 \
        --all_test

    srun -u "$PYBIN" sample.py \
        --config "$cfg" \
        --checkpoint "$out/best.pt" \
        --n_samples 10 \
        --all_test \
        --ddim

    echo ""
    echo ">>> Evaluating: $cfg"
    srun -u "$PYBIN" evaluate.py \
        --config "$cfg" \
        --ddpm_samples "$out/samples_ddpm.npz" \
        --ddim_samples "$out/samples_ddim.npz" \
        --real_npz     "$data/test.npz" \
        --out_dir      "$out/eval"
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. Single-node runs  (default + calendar)
#    Preprocess once — both configs share ./data
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "======================================================"
echo "STEP 1/6: Preprocess single-node data (pnode 48806)"
echo "======================================================"
srun -u "$PYBIN" data/preprocess.py --config configs/default.yaml

echo ""
echo "======================================================"
echo "STEP 2/6: Train — default (DA only, single node)"
echo "======================================================"
srun -u "$PYBIN" train.py --config configs/default.yaml
run_eval configs/default.yaml runs data

echo ""
echo "======================================================"
echo "STEP 3/6: Train — calendar (DA + calendar, single node)"
echo "======================================================"
srun -u "$PYBIN" train.py --config configs/calendar.yaml
run_eval configs/calendar.yaml runs/calendar data

# ══════════════════════════════════════════════════════════════════════════════
# 2. Multi-node runs  (big + big_calendar)
#    Preprocess once — both share ./data/big
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "======================================================"
echo "STEP 4/6: Preprocess multi-node data (15 nodes)"
echo "======================================================"
srun -u "$PYBIN" data/preprocess.py --config configs/big.yaml

echo ""
echo "======================================================"
echo "STEP 5/6: Train — big (DA only, 15 nodes)"
echo "======================================================"
srun -u "$PYBIN" train.py --config configs/big.yaml
run_eval configs/big.yaml runs/big data/big

echo ""
echo "======================================================"
echo "STEP 6/6: Train — big_calendar (DA + calendar, 15 nodes)"
echo "======================================================"
srun -u "$PYBIN" train.py --config configs/big_calendar.yaml
run_eval configs/big_calendar.yaml runs/big_calendar data/big

echo ""
echo "======================================================"
echo "All runs complete: $(date)"
echo "Results:"
echo "  Single-node baseline:   runs/eval/"
echo "  Single-node calendar:   runs/calendar/eval/"
echo "  Multi-node baseline:    runs/big/eval/"
echo "  Multi-node calendar:    runs/big_calendar/eval/"
echo "======================================================"
