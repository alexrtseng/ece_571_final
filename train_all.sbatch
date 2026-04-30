#!/bin/bash -l
#SBATCH --job-name=rt-diffusion
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL

set -euo pipefail

mkdir -p logs

echo "======================================================"
echo "Node:    $(hostname)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Started: $(date)"
echo "======================================================"

# ── Modules ───────────────────────────────────────────────────────────────────
module purge

module load proxy/default || { echo "WARNING: proxy/default not available"; }

# Try common Gurobi versions on Della; update if the cluster has a newer one.
module load gurobi/11.0.1 2>/dev/null || \
module load gurobi/10.0.3 2>/dev/null || \
module load gurobi/9.0.1  2>/dev/null || \
    echo "WARNING: No compatible Gurobi module found. Battery eval will fail."

# ── Python environment ────────────────────────────────────────────────────────
# Set up on Della once with:
#   module load python/3.11
#   python -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements.txt
source .venv/bin/activate

# ── Thread throttles ──────────────────────────────────────────────────────────
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export TORCH_MATMUL_PRECISION=tf32

# ── Runner knobs (override with --export on sbatch command line) ──────────────
# Which configs to train — comma-separated names, or "all"
export CONFIGS=${CONFIGS:-all}
# DDPM samples per test-set day (sampling + backtest)
export N_SAMPLES=${N_SAMPLES:-20}
# Backtest date range (must lie within the test split: 2025-01 → 2025-10)
export BACKTEST_START=${BACKTEST_START:-2025-01-01}
export BACKTEST_END=${BACKTEST_END:-2025-10-31}

PYBIN="$(command -v python)"

# ── Diagnostics ───────────────────────────────────────────────────────────────
nvidia-smi || true
$PYBIN -c "
import sys, torch
print(sys.executable)
print('Torch', torch.__version__,
      '| CUDA:', torch.cuda.is_available(),
      '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
"

# ── Build runner args ─────────────────────────────────────────────────────────
RUNNER_ARGS=(
    "--n_samples"      "$N_SAMPLES"
    "--backtest_start" "$BACKTEST_START"
    "--backtest_end"   "$BACKTEST_END"
)

if [ "$CONFIGS" != "all" ]; then
    RUNNER_ARGS+=("--configs" "$CONFIGS")
fi

# ── Launch ────────────────────────────────────────────────────────────────────
srun -u "$PYBIN" runner.py "${RUNNER_ARGS[@]}"

echo ""
echo "======================================================"
echo "All runs complete: $(date)"
echo "======================================================"
