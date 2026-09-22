#!/bin/bash
#SBATCH --job-name=gan-reg-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

MODE="${MODE:-wgan_gp}"
STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GPUS="${SLURM_GPUS_ON_NODE:-4}"

# Configure the cluster-specific Python/CUDA environment before this line.
# Example:
# module load python
# source .venv/bin/activate

torchrun \
  --standalone \
  --nproc-per-node="${GPUS}" \
  -m gan_bench.train \
  --mode "${MODE}" \
  --device cuda \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --output "results/${MODE}-${GPUS}gpu-${SLURM_JOB_ID}.json"
