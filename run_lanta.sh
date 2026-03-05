#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH -J GDmicro_experiment
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

ml Mamba/23.11.0-0
eval "$(conda shell.bash hook)"
conda activate gdmicro37

# safe to enable strict mode AFTER conda
set -euo pipefail

export MPLBACKEND=Agg
export WANDB_MODE=${WANDB_MODE:-online}

echo "Welcome to LANTA"
echo "Run directory: $PWD"
echo "Job ID: $SLURM_JOB_ID"

bash run_GDmicro_demo.sh