#!/bin/bash

set -euo pipefail

# --------- Config (override via env if needed) ---------
PARTITION=${PARTITION:-gpu}
GPUS_PER_JOB=${GPUS_PER_JOB:-1}
CPUS_PER_GPU=${CPUS_PER_GPU:-4}
MEMORY=${MEMORY:-32G}
TIME_LIMIT=${TIME_LIMIT:-24:00:00}
CONDA_ENV=${CONDA_ENV:-gdmicro37}
WANDB_ENABLE=${WANDB_ENABLE:-0}
WANDB_MODE=${WANDB_MODE:-offline}
LOG_DIR=${LOG_DIR:-logs/parallel_demo}

mkdir -p "$LOG_DIR"

# format: job_name|command
JOBS=(
  "T2D_10fold|python GDmicro.py -i Input_files/T2D_10fold.csv -t 1 -d T2D -o GDmicro_new_pipe/T2D_10fold --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "CRC_10fold|python GDmicro.py -i Input_files/CRC_10fold.csv -t 1 -d CRC -o GDmicro_new_pipe/CRC_10fold --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "IBD_10fold|python GDmicro.py -i Input_files/IBD_10fold.csv -t 1 -d IBD -o GDmicro_new_pipe/IBD_10fold --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "Obt_10fold|python GDmicro.py -i Input_files/Obt_10fold.csv -t 1 -d Obt -o GDmicro_new_pipe/Obt_10fold --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "cirrhosis_10fold|python GDmicro.py -i Input_files/cirrhosis_10fold.csv -t 1 -d cirrhosis -o GDmicro_new_pipe/cirrhosis_10fold --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "CRC_FRA_LOSO|python GDmicro.py -i Input_files/CRC_FRA_LOSO.csv -s 10 -d CRC -o GDmicro_new_pipe/CRC_FRA_LOSO --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "CRC_GER_LOSO|python GDmicro.py -i Input_files/CRC_GER_LOSO.csv -s 10 -d CRC -o GDmicro_new_pipe/CRC_GER_LOSO --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "CRC_CHI_LOSO|python GDmicro.py -i Input_files/CRC_CHI_LOSO.csv -s 10 -d CRC -o GDmicro_new_pipe/CRC_CHI_LOSO --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "CRC_USA_LOSO|python GDmicro.py -i Input_files/CRC_USA_LOSO.csv -s 10 -d CRC -o GDmicro_new_pipe/CRC_USA_LOSO --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "CRC_AUS_LOSO|python GDmicro.py -i Input_files/CRC_AUS_LOSO.csv -s 10 -d CRC -o GDmicro_new_pipe/CRC_AUS_LOSO --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "IBD_UK_LOSO|python GDmicro.py -i Input_files/IBD_UK_LOSO.csv -s 10 -d IBD -o GDmicro_new_pipe/IBD_UK_LOSO --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "IBD_DK_LOSO|python GDmicro.py -i Input_files/IBD_DK_LOSO.csv -s 10 -d IBD -o GDmicro_new_pipe/IBD_DK_LOSO --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
  "CRC_IND_Additional|python GDmicro.py -i Input_files/CRC_IND_additional.csv -s 10 -b 32 -d CRC -o GDmicro_new_pipe/CRC_IND_Additional --wandb ${WANDB_ENABLE} --wandb_mode ${WANDB_MODE}"
)

submit_one() {
  local job_name="$1"
  local command="$2"

  sbatch \
    --partition="$PARTITION" \
    --gres="gpu:${GPUS_PER_JOB}" \
    --cpus-per-gpu="$CPUS_PER_GPU" \
    --mem="$MEMORY" \
    --time="24:00:00" \
    --job-name="GDm_${job_name}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --wrap="bash -lc '
      set -euo pipefail
      cd \"$PWD\"
      if command -v ml >/dev/null 2>&1; then ml Mamba/23.11.0-0 || true; fi
      eval \"\$(conda shell.bash hook)\"
      conda activate ${CONDA_ENV}
      export MPLBACKEND=Agg
      export WANDB_MODE=${WANDB_MODE}
      echo Running ${job_name}
      ${command}
    '"
}

for item in "${JOBS[@]}"; do
  job_name="${item%%|*}"
  command="${item#*|}"
  submit_one "$job_name" "$command"
  echo "Submitted: $job_name"
done

echo "All jobs submitted. Check queue with: squeue -u $USER"
