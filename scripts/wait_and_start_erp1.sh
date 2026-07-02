#!/bin/bash
set -e
LOG=/hy-tmp/dimsum/erp1.log
PREPARE_LOG=/hy-tmp/dimsum/prepare_data.log
TRAIN_JSONL=/hy-tmp/dimsum/data/prepared/train.jsonl
VAL_JSONL=/hy-tmp/dimsum/data/prepared/val.jsonl

echo "[$(date)] Waiting for dataset preparation..." | tee -a "$LOG"
while [ ! -f "$TRAIN_JSONL" ] || [ ! -f "$VAL_JSONL" ]; do
  sleep 30
  if ! pgrep -f prepare_data_remote.py > /dev/null; then
    echo "[$(date)] prepare_data_remote.py no longer running; checking files..." | tee -a "$LOG"
    if [ ! -f "$TRAIN_JSONL" ] || [ ! -f "$VAL_JSONL" ]; then
      echo "[$(date)] ERROR: prepare script exited but JSONL files missing. Check $PREPARE_LOG" | tee -a "$LOG"
      exit 1
    fi
    break
  fi
done

echo "[$(date)] Dataset ready. Train samples: $(wc -l < $TRAIN_JSONL), Val samples: $(wc -l < $VAL_JSONL)" | tee -a "$LOG"
echo "[$(date)] Starting erp1 training..." | tee -a "$LOG"

export HF_HOME=/hy-tmp/hf_cache
export TORCH_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TRAIN_JSONL=$TRAIN_JSONL
export VAL_JSONL=$VAL_JSONL
export OUTPUT_DIR=/hy-tmp/dimsum/outputs/base-erp1
export MODEL_NAME=/hy-tmp/whisper-small-local
export BATCH_SIZE=4
export GRAD_ACCUM=4
export LR=1e-4
export NUM_EPOCHS=2
export USE_BF16=1
export USE_EMA=1
export T_0=200
export T_MULT=2

python3 -u /hy-tmp/dimsum/scripts/train_erp1.py >> "$LOG" 2>&1

echo "[$(date)] erp1 training finished." | tee -a "$LOG"
