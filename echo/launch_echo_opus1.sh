#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torchaudio/lib:$LD_LIBRARY_PATH
export HF_HOME=/hy-tmp/hf_cache
export TORCH_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/hy-tmp/dimsum/scripts:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TAU_STAR_JSON=/hy-tmp/dimsum/outputs/echo-opus-1/tau_star_stable.json
export OUTPUT_DIR=/hy-tmp/dimsum/outputs/echo-opus-1
export NUM_EPOCHS=15
export T0_EPOCHS=4
export T_MULT=2
export LR=5e-6
export TAU_LR=3e-4
export BATCH_SIZE=6
export GRAD_ACCUM=3
export SPARSITY_LAMBDA=0.01
export INIT_K_RATIO=0.4
export USE_EMA=0
export USE_FLASH=0  # BUG-3 防护: Triton 3.3.0 不支持 Blackwell SM_120, 升级 3.3.1+ 后再开

cd /hy-tmp/dimsum

# Kill leftovers (use exact match to avoid self-kill)
pkill -9 -f 'train_echo_opus1\.py' 2>/dev/null
sleep 1

echo "[launch] Starting echo-opus-1 @ $(date)"
echo "[launch] batch=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM}, epochs=${NUM_EPOCHS}"
echo "[launch] k_ratio=${INIT_K_RATIO}, sparsity_λ=${SPARSITY_LAMBDA}, T0=${T0_EPOCHS}ep, Tmult=${T_MULT}"
echo "[launch] Log: /hy-tmp/dimsum/train_echo_opus1.log"

python3 -u scripts/train_echo_opus1.py > /hy-tmp/dimsum/train_echo_opus1.log 2>&1

echo "[launch] EXIT CODE: $?"