#!/bin/bash
# erp1 -> erp2 autonomous pipeline for DimSum
# Runs on remote server, monitors erp1, auto-launches erp2.
set -e

ERP1_LOG=/hy-tmp/dimsum/erp1.log.old5
ERP2_LOG=/hy-tmp/dimsum/erp2.log
OUTPUT_DIR=/hy-tmp/dimsum/outputs/erp2_tau_star.json
COMPLETE_MARKER=/hy-tmp/dimsum/.pipeline_done

echo "[pipeline $(date)] Starting erp1 -> erp2 pipeline"
echo "[pipeline $(date)] ERP1_LOG=$ERP1_LOG"

# If marker exists from previous run, clean
rm -f "$COMPLETE_MARKER"

# If erp2 output already exists, we're done
if [ -f "$OUTPUT_DIR" ]; then
    echo "[pipeline $(date)] erp2 already complete at $OUTPUT_DIR"
    touch "$COMPLETE_MARKER"
    exit 0
fi

# --- PHASE 1: Wait for erp1 ---
echo "[pipeline $(date)] PHASE 1: Waiting for erp1 to complete..."
while true; do
    # Check if train_erp1 process still running
    if pgrep -f "train_erp1.py" > /dev/null; then
        # Still running, wait
        if [ -f "$ERP1_LOG" ]; then
            LAST_LINE=$(tail -1 "$ERP1_LOG" 2>/dev/null || echo "")
            echo "[pipeline $(date)] erp1 running... $LAST_LINE"
        fi
        sleep 60
    else
        echo "[pipeline $(date)] erp1 process not found. Checking output..."
        if [ -d /hy-tmp/dimsum/outputs/base-erp1/best_model ]; then
            echo "[pipeline $(date)] erp1 output found at /hy-tmp/dimsum/outputs/base-erp1/best_model/"
            echo "[pipeline $(date)] erp1 stage complete!"
            break
        elif [ -d /hy-tmp/dimsum/outputs/base-erp1/final_model ]; then
            echo "[pipeline $(date)] erp1 output (final) found"
            break
        else
            echo "[pipeline $(date)] WARNING: erp1 ended but no output found. Last 20 lines:"
            tail -20 "$ERP1_LOG" 2>/dev/null || echo "(log missing)"
            echo "[pipeline $(date)] Maybe restart needed. Exiting."
            exit 1
        fi
    fi
done

# Brief pause for system to settle
sleep 10

# --- PHASE 2: Run erp2 (tau-star estimation) ---
echo "[pipeline $(date)] PHASE 2: Starting erp2 (τ* estimation, 5 runs)..."

export HF_HOME=/hy-tmp/hf_cache
export TORCH_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com

python3 -u /hy-tmp/dimsum/scripts/compute_tau_star_iop.py >> "$ERP2_LOG" 2>&1
ERP2_EXIT=$?

if [ $ERP2_EXIT -ne 0 ]; then
    echo "[pipeline $(date)] ERROR: erp2 failed with exit code $ERP2_EXIT"
    tail -20 "$ERP2_LOG"
    exit 1
fi

echo "[pipeline $(date)] erp2 complete! Output at $OUTPUT_DIR"
echo "[pipeline $(date)] Summary:"
python3 -c "
import json
with open('$OUTPUT_DIR') as f:
    data = json.load(f)
for m in data:
    print(f'  {m[\"module_name\"]}: tau_mean={m[\"tau_mean\"]}, tau_std={m[\"tau_std\"]}')
"

touch "$COMPLETE_MARKER"
echo "[pipeline $(date)] Pipeline complete. Server can be shut down."
