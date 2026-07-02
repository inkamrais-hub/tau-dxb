#!/bin/bash
export BATCH_SIZE=6
export GRAD_ACCUM=2
cd /hy-tmp/dimsum
rm -f erp3.log
nohup python3 -u scripts/train_erp3.py > erp3.log 2>&1 &
echo PID=$!
