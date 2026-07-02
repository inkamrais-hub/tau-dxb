#!/bin/bash
pkill -f train_erp1 2>/dev/null || true
sleep 2
mv /hy-tmp/dimsum/erp1.log /hy-tmp/dimsum/erp1.log.old4 2>/dev/null || true
cd /hy-tmp/dimsum
nohup bash /hy-tmp/dimsum/scripts/wait_and_start_erp1.sh > /hy-tmp/dimsum/wait_and_start_erp1.log 2>&1 &
sleep 10
ps aux | grep train_erp1 | grep -v grep
echo "---"
tail -30 /hy-tmp/dimsum/erp1.log
echo "---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
