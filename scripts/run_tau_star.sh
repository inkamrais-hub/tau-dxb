#!/bin/bash
cd /root/dimsum && python3.11 -u scripts/compute_tau_star_b.py > /tmp/tau_b.txt 2>&1
echo "EXIT: $?"
ls -la /root/dimsum/outputs/whisper_small_tau_star_b.json 2>&1
