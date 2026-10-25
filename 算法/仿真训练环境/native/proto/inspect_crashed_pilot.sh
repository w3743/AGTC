#!/bin/bash
export PATH=/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/runs/lstm_pilot_5ms || exit 1
ls -la | head -12
echo '--- identity ---'
cat identity.json 2>/dev/null | head -40
echo '--- metrics last 6 ---'
tail -6 metrics.jsonl 2>/dev/null
echo '--- log tail 50 ---'
tail -50 ../lstm_pilot_5ms.log 2>/dev/null
