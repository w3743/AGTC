/opt/venv/bin/python -B -m unittest discover -s '/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/tools' -p 'test_*.py' -v > /mnt/c/pioneer/agtc-native-acceptance-backup/root_docs/orchestration32_wsl.log 2>&1
agtc_rc=$?
printf '%s\n' "$agtc_rc" > /mnt/c/pioneer/agtc-native-acceptance-backup/root_docs/orchestration32_wsl.exit
exit "$agtc_rc"
