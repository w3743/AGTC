/opt/venv_mujoco315_smoke/bin/python -B -m unittest discover -s '/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/tools' -p 'test_*.py' -v > /mnt/c/pioneer/agtc-native-acceptance-backup/root_docs/validation_native_v2_frozen/wsl_stdlib.log 2>&1
agtc_rc=$?
printf '%s\n' "$agtc_rc" > /mnt/c/pioneer/agtc-native-acceptance-backup/root_docs/validation_native_v2_frozen/wsl_stdlib.exit
tail -5 /mnt/c/pioneer/agtc-native-acceptance-backup/root_docs/validation_native_v2_frozen/wsl_stdlib.log
exit "$agtc_rc"
