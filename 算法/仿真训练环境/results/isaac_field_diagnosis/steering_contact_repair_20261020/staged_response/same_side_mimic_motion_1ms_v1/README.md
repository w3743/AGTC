# Same-side mimic 1 ms response pair (prepared; no motion launched yet)

This independent version uses the frozen `response_runner.py` collector logic and the public `same_side_mimic_v2` hook. It is pinned to the same eight actions, seed 90231, reset state, `1 s` zero-action settle + `2 s` drive + `5 s` brake, flat mesh, elliptic cone, `impratio=1`, `mu=.8`, `kf=1000`, P=.04, side I=.1 and 14 per-wheel 2 Nm effort caps. The only treatment difference is whether the same-side hook is enabled. The original 2 ms source, output and protocol remain unchanged.

This pair changes the fixed time resolution from the previous pair only: `physics_dt=.001 s`, decimation 50, policy `dt=.05 s`. It imports the same source collector (`staged_response/staged_motion_response_probe.py`), same hook, and unchanged `same_side_mimic_v2/protocol.json`; the frozen residual thresholds are unchanged. Expected per run: 1,280 policy rows and 64,000 native physics qd rows; each case's last one-second drive window has 1,000 native qd rows × 12 pairs and 20 policy rows × 12 pairs.

CPU preflight:

```bash
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境" && bash -n results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/same_side_mimic_motion_1ms_v1/run_static_preflight.sh && bash results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/same_side_mimic_motion_1ms_v1/run_static_preflight.sh'
```

After the parent explicitly confirms the GPU queue is free and authorizes these two cases, run them serially:

```bash
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境" && RUN_STAGED_MIMIC_RESPONSE_1MS=1 bash results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/same_side_mimic_motion_1ms_v1/run_baseline_1ms_elliptic_1_mu0_8_kf1000.sh'
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境" && RUN_STAGED_MIMIC_RESPONSE_1MS=1 bash results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/same_side_mimic_motion_1ms_v1/run_mimic_v2_1ms_elliptic_1_mu0_8_kf1000.sh'
```

The launch token guards the run shells; the runner records a fresh actual exit JSON per case and refuses existing non-empty outputs. Do not select production or call this friction calibration from response alone. The planned 1 ms versus .5 ms yaw convergence belongs to the separate frozen protocol.
