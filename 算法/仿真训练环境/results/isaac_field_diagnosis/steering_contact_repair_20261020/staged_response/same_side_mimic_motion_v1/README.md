# Same-side mimic motion pair (prepared)

This pair reuses `../staged_motion_response_probe.py` for the same eight actions and `1 s` zero-action settle, `2 s` command, `5 s` braking schedule. Both runs use seed `90231`, the same flat Field mesh, `dt=0.002 s`, decimation `25`, policy `dt=0.05 s`, elliptic cone, `impratio=1`, config `mu=0.8`, runtime `kf=1000`, the existing `P=0.04`, side `I=0.1`, and all 14 wheel actuators capped at `2 Nm`. The two fresh runners differ only in whether the same-side mimic builder hook is installed.

Both modes record the same 12 wheel-0/follower relationships. Native `joint_qd_start` and `joint_q_start` offsets are resolved once after model creation; a post-actuator callback reads the current `NewtonManager.get_state_0().joint_qd` and writes 12 pair residuals per physics step to a Warp tape. The callback does no CPU synchronization. Each policy trajectory row records all 12 `q` pair residuals from the native state after `env.step`. The candidate additionally records the actual Newton model, host MuJoCo, and MJWarp equality counts and solver parameters; it stops before motion unless all 12 expected rows and their configured `solref`/`solimp` survive.

The last second of each 2 s drive is summarized against the v2 mimic protocol: native qd residual absolute p95 at most `0.09259 rad/s`, absolute max at most `0.46296 rad/s`, policy q residual absolute max at most `0.02 rad`, finite samples, and the 14-wheel native effort cap. These are response gates, not evidence of real-vehicle calibration or a production repair. Compare the baseline and candidate only after confirming all eight pre-settle state hashes match; the existing 5 ms pyramidal runs are not a single-variable baseline for this 2 ms elliptic pair.

## CPU static check

```bash
wsl.exe -d Ubuntu-24.04 -- bash -n results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/same_side_mimic_motion_v1/run_baseline_2ms_elliptic_1_mu0_8_kf1000.sh
wsl.exe -d Ubuntu-24.04 -- bash -n results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/same_side_mimic_motion_v1/run_mimic_v2_2ms_elliptic_1_mu0_8_kf1000.sh
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境" && scripts/isaaclab_python.sh results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/same_side_mimic_motion_v1/response_runner.py --static-check'
```

## Authorized motion commands

These fixed commands refuse any prior output, log, or exit record. Run sequentially only while the GPU queue is reserved for this pair:

```bash
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境" && RUN_STAGED_MIMIC_RESPONSE=1 bash results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/same_side_mimic_motion_v1/run_baseline_2ms_elliptic_1_mu0_8_kf1000.sh'
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/e/Pioneer/先进通用履带底盘工程/算法/仿真训练环境" && RUN_STAGED_MIMIC_RESPONSE=1 bash results/isaac_field_diagnosis/steering_contact_repair_20261020/staged_response/same_side_mimic_motion_v1/run_mimic_v2_2ms_elliptic_1_mu0_8_kf1000.sh'
```
