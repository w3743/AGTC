# Timeout trajectory audit (2026-10-20)

Read-only audit of saved final development evaluation trajectories. No simulation, training, GPU, source edit, or modification of the original evaluation files.

Protocol: 20s max episode (400 steps at 0.05s); goal tolerance 0.4m, hold 0.3s.

All 8 seeds:

| Set | Seed | Start (m) | End (m) | Min (m) | Elapsed (s) | Result | Entered 0.4m / max hold (s) |
|---|---:|---:|---:|---:|---:|---|---|
| free | 87001 | 2.418 | 1.912 | 1.563 | 20.00 | time_limit | no / 0.00 |
| free | 87002 | 1.507 | 2.350 | 0.819 | 20.00 | time_limit | no / 0.00 |
| free | 87003 | 1.425 | 1.552 | 1.235 | 20.00 | time_limit | no / 0.00 |
| free | 87004 | 1.208 | 0.280 | 0.273 | 1.85 | goal_reached | yes / 0.30 |
| field | 82001 | 1.260 | 1.667 | 1.028 | 20.00 | time_limit | no / 0.00 |
| field | 82002 | 2.142 | 2.812 | 1.786 | 20.00 | time_limit | no / 0.00 |
| field | 82003 | 2.140 | 0.378 | 0.346 | 5.05 | goal_reached | yes / 0.30 |
| field | 82004 | 1.645 | 2.350 | 1.414 | 20.00 | time_limit | no / 0.00 |

Timeout windows: start/end distances and changes are actual saved samples; positive distance change means the robot ended farther from the goal; a negative change means it ended closer. Slope is OLS distance vs time. Actual v/w are chassis state linear m/s and angular rad/s; L/R values are commanded track speeds in m/s.

| Set | Seed | End window | Distance start→end (m) | Δ distance (m) | Slope (m/s) | Mean actual v / w | Mean L / R command (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| free | 87001 | 5s | 1.654→1.912 | +0.258 | +0.044 | +0.182 / -0.061 | +0.333 / +0.120 |
| free | 87001 | 10s | 1.594→1.912 | +0.319 | +0.019 | +0.057 / -0.052 | +0.166 / -0.001 |
| free | 87002 | 5s | 2.052→2.350 | +0.299 | +0.061 | -0.068 / -0.038 | -0.036 / -0.145 |
| free | 87002 | 10s | 1.783→2.350 | +0.567 | +0.054 | -0.063 / -0.038 | -0.030 / -0.139 |
| free | 87003 | 5s | 1.435→1.552 | +0.117 | +0.021 | -0.095 / -0.033 | -0.051 / -0.141 |
| free | 87003 | 10s | 1.505→1.552 | +0.048 | +0.002 | -0.082 / -0.033 | -0.049 / -0.136 |
| field | 82001 | 5s | 1.293→1.667 | +0.374 | +0.072 | -0.117 / -0.034 | -0.058 / -0.135 |
| field | 82001 | 10s | 1.221→1.667 | +0.446 | +0.045 | -0.118 / -0.034 | -0.056 / -0.138 |
| field | 82002 | 5s | 1.916→2.812 | +0.896 | +0.183 | +0.276 / -0.069 | +0.460 / +0.195 |
| field | 82002 | 10s | 1.863→2.812 | +0.950 | +0.088 | +0.161 / -0.063 | +0.321 / +0.090 |
| field | 82004 | 5s | 1.839→2.350 | +0.510 | +0.104 | +0.317 / -0.062 | +0.375 / +0.145 |
| field | 82004 | 10s | 1.840→2.350 | +0.509 | +0.024 | +0.166 / -0.055 | +0.227 / +0.029 |

Interpretation:

- The six timeouts used the full 20s. None entered the 0.4m tolerance, so none was close enough to be a missed 0.3s hold.
- The last-window distance trends distinguish continued progress from worsening/stalling; they do not establish what a longer rollout would have done.
- The compact trajectory stores left/right speed commands and chassis `actual_v_w`, but not separate measured left/right track feedback.
- `task.py` exposes `episode_time_ratio=min(1,time/max_time)` to the policy. Changing max_time to 40/60s changes this input before t=20, so it is a mixed comparison. For a clean timeout test, keep this input at the original 20s scaling while extending only the termination cap; compare same model, reset initial condition, goal, and environment at 20/40/60s in real rollouts. No extension was run here.

Input SHA-256:
- `protocol.json`: `2735ca85917643e49e27088957d00b1576f37902ce9664203253ae698181d5b6`
- `evaluation_free_final_87001_87004.json`: `752ec2fb20a8041a0dff27328cd95a9c0f70894c55dd0042d7f6aeb17f583e1a`
- `evaluation_field_final_82001_82004.json`: `1da0f1fff010bdc8f21e9db18e2fd5de413772e1f6b2789654b85e67971ad45b`

