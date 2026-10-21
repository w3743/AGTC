# Physics timestep convergence: native wheel-speed and contact summary

CPU-only post-processing of the three existing v2 raw runs; no simulation was rerun. The comparison window is the final 1.0 s of differential drive, `simulation_time_s in (8.0, 9.0]`.

All cases share the same reset-before-settle state hash and fixed comparison-contract hash. The experiment used a 1 s zero-action settle with PI still enabled and no integral reset at settle end, followed by 8 s of `[0.1, -0.1, 0, 0]` drive. This was not a P-only settle.

## Native wheel response

QD/effort summaries pool all seven wheel samples per side over the window. `Δqd RMS` pools adjacent native physics-step differences for each wheel; side-mean qd variability is reported separately in JSON.

| dt | side | native qd mean ± std (rad/s) | range (rad/s) | adjacent Δqd RMS (rad/s) | mean abs target error (rad/s) | effort mean ± std (Nm) |
|---:|:---|---:|---:|---:|---:|---:|
| 5 ms | left | 9.6848 ± 7.2444 | 22.5338 | 7.4895 | 6.4982 | 0.4193 ± 0.2895 |
| 5 ms | right | -9.1684 ± 6.3642 | 21.4358 | 6.8965 | 5.6652 | -0.4163 ± 0.2565 |
| 2 ms | left | 9.2379 ± 3.7883 | 19.1338 | 2.5984 | 3.1306 | 0.4429 ± 0.1524 |
| 2 ms | right | -9.2596 ± 3.8669 | 18.5533 | 2.6952 | 3.2068 | -0.4559 ± 0.1539 |
| 1 ms | left | 9.7272 ± 3.5533 | 21.4518 | 1.2632 | 2.8199 | 0.4420 ± 0.1412 |
| 1 ms | right | -9.7736 ± 3.9826 | 22.2091 | 1.1891 | 3.1363 | -0.4140 ± 0.1597 |


## Temporal fluctuation read

Both sides show reduced temporal fluctuation as the physics step is refined: per-step mean-of-seven-wheel qd standard deviation falls from 5 ms to 2 ms to 1 ms (left 4.39 → 2.17 → 1.73 rad/s; right 4.17 → 2.69 → 2.27 rad/s). Adjacent per-wheel Δqd RMS also falls (left 7.49 → 2.60 → 1.26; right 6.90 → 2.70 → 1.19 rad/s). Pooled per-wheel qd standard deviation is lower at both finer steps than at 5 ms, though right-side 1 ms is slightly above 2 ms. Pooled per-wheel min-to-max range is non-monotonic from 2 ms to 1 ms on both sides (left 19.13 → 21.45; right 18.55 → 22.21 rad/s), so individual extrema remain and range should not be described as monotonically converged.

## Body yaw, normal load, and convergence

| dt | end-last-1s mean body yaw rate (rad/s) | raw policy yaw std (rad/s) | mean total wheel upward-force aggregate (N) | left/right mean upward-force aggregate (N) |
|---:|---:|---:|---:|:---|
| 5 ms | -0.070205 | 0.020176 | 296.920 | 151.909 / 145.011 |
| 2 ms | -0.096045 | 0.019913 | 316.152 | 158.178 / 157.973 |
| 1 ms | -0.091466 | 0.023403 | 301.662 | 152.438 / 149.225 |

Protocol check: |2 ms − 1 ms| = 0.004580 rad/s; allowed tolerance = 0.010000 rad/s. **2 ms passes** the prespecified convergence criterion. The 5 ms vs 1 ms difference is 0.021260 rad/s. The mean yaw remains weak at 2 ms (-0.096045 rad/s).

Normal force is the per-wheel-body upward world-force aggregate across all current contact counterparts, not a ground-only or per-contact-pair force. Every case has full raw sample counts, native qd and control targets available, max uncapped effort-formula residual 0 Nm, and applied effort below the 2 Nm cap; no no-contact native substeps occurred, so no-contact behavior is not tested here.

Reset-before-settle canonical initial-state SHA256 (all three): `7d7cd17756b8eee3eedc39828b8a2e90537820f208daf8cd611a40d657e8e8af`.
Fixed contract SHA256 (excluding dt): `6eefae6814dd7fbfda7845238d86a94b6b188d5d09602174cb14e092ab87aa69`.
Decision protocol SHA256: `cf33f05e172f95b05f81c23d527525b9b1f435d426bcf33d4fb96abd93531339`.
