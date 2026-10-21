# Front-guide fixed-action step-contact pair (attempt 3)

This record compares the free-rotating guide-wheel diagnostic mode with the configured same-side-track-speed coupling under one fixed action. It is a physics contact probe, not a trained-policy comparison, a continuous-track model, a step-height limit, or a claim of successful obstacle traversal.

## Reproducible setup

- Isaac Lab/Newton in Ubuntu-24.04; one environment per run, reset seed `87001`, environment seed `80003`.
- Start pose `(x, y, yaw) = (2.0 m, 2.0 m, 0 rad)` and goal `(6.0 m, 2.0 m)`.
- A static Isaac Lab USD cuboid is centered at `(3.0 m, 2.0 m, 0.03 m)`, size `(0.8 m, 2.0 m, 0.06 m)`. Its front and rear edges are `x=2.6 m` and `x=3.4 m`. The imported Newton box shape has collision enabled and uses the configured default material (`mu=0.8`); no per-probe friction override was applied.
- `plants=0`, roughness/rocks disabled, `goal_distance=None`, `auto_reset=False`, maximum episode time `12 s`.
- The same normalized action `[0.05, 0.05, 0, 0]` was applied for 200 policy steps: `dt=0.001 s`, decimation `50`, policy period `0.05 s`, total simulated time `10 s` per run. No checkpoint was loaded.
- Guide geometry and mass were unchanged: front guides at `x=0.42 m`, `z` offset `0.05 m` above support-wheel center, radius `0.045 m`, width `0.06 m`, mass `0.25 kg` each. The free/coupled modes intentionally have distinct robot USD/spec hashes, while the field USD content hash is identical.

## Run integrity

`pair_manifest.json` records actual process exits: free probe `0`, coupled probe `0`, CPU pair analyzer `0`, outer launcher `0`. All 200 rows in each run are finite; the CPU analyzer reports every pairing-contract check true. Both runs used the same probe and effective production source hashes and identical reset initial state. The field USD SHA256 matches in both runs: `a57af01de1c47d3ce71458713e782ac219ae2e6b005cff35752c861c7023e57a`.

The active probe source SHA256 is `cb9db08722c11cb3bdf110592870cd087d9d978922da8f2bb59c9992fb6d10cd`. Effective source hashes were:

- `isaac_field/env.py`: `1403bccf090f48638fb34a9923e0a23039ebf7c13023dcc03a56a9da542c1e5d`
- `isaac_field/assets.py`: `d6e5f07e6c6170dd04dfb7c7febd187a811eaa8886fea1e976b2bc3521da41f0`
- `isaac_field/newton_drive.py`: `97fd298606c4e3d00759bed2d50ca92d88675bc8b78d6cd8cfc2890423538789`
- `isaac_field/task.py`: `ba63a0a75fcb854e2c6292c679fdf113ea4655e9c1184d6767d2af6069dea2a9`

Earlier attempts are retained alongside this directory. Attempt 1 stopped on a probe joint-name lookup assumption before any physics samples. Attempt 2 completed the free run but its coupled run stopped before stepping because an extra `eq_objtype` filter excluded equality rows. Attempt 3 uses the already-passing native-test mapping contract and does not alter physics, actions, timing, or geometry.

## Observed result

Newton contact identity snapshots contained an actual front-guide-to-diagnostic-step shape pair in 187 of 200 policy-step samples in each mode, first appearing at policy step 14 and still present at step 200. The pair uses guide collision shape IDs `17`/`18` and step shape ID `0`. Support-wheel-to-step pairs also appeared in 176 free-mode and 175 coupled-mode snapshots. These are per-policy-step snapshots of the final physics substep; they do not establish uninterrupted contact, contact force, force direction, or why the vehicle stopped.

| Measurement | Free guides | Same-side coupled guides |
|---|---:|---:|
| Finite policy-step rows | 200/200 | 200/200 |
| Guide-step pair snapshots | 187/200 | 187/200 |
| Guide relations per world | 0 | 2 |
| Support-wheel relations per world | 12 | 12 |
| Total runtime equality rows per world | 12 | 14 |
| Maximum guide applied effort | 0 Nm | 0 Nm |
| Maximum guide generalized actuator force | 0 Nm | 0 Nm |
| Maximum support-wheel applied effort | 0.9772 Nm | 0.9334 Nm |
| Maximum guide-to-track angular-speed residual | Not applicable | 0.0921 rad/s |
| Maximum root x during run | 2.2538 m | 2.2535 m |
| Final root x | 2.2524 m | 2.2519 m |
| Final conservative whole-proxy rear-x lower bound | 1.7274 m | 1.7269 m |

The coupled-mode runtime audit found 12 support-wheel relations plus 2 guide relations, for 14 total per world. The guide equalities point to the same-side wheel-0 leaders and use coefficient `1.0` (`Rmain/Rguide = 0.045/0.045`). No separate guide actuator or USD DriveAPI was present; guide `qfrc_actuator` and applied effort remained zero across all samples. Driven-wheel effort remained below the unchanged `2 Nm` per-wheel cap.

The guide wheels did make contact with the imported step shape, but the vehicle did **not** traverse the obstacle. Its root never reached the step front edge (`x=2.6 m`), and the conservative full-proxy rear bound remained far before the step rear edge (`x=3.4 m`). Thus this test establishes sampled guide-step contact under the fixed action, not successful climbing or passage. Free and coupled runs also show nearly the same root displacement; this small fixed-action pair is not evidence of a meaningful performance gain.

## Files

- `pair_comparison.json`: CPU-checked paired result and all machine-readable checks.
- `paired_trajectory_deltas.jsonl`: per-policy-step coupled-minus-free recorded differences.
- `free_run_003/` and `coupled_run_003/`: raw 200-row samples, probe summaries, logs, actual-exit records, source copies, and run manifests.
- `pair_manifest.json`, `pair_analysis.log`, and `pair_analysis.exit.json`: pair commands, hashes, and analyzer result.
