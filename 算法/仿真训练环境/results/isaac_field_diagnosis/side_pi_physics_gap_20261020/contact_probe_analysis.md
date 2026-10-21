# Contact probe analysis — `diff_left_plus_0.10`

The one authorized contact-instrumented case completed on 2026-10-20 with actual exit code 0. It used a one-second P-only settle followed by eight seconds of normalized action `[0.10, -0.10, 0, 0]`, with no plants on the flat base surface. The runner captured 180 policy-rate contact samples (20 settle, 160 drive) and 1,800 native effort samples. The 41-source training snapshot matched before the run and the probe confirmed source hashes remained unchanged after it.

## Contact and effort

Across the final 20 drive samples, the fourteen per-wheel aggregate upward normal-force components averaged **293.111 N**, compared with the 30 kg vehicle’s **294.3 N** weight (−1.189 N, −0.40%). The 20-sample total ranged from 210.866 to 362.710 N. This is consistent with adequate *aggregate vertical support* in this flat case. It does not show that each wheel stayed continuously loaded: per-wheel upward-force readings were exactly zero in 4–10 of the final 20 samples.

| Wheel | Mean upward component (N) | Zero readings / 20 | Mean wheel qd (rad/s) | Mean geometric bottom gap (mm) |
|---|---:|---:|---:|---:|
| L0 | 16.662 | 9 | 10.864 | 3.911 |
| L1 | 15.810 | 9 | 10.851 | 3.919 |
| L2 | 23.359 | 8 | 9.545 | 3.928 |
| L3 | 17.645 | 8 | 10.146 | 3.936 |
| L4 | 36.337 | 4 | 6.704 | 3.945 |
| L5 | 21.933 | 6 | 8.931 | 3.953 |
| L6 | 17.223 | 7 | 9.842 | 3.962 |
| R0 | 14.086 | 10 | −10.887 | 3.945 |
| R1 | 19.837 | 8 | −9.667 | 3.954 |
| R2 | 23.163 | 6 | −7.233 | 3.962 |
| R3 | 21.229 | 8 | −8.098 | 3.971 |
| R4 | 26.682 | 8 | −8.894 | 3.979 |
| R5 | 16.408 | 9 | −10.351 | 3.988 |
| R6 | 22.736 | 8 | −10.109 | 3.996 |

The reported bottom gap is an oriented-cylinder geometric support estimate relative to the flat base terrain. Values over the final second ranged from 3.313 to 4.244 mm; this estimate excludes the solver’s contact margin/gap and is not a solver penetration or separation measurement.

The probe recorded 8,029 wheel-involving Newton contact-manifold rows across all policy samples. Every counterpart label matched the ground shape, with zero unresolved labels. Those are repeated rows across samples, not 8,029 unique physical pairs; the sensor output remains aggregated per sensing wheel body. The actual post-actuator effort maximum was **0.8961 Nm**, below the 2 Nm native cap, and all 1,800 effort values were finite. Each of the 180 policy steps observed exactly ten effort callbacks.

## Wheel response and body yaw

The left/right target roller speeds were **+9.259 / −9.259 rad/s**. Last-second means of the seven wheel speeds per side were **+9.555 / −9.320 rad/s**, close to their targets; individual wheel means were more variable, as the table shows. The body yaw changed **−0.6513 rad** during the eight-second drive. Its last-second mean yaw rate was **−0.07255 rad/s** against a nominal no-slip kinematic command of **−2.03252 rad/s**, or **3.57%** by that last-second rate comparison. Final measured task velocity was `v=−0.02333 m/s`, `w=−0.06588 rad/s`. The turn direction was correct, but this case does not establish why chassis yaw remains far below the no-slip command.

## Friction-vector limits

For the final second, the per-wheel aggregate friction vectors’ projections onto body-forward averaged **+61.767 N** for the left side and **−56.945 N** for the right side, for a net **+4.822 N**. Their mean absolute world-axis components were left `|Fx|=8.133, |Fy|=7.636, |Fz|=0.0075 N` and right `|Fx|=8.065, |Fy|=6.513, |Fz|=0.0124 N`; the sampled vectors are not systematically confined to one world axis. At the final sample, body-forward was `[0.7953, −0.6062, 0.0007]`; wheel L0 had world friction vector approximately `[0, −17.6805, 0.0007] N`, whose recorded forward projection was `+10.7188 N`.

Isaac Lab documents `net_friction_forces_w` as world-frame force. Newton’s sensor kernel takes the top three components of the contact spatial force, removes the component along the contact normal, and aggregates the resulting tangential vectors per sensing body. The complete solver-to-sensor force-frame transformation was not independently verified in this inspection; therefore these are API-field aggregate projections, not definitive per-pair or hardware longitudinal traction measurements. No contact force is attributed to an individual pair.

The probe supports aggregate vertical contact on the flat case and near-target mean wheel qd, while body yaw remains weak. The measurements do not isolate the cause of the wheel-to-body yaw gap. No further simulation was run for this analysis.

## Evidence

- [Probe summary](contact_probe_runs/diff_left_plus_0.10_20261020/contact_transfer_probe_summary.json)
- [Actual exit record](contact_probe_runs/diff_left_plus_0.10_20261020/contact_transfer_probe_exit.json)
- [Runner log](contact_probe_runs/diff_left_plus_0.10_20261020.log)
- [Policy-rate raw contact samples](contact_probe_runs/diff_left_plus_0.10_20261020/policy_step_contact_samples.jsonl)
- [Per-physics-substep effort tape](contact_probe_runs/diff_left_plus_0.10_20261020/physics_substep_applied_effort.jsonl)
- [Newton force-projection source inspection](newton_contact_force_projection_inspection.json)
- [Probe launcher repair and freeze manifest](contact_probe_launcher_repair_manifest.json)