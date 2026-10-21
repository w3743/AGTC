# Contact and wheel-axis telemetry v1

This single diagnostic case imports the frozen `contact_friction_v4.py` at runtime and replaces only its `mjw_contact_parameter_snapshot` callback in memory. It calls the unchanged v4 `main()` with the original contract: 2 ms physics step, 25 substeps per 50 ms policy step, elliptic cone, `impratio=1`, seed 90231, the same start, `[0.1, -0.1, 0, 0]` command, 1 s settle, and 8 s drive. V4 retains its original `EXP_ROOT`, `OUT_ROOT`, helper path, and runner-hash source; the new case gets a unique subdirectory in the existing `contact_probe_runs_v4/` results root. No production source, USD, solver setting, action, or training artifact is changed.

At the existing four v4 snapshot times, the wrapper appends the actual host `mj_model`, `mjw_model`, and `mjw_data` geometry/joint/passive arrays, valid contact frame/position/friction rows, active EFC rows (`vel`, `aref`, `force`, `D`, `state`, `type`, `id`), and contact-to-EFC row addresses when exposed by the installed build. Each snapshot records the wrapper SHA-256 and sample time. The EFC record explicitly refers to the pre-integration solve; the pose and qvel sampled after `env.step()` can be post-integration and must not be combined as if simultaneous.

The `wheel_cylinder_axis_to_joint_axis` section derives the MuJoCo cylinder's local +Z axis through the actual `geom_quat` into its body-local frame, then compares that vector with the matching body's `jnt_axis`. This checks the converted runtime geometry rather than only the USD authoring axis. Missing runtime fields are recorded as `NA` or an explicit status; they are not inferred from authoring files.

Run only once after authorization from the WSL project checkout:

```bash
RUN_CONTACT_AXIS_TELEMETRY_V1=1 /bin/bash results/isaac_field_diagnosis/steering_contact_repair_20261020/contact_axis_telemetry_v1/run_contact_axis_telemetry_v1.sh
```

The runner refuses to overwrite a prior case. Raw v4 samples, native qd/effort output, the augmented `mjwarp_contact_parameter_snapshots.jsonl`, and `telemetry_run_identity.json` are saved under `contact_probe_runs_v4/axis-telemetry-v1-attempt3__baseline__dt-2ms__cone-elliptic__impratio-1__mu-0.8__kf-1000/`; the wrapper's `.log` and `.exit.json` sit beside it in the same results root. The first two hook failures remain preserved under their original `axis-telemetry-v1__...` and `axis-telemetry-v1-attempt2__...` case/log/exit paths.

The successful attempt returned actual exit 0 and the v4 summary is `complete` (180 policy samples, 4500 native substeps, max effort 0.96728 Nm). Its four contact snapshots contain 36/46/44/43 active contact records and 96/87/66/51 active EFC rows. Cross-correlating captured wheel contact geom/body IDs with model arrays shows all 14 collision geoms are MuJoCo cylinders (`geom_type=5`); each actual `geom_quat` maps cylinder-local +Z to body-local +Y, parallel to its matching `jnt_axis=[0,1,0]` (absolute dot 1.0). All 22 `dof_damping`, `dof_frictionloss`, and `dof_armature` entries are zero. These observations rule out a converted wheel-axis mismatch and hidden MuJoCo DOF damping/friction/armature as explanations in this model; they do not rule out contact friction/solver behavior or validate real vehicle turning performance.

For the EFC check, each wheel-ground pair was joined by `contact.efc_address[:dim]`; the two tangent rows were transformed using the stored contact-frame tangent axes and projected onto the sampled root-forward/lateral directions. Only EFC-mapped pairs with normal multiplier above 1 mN are counted as load-bearing below. Rows with no EFC address or zero normal force are not treated as evidence of no slip.

| Sample | Wheel-ground records / mapped EFC / load-bearing | Forward slip, median / p90 (m/s) | Lateral slip, median / max (m/s) | `|Ft|/(μFn)` | `Ft·vt` sign | `D_t1 == D_t2` |
|---|---:|---:|---:|---|---|---|
| settle end | 36 / 32 / 32 | ~0 / ~0 | ~0 / ~0 | median 0.00017 | at numerical zero | exact, 32/32 |
| drive 2 s | 46 / 29 / 23 | 0.236 / 0.444 | 0.0137 / 0.0206 | 1.000 within 1e-7 | negative, 23/23 | exact, 29/29 |
| drive 4 s | 44 / 22 / 20 | 0.134 / 0.171 | 0.0106 / 0.0155 | 1.000 within 1e-7 | negative, 20/20 | exact, 22/22 |
| drive 8 s | 43 / 17 / 7 | 0.410 / 0.464 | 0.0139 / 0.0253 | 1.000 within 1e-7 | negative, 7/7 | exact, 17/17 |

The load-bearing solved contacts therefore show real forward rolling-direction slip and friction saturation, not a static-contact proxy artifact; lateral tangent slip is much smaller in these snapshots. EFC stored velocity/force are from the pre-integration solve and `Ft·vt` is computed in the contact tangent rows; this shows dissipative sign for these samples, not a general proof of maximum-dissipation correctness or calibrated steering response.

The per-side counts and normal multipliers also show why a single end sample cannot establish persistent side loss:

| Snapshot | Left pairs / mapped / no EFC / mapped zero-Fn / loaded; ΣFn (N) | Right pairs / mapped / no EFC / mapped zero-Fn / loaded; ΣFn (N) |
|---|---|---|
| settle end (total 1 s) | 18 / 16 / 2 / 0 / 16; 149.03 | 18 / 16 / 2 / 0 / 16; 145.31 |
| drive 2 s (total 3 s) | 22 / 11 / 11 / 1 / 10; 173.39 | 24 / 18 / 6 / 5 / 13; 194.73 |
| drive 4 s (total 5 s) | 21 / 3 / 18 / 0 / 3; 124.41 | 23 / 19 / 4 / 2 / 17; 167.16 |
| drive 8 s (total 9 s) | 18 / 17 / 1 / 10 / 7; 205.05 | 25 / 0 / 25 / 0 / 0; 0.00 |

At the final sample only the left side carried positive normal multipliers, but the preceding 4 s drive sample was opposite (3 left, 17 right loaded), and the 2 s drive sample was close to even (10/13); these are four sampled instants, not a continuous load history. At total 9 s the root was nearly level (roll −0.00036 rad, pitch +0.000015 rad) while angular velocity was `[+0.0385, −0.0343, −0.0812]` rad/s and linear velocity `[+0.01957, −0.01440, −0.00270]` m/s. This is compatible with a transient loading change and does not prove persistent right-side lift or its cause.
