# MJW v6 verified-input physics gate

All 1, 3, and 64 world cases passed. The driver validated all 12 runtime inputs against the same prelaunch manifest (SHA-256 e623b12ce038165204d2ee79aee47a3847fc2c087557662875507ba452c59b50) before backend imports and result-directory creation. Each result records those comparisons.

| Worlds | Policy steps | Contacts during settling | Max penetration | Max continuous mimic residual | Actual wheel / gimbal clamp | Capacity / outer | Reset fields preserved | LS limit samples |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 20 | yes | 5.419 mm | 0.0000451 rad | 2 / 1 Nm | 0 / 0 | N/A (one world) | 799 |
| 3 | 20 | 3 / 3 | 5.433 mm | 0.0002511 rad | 2 / 1 Nm in all worlds | 0 / 0 | 44 | 1,978 |
| 64 | 20 | 64 / 64 | 5.404 mm | 0.0003487 rad | 2 / 1 Nm in all worlds | 0 / 0 | 1,386 | 37,637 |

Each case performed 1,000 policy physics substeps plus two diagnostic substeps. All three used cuda:0, Newton 1.7.0, Warp 1.18.0, MuJoCo 3.15.0, and MuJoCo Warp 3.15.0; warning mask 3071; exported geom gap and margin both zero. Mimic residuals are raw continuous values from stored reduced joint coordinates and actual MuJoCo Warp qpos, not wrapped IK values.

The full result JSON, trajectory NPZ, stdout, stderr, exit code, command, copied prelaunch manifest and preregistration are retained under each worldN attempt directory. The NPZ does not contain full body_q state or a body-label mapping; it cannot directly create a complete scene snapshot for a viewer.

The earlier unverified attempts and their original manifest bytes remain unchanged in the separate mjw_v6_physics_gate_prepared package. They are not counted as this gate.