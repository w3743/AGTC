# Front-guide environment integration native validation

The final CPU validation passed with 336 tests passed, 17 skipped, and 8 warnings. `compileall` and the source-hash check both returned 0. The isolated native front-guide/stability suite passed 2 tests on `cuda:0`; the existing Isaac/Newton integration suite passed all 14 tests on `cuda:0`.

The native checks imported two passive guide joints in a two-world model, confirmed 30 kg total mass, retained 14 driven support wheels and 12 same-side mimic relations per world, resolved guide joint/DOF indices through Newton's official per-world MuJoCo mappings, and found no actuator or joint-equality membership for the guides. After wheel commands, Newton actuator generalized force at both guide DOFs was zero. Rough reset, finite state, guide/support clearances, live contact-material metadata (`mu=0.8`, `kf=1000`), reward-component reconciliation, body-frame angular-rate conversion, partial-reset statistics, and the adapter's actual `Scene.terrain_seed` were verified.

Two native failures are preserved as separate attempts. The first exposed IsaacLab `configclass` mutating a frozen spec and config; `make_env` now serializes those values as plain mappings in the config and restores their typed objects on the environment. The second was a test-harness namespace error from matching USD names directly in a MuJoCo template; the passing test uses Newton's per-world joint and DOF maps.

The guides remain discrete free-rotating proxies, not a continuous track. This validation establishes their imported contact material and geometry, but does not measure contact load share or hardware calibration. No training, held-out evaluation, or FPS measurement was run.

Full per-attempt logs, exact commands, exit codes, and final source hashes are recorded in `summary.json` and the neighboring attempt directories.