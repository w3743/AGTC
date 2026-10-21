# Same-side mimic v2 diagnostic

This result-only probe reuses the frozen v4 motion collector and v1 same-side mimic setup, but sets the softness of its 12 matching MuJoCo JOINT equality rows through the public Newton `ModelBuilder.add_custom_values` schema. It does not import Newton's private `_src` helpers or modify the source USD, vendor code, production physics, training, or policy.

The treatment is fixed in `protocol.json`: 2 ms physics step, elliptic cone, impratio 1, µ=.8, kf=1000, the v4 1 s settle plus 8 s `[.1, -.1, 0, 0]` action, the same 14 per-wheel PI actuators and 2 Nm caps, and equality solref `[.004, 1]` / solimp `[.999, .9999, .001, .5, 2]`. The expected pre-motion state hash is inherited from the successful v1 case; the final case must record and compare it.

Before motion the runner verifies the actual CPU builder schema and solver, then the live environment's mimic/equality pair mapping, exact row counts (`12`, not `24`), and Newton/host MuJoCo/MJWarp softness values. After the single authorized run it reports last-second qd differences from every native physics callback and q differences from the final 20 policy steps against the predeclared protocol thresholds.

Run `bash run_static_preflight_attempt2.sh` for CPU-only schema/solver validation and `bash run_same_side_mimic_v2.sh` for the one authorized physical case. Original and failed v1 attempts remain under `../same_side_mimic_v1/` untouched.
