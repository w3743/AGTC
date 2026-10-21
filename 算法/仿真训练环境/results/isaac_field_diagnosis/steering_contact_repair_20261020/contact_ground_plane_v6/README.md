# Flat-mesh versus native-plane contact probe (v6)

This is an isolated one-case diagnostic copied from the v4 contact probe. It
does not edit production files, the 41-file frozen training snapshot, weights,
protocols, or previous result directories. The v6 simulation completed with
the pre-motion geometry/material/solver audit passing.

## Fixed case

The case holds the v4 conditions: 30 kg vehicle asset and 14 proxy wheels,
`dt=0.002 s`, decimation 25 / policy period 0.05 s, elliptic cone,
`impratio=1`, Newton collision pipeline, `mu=.8`, runtime `kf=1000`, PI gains
`P=.04`, `Ki=.1`, 2 Nm joint/actuator caps, seed 90231, start `[50,50,0]`,
action `[.10,-.10,0,0]`, 1 s settle and 8 s drive. It keeps the v4 policy-step
and post-actuator native wheel-q̇/effort/contact and MJWarp contact-parameter
sampling paths.

The single declared treatment is
`geometry_treatment="flat_mesh_to_native_plane"`: the imported `/Field/Ground`
mesh remains visible but has `COLLIDE_SHAPES` disabled; the hook adds one static
+Z Newton plane at the same world transform and copies the ground's six
material fields. The plane remains in the `COLLIDE_SHAPES` raycast BVH, so flat
in-field ray readings should agree geometrically, while rays outside the finite
100 m mesh may hit the plane. This is a plant-free, flat-field motion diagnostic;
it does not establish rough-terrain, boundary, plant, or full sensor equivalence.

Before reset or any action, the probe fails closed unless the builder hook ran
once for world 0, exactly one source ground mesh and one plane remain, the
original ground mesh is visible/non-colliding, the plane is colliding/not
rendered, and all 14 wheel joints and wheel-link bodies survive. A second
finalized-model check requires all six material fields (`mu`, `kf`, `ke`, `kd`,
margin, gap) to match between retained mesh and plane; it checks wheel/ground/
plane `mu=.8` and `kf=1000`, and requires the retained mesh, plane, and all
selected wheel shapes to report MuJoCo `solref_mode=2`
(`SOLREF_MODE_MJCF_DEFAULT`). The plane hook sets the known v4-observed mode 2
explicitly; the final-model audit verifies the actual effective result instead
of assuming the builder default.

## Contract and comparison rule

Both comparison hashes include `geometry_treatment`, active contact
representation, and the v4-observed solref-mode contract. The source USD is
recorded as `flat_field_source_usd_sha256`, not as the active collision
geometry. Thus a matching v4 source-USD hash or initial-state hash cannot make
this plane case fixed-contract-equivalent to the v4 mesh case. Compare motion
only as the intended geometry A/B after checking the actual config, finalized
material/mode audit, and reset-state hash. Do not claim the plane result is a
same-geometry pair, production fix, or learned-vehicle result.

## Result

The runner completed with real exit code 0 and status `complete`. It captured
all 180 policy steps (1 s settle + 8 s drive) and 4,500 native physics
substeps; maximum actual actuator effort was 0.9374 Nm, below the unchanged
2 Nm cap.

The final-model audit found ground flags 5 (visible, collision disabled) and
plane flags 2 (collision enabled, not rendered). The retained mesh and plane
had identical finalized values for all six material fields: `mu=.8`,
`kf=1000`, `ke=2500`, `kd=100`, `margin=.002`, `gap=.005`. All 14 wheel joints,
wheel-link bodies, and proxy shapes remained present; ground, plane, and the 14
wheel shapes all reported `solref_mode=2`.

Each of the four contact snapshots had 28 valid plane-wheel contacts and no
active ground-mesh contacts. Across every snapshot, the set of contact
`solref`, `solimp`, and `solreffriction` triples was exactly equal to v4's set
(11 distinct triples), showing the geometry hook preserved the observed normal
and friction contact parameters. The v4 and v6 initial-state hashes also
matched. Both full and fixed comparison-contract hashes differ, as required by
the explicit geometry treatment.

The last-second mean yaw rate was −0.079109 rad/s in v4 and −0.076355 rad/s in
v6. This single geometry substitution did not materially recover the weak
turning response; do not treat the small difference as an improvement. The
result narrows the diagnosis by ruling out this flat-mesh-to-plane geometry
change as a sufficient fix under the tested conditions.

Raw output is in
`contact_probe_runs_v6/flat-mesh-to-native-plane__dt-2ms__cone-elliptic__impratio-1/`;
the runner's `.log` and `.exit.json` sit beside `contact_probe_runs_v6/`.

## Files and reproduction command

- `contact_ground_plane_v6.py`: v4-derived one-case probe with a constructor-only
  `newton_builder_world_hook(plane_hook)` insertion and fail-closed geometry
  audit.
- `run_contact_ground_plane_v6.sh`: no-argument runner with a distinct v6
  output root and refusal to overwrite existing artifacts.
- `../staged_plane/staged_plane_hook.py`: shared result-only plane hook; its
  source hash is recorded by the probe.

The authorized run used this command from the repository root in Ubuntu 24.04:

```bash
RUN_CONTACT_GROUND_PLANE_V6=1 bash results/isaac_field_diagnosis/steering_contact_repair_20261020/contact_ground_plane_v6/run_contact_ground_plane_v6.sh
```

The runner writes under `contact_ground_plane_v6/contact_probe_runs_v6/` and
pins the `flat-mesh-to-native-plane__dt-2ms__cone-elliptic__impratio-1` case.
The runner refuses arguments and refuses to overwrite the existing result.

## Static checks performed

The copied probe and plane hook passed Python `compile()` syntax validation,
and the copied shell runner passed Git Bash `bash -n`.
A CPU fake final model with one ground, one plane, and 14 wheel shapes passed
the audit; separate fake `kf` and solref-mode mismatches were rejected, and the
geometry treatment was verified in the full-contract source. The runner was
then run with the explicit `Ubuntu-24.04` WSL distribution and exited 0. WSL
printed a UTF-16 localhost/NAT warning during startup; it was non-fatal and the
actual process exit file confirms successful completion.
