# Contact-probe API findings

This is source/API preparation only. No scene or simulator was created.

The frozen field setup selects Newton MJWarp with `use_mujoco_contacts=False`. In this mode the Isaac Lab Newton manager uses its Newton `CollisionPipeline`; `NewtonManager.get_contacts()` returns the active `Contacts` buffer. The installed `Contacts` API exposes `rigid_contact_count`, `rigid_contact_shape0`, `rigid_contact_shape1`, and `rigid_contact_normal`. These support recording pair identities, labels, and normals for the wheel-involving manifold records. They do not by themselves give a trustworthy per-wheel-ground force attribution.

The installed Newton `Contacts` type has both an older `rigid_contact_force` vector marked “currently unused” and an extended `force` spatial-vector array. `SensorContact.update()` explicitly requires `Contacts.force`; its kernel takes the top three force components for each contact record and accumulates forces onto each sensing body. It also decomposes each contact force into normal and tangential parts using the contact normal. The Isaac Lab public outputs are therefore useful for this probe:

- `ContactSensor.data.net_normal_forces_w`: aggregate world-frame normal force per sensing wheel body, in N.
- `ContactSensor.data.net_friction_forces_w`: aggregate world-frame friction force per sensing wheel body, in N, when `track_friction_forces=True`.

The forward projection of the latter will be labelled as an aggregate sensor-body quantity. It combines every counterpart touching that wheel and cannot be described as ground-only traction unless the observed pair identities justify that interpretation. The probe will keep contact-pair identities separate from sensor aggregates and will not serialize a made-up per-pair force.

The optional per-counterpart force matrix is deliberately excluded. The local official Newton sensor test explicitly skips its `use_mujoco_contacts=False` filtered-matrix case because reporting is nondeterministic across hardware. The existing field ground-contact probe also documents inconsistent pair-count and selected-wheel sensor reporting. This probe records the aggregate per-wheel vectors and raw pair labels together so discrepancies remain visible.

The effort tape will reuse `tests/newton_effort_capture.py` with the official `NewtonManager.register_post_actuator_callback`. That callback observes the actual post-actuator effort once per internal physics iteration. Contact sensor readings are sampled immediately after each policy `env.step` at 50 ms; each row records the matching final physics-iteration index (one reading per decimation window). Thus the force samples are policy-rate aggregate snapshots, while effort is recorded at the 5 ms actuator-call rate.

The wheel-bottom gap is a geometric support-gap estimate from live wheel-body pose, cylinder radius/width/axis, and the simulator's flat `Terrain.height()`. It explicitly excludes Newton contact margin/gap and is not a solver penetration/separation measurement.

Relevant local sources:

- `isaac_field/env.py`: current MJWarp/collision settings, wheel-contact geometry and v6 drive config.
- `external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py`: `get_contacts()`, contact-sensor registration, and post-solve sensor update.
- `external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager.py`: `use_mujoco_contacts=False` selects the Newton collision pipeline.
- `external/IsaacLab/source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor/contact_sensor.py` and `contact_sensor_kernels.py`: sensor reads Newton contact forces and aggregates normal/friction vectors.
- `external/IsaacLab/source/isaaclab_newton/test/sensors/test_contact_sensor.py`: ordinary aggregate sensor coverage and the filtered-matrix skip note.
- `results/isaac_field_diagnosis/ground_contact/ground_contact_tail_pose_summary_20261017.json`: previous diagnostic evidence; not rerun here.

`inspect_newton_contact_api.py` records the exact installed Newton source paths and SHA-256 values at runtime without starting Isaac Sim. `contact_transfer_probe.py` is guarded by `RUN_CONTACT_TRANSFER_PROBE=1`, validates the training trial's frozen source snapshot, and writes only to a fresh directory under `contact_probe_runs/`. It remains prepared only until the parent authorizes execution.
