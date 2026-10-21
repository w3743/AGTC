# Post-actuator qd sampling audit

Read-only source audit for the Newton/MJWarp A/B probe. No simulator or GPU was started and no probe/source files were changed.

## Result

For this environment, sample the active pre-solver joint velocity from Newton state 0. MJWarpManager._build_solver sets NewtonManager._use_single_state = True (external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager.py:46–58). In the single-state branch, _run_solver_substeps uses backend.state_0 as both solver input and output in place (newton_manager.py:1698–1706); it does not ping-pong state 0/state 1. get_state_0() returns that active object (:1818–1820); get_state_1() is the next-state object (:1837–1839) and is not the sample to use here.

One simulation iteration runs NewtonActuatorAdapter.step(state_0, ...), then post-actuator callbacks, then the solver (newton_manager.py:1751–1765). The adapter passes the same sim_state into each Newton actuator (external/IsaacLab/source/isaaclab/isaaclab/actuators/newton/adapter.py:136–163); Newton's actuator reads sim_state.joint_qd as its current velocity (newton/_src/actuators/actuator.py:453–465). The IsaacLab qd binding is a direct view from root_view.get_dof_velocities(get_state_0()) (external/IsaacLab/source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation_data.py:1625). Thus the qd sampled in post-actuator is the same state-buffer qd the native PD just used, before this iteration's solver/contact update.

The CUDA graph reuses the same state-0 array addresses, but the MJWarp path updates that buffer in place; a captured pointer is therefore live across graph replays, not a stale per-tick snapshot. Do not read state 1 for this path. For this controller, the native wheel actuator has zero position gain, velocity gain 0.04, and a 2 Nm actuator/joint effort limit (isaac_field/env.py:167–170); paired same-slot qd, target velocity, integral feedforward, and applied effort can test the unsaturated law. Existing confirmation found effort below saturation in the probe cases.

## Safe sample boundary

Register before the first simulation step / graph capture. In one post-actuator callback, launch a graph-safe Warp copy kernel that writes qd, target velocity, integral feedforward, and applied effort to the same preallocated tape row; advance its slot with a device counter, as the existing effort-capture utility does. Read or convert the tape only outside the captured callback. This preserves one-to-one tick alignment without Python-side device reads in the graph.

The environment sets manager decimation from the DirectRLEnv config (external/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py:182–183), and Newton's full path loops that many actuator/solver iterations inside one graph (newton_manager.py:1741–1769). Therefore post-actuator runs for each physics iteration and captures the qd used by that iteration. A later row's qd is the previous iteration's post-solver result. The post-step callback runs only after the whole decimation loop (:1767–1769), so it is not a per-inner-iteration qd tap.
