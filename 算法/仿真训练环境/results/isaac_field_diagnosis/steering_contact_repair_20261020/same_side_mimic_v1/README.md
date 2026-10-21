# Same-side mimic diagnostic — prepared, not run

This result-only treatment adds six wheel-joint mimic relations per side through
the public Newton builder-world hook. Wheel 0 is the leader; wheels 1–6 are
followers with `coef0=0`, `coef1=+1`. All 14 original per-wheel PI actuators and
their 2 Nm caps stay enabled, so this isolates the constraint graph while
retaining the existing drive torque budget. It does not represent a single
motor driving a real continuous track.

The collector is the frozen v4 runner imported and narrowly patched in memory
by `contact_mimic_v1_runner.py`; the v4 file and production USD remain unchanged.
The summary adds the mechanism treatment to both canonical comparison
contracts, while preserving the original robot USD hash as source provenance.
It records the final Newton mimic count, host MuJoCo and MJWarp equality counts
and types, and after each policy step the native q/qd values and follower-minus-
master residuals. The hook itself does not collect body velocities or external
forces.

Resetting every wheel position and velocity to zero satisfies the zero-offset
relations. For any nonzero reset, positions and velocities must be initialized
consistently with the mimic coefficients.

The authored asset has the same local +Y wheel axes and 45 mm radii, which
supports `coef1=+1`; execution remains gated on B's runtime axis-sign audit and
explicit parent authorization. No simulation or GPU run has been started.

`contact_mimic_v1_runner.py --static-check` performs only source-patch, fake
builder, and CPU API checks. `run_same_side_mimic_v1.sh` refuses to run unless
`RUN_CONTACT_MIMIC_V1=1` is set and refuses to overwrite any prior attempt.
