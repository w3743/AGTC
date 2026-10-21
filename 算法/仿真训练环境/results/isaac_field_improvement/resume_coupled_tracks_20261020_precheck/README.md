# Coupled tracks resume precheck

Observed at 2026-10-20 05:30:37 UTC. This is a read-only precheck; no model reload, training, evaluation, GPU work, 830xx seeds, FPS work, or child preparation was performed.

The completed parent run is internally consistent: protocol SHA-256 2735ca85917643e49e27088957d00b1576f37902ce9664203253ae698181d5b6, zero-step baseline SHA-256 beda8fde92236b558433a148245094caeefb88d7c23b9ea73ed94c4cd596f1d5, final stage_04_turn90.zip SHA-256 6b4c85884a51c2e23f557eb98bc04ea63905fc8f20d4379715fb9fe72b75cdc5. Recorded process and Python exits are 0 at 196,608 timesteps.

The existing CPU-only validation passed and matches these hashes. It confirms CPU loading, no CUDA devices, Adam state for 37 parameter entries at step 7,607, all finite, and the direct-track-speed-v1 action contract. Protocol observations are lidar 1x400, depth 1x160x88, detections 16x8, state 33, feedback 37. The drive contract is independent_left_right_track_speed_coupled_pi_v7 with Newton/MJWarp at 1 ms physics and 50 ms policy steps.

All 42 archived source copies match their protocol hashes. The live tree differs in exactly four expected viewer/UI files: viewer.py, network_2d.py, network_2d.html, and THIRD_PARTY_NOTICES.md. The other 38 sources match.

The Ubuntu-24.04 process scan found no training process. It found the existing viewer, PID 326346, using this final checkpoint on port 8769; it was left running.

Continuation caveat: the existing scripts/train_isaac_foundation.py resume path only accepts a different prepared v2 parent, failed/interrupted status, a 53,248-step baseline, and the exact 8,192-step near checkpoint. It will reject this completed fresh-coupled parent. The checkpoint itself is valid continuation input, but a separate child and continuation path must explicitly support this protocol while preserving the parent.
