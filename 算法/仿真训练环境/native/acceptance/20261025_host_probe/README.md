# Host probe evidence recovery

This directory was reconstructed after the original probe-capture files were unexpectedly lost. It is not the original live capture.

The four run artifacts below were copied byte-for-byte from the static pilot01 run. Checkpoint files were not copied; their hashes and CPU-read metadata are in host_stop_reconstruction.json.

The historical 16:22 and 16:28 Beijing observations are identified as recovered tool/conversation notes, at minute precision. The JSON records the reconstruction time separately. PID 48968 was later observed absent after a non-forced stop attempt; this is an observation of absence, not proof that a forced stop succeeded.

The static pilot manifest records exit_code 1 at iteration 20. The cause is unresolved. This evidence does not claim natural completion, a head-pipe cause, or exclusive/idle desktop GPU state.

Package versions were re-read from WSL /opt/venv distribution metadata. This was a CPU-side metadata query only; no CUDA probe, training, benchmark, or process termination was performed.

See host_stop_reconstruction.json for byte sizes, SHA256 values, metrics summary, checkpoint metadata, provenance, and limitations.
