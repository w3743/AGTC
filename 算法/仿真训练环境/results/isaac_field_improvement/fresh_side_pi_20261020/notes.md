# Fresh side-PI training notes

This trial uses a new random RecurrentPPO initialization and the current frozen sensor-minimal-v6 policy contract. It does not resume or transfer any earlier checkpoint. The four-stage 196,608-step curriculum, formal zero-step baseline, 29-checkpoint schedule, and paired development evaluation are unchanged.

The simulation drive uses native per-wheel P=0.04 and a default side-integral Ki=0.1, updated every policy step using the actual `step_dt`; Newton's combined per-wheel effort clamp remains 2 Nm. Ki, the side-mean aggregation, and all proxy motor parameters are uncalibrated simulation assumptions, not real-vehicle controller settings.

The isolated five-second braking probe passed the final zero-command and terminal velocity gates, but the PI run showed about 0.0729 m reverse-speed integral and reached about -0.130 m/s, compared with near-zero reverse integral for P-only. This trial tests learning under the changed execution loop; it does not imply improved braking or real-vehicle safety.

Training uses an empty field (`plants=0`). Its free-space and 16-plant paired development results are not blind validation and cannot be described as a safety or field-capability improvement. The 830xx seed range remains unused and unobserved.
