# Windows GPU Engine after-snapshot

After the owned Viser preview tab was closed, three read-only CIM samples were captured at 22:41:52.9, 22:41:57.7, and 22:42:00.9 (local time). Actual spacing was about 4.8 s and 3.2 s because each CIM collection took time in addition to the requested 1 s pause.

| Process / engine | Before (22:38 samples) | After (three samples) |
| --- | ---: | ---: |
| Edge PID 36908 | 61%, 75% | Process no longer existed; no matching engine rows |
| CloudMusic PID 23676 | 31%, 19% | Process remained, but was outside the reported top 8 in all three samples; its lower-ranked utilization is not retained in this first after file |
| dwm PID 2324 | 5%, 2% | 27%, 1%, 4% |
| ChatGPT PID 30068 | Not in before top 8 | 9%, 1%, 3% |

The GPU Engine rows contain raw LUID `0x00000000_0x00018384`. A prior DXGI inventory associated this LUID with Intel Graphics, but this run did not re-establish that mapping: current `Win32_VideoController` reports Intel, NVIDIA RTX 5070 Ti, and Meta Virtual Monitor names without a direct LUID join. Treat the Intel label as historical evidence, not a current-session mapping. The NVIDIA `nvidia-smi` read-only query was 0% utilization, 200/12227 MiB, 53–54 °C across these samples.

The Edge GPU process leaving and the high Edge rows disappearing are consistent with the preview tab contributing to the earlier load. These short samples do not prove causality: other browser pages and desktop activity may still affect DWM and GPU counters. CPU load varied 67%, 6%, and 1%; free RAM was about 60,960–61,009 MiB. This is monitoring evidence, not a performance benchmark.

Raw samples: [before](windows_gpu_engine_snapshot.json), [after](after_gpu_engine_samples_20261017.json). The reusable read-only collector is [collect_windows_perf.ps1](collect_windows_perf.ps1); it defaults to three samples and accepts `-FocusProcessIds` for exact PID readings. The collector itself keeps raw LUID/physical-engine identifiers and does not infer adapter names.
