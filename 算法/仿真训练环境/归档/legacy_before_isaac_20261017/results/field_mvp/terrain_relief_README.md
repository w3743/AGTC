# Terrain relief snapshot (2026-10-17)

This snapshot records the new default `ground_amplitude=0.25 m` on the 100 m field with master seed 7. The amplitude is a bound around base height (approximately ±0.25 m), so the theoretical peak-to-trough limit is 0.5 m.

## Surface statistics

`terrain_relief_summary.json` is the machine-readable source. The sampled 0.5 m grid (40,401 points) ranged from −0.240781 m to +0.224833 m, for a sampled full-field peak-to-trough range of 0.465613 m. Its maximum sampled slope magnitude was 0.042798; the analytic global bound for this seeded terrain was 0.057297. The generated wavelengths were 27.72, 30.14, 20.00, and 30.54 m.

Random horizontal segments were sampled with uniformly random directions and starts kept far enough from the boundary for both endpoints to remain inside the field. Across 100,000 segments per distance, absolute height differences were:

| Segment length | 95th percentile | Maximum sampled |
| --- | ---: | ---: |
| 5 m | 0.12905 m | 0.20414 m |
| 10 m | 0.23030 m | 0.35682 m |

The whole-field range is a sampled extremum statistic, not a claim that every local interval has this relief. The segment results are a finite random sample and do not guarantee a bound for every path.

## Ray-cast spot check

A separate 1,000-ray audit compared the new cast path against a 0.01 m scalar sampling reference using the same seeded terrain, 0.4 m origin clearance, directions from −10° to +10° pitch, and 8 m maximum range. Both found 347 first crossings; there were no missed crossings or extra cast hits. Maximum distance difference was 0.00998 m, limited by the 0.01 m reference spacing. This is a spot check, not an exhaustive proof; non-monotone rays use finite 0.5 m horizontal sampling.

## CPU benchmark

The headless benchmark was run with:

```powershell
.\.venv-field\Scripts\python.exe -m field_sim benchmark --steps 300 --policy-steps 200 --paced-seconds 5 --checkpoint results/field_mvp/training_terrain/checkpoint.zip --output results/field_mvp/benchmark_relief.json
```

The CLI limits BLAS worker threads to one. On this Windows host CPU, environment stepping averaged 1.733 ms (P95 8.885 ms; 577 Hz). Cached policy inference P95 was 0.705 ms. The paced run completed 100 steps at 19.999 Hz against a 20 Hz target with zero deadline misses; total loop P95 was 25.305 ms. Initial reset plus scene validation took 183.21 ms.

This measures only the headless host CPU path. It excludes browser rendering, HTTP serialization, real perception inference, and RK3588 hardware. The checkpoint is a short-trained model and this timing does not establish task competence. Full raw benchmark values are in `benchmark_relief.json`.
