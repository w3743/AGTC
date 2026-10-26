# MuJoCo-Warp 候选探针证据

本目录保存外部 `audit_stack/mjw_candidate` 的已完成 CPU/GPU 探针输出、当前候选和探针源码，以及 v3 冻结依赖。文件来源、原始大小与 SHA-256、复制后 SHA-256 见 `copy_manifest.json`。NPZ 是探针轨迹；所有 `*.pt`、`__pycache__`、`*.pyc` 与外部旧诊断 `run/` 均未复制。

## 探针记录

表中时间取对应 JSON 的 `started_utc`；目录名中的时刻是命名标签，不能替代记录时间。对没有 `started_utc` 的旧单文件结果，不根据文件名或文件系统时间推定启动时刻。

| 原始记录 | JSON 启动时间（UTC） | 记录状态与范围 |
|---|---|---|
| `probes/cpu/cpu_preflight_20261025_1942/result.json` | `2026-10-25T11:34:16.658759+00:00` | CPU，3 worlds、1 policy step/50 子步，revision v2，passed。 |
| `probes/cpu/cpu_preflight_capacity_20261025_1953/result.json` | `2026-10-25T11:40:19.496664+00:00` | CPU，3 worlds、1 policy step/50 子步，revision v3，passed；输出仍记录 LS_ITERATIONS raw overflow 1024。 |
| `probes/gpu/gpu_probe_20261025_1943/` | 无 result JSON；见 `command.txt` | 启动尝试退出 1，`FileExistsError` 因结果目录已存在，未完成物理探针。 |
| `probes/gpu/gpu_probe_20261025_1944/result/result.json` | `2026-10-25T11:44:36.282850+00:00` | GPU，3 worlds、120 policy steps/6000 子步，revision v3，failed：LS_ITERATIONS overflow 120/120 policy samples，raw 1024；最大 mimic residual 约 12.414 rad。 |
| `probes/cpu/cpu_v4_reset_eq_20261025_195946/result.json` | `2026-10-25T12:00:00.970072+00:00` | CPU，3 worlds、2 policy steps/100 子步，revision v4，passed；此短 probe 记录零 overflow。 |
| `probes/gpu/gpu_probe_v4_20261025_200019/result/result.json` | `2026-10-25T12:01:23.058947+00:00` | GPU，3 worlds、120 policy steps/6000 子步，revision v4，failed：reset 后选中 world 的 reduced joint state 未同步到两份 ping-pong state。 |
| `probes/cpu/cpu_v4b_smoke_20261025_200300/result.json` | `2026-10-25T12:02:26.218913+00:00` | CPU，3 worlds、2 policy steps/100 子步，revision v4，passed；此短 probe 记录零 overflow。 |
| `probes/gpu/gpu_probe_v4b_20261025_200246/result/result.json` | `2026-10-25T12:03:50.085902+00:00` | GPU，3 worlds、120 policy steps/6000 子步，revision v4，probe status passed。原始输出仍包含非零 policy-sample overflow 摘要与 raw 最大值 1024；该短探针不能作为稳定性、等价性或训练能力验收。 |

最早的 `cpu_probe_result*.json` 是原字节保留的历史单文件结果，缺少可靠 `started_utc`；请直接查其 `identity.backend_revision` 与结果字段，不要从文件名猜时间。`gpu_probe_20261025_1943` 保留了失败命令和退出输出，因为它没有生成 `result.json`。

## 源码与可复现性边界

`source/` 中的候选实现和 probe 脚本是复制时可读取到的当前源码。`dependencies/frozen_v3/` 是从 `20261025_v3_formal_rejected/source_snapshot/.../native/proto/` 复制的 `newton_backend.py` 和 `physics_backend.py`，不是仓库当前正在重构的基类。探针 JSON 内已有的 `git_commit_readonly`、`source_files` 和 hash 字段按原字节保留。

这些 CPU/GPU 探针目录没有一致的启动时源码 snapshot/hash。v4/v4b 当前 candidate/probe 文件是运行后复制；audit agent 表示该轮前后文件未改，但本目录没有独立的启动前 hash 证据。因此不声称完整 launch freeze 或逐运行源码精确对应。旧 v3/v4 failed 结果也不能仅凭当前复制源码重建出确切的启动时源树。本目录的 `copy_manifest.json` 只证明“本次归档复制的字节”与当前外部源文件一致，不证明它们就是每次探针启动时的字节。

当前 v4b probe 仅是有界物理诊断，不是正式训练、性能基准或能力评估。不能把 status `passed` 解释为多环境长时间稳定、无 solver warning、与 XPBD/MuJoCo CPU 等价或训练可用；原始 overflow、有限样本和失败记录均保留供审查。正式仓库集成需另做完整源码冻结与复测。

## 独立 XPBD diagnostic60 记录

`v3_diagnostic60/` 是仓库原有 `native/acceptance/20261025_v3_diagnostic60` 的轻量证据副本，保留原字节，不包含 `latest.pt` 或 `checkpoint_*.pt`；原始目录未修改。该单独 run 的 audit 记录 60/60 iterations、每轮 2048 samples、exit 0、0 nonfinite reset/action/event，source manifest stable，Git commit `6e4a438285ec15d160c47bb47ce63b24f9b83477`。它是一次未复现诊断，不能推翻另一独立 v3 formal run 的失败：`20261025_v3_formal_rejected/outcome_summary.json` 记录只完成 57 轮，iteration 58/step 21、env 11 非有限状态、exit 3、`formal_eligible=false`；该记录的 launch commit 为 `c94553e2b4831848b66700d4dd0d4f9e977b343d`，finish commit 为 `5e29a0f8aea88516731cd04fdf5eb36237931058`，且原始 source manifest 前后相同。两份证据描述不同 run；clean 60 轮只说明这一次有界诊断未触发失败。

复制清单不包含 README 自身或 `copy_manifest.json` 自身。完整 `.pt` checkpoint 原样保留在原 `20261025_v3_diagnostic60/run/`，未复制、未暂存、未修改。