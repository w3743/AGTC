# Single-preview local autoscale backend evidence

This record covers the backend/API implementation and focused tests for `detail_scale=local|global`. The default remains `global`; local mode computes the selected activation or parameter's finite max absolute value, applies it symmetrically to the preview and `detail.legend`, and uses scale 1 when no nonzero finite values exist. Main-network node images and the top-level global legend continue to use the frame's shared adaptive scale.

The response adds canonical `detail.tensor` plus a `detail.legend` containing the palette name/stops, symmetric bounds, zero, mode, scale, and observed maximum. PNG caching continues to key by frame revision, layer, tensor, and numeric scale.

Validation ran on CPU with Python 3.11. The final recorded attempt is in `attempt_02/`: `python -m pytest -q tests/test_network_2d.py` exited 0 with 17 passed and one existing NumPy overflow warning from the extreme-value statistics test; `python -m compileall -q isaac_field/visualization/network_2d.py tests/test_network_2d.py` exited 0. Attempt 1 is retained as an earlier successful run before the final palette metadata assertion was added.

Integrity checks in `integrity_check.json` report zero SHA-256 changes across all 42 `source_snapshot` files and the three protected trial artifacts. Their before/after manifests are `source_snapshot_before.json`, `source_snapshot_after.json`, `protected_artifacts_before.json`, and `protected_artifacts_after.json`. The protected artifacts are the trial protocol, untrained baseline, and `stage_04_turn90.zip` final checkpoint. Current source hashes are in `workspace_source_sha256_after.json`.

This subtask modified only `isaac_field/visualization/network_2d.py` and `tests/test_network_2d.py`; the root agent is editing `isaac_field/visualization/network_2d.html` in parallel. No model, training protocol, evaluation, or AGENTS file was changed here. No GPU or training was run.

## 前端与真实部署验收

前端详情默认勾选“单图自适应色标”，独立显示当前激活/参数的色条；取消勾选切回全局统一。异步请求同时检查所选层、张量及色标模式，避免旧响应覆盖新选择。主图继续共用原全局色标。Node内联JS语法检查exit0，`run_viewer.sh`的bash语法检查exit0。

本项目旧viewer经核实命令后停止并重启为final196608模型；部署记录WSL PID323976/exec19150，后续须查实际进程，当前暂停0秒。浏览器实测60卡片、71连线；切层不改变连线颜色/线宽/透明度，没有related/muted线class。详情local/global切换时，主图60张图片逐项完全相同、全局±2.27487不变，单张预览图片改变。深度图单图范围±0.952539，卷积weight±1.04657、bias±0.0202728；不同范围均显示对应色条。console warn/error为空。

截图 `preview_local_scale.png`，结构化记录 `browser_acceptance.json`，真实API响应 `runtime_local_depth.json`、`runtime_global_depth.json`、`runtime_local_weight.json`、`runtime_local_bias.json`。无新增训练，不用该UI检查推断策略能力。旧42快照及protocol/baseline/final哈希不变，当前source相对训练快照的预期差异是network_2d.py与network_2d.html。
