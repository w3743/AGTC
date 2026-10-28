# 能力验收剩余缺口：depth 160×88 与 detections 16×8（设计 + CPU 估算，不实现）

> 现状：`capability_metrics_allowed=false`；depth 为常数 0.5 占位，detections 全零占位。
> 本文只做设计与 CPU 侧成本估算（不跑 GPU、不实现）；参考实现位置：
> `isaac_field/env.py:363-375`（RTX Camera depth/detector 配置）、`task.py`（深度内参、检测过滤/噪声）、`task.py:1264`（detection 消费）。

## 1. depth 160×88（Warp raycast 可行，但需先解决"解析 vs 栅格化"域差）

**参考链路**：Isaac RTX `Camera` + `distance_to_image_plane`，160×88，2.5 Hz（每 8 个 policy step），latency 300 ms；归一化 `[0,1]`（`depth_range=5 m`）；内参由 352×640 原生内参缩放（fx≈98.32、cx≈44.01、cy≈80.56；`task.py:769-780` 已有 `depth_camera_xy`）。

**拟实现**：同一套解析几何（leaf patch + stem cylinder + terrain + rocks）逐像素 raycast；每 env 14,080 条射线，64 env = 901,120 条。

**成本估算（CPU 估算 + 已有 warps 实测换算）**：
- 已有实测：解析 raycast（terrain + 400 plants canopy-only）CUDA **1.536 ms/帧**（901k 射线，≈587 M rays/s，平均 3.6 株候选/射线）；CPU 258 ms/帧。
- 加上每株 8 个 leaf patch 后，图元测试数约 ×2.7（≈32 图元/射线）→ 估 **CUDA 4–6 ms/帧**；2.5 Hz 摊销到 20 Hz policy step ≈ **0.5–0.8 ms/step**（相对 50 ms 预算可接受）。
- CPU 侧：估 **1–2.5 s/帧**，不可用于训练循环，只做正确性对照。
- 结论：**GPU 成本可行**；瓶颈在等价性与域差，不在算力。

**域差与验证**：
- RTX 是光栅化 + Z-buffer，解析是 ray-三角形/椭球求交：边缘像素、薄叶片遮挡、亚像素覆盖、无抗锯齿 → 系统性差异。
- RTX 深度是 **轴向 Z**；解析 raycast 给径向距离 → 需按内参反投影换算（每像素 cos 因子），否则全场偏差可达数 %。
- 材质/相机畸变/TAA 在解析侧不存在。
- 建议验证顺序：① CPU 单平面/单叶解析 vs 独立栅格化（算法等价）；② 固定相机位姿下与 Isaac RTX 离线渲染逐像素对照（域等价），容差建议：平面区域 ≤1e-3 m、边缘 1px 排除、p95 ≤2e-3 m。
- 工作量：实现 + CPU 验证 **2–3 人日**；与 Isaac RTX 逐像素对照 **1–2 人日**（依赖 GPU 时段）。

## 2. detections 16×8（可从解析几何推导，但语义依赖下游检测器）

**参考链路**：Isaac 64×64 instance-segmentation 相机（5 Hz、latency 120 ms）；`task.py` 从 mask 产出 bbox/class/range，再经过 `class_error=0.015`、`detection_drop=0.08`、`range_noise=0.01` 等噪声/过滤（`task.py` 检测消费段）。

**拟实现（解析投影）**：
1. 用与 depth 相同的 raycast（64×64=4,096 射线/env）或每株几何的解析投影得到每个像素的 plant id → mask；
2. mask → 每株一个 bbox（规则需与 Isaac 的 mask→bbox 完全一致）；
3. class 由 `scene.leaf_dry` 推导、range 取最近命中 `t`，再复用 `task.py` 的噪声/过滤（drop、class error、range noise）；
4. 输出 16×8 结构化张量（box 4 + class + confidence + range + 保留位，按 `task.py` 实际 dtype/布局）。

**成本估算**：262k 射线/帧（64 env）→ CUDA ≈**0.4–0.6 ms/帧**（5 Hz → 0.1 ms/step 摊销）；CPU ≈**70–150 ms/帧**（可做对照）。

**风险**：detections 不是"纯几何"输出——它依赖 instance-seg 的像素级 mask 边缘、遮挡顺序、检测器/规则生成的 confidence；解析投影通常更"干净"。**结论：可做，但语义等价门槛比 depth 高**，需要先确认 `task.py` 中 mask→检测框/置信度的确切规则，并做离线对照。

**工作量**：实现 + CPU mask/bbox 对照 **2–3 人日**；与 Isaac 端到端对照 **≈1 人日/轮**。

## 3. 与 Isaac RTX 的域差登记方式（contract/verification 新增字段）

每个 RTX 通道在 contract/verification 中增加：

```json
{
  "channel": "depth",
  "isaac_source": "RTX Camera distance_to_image_plane, 160x88, 2.5 Hz, latency 0.3 s",
  "native_source": "warp analytic raycast (leaf patch + stem + terrain)",
  "expected_differences": ["rasterization vs raycast", "axial-Z vs radial range", "no AA/material/TAA",
                            "edge/occlusion boundary pixels"],
  "verification": {
    "algorithm_equivalence": {"reference": "CPU offline rasterizer", "max_abs_err_m": null, "p95_m": null},
    "domain_equivalence": {"reference": "Isaac RTX offline capture", "max_abs_err_m": null, "p95_m": null,
                            "edge_policy": "exclude 1px border"},
    "status": "not_run"
  },
  "acceptance": {"usable_for_capability": false, "tolerance": "plane <=1e-3 m, p95 <=2e-3 m"}
}
```

规则：
1. 未完成逐像素/逐检测对照前一律 `usable_for_capability=false`；
2. 报告必须区分「算法等价」（解析 vs 离线栅格化）与「域等价」（vs Isaac RTX）；
3. 域差若不可消除（抗锯齿/材质），需在验收口径中写明"允许的差异统计量"而不是伪装成逐像素相等；
4. detections 额外要求 mask→bbox/confidence 规则与仓库 `task.py` 一致，并在 contract 里记录规则来源。

## 4. 汇总

| 通道 | 算力可行性 | 主要风险 | 估算工量 |
|---|---|---|---|
| depth | GPU 4–6 ms/帧（0.5–0.8 ms/step） | 光栅化 vs raycast 的边缘/遮挡域差、轴向 Z 换算 | 2–3 人日 + 1–2 人日对照 |
| detections | GPU 0.4–0.6 ms/帧（0.1 ms/step） | mask→bbox/confidence 语义依赖，域差更难消除 | 2–3 人日 + 1 人日/轮对照 |

合计 **6–9 人日**（不含 GPU 时段等待）；建议顺序：depth 先做（几何语义清晰、可逐像素对照），detections 后做（先确认 task.py 规则）。
