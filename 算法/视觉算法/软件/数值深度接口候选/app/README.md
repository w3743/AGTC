# 双目深度应用文件

板端状态、当前生产链路和验证限制见[上级说明](../README.md)。部分模块是离线实验，不代表已接入当前服务。

| 文件 | 用途 |
| --- | --- |
| `server.py` | 板端双目采集和伪彩深度服务。 |
| `index.html`、`ui.css`、`ui.js` | 主网页及共享样式、交互。 |
| `calibration.html`、`calibration_nav.js` | 在线标定页面及导航。 |
| `calibration.npz` | 保存的相机标定参数。 |
| `calibration.py`、`calibration_live.py`、`calibration_apply.py` | 标定计算、在线协调与参数切换。 |
| `depth.py`、`fast_depth.py` | 校正双目深度和 CPU SGBM 深度处理。 |
| `depth_wire.py`、`depth_adapter.py` | 数值深度帧编码与离线消费适配。 |
| `benchmark_depth.py`、`benchmark_npu.py` | 固定图像深度性能与板端 RKNN 推理基准。 |
| `lightstereo_depth.py`、`lightstereo_depth.preparallel.py` | LightStereo NPU 深度后端及并行前的历史版本。 |
| `hardware.html`、`hardware_video.py` | 硬件状态页面和板端视频流。 |
| `native_roi.py`、`roi_depth.py`、`range4k.py` | 高分辨率 ROI 深度实验及原始图像到深度网格的坐标映射。 |
| `npu_live.py` | 按需 NPU 深度预览实验。 |
| `pointcloud.py` | 将深度帧导出为点云。 |
| `probe_depth.py` | 采样和保存数值深度 HTTP 数据。 |
| `realtime.py` | 界面叠加和板端资源采样辅助逻辑。 |
| `README.md` | 本目录索引。 |
