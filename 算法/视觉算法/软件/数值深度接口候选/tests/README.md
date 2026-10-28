# 深度接口测试

本目录包含数值深度、图像处理和板端接口的自动化测试。结果须以实际测试输出为准。

| 文件 | 用途 |
| --- | --- |
| `test_depth_adapter.py` | 深度数据消费适配测试。 |
| `test_depth_benchmark.py`、`test_depth_npu_benchmark.py` | CPU 深度与 NPU 基准测试。 |
| `test_depth_contract_acceptance.py` | 深度接口契约验收检查。 |
| `test_depth_http_transport.py`、`test_depth_producer_http.py` | HTTP 传输与生产者接口检查。 |
| `test_far_routes.py` | 远距相关路由测试。 |
| `test_hardware_monitor.py` | 硬件监测接口测试。 |
| `test_native_roi.py`、`test_roi_depth.py` | ROI 深度处理测试。 |
| `test_pointcloud_geometry.py` | 点云几何计算测试。 |
| `README.md` | 本目录索引。 |
