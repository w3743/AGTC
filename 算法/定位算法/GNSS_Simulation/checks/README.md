# GNSS 仿真检查脚本

本目录包含仿真实现的检查和诊断入口，结果不代表实车验收。

| 文件 | 用途 |
| --- | --- |
| `check_estimators.py` | 检查各定位估计方法在探针场景下的行为。 |
| `check_geometry.py` | 检查卫星可见性、几何及 DOP。 |
| `check_observations.py` | 检查生成观测与误差来源。 |
| `check_physics_limits.py` | 检查物理模型及边界条件。 |
| `debug_ppp.py` | PPP 估计过程诊断。 |
| `debug_spp.py` | SPP 解算过程诊断。 |
| `smoke_experiment.py` | 用较小规模实验验证基本运行链路。 |
