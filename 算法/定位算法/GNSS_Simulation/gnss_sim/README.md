# GNSS 仿真 Python 包

本目录提供仿真数据和估计方法；整个项目的运行与结果边界见[上级 README](../README.md)。

| 文件 | 用途 |
| --- | --- |
| `__init__.py` | Python 包入口。 |
| `config.py` | 场景、探针、实验与定位模式配置。 |
| `constellation.py` | GNSS 卫星星座与轨道模型。 |
| `estimators.py` | SPP、SBAS、DGNSS、RTK、PPP 定位估计。 |
| `experiments.py` | 实验执行和结果统计汇总。 |
| `geodesy.py` | WGS84 地理参数及坐标、时间转换工具。 |
| `metrics.py` | CEP、RMS、分位误差等定位评价指标。 |
| `models.py` | 仿真误差源模型。 |
| `observation.py` | 合成伪距及载波相位观测。 |
