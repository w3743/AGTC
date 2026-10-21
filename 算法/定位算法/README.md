# 定位算法

本目录包含 GNSS 场景仿真项目和一个独立的平面多基站测距示例；两者都不是已交付的实车定位系统。

## 当前入口

- [GNSS 仿真项目](GNSS_Simulation/README.md)：当前维护的农田场景 GNSS 定位精度仿真。
- [基站测距示例](positioning.py)：基于 NumPy 和 SciPy 的平面多基站测距与最小二乘示例，坐标和测距值均为模拟输入。

`positioning.py` 在导入时也会执行示例。安装 NumPy、SciPy 后，在本目录运行 `python positioning.py`。

## 历史材料

- [原始历史包](归档/GNSS_Sitmulation.zip)：保留原始字节，仅供历史查阅，不作为当前入口。

仿真结果不代表生产部署、实车定位能力或现场精度验收；这些结论须以相应模块的实际证据为准。
