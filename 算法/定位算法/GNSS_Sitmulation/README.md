# 农田场景 GNSS 土壤探针定位精度仿真

在 **100 m × 100 m、无高大作物遮挡**的农田里，用 GNSS 给随机布设的**多个土壤探针**
定位，把真实世界的误差源（电离层、对流层、地表多路径、天线特性、星历钟差、
基准站误差……）都带进仿真，回答一个问题：**定位精度到底有多少？**

## 快速开始

```bash
pip install numpy pandas matplotlib

python run_sim.py          # 跑全部实验，输出 results/*.csv     (~8 min)
python make_figures.py     # 画图，输出 figures/*.png
python report.py           # 生成中文报告 report.md
```

自检脚本（建议先跑，确认模型量级正确）：

```bash
python checks/check_geometry.py       # 星座几何：可见数/DOP/GEO定点经度
python checks/check_observations.py   # 误差源量级：电离层/对流层/多路径/噪声
python checks/check_estimators.py     # 解算器：各模式单探针误差
python checks/check_physics_limits.py # 解析极限：Fresnel 反射、双射线延迟、
                                      #   对流层/Klobuchar、消电离层组合、
                                      #   坐标链、轨道周期、模糊度固定
```

`checks/` 下另有 `smoke_experiment.py`（小规模冒烟测试）与
`debug_spp.py` / `debug_ppp.py`（定位异常逐项归因，开发时用于定位
“广播钟差漂移量纲错误”“模糊度随时间重算”等问题）。

> 若环境中没有 matplotlib，`make_figures.py` 会自动改用工作区内的本地依赖目录
> `.pylibs`（可用 `pip install --target .pylibs matplotlib` 创建）。

## 主要结论（详见 report.md）

| 模式 | 水平误差 95% | 说明 |
|---|---|---|
| 多系统单点定位 (SPP) | ~2.4 m（垂直 8.5 m） | 只靠广播星历 + Klobuchar |
| 星基增强 (SBAS) | ~0.8 m | 无需通信链路 |
| 本地基准站伪距差分 (DGNSS) | ~0.47 m（1 h 平均 0.13 m） | 需要基准站与链路 |
| 静态 RTK (10 min，坐标已标定) | ~0.4 cm（中位 0.2 cm，垂直 2.0 cm） | 基准站坐标必须标定 |
| 静态 RTK (基准站坐标随手测) | ~2.7 m | 基准站坐标误差吃掉全部优势 |
| 单基站长基线 RTK (15 km) | ~0.6 m（仅 47% 可固定） | 大气去相关，垂直方向偏差明显 |
| PPP (30 min 静态) | ~0.11 m（中位 3.8 cm） | 无需基准站，需收敛时间 |

**核心结论：开阔农田的定位瓶颈不是遮挡与多路径，而是改正信息来源。**

## 代码结构

```
gnss_sim/
  geodesy.py         WGS84 坐标/时间基准、ENU 局部坐标系、仰角映射
  constellation.py   GPS/BDS-3/Galileo/GLONASS 星座构造与 J2 轨道传播、DOP
  models.py          误差源物理模型
                       · 电离层：薄壳 VTEC 真值 + Klobuchar 广播模型
                       · 对流层：Saastamoinen 干湿分量 + AR(1) 时变 + 梯度
                       · 多路径：Topp 土壤介电常数 → Fresnel 复反射 → 双射线
                                  + 粗糙面漫反射（相干 + 非相干）
                       · 天线：增益方向图、左右旋抑制、相位中心变化 PCV
                       · 噪声：C/N0 模型与码/相位噪声
                       · 星历钟差：SISRE 量级轨道/钟差误差
  observation.py     观测合成（真值/模型分离），探针与基准站接收机状态
  estimators.py      解算器：SPP / SBAS / DGNSS / RTK(条件方差序贯固定) / PPP
  metrics.py         CEP/RMS/95 分位、相对精度、管理分区归属错误率
  experiments.py     蒙特卡洛实验驱动与统计汇总
run_sim.py           6 组实验的主入口
make_figures.py      9 张结果图
report.py            中文分析报告生成
checks/              三个模型自检脚本
results/             仿真输出 CSV
figures/             图
```

## 建模要点

1. **真值/模型分离**：观测量由真实几何与真实误差源合成，解算器只使用它
   “应该知道”的模型量（广播星历、Klobuchar、标准大气等），两者之差自然形成
   定位误差，不人为注入误差。
2. **物理驱动**：土壤含水率通过 Topp 式影响介电常数，经 Fresnel 反射与双射线
   干涉决定多路径，而不是给一个经验误差值。
3. **公共随机数**：敏感性分析的各变体共用同一批随机布局、大气与星历实现，
   使差异只来自被改变的那个因素。
4. **基准站真实性**：基准站有自己的接收机、噪声与多径；“基准站坐标由单点定位
   测得”作为一个独立场景，用于暴露“相对精度 ≠ 绝对精度”这一常见误区。

## 模型局限

多路径为解析模型并按实测量级标定（绝对量值 ±50% 不确定度，趋势可靠）；
模糊度固定为条件方差序贯取整 + 残差检验，非完整 LAMBDA；PPP 为静态多历元
最小二乘（浮点模糊度），未模拟滤波过程噪声。详见 `report.md` 第 6 节。

> **模型修正记录**：本工程经过一次独立对抗性复核。复核确认星座/坐标/时间链、
> Klobuchar、Saastamoinen、Topp、J2、观测方程符号、消电离层组合与模糊度固定
> 逻辑均正确，并发现一处公式性错误——地表反射的 Fresnel 系数曾把**卫星仰角**
> 当作**入射角**代入（正确关系 cos θi = sin el），使反射率的仰角依赖方向被反转。
> 该错误已修正，并在 `checks/check_physics_limits.py` 中补充了正入射/掠射/
> 理想导体极限与单调性断言（全部通过）；`results/` 与 `report.md` 均为修正后重跑结果。
