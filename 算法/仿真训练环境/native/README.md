# Native 训练栈

本目录的当前正式入口是 WSL2 ARM64 上的 Native V2：Newton MuJoCo Warp V6、MuJoCo 3.15.0 匹配环境、普通 eager GPU、NOT_GUARANTEED。协议已冻结；当前正式 300 轮 trial 仍在运行，尚未有最终通过结论。状态、身份、性能记录和历史失败见 [迁移与验收说明](MIGRATION_ACCEPTANCE.md)。

## 当前验证状态

- Native V2 protocol SHA256：83db5a6e512d7d7244bc47ddf2059720c757504270e9bbf9129d0c7d8f92f87e。
- Fast-stop task identity SHA256：209016904867c97633eb006cbb8f52fc75d95f82d74463e7f1d05ad49ecbc1b2。
- V6 的 1/3/64-world 前置物理门、同身份 resume 和 64→1 strict-evaluation smoke 已通过；正式 run 必须单独完成完整 300 轮门。
- Windows 11 ARM64 / RTX Spark N1X 上实际训练路径是 Ubuntu 24.04 ARM64 WSL2，不是 Windows 原生 CUDA Python。

## 目录结构

~~~text
native/
├── proto/     # 物理、观测、策略与环境原型
├── train/     # V2 固定训练/评估 runner、训练入口与协议校验
├── viz/       # 历史资产与 2D/状态可视化组件
└── remote/    # 远程执行模板
~~~

runs、checkpoint、日志与实时状态不纳入版本库。

## 匹配环境

当前实测解释器为 /opt/venv_mujoco315_smoke/bin/python。核心版本为 Python 3.12.3、torch 2.14.1+cu130、NumPy 2.5.4、Warp 1.18.0、Newton 1.7.0、MuJoCo 3.15.0、mujoco-warp 3.15.0、VQF 2.1.2。requirements-native-v2.txt 记录实际观察到的版本；现环境为隔离 overlay，未验证全新安装重建。旧 /opt/venv 与 MuJoCo 3.17.0 属于旧 XPBD，不得用于 V2 runner。

## 训练与评估

正式训练使用新的唯一 tag，每次由 runner 创建新的时间戳目录，从头开始，不续训旧 XPBD checkpoint：

~~~bash
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native
PYTHON=/opt/venv_mujoco315_smoke/bin/python bash train/run_native_v2_trial.sh <unique-tag> \
  --live-state-file /mnt/c/pioneer/agtc-native-acceptance-backup/native_v2_live/state.json \
  --live-state-interval 5
~~~

runner 固定 64 env × 32 steps × 300 iterations、seed 0、LSTM-512、fast_stop_v2、五路真实观测、1 ms × 50 与 zero-nonfinite gate；仅 live-state viewer 参数可作为附加项。

单独开发评估：

~~~bash
PYTHON=/opt/venv_mujoco315_smoke/bin/python bash train/run_eval_v2.sh <checkpoint.pt> <new-output-dir> pilot
~~~

默认 pilot 使用 seed 87010–87013、最多 1200 policy steps。runner 在加载评估环境前校验协议与 checkpoint contract；物理或 task 不匹配会拒绝运行。16-seed 开发评估使用显式 formal 模式，仍不构成 held-out 能力验收。

## 实时界面与能力边界

实时 viewer 消费训练过程每 5 policy steps 发布的 raw state、目标与每环境任务时间；静态车辆 geometry 使用冻结 CPU 模型导出的 19-body primitives/local transforms，地面绘制实际 flat plane。旧 Isaac USD 缺少两个云台和两个前导轮共 4 个 body 的几何，旧 viewer 方块 fallback 不再用于 Native V2；旧 USD 不作为车辆或地面的真值。界面不显示传感器图像，也不声明 CPU 几何导出与 GPU shape arrays 逐字节相同。

代理物理、解析传感器与 RTX/实车域之间仍有差异。capability_metrics_allowed=false；平台链路或吞吐通过不能代替导航能力、完整地形/植物碰撞或实车验收。

V1/V1.1 和 Newton XPBD 历史入口与失败证据保留，冻结协议不编辑。5 ms/10 只能作实验配置。