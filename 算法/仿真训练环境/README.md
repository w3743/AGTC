# 仿真训练环境

本目录汇集履带底盘的仿真任务、训练程序及运行证据。不同文档记录不同时间的方案与状态，应结合文档日期和对应证据阅读，不把历史运行状态当作当前实况。

**2026-11-07 方向更新：** 用户决定弃用 Isaac。现有 Isaac 文档、脚本与实验记录保留用于追溯，不作为后续训练的默认入口。原生训练栈及 Newton、MuJoCo 相关资料见 [native 训练说明](native/train/README.md) 和 [原生训练栈记录](NATIVE_STACK_PLAN.md)；最终采用的替代后端及迁移验证状态仍需明确，历史结果不自动证明现行方案可用。

| 同目录文件 | 用途 |
| --- | --- |
| `AGENTS.md` | 本目录的协作和技术约束。 |
| `使用说明.md` | Isaac Lab 安装和使用的历史说明，含其他平台记录。 |
| `ISAAC_LAB.md` | Isaac Lab 项目状态与历史实验记录。 |
| `NATIVE_STACK_PLAN.md` | 原生 ARM64 训练栈迁移的历史计划和边界。 |
| `PERF_PLATFORM.md` | 训练平台性能改造过程和实测记录。 |
| `PERF_SUMMARY.md` | 性能改造的状态入口及证据索引。 |
| `WSL_TRAIN_PLAN.md` | WSL 原生训练迁移决策和准备记录。 |
| `pyproject.toml` | `isaac-field` Python 项目的构建、依赖及包配置。 |
| `归档/Isaac/安装IsaacLab.cmd` | 历史 Isaac Lab 安装入口，保留原字节，不再作为运行入口。 |
| `归档/Isaac/启动IsaacLab.cmd` | 历史 Isaac Lab 启动入口，保留原字节，不再作为运行入口。 |

子目录中的源码、工具和项目资料按各自说明使用。原始`results/`训练结果与运行日志已按用户要求删除，删除范围和保留摘要见[训练结果汇总](训练结果汇总.md)。其它文档中保留的`results/`路径仅作历史溯源，不表示对应文件仍存在；`归档/`和`native/acceptance/`中的记录按各自说明管理。`external/`中的第三方代码按项目固定版本管理。

## 自主运行集成交接

仿真环境维护、集成评估和板端回放交接由「AGTC 自主运行集成与离线回放」负责，实车验收由实物测试负责人执行。神经网络结构与策略设计由「继续神经网络研究」负责。模型交付沿用 [Native V2 任务契约](native/train/PROTOCOL_NATIVE_V2.md)，提供检查点、任务契约哈希、观测定义和动作顺序。当前五路输入为 LiDAR、深度、检测、状态和反馈，四路归一化动作是左右履带速度及云台偏航、俯仰角速度。接口若需改变，先对齐训练、回放和板端消费端，再评估新模型。 [动作定义](isaac_field/action_contract.py)

| 验收阶段 | 通过条件与本机状态 |
|---|---|
| CPU 契约与门禁 | 2026-11-04 在本工作区运行 Native 标准库检查，任务身份 12 项、V2 协议 7 项、可在 Windows 执行的启动器检查 3 项通过。依赖原 WSL 发行版的用例跳过。本机 WSL 列表查询报告未安装，且工作树没有本轮可用的正式模型检查点。 |
| 仿真闭环 | 在原训练主机取回匹配契约的检查点与 run 证据，先复用现有 V2 评估入口跑 4 个开发种子。按协议检查到点、低速保持、非有限数据、碰撞和失败原因。开发结果达标后再做独立场景验证，基础场景不能覆盖整田或避苗验收。 [开发评估入口](native/train/README.md#开发评估) |
| 板端回放 | 使用与检查点契约一致的传感器录包，依次核对时间戳、坐标系、单位、传感器有效标记、四路输出和左右履带映射。核查输入过期、通信超时或策略输出异常时，驱动侧按已确认的停止条件处理，记录端到端时延和停驱证据。 [RK3588 平台](../RK3588平台/README.md) · [固件交接](../../固件/README.md) |
| 实车自主运行 | 预充合闸、驱动退出和制动能量路径达到放行条件后，由实物测试负责人按确认工况验收到点、停车及异常退出。当前仿真训练链路和实车自主能力分别判定。 [整车电力边界](../../电力/基本设计.md) |

### CAN 回放与固件交接

2026-11-04 CPU 回放核对通过。仿真目录的 [黄金向量](tests/fixtures/foc_drive_feedback_v1_golden_vectors.json) 与 [FOC C 接口源向量](../../电力/FOC驱动模块/软件/golden_vectors.json) 字节一致。7 帧 CRC、左轮正转和右轮反转命令编码、同序号反馈拼接以及反馈过期后的失效处理均已实跑。回放中两路速度命令 ID 分别为 `0x501`、`0x502`，反馈超出100ms后上层适配器不再返回该样本。

板端联调保持待验收。现有 FOC v1.0 的 [CAN 数据层](../../固件/foc_driver/DataLayer/Inc/dl_can.h) 声明旧的 `0x180` 状态与 `0x200` 命令接口。工程源码中未找到新协议编解码器接入该固件的记录。上板前先绑定原理图和烧录固件版本，由固件实施方交付同一协议下的命令接受、左右轮方向、机械 RPM 单位、TTL 到期和停驱实测。上层丢弃过期反馈的结果只覆盖适配器行为，驱动侧命令超时和停驱单独验收。

### 履带速度与机械转速交接

2026-11-04 离线核对了 [机械参数基线](../../机械/参数基线.json)、[仿真速度尺度](native/proto/physics_backend.py) 与 [CAN 编码](isaac_field/telemetry.py)。仿真前两维动作的满量程为 15 km/h（约 4.167 m/s）。按 4:1 减速和 60 mm 名义主动轮直径，履带速度 `v` 转电机机械转速的计算是 `n = 60 × 4 × v / (π × 0.060)`。以项目名义值计算，1 m/s 约 1273 RPM，15 km/h 约 5305 RPM。仿真 45 mm 支承轮代理半径与名义主动轮半径 30 mm 各自用于对应模型。将 45 mm 代入板端换算会得到目标转速的三分之二。

已用现有编码器核对 0、0.05、1 和 4.167 m/s 的左右路带符号机械转速，共 8 个命令。签名和取整均在协议字段内。回放、台架和实车应分别记录归一化动作、限速后的目标履带速度、指令与反馈的机械 RPM。上板验收须核实左右路正方向、实际减速比和有效节圆，并在各侧统一应用该工况批准的限速和加减速边界。仿真满量程与协议字段宽度各自属于输入约定。停车判定使用车体运动证据，履带打滑或轮速归零时仍须确认车体实际停稳。

### 本机闭环前置检查

2026-11-04 追踪了现有回放入口。RSL 推理将有限动作裁到 `[-1, 1]`，LASSC 消费端检查输入和动作，回放仅在收到就绪结果后调用 `action_commands`。转换函数以 `cfg.max_speed` 缩放前两维，按名义主动轮和减速比得到两侧机械 RPM，再生成 CAN 候选帧。这里没有独立的部署限速或加减速逻辑。仿真物理后端的速率限制属于另一条路径。 [推理](isaac_field/rsl_rl_inference.py) · [消费端和转换](scripts/lassc_consumer.py) · [回放入口](scripts/lassc_replay.py)

| 证据范围 | 结果 |
| --- | --- |
| 离线执行通过 | 从源码隔离运行转换函数，取测试上限 0.5 m/s，动作 `[1, -0.5, 0, 0]` 得到 `[0.5, -0.25]` m/s，对应 `0x501 / +637 RPM` 和 `0x502 / -318 RPM`，返回 `transmitted=False`。转换函数拒绝 NaN、Inf 和越界输入，不返回候选帧。另以 CPU Torch 和注入的 actor 隔离验证 RSL `_predict`。原先裁剪会把 ±Inf 变成满幅动作，现先检查原始输出。复测拒绝 ±Inf 与 NaN，有限越界值仍裁剪，actor 隐状态恢复。 |
| 消费端过期门禁 | 深度超过配置年龄，或 LiDAR、相机帧过期，或当前姿态反馈无效、反馈快照时刻与控制时刻不符，消费端返回未就绪。策略异常时恢复 runner 状态并抛出异常，回放在候选帧生成前退出。已复现原转换入口在没有 FOC 反馈时仍生成两路运动候选。现由 `action_commands` 按回放控制时刻从同一个 `TelemetryAssembler` 检查双侧配对反馈，沿用其 100 ms 默认窗口及 RPM、Iq 有效位。任一侧缺失、无效或过期时抛出带侧别的错误，整个调用在编码前退出。门禁只作用于候选帧生成，纯观测打包仍按原规则。CPU 隔离执行 8 项通过，覆盖双侧新鲜、左和右分别过期 101 ms、缺失和 RPM 无效。用原始左侧黄金向量及由其重算右侧 ID 和 CRC 的合成右侧样本。证据止于实际 CAN 解码、反馈配对和候选帧生成函数。 |
| 本机完整回放受阻 | `lassc_replay.py` 在导入 `isaac_field/feedback.py` 的 `scipy.spatial.transform.Rotation` 时失败。当前本机可用的工作区 Python 同时缺 PyTorch，仿真工作树无匹配的 RSL 检查点。完整消费链、实际总线停发和零速发送均待验证。 |
| 板端待证 | 绑定控制周期、时钟映射、两侧有效位及新鲜度、部署限速和加减速、左右方向、CAN 会话与 TTL。用健康录包先确认两侧方向和机械 RPM，再从已有非零指令分别注入深度、LiDAR、相机、姿态及任一侧电机反馈过期，以及策略 NaN/±Inf。逐项捕获最后一条非零帧、零速帧或停发、驱动 TTL 到期、两侧退出和车体停稳时间。台架须分别验证上层正常发零与上层停发时的驱动退出。 |
| 正式仿真待检查点 | 取得匹配 Native V2 契约的正式检查点和运行证据后，按上表的仿真闭环阶段执行四开发种子评估。 |

在已有 CPU 回放环境中，从仓库根目录运行以下命令。将 `$rslCheckpoint` 指向该环境中的兼容 RSL 检查点，先验证依赖和文件，再执行回放和专项回归。正式 Native V2 评估另按其任务契约校验检查点。

```powershell
Set-Location '算法/仿真训练环境'
$rslCheckpoint = 'C:\path\to\compatible-rsl-checkpoint.pt'
python -c "import torch; from scipy.spatial.transform import Rotation"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $rslCheckpoint -PathType Leaf)) { throw 'Replay environment or RSL checkpoint is missing' }
python scripts/lassc_replay.py --checkpoint $rslCheckpoint --output "$env:TEMP/lassc-replay.json"
python -m pytest -q tests/test_rsl_rl_inference.py -k rejects_nonfinite_actor_output_before_clipping
python -m pytest -q tests/test_lassc_replay.py -k requires_fresh_valid_feedback_on_both_sides
```

额度控制采用逐级门禁。先用已有源码、冻结协议、检查点和录包完成可运行的小检查，再扩展评估或训练。已有训练任务和研究对话继续原分工。
