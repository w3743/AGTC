# PROTOCOL_NATIVE_V2 — 原生GPU训练平台验收（DRAFT）

> protocol_sha256:
> 状态：DRAFT。候选后端GPU正确性检查完成后才冻结本文件并生成独立SHA。不得改写V1/V1.1或其原始成绩。

本协议只验收本机Windows 11 ARM64 / Ubuntu-24.04-Arm64 WSL2的原生训练平台与吞吐。Isaac Sim/Kit不参与训练。`capability_metrics_allowed=false`；不据吞吐或开发评估宣布导航、实车或Isaac物理等价。

## 固定配方

- 候选physics名字`newton_mujoco_warp`，Newton1.7.0 + MuJoCo3.15.0 + mujoco-warp3.15.0 + Warp1.18.0，GPU separate_worlds与MuJoCo内部接触；实际revision/solver参数必须进入任务身份，不能借用XPBD名字。主验收固定普通eager GPU执行、NOT_GUARANTEED默认数值模式；种子不保证逐轨迹bitwise复现。Graph仍为实验，不进入本V2正式训练。
- 64 env × 32 steps × 300 iter =614,400 samples，从头训练，seed0、threads8、1ms×50、20Hz策略。模型为五路编码器/LSTM512/PPO，epochs4、lr3e-4、clip.2、value_coef.5、entropy_coef.01、grad_clip.5；不更改奖励以使平台验收通过。
- 显式fast_stop_v2：训练位置容差.4m、低速hold.3s、60s为terminal failure；开发评估保持位置≤.4m、XY速度<.05m/s、全角速度<.1rad/s持续.5s，至多1200步。成功定义不可混算。
- 五路real：LiDAR(1,400)/depth(1,160,88)/detections(16,8)/state(33)/feedback(37)，沿用已登记的解析算法与VQF传感器域差。关节力矩反馈来源必须真实施力或明确标为估计，不能将未施加的PI输出称实测。

## 物理前置门

修正车身全尺寸惯量并逐数组读回；显式接触gap/margin与mu=.8；14轮与2云台执行器的实际2Nm/1Nm限幅必须在过限命令下验证。记录实际质量/惯量/几何/材料、mimic数量与残差、求解器参数和版本。平面代理不等于完整地形/植物物理。

先验收1/3/64 world的接触、有限性、不同控制读回与部分reset隔离。重置需清相应MuJoCo内部缓存，保留其他world状态。最终GPU探针长度、地面穿透和mimic容差在冻结前写实测依据，不在观察结果后放宽。

CUDA Graph严格逐轨迹对照未通过；同默认模式的eager重复也会分叉，原因未单独归因。RUN_TO_RUN初始化在本机约14GiB WSL内存下未完成并安全中止。Graph的隔离物理加速只作为实验，不作为本协议验收数据。

## 正式训练与证据

非有限状态、动作、观测、reward、loss、策略/优化器张量零容忍；出现即失败、保留原始证据。不扩大异常预算、覆盖失败目录或用续训凑足正式300轮。exit0还需核验完整1..300整数metrics/2048 samples每轮/同task SHA/summary/零异常和完整checkpoint；失败时未完成iteration不得标为已完成，故障模型不得默认为可恢复。

记录启动时源raw-byte清单与快照、Git commit/限定dirty范围、wrapper/协议SHA、实际argv、依赖、种子、GPU竞争进程前后观察、1Hz采样、原始stdout/stderr、真实退出和检查点SHA。模型按LFS保存，原始日志/协议不规范化改写。完整launcher墙钟与训练loop/iteration窗口分别报告。

短测固定同子进程前2轮warmup、后5轮实测，两个独立子进程全报告；以总样本/总时间加权。collection、iteration window、全部样本/完整进程墙钟分列；桌面/下载、时钟/负载波动如实记录，不挑最快或以历史异配Isaac成绩宣称严格倍数。

正式完成后验证同身份checkpoint恢复与64→1环境评估链路；开发4-seed固定87010–87013。完整16-seed87010–87025单独标为开发/平台评估，保留830xx未观察，不把已看过的种子称held-out。域差和capability标志保持false，不以成功率充当平台通过条件。
