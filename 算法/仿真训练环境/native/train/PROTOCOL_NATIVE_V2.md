# PROTOCOL_NATIVE_V2 — 原生 GPU 训练平台验收

> protocol_sha256: 83db5a6e512d7d7244bc47ddf2059720c757504270e9bbf9129d0c7d8f92f87e
> 冻结日期：2026-10-25。正文变化必须建立新协议版本；V1/V1.1及原始成绩保留原字节。

## 范围与身份

本协议验收本机 Windows 11 ARM64 / RTX Spark N1X / Ubuntu-24.04-Arm64 WSL2 的训练链路、稳定性与实测吞吐。训练不启动 Isaac Sim/Kit。capability_metrics_allowed=false；不以吞吐或开发评估宣布导航、实车、完整地形/植物碰撞或 Isaac 物理等价。Windows ARM64 原生 CUDA Python 未实测。

唯一正式 task identity SHA256 为 `209016904867c97633eb006cbb8f52fc75d95f82d74463e7f1d05ad49ecbc1b2`，schema 为 native_task_contract_v2；物理后端为 newton_mujoco_warp，revision 为 newton_mujoco_warp_v6_deterministic_identity，实际 device=cuda:0。每 world 拓扑为19 bodies/19 joints/24 DOF/14 mimic equalities/16 actuators；并行数规范化后64→1须保持相同SHA。其他物理、任务、奖励或通道身份拒绝。

实际版本固定 Newton1.7.0、MuJoCo3.15.0、mujoco-warp3.15.0、Warp1.18.0；测量环境为 Python3.12.3、torch2.14.1+cu130、NumPy2.5.4、VQF2.1.2。requirements-native-v2.txt登记已观察匹配环境；本机为隔离overlay，未验证全新安装。

## 固定配方

- 64 env × 32 steps × 300 iter = 614,400 samples，从头训练、不resume，seed0、threads8、1ms×50、20Hz策略；每20轮保存检查点。
- 五路编码器/LSTM512/PPO：epochs4、lr3e-4、clip.2、value_coef.5、entropy_coef.01、grad_clip.5；不更改奖励以使平台验收通过。
- fast_stop_v2：训练位置容差.4m、低速hold.3s、60s为terminal failure。开发评估采用位置≤.4m、XY速度<.05m/s、全角速度<.1rad/s持续.5s、至多1200步；训练与评估成功定义不能混算。
- 五路real：LiDAR(1,400) Warp、depth(1,160,88) analytic Warp、detections(16,8) analytic detector Warp、state(33)、feedback(37) VQF。沿用登记解析算法与传感器域差。力矩反馈来自实际QFRC_ACTUATOR，不把未施加的侧向PI输出称实测。
- 普通eager GPU；数值模式NOT_GUARANTEED，seed不保证bitwise逐轨迹复现。Graph请求和实际启用均为false。Graph严格对照未通过，eager重复也分叉；RUN_TO_RUN初始化因本机内存压力被SIGTERM终止，未产生policy数据，不作为通过或吞吐证据。

## 物理与运行配置

SolverMuJoCo使用MuJoCo-Warp内部接触、separate_worlds=true、update_data_interval=1；Newton solver、implicitfast integrator、elliptic cone、impratio10、outer iterations40、line-search iterations8，nconmax64/world、njmax256/world。实际参数、模型数组与版本记录到identity，不只保存请求值。

车身惯量采用全尺寸盒体公式，diag=(.2475,1.4575,1.606) kg m²；gimbal_pitch diag=(.0000133333333,.0000133333333,.0000133333333)。显式摩擦mu=.8、接触gap=margin=0。14条mimic在CPU与GPU导出数组均显式设solref=(.004,1)、solimp=(.999,.9999,.001,.5,2)。14轮/2云台实际执行器限幅2Nm/1Nm；轮组速度servo、云台位置servo，不施加侧向PI torque feed-forward。

仅关闭官方per-model warn_overflow中的LS_ITERATIONS控制台bit，实际mask=3071；其余warning bits保持。每1ms先清overflow再由设备端累计fresh capacity/outer/LS事件，在policy边界读取；不清除或隐瞒LS事件。runtime_diagnostics记录mask和执行模式，这些不改变物理task SHA；原始源码及运行profile仍必须单独冻结。行搜索上限事件属于报告项，不能冒称求解器每步完全收敛；容量溢出为失败。

## 前置物理门（本次新探针执行前登记）

1/3/64 world分别20 policy steps、每步50子步，共1000子步；使用最终v6源码、相同物理配置和eager/NOT_GUARANTEED模式。steps0–3 zero，4–11不同world固定输入，12–19线性ramp；ramp前reset选定world。保存输入、world映射、数组来源、启动前源码快照、原始结果和真实退出。额外限幅单子步须单列，不冒充20步中的数据。

每采样step状态/控制/真实力矩全有限；capacity overflow=0。保守地面clearance≥−.015m，报告几何计算方法；连续reduced q的未包角affine mimic residual最大值≤.001rad，不能用wrapped IK替代。过限命令下真实QFRC_ACTUATOR分别≤2/1Nm+1e-5Nm。逐world记录真实接触缓冲及稳态阶段接触；刚reset悬空时不要求立即接地。选定world reset不改其他world状态/控制原字节，reset后继续step有限且控制有效；1world跨环境隔离项记N/A。

上述阈值于本次v6探针执行前登记，旧XPBD/v4候选结果不回填为本门通过。短探针不替代正式300轮。

## 正式训练与证据门

非有限状态、动作、观测、reward、loss、策略/优化器张量零容忍；出现即失败。不得提高异常预算、覆盖失败目录、自动重启或用续训凑足正式300轮。真实exit0还须核验完整整数metrics 1..300、每轮2048 samples、相同task SHA、summary、零异常、无capacity truncation、完整checkpoint及CPU安全读回张量有限。

每轮physics_solver_events须记录102400 world-substeps及capacity/outer/LS计数；各计数为合法整数且不超过world-substeps，capacity必须为零；outer/LS全量报告，不能删去非零事件。失败未完成iteration不得称已完成，故障模型不得默认为可恢复。

保留启动前源raw-byte清单与快照、Git commit与限定dirty范围、协议/wrapper SHA、实际argv、依赖/seed、Win/WSL竞争进程前后观察、1Hz GPU采样、原始stdout/stderr、真实退出与checkpoint SHA。源运行前后必须稳定；模型按LFS保存，原始日志、冻结协议与源快照不规范化改写。桌面/下载活动和时钟波动如实记录，不声称全系统完全空闲。

完整wrapper墙钟、训练loop和iteration窗口分别报告；sample为一个环境完成一个0.05s策略步。短benchmark固定同子进程前2轮warmup、后5轮测量，至少两个独立子进程全报告，以总样本/总时间加权；分别报告collection、iteration window、含warmup全部样本/完整进程墙钟。使用主机计时，不称kernel时间；不挑最快，不以历史异配Isaac成绩宣称严格等价倍数。

正式完成后核验同身份checkpoint的policy/optimizer续训状态延续和64→1评估链路；检查点未保存RNG状态，不能声称bitwise随机流续接。开发4-seed固定87010–87013、1200步；16-seed87010–87025可另报开发结果，830xx保持未观察，不把已观察种子称held-out。capability_metrics_allowed保持false，成功率不是平台通过条件。
