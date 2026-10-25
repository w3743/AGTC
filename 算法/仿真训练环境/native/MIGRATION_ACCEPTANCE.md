# RTX Spark N1X 原生训练迁移与验收

> 当前状态：正式验收未通过。v3短探针/短测通过，但全新300轮在第58轮出现1个环境的非有限物理状态，严格门正确以exit3拒绝。正在定位长期稳定性；旧pilot06的330.2 samples/s和7.7倍也无效。

目标机器为Windows 11 ARM64 / RTX Spark N1X，实际验收路径为Ubuntu-24.04-Arm64 WSL2、ARM64 Python、PyTorch CUDA、Newton XPBD、Warp与VQF。训练不启动Isaac Sim/Kit；Windows ARM64原生CUDA Python路径未实测。

## 当前实现与物理修复

入口为`train/train_wsl.py` → `proto/native_env.py` → Newton物理与Warp/解析传感器 → 五路编码器 → LSTM-512 → PPO。仍有主机/GPU往返；移除的是Kit/PhysX逐物理子步fetch_results路径，不宣称全流程没有同步。

正式配方为64 env × 32 steps × 300 iterations = 614,400 samples，seed 0，1 ms × 50（20 Hz策略），LSTM-512，显式`fast_stop_v2`。五路real观测为LiDAR `(1,400)`、depth `(1,160,88)`、detections `(16,8)`、state `(33)`、feedback `(37)`。原型CLI仍保留simplified/占位通道兼容默认值，正式复现必须使用正式runner。

本轮修正：

- Newton每个1 ms子步实际生成contacts并传入XPBD；旧`contacts=None`跳过刚体接触，车辆可跌穿地面。
- 按实际world/joint/coordinate映射分别控制和读取轮组、云台与车体；partial reset只写选定articulation，其他world状态/控制保持原字节。模型、状态、控制和contacts均在所请求device。
- 每个ShapeConfig显式设摩擦0.8，身份读取模型实际mu/ke/kd；旧构造器局部默认配置使动态形状实际mu=1.0。
- 新物理revision为`newton_xpbdg_v3_contacts_worldmap_materialcfg`。checkpoint的`native_task_contract_v2`核对任务、奖励/终止、物理和每环境拓扑；64环境训练可转1环境评估，不同revision、任务或物理步长拒绝。
- 正式runner完整记录日志和真实退出码，固定配方不可被extra参数覆盖；使用`--nonfinite-limit 0`。验收重读完整300轮metrics、summary、异常事件与checkpoint，不能只凭exit0或目录名通过。旧manifest不回填。
- 评估入口统一1 ms/50，核对任务SHA、检查点、每个种子及每行结果；有运行时/非有限异常则拒绝。旧checkpoint仅能显式选择legacy探索评估，不能得到strict_match。

冻结V1/V1.1协议和历史模型/成绩保留原字节。冻结正文仍有旧5 ms示例与wsl_train路径，当前执行以此说明及runner为准。没有改变奖励或续训旧错误物理模型。

## v3接触/环境隔离修复后的实测（正式未通过）

源码提交`c94553e2b4831848b66700d4dd0d4f9e977b343d`；相关源码字节在运行前后稳定。source manifest SHA256为`dad7add0aec5e0f5d84289972ee00ec7bca8b68c34d928f5387338b6c19da9a6`，task identity SHA256为`f04ab2db16e4bcfd5c5c0ade1fe3eb39687ea20c2f77440d3e9fccc71d76a329`。

两次独立子进程使用同配置、seed 0，各连续7轮，预先固定前2轮warmup、后5轮测量（10,240 samples）。没有清空系统/Warp编译缓存。两次五路均real、非有限状态/动作/恢复计数均0，checkpoint策略与优化器共132张量均有限。Windows/WSL前后快照及GPU进程列表未发现其他训练/推理任务；桌面与下载活动保留。

| 口径 | rep01 | rep02 |
|---|---:|---:|
| Collection samples/s | 317.599（32.2419 s） | 191.149（53.5707 s） |
| PPO update samples/s | 2280.420（4.4904 s） | 1375.106（7.4467 s） |
| Iteration window samples/s | 278.538（36.7634 s） | 167.671（61.0720 s） |
| 完整launcher wall | 72.9202 s | 100.8975 s |
| 全部14,336 samples / launcher wall | 196.598 samples/s | 142.085 samples/s |

rep02的iteration吞吐较rep01低约39.8%。这是已观察波动，原因尚未定位；不能据两次短测宣称稳定提速倍数。所有重复均保留，不挑最快值。

原始结果位于[rep01](acceptance/20261025_v3_rep01/benchmark_result.json)、[rep02](acceptance/20261025_v3_rep02/benchmark_result.json)，以及[两轮对比](acceptance/20261025_v3_repeats_comparison.json)。相邻同名前缀文件保留Win/WSL观察、只读summary和CPU checkpoint审计。此前2+1轮scout单独存于`acceptance/20261025_v3_eager_baseline`，不混入两轮统计。

正确性检查独立于吞吐：6项真实Newton CPU隔离测试；GPU 3 world × 1400策略步（200零动作+600固定+600随机），实际contact存在，无非有限状态，最低保守地面间隙约−0.000124 m（容差−0.015 m）；GPU 64 world × 8步，64份映射独立，contact存在且无非有限状态。短探针不代表300轮学习已完成。

全新正式trial `native_v1_1_fixed_physics_v3_20261025_184052_20261025_184704`使用相同v3源码，从头开始，目标300轮。北京时间18:58:07在iter58/step21出现env11非有限pos/速度/轮速，动作有限；训练按零异常门exit3，formal_eligible=false，只完成57轮，完整wrapper wall666.72 s。最后周期checkpoint_40的132个策略/优化器张量均有限，不代表物理有效。原始run、异常与检查点全部保留，不能将失败前窗口速率称为正式300轮成绩，亦不能通过扩大异常预算转为通过。

另确认两个模型语义问题：Newton1.7 XPBD不执行joint_effort_limit，配置的2 Nm不能视为实际限幅；CHASSIS_I使用半尺寸代入全尺寸盒体公式，各轴比正确惯量小4倍。这些问题尚不能证明是NaN触发源，但必须修复/登记；v3短测仅为该实现的真实观测，不作为最终正确物理平台成绩。后续新revision与trial独立记录。

## 计时定义与历史数据

一次sample是一个环境完成一个0.05 s策略步。collection速率为总样本/总采集秒；PPO速率为rollout样本/更新秒（不是optimizer内部多epoch总处理量）。iteration window包含采集与PPO，在后置有限性检查/checkpoint I/O前计时。launch速率用含warmup的全部样本/完整子进程启动至退出墙钟，包含初始化、编译、检查、保存与采样收尾。它不是全新安装/首次下载耗时。

以上使用主机`perf_counter`，没有CUDA event或专门追加显式同步，不是kernel时间。跨轮计算使用总样本/总时间加权，不能平均速率。若用只测量部分的10,240 samples除完整launcher wall，必须另标保守口径，不能称全部训练速率。

历史Isaac perf为32 env、5 ms/10：393,216 / 9218.8 s = 42.65 samples/s；任务、并行数、步长及物理保真与新栈有差异，不能作为严格物理等价对照。M3b约328.2 samples/s为占位通道短原型，不能作为完整栈成绩。

旧`native_v1_1_pilot06_20261025_170308`虽然原始exit0、formal_eligible=true，iter185有异常恢复（world0读数广播将计数放大至64），存在NaN与约−108 m高度。该330.2 samples/s / 7.7倍无效；其4-seed结果也不能归因于策略、奖励或数据量。旧run、checkpoint和manifest原样保留，另存rejected审计。旧pilot01实际simplified、最后20轮、exit1且无完成摘要，也不记正式完成。

## 本机环境与复现

实测Python为WSL `/opt/venv/bin/python`：Python3.12.3、torch2.14.1+cu130、Warp1.18.0、Newton1.7.0、MuJoCo3.17.0、NumPy2.5.4、VQF2.1.2。`requirements-native.txt`固定已观察核心版本；根项目依赖包含Isaac Lab，不用于准备native环境。本轮未重建干净环境安装，不能声称干净安装已通过。ovphysx/ovstage不是Newton必需依赖。

```bash
# 在“算法/仿真训练环境”目录；安装示例，本轮没有重跑
agtc_venv_dir="$HOME/venvs/agtc-native-$(date -u +%Y%m%dT%H%M%SZ)"
python3.12 -m venv "$agtc_venv_dir"
"$agtc_venv_dir/bin/python" -m pip install -r native/requirements-native.txt

cd native
export PATH=/usr/lib/wsl/lib:$PATH
unset CUDA_VISIBLE_DEVICES
# 正式300轮；tag必须全新
PYTHON=/opt/venv/bin/python bash train/run_native_v1_trial.sh "reproduce_$(date -u +%Y%m%dT%H%M%SZ)"

# 先停止其他训练/推理；每次输出目录必须全新
agtc_benchmark_tag=$(date -u +%Y%m%dT%H%M%SZ)
/opt/venv/bin/python -B tools/benchmark_training.py \
  --out "acceptance/rep_${agtc_benchmark_tag}_01" --warmup-iterations 2 --iterations 5 \
  --seed 0 --threads 8 --exclusive-window
/opt/venv/bin/python -B tools/benchmark_training.py \
  --out "acceptance/rep_${agtc_benchmark_tag}_02" --warmup-iterations 2 --iterations 5 \
  --seed 0 --threads 8 --exclusive-window
/opt/venv/bin/python -B tools/benchmark_training.py --summarize "acceptance/rep_${agtc_benchmark_tag}_01"
```

```powershell
wsl -d Ubuntu-24.04-Arm64 -- /opt/venv/bin/python /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/tools/benchmark_training.py --help
```

`--exclusive-window`是操作者声明，需同时保留Win/WSL进程观察与GPU采样。benchmark是7轮性能测量，不能冒充300轮正式trial。正式runner的extra参数只允许观察器选项。

部分原型模块仍含本机`/mnt/c/pioneer/...`路径。换机须核对路径、ARM64 CUDA wheel和验证JSON，不能把runner相对路径等同于全栈任意目录可运行。WSL系统需Git/LFS；本机补装Ubuntu包git-lfs3.4.1-1ubuntu0.4，没有更改torch或驱动。

## 保真、能力与Git边界

Newton使用盒体/圆柱轮代理、平面接触及速度目标驱动，未实现完整Isaac场景的地形/植物物理碰撞和侧PI力矩控制。解析传感器中的地形/植物不代表已有相应物理碰撞。depth/detections有参考算法对照，但RTX遮挡、薄叶边缘和材质域差未逐像素验收。`capability_metrics_allowed=false`保持不变；平台链路/吞吐通过不能替代16-seed配对能力或实车验收。

延续`perf/max-throughput`和[PR #24](https://github.com/w3743/AGTC/pull/24)，通过PR保留main历史，冻结材料不改写。模型按Git LFS保存；相关源、协议、argv、包版本、真实退出、checkpoint SHA与原始计时分别保留。来源工具读取明确HEAD tree并比较实际源raw bytes，不调用全工程git status；`relevant_sources_dirty=false`只描述所列源相对提交，不声称整个共享工作区干净。正式长测另保留源文件原字节快照及wrapper/protocol SHA。

CPU标准库回归Windows32项全过，WSL32项中2项PowerShell专用跳过，其余通过；可选Newton实际物理6项另跑，不进入无GPU依赖的CI测试目录。Repository checks只验证仓库规则，最终状态以PR当前提交为准，不能代替本机训练验收。历史报告EOF的精确属性例外用于保留原字节，没有全局放宽检查。
