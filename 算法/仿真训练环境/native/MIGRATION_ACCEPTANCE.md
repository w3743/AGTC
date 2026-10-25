# RTX Spark N1X 原生训练迁移与验收

目标：Windows 11 ARM64 的 Ubuntu-24.04-Arm64 WSL2，ARM64 Python、PyTorch CUDA、Newton XPBD 与 Warp。训练不启动 Isaac Sim/Kit。Isaac 历史运行仅作吞吐参照，不能据此声称物理等价。

## 路径与配置

`train/train_wsl.py` → `proto/native_env.py` → Newton XPBD 物理、Warp LiDAR/解析深度/检测、VQF → 五路编码器 → LSTM-512 → PPO。仍存在主机/GPU 数据往返；消除的是 Kit/PhysX 每物理子步 fetch_results 路径，不宣称整个流程没有同步。

正式配置：1 ms × 50、64 env × 32 steps = 2048 samples/iteration、seed 0、LSTM-512、显式 fast_stop_v2。五路真实观测为 LiDAR `(1,400)`、depth `(1,160,88)`、detections `(16,8)`、state `(33)`、feedback `(37)`。5 ms × 10 有历史非有限状态记录，仅属实验配置。

CLI 默认保留 simplified/部分占位通道的原型兼容性，必须使用正式 runner 或完整参数。目录名、协议副本与五路 mode=real 不能证明任务配方正确。

## 本轮修正

- 旧 pilot01 虽有五路真实观测，训练器没有传 task_backend，实际是 simplified。最后完整 checkpoint_20、目标300迭代但 exit 1、无完成摘要，不作为 V1.1 成绩。
- 新训练显式 fast_stop_v2，checkpoint 写入 native_task_contract_v2 身份。训练/resume/eval 核对任务、奖励/终止、物理和每环境拓扑，允许训练64环境→评估1环境；排除 profile、world count 和运行元数据。
- 旧 checkpoint 缺少新版任务身份时默认拒绝。只有评估可同时显式 --task-backend 与 --allow-legacy-checkpoint-contract 探索使用，输出 legacy 状态，不能得到 strict_match。
- 正式启动器完整消费日志并保留退出码，取消截取前8行的管道；评估不再静默使用旧5ms/10配置。
- 冻结协议、旧模型与历史成绩不回填。冻结 V1.1 正文仍有旧5ms示例与 wsl_train 路径，执行以本说明及修正runner为准。

## 本机环境与复现

本机 Python 是 WSL 的 `/opt/venv/bin/python`。不要安装仓库根项目默认依赖来准备 native 环境，根项目包含 Isaac Lab。`requirements-native.txt` 记录已观察的核心版本：Python3.12、torch2.14.1+cu130、Warp1.18、Newton1.7、MuJoCo3.17、VQF2.1.2；CPU导入和实际训练分别验收。本轮没有新建干净环境安装，不能声称已通过干净安装验收。ovphysx/ovstage 不是 Newton XPBD 必需依赖。

```bash
# 从“算法/仿真训练环境”目录准备新环境；安装步骤本轮未重跑
python3.12 -m venv <new-venv>
<new-venv>/bin/python -m pip install -r native/requirements-native.txt

cd native
PYTHON=/opt/venv/bin/python bash train/run_native_v1_trial.sh <new-tag>
/opt/venv/bin/python tools/benchmark_training.py --help
```

```powershell
wsl -d Ubuntu-24.04-Arm64 -- /opt/venv/bin/python /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/tools/benchmark_training.py --help
```

当前部分原型模块仍含本机 `/mnt/c/pioneer/...` 引用，runner使用相对路径不代表整栈已能在任意目录运行。换机须检查路径、ARM64 CUDA wheels 和验证 JSON。本轮只验收本机 WSL2 ARM64；Windows ARM64 原生 CUDA Python 路径未实测。

## 吞吐口径

一次 sample = 一个环境完成一个0.05s policy step。

- collection samples/s = 总samples / 总collection秒，不含PPO。
- iteration-window samples/s = 总samples / 总iteration wall秒，含策略、环境与PPO；训练器在后续参数有限性检查、checkpoint保存之前读该计时，不是完整进程耗时。
- launch samples/s = 含warmup的全部samples / 子进程启动至退出墙钟，含初始化、编译、检查和checkpoint I/O。
- 固定warmup先验指定、单列；使用总样本/总时间加权，报告所有重复，不挑最快轮。

这些是主机墙钟，非CUDA event/kernel时间。历史Isaac的42.65 samples/s来自393216/9218.8s整段训练，比较时必须保留任务、并行数、物理和传感器差异。旧M3b的328.2 samples/s属于占位通道短原型，不能充作完整栈新测数据。

## 物理与能力边界

Newton使用盒体/圆柱轮代理、平面接触和速度目标驱动；未实现与完整Isaac场景相同的地形/植物物理碰撞及侧PI力矩控制。传感器中的解析地形/植物不代表已有相应物理碰撞。

解析depth/detections已有独立参考算法对照；RTX遮挡、薄叶边缘、材质、光栅化域差未逐像素验证。capability_metrics_allowed=false保持不变。短训练/resume/短评估只验收平台链路，不能替代16-seed配对能力评估、导航能力或实车验收。

## Git与证据

延续perf/max-throughput与[PR #24](https://github.com/w3743/AGTC/pull/24)，保留main历史。代码变更、验证及证据分项提交；只暂存本轮文件。模型按Git LFS保存；原始日志/JSON与源码/检查点SHA可追溯。Repository checks只验证仓库规则，不代替本机训练验收。

共享目录曾出现新增未提交文件与修改回退，原因未明。本轮改为串行写入、外部备份、分项提交。旧probe取样目录已丢失，恢复内容明确区分原始run文件副本、由工具记录重建的观察以及重新采样，不能冒充原取样文件。

## WSL Git 身份采集依赖

本轮发现WSL内缺少git-lfs，Linux git status在读取带LFS属性的工作区时失败，未跳过此检查。已在Ubuntu-24.04-Arm64执行apt-get install -y git-lfs，安装包3.4.1-1ubuntu0.4（git-lfs/3.4.1 linux arm64）；没有调整PyTorch或驱动。复现机应先安装系统Git/LFS，且不要与Windows Git并发刷新同一个工作区索引。大量历史文件可能使首次Linux Git身份采集耗时；该准备耗时不属于训练子进程启动至退出的launch测量。

## Git 原始输入身份与测量窗口

全工程Linux git status曾阻塞于LFS过滤进程，已取消本次工具自己的探针子树，未停止新训练。测量工具改为读取明确提交的HEAD tree，对实际采集的源码/验证JSON/协议原始字节计算Git blob摘要；同时保留SHA256输入清单。relevant_sources_dirty仅描述raw_source_bytes相对该提交的变化，不声称整个工作区干净，也不枚举无关CAD/历史模型。完整工作区与PR同步由Git协调步骤另行检查。

2026-10-25 17:03:08（北京时间）另一入口启动native_v1_1_pilot06_20261025_170308，优先保留这次新训练自然运行。该run的contract为fast_stop_v2、1ms/50、五路real与LSTM512；启动时本轮未采集完整源码清单，运行中核对相关源与f0e3bea7一致，只能作为运行中观察，不补造启动时冻结证据。两次独立GPU复测须等待其退出，不能同时运行后再宣称无竞争训练负载。

CI首次新CPU回归27项通过（2项PS专用跳过），随后在两个历史报告末尾空行规则失败；未改写报告，只为DEPTH_DETECTIONS_REPORT.md与REPORT.md设置精确blank-at-eof例外，其余仓库检查保留。后续Git身份补充回归后Windows本轮28项全过；最终GitHub结果按PR当前提交核对。
