# RTX Spark N1X 原生训练迁移与验收

本机目标为 Windows 11 ARM64 上的 `Ubuntu-24.04-Arm64` WSL2。训练走 ARM64 Python、PyTorch CUDA、Newton XPBD 和 Warp，不启动 Isaac Sim/Kit。Isaac 历史运行仅作吞吐参照。一次 sample 指一个环境完成一个 0.05 s policy step。

## 训练路径

`train/train_wsl.py` → `proto/native_env.py` → Newton XPBD 物理、Warp LiDAR/深度/检测和 VQF → 五路编码器 → LSTM-512 → PPO。Newton 仍有主机与 GPU 数据往返；这里消除的是 Kit/PhysX 每物理子步的 `fetch_results()` 路径，不宣称整个训练流程没有同步。

正式物理为 1 ms × 50；64 环境 × 32 步为 2048 samples/iteration。5 ms × 10 曾出现非有限状态，不用于正式训练。五路观测为 LiDAR `(1,400)`、depth `(1,160,88)`、detections `(16,8)`、state `(33)`、feedback `(37)`；正式命令必须显式选择真实传感器与 `fast_stop_v2` 任务。命令行默认值保留早期原型兼容性，不能只凭输出目录名判断正式试验。

## 本轮发现与修正

旧 pilot01 的五路观测均为真实实现，但训练器没有传入 `task_backend`，实际使用简化任务。旧记录不可作为 V1.1 的 fast-stop 训练成绩。本轮新运行使用显式任务选择；训练、resume 与评估核对契约身份，物理或任务不符须拒绝。旧模型和冻结协议不回填、不覆盖。

启动器必须完整消费训练输出、保存真实退出码，禁止以截取前八行输出的方式运行长训练。评估脚本使用正式 1 ms/50 配方，禁止静默切到旧 5 ms/10 配方。

## 本机复现

当前已验证的 Python 为 WSL 中 `/opt/venv/bin/python`。**不要安装仓库根项目的默认依赖来准备 native 环境**，根项目包含 Isaac Lab；原生依赖应使用独立环境。本机版本以验收目录中的依赖记录为准。

[requirements-native.txt](requirements-native.txt) 保存本机观察到的核心依赖版本，CUDA torch 使用官方 cu130 wheel 索引。现有环境的导入和训练会实测；本轮没有重新创建干净环境安装，不能把该清单称为已通过干净安装验收。ovphysx/ovstage 不属于 Newton XPBD 训练的必需依赖。

```bash
python3.12 -m venv <new-venv>
<new-venv>/bin/python -m pip install -r native/requirements-native.txt
```

```powershell
wsl -d Ubuntu-24.04-Arm64 -- /opt/venv/bin/python /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/tools/benchmark_training.py --help
```

正式 trial 的入口为 `train/run_native_v1_trial.sh`；WSL 启动时明确指定发行版，不重启 WSL 或驱动。新 benchmark 的具体命令、warmup、测量轮次和原始产物见本轮验收记录。训练输出使用新目录，禁止覆盖历史数据。

当前部分 prototype 模块仍含本机 `/mnt/c/pioneer/...` 路径。启动器自身使用相对路径并不代表整个原型已经可迁移到任意目录。换机时须核对这些引用、CUDA wheel 的架构以及验证 JSON；本轮仅验收本机 WSL2 ARM64，Windows ARM64 原生 CUDA Python 路径没有实测。

## 数字的口径

- collection samples/s = 总 samples / 总 collection 秒；不包含 PPO。
- iteration samples/s = 总 samples / 总 iteration wall 秒；包含策略、环境与 PPO，使用训练器主机墙钟。训练器的参数有限性检查及 checkpoint 保存发生在该计时之外，不能称完整进程耗时。
- launch samples/s = 包含 warmup 的全部 samples / 子进程启动至退出墙钟；包含初始化、编译、检查和 checkpoint I/O。
- 预先指定的 warmup 单列；采用时间加权总吞吐，不平均各轮速率。重复运行逐项报告，不挑最快轮。

这些不是 CUDA event/kernel 时间。历史 Isaac 的 42.65 samples/s 是 393216 samples / 9218.8 s 的整段训练数字；与新测 iteration 或 launch 数字比较时必须保留口径、任务、传感器和代理物理差异。旧 M3b 的 328.2 samples/s 来自占位通道的短原型，不能当作本轮完整栈验收数字。

## 物理与能力边界

Newton 当前使用盒体/圆柱轮代理、平面接触和速度目标驱动；未实现与 Isaac 完整场景相同的地形碰撞、植物接触与侧 PI 力矩控制。观测场景有解析地形和植物，不能将传感器几何当成已实现物理碰撞。

depth/detections 的解析算法已有独立参考对照，RTX 的遮挡、薄叶边缘、材质及光栅化域差仍未逐像素比较。`capability_metrics_allowed=false` 保持不变。短训练、resume 和短评估只验收平台链路；不代表 16-seed 正式能力评估、导航学会或实车验收。

## Git 与证据

延续现有 `perf/max-throughput` 与 [PR #24](https://github.com/w3743/AGTC/pull/24)，保留 main 历史。只提交本轮责任文件；已有未提交修改、历史协议、模型和实验快照保留原字节。模型使用 Git LFS，测量日志与 JSON 以原始文件和 SHA256 追溯。GitHub `Repository checks` 与本机训练验收分别记录。
