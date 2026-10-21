# Gazebo 农田短训练入口

该入口将 Gazebo Harmonic 场景适配到现有 `field_sim` 四维动作和四项 observation，再调用 `field_sim.policy.make_model` 的 CPU `RecurrentPPO`。默认 `all` 依次执行传感器 smoke、短训练、同一新 checkpoint 的基准；训练与基准各自启动/关闭自己的环境实例。脚本不会连接旧的 `8767` 服务，Gazebo 分区和 ROS domain 独立设置为每次运行的私有值。

## 启动

Windows 双击仓库根目录的 `启动Gazebo短训练.cmd`。它会调用 PowerShell/WSL wrapper，默认使用 100 m、400 株、20 秒 episode、20 Hz decision step、seed 7、mixed 任务和 512 个训练步，并将全部文件放到新的 `results/gazebo_field_training/run_*` 目录。

也可以在 PowerShell 中指定工作量或只跑某个阶段：

```powershell
.\scripts\field_train_gazebo.ps1 -Command all -Mode mixed -Steps 512 -NSteps 128
.\scripts\field_train_gazebo.ps1 -Command smoke -Mode mixed
.\scripts\field_train_gazebo.ps1 -Command train -Mode mixed -Steps 512
.\scripts\field_train_gazebo.ps1 -Command benchmark -Checkpoint .\results\gazebo_field_training\run_...\train\checkpoint.zip
```

WSL 中等价命令：

```bash
bash scripts/field_train_gazebo.sh all --backend gazebo --mode mixed \
  --size 100 --plants 400 --max-time 20 --dt 0.05 --seed 7 \
  --steps 512 --n-steps 128 --output "$PWD/results/gazebo_field_training/manual-run"
```

`simulation.train_field_gazebo` 命令也可直接接收 `all`、`smoke`、`train`、`benchmark`；省略子命令时默认为 `all`。默认 backend 是 `gazebo`，可用 `--backend fieldsim` 做 FieldSim 回退/代码路径检查。输出目录会解析为绝对路径；PowerShell wrapper 为避免覆盖，默认创建带时间戳和进程号的新目录。

`--monitor-port` / `-MonitorPort` 可选启用只读训练监看端口；默认关闭。根目录的短训练 `.cmd` 默认使用 `8768`。该端口必须空闲；绑定失败会报告错误并写入 `failure.json`，不会接管已有服务或复用旧状态。监看器读取当前 Env 的 `state_payload` 与 `scene_payload`，最多按 5 Hz 发布训练阶段和进度，不会向 Env 发送动作或改变训练轨迹。训练时如果还运行 Web 前端，可连接到同一端口观察当前训练车辆。

手动启用监看器的 PowerShell 示例：

```powershell
.\scripts\field_train_gazebo.ps1 -Command all -Mode mixed -Steps 512 -MonitorPort 8768
```

## 场景任务和策略

`mixed` reset 使用独立 seed 的任务 RNG：首次以 60% 概率选 goto，第二次 reset 保证覆盖另一类，后续 reset 按 60% goto 独立采样，确保短训练确实接触 search 和 goto。Goto 起点附近会尝试采样 1–5 m 的可通行 goal；只在任务重置时读取场景导航信息，并通过 `set_goal` 设置目标，策略 observation 不添加地图、植物标签或真值坐标。Env 侧 observation/action shape 与 `FieldEnv` 对齐；导出的实际空间需符合当前训练 checkpoint。

训练 wrapper 在未收到显式 `start` 时会请求 `random_start=True`。FieldEnv 的随机起点会选择空地；课程起点采样则按seed随机化位置和朝向。现有检查覆盖真实 FieldEnv 与 mock wrapper 的起点／参数传递，但没有确认 Gazebo 物理场景中的随机spawn，不能把它记作 Gazebo 验收结果。

Smoke 在 search 与 goto 各执行 32 个小幅连续动作，验证 observation space 和有限值、策略可观察的车体/云台运动及实际 LiDAR/depth sequence 推进。Goto 段再把 goal 放在当前位置并发零动作，等 `goal_reached` 与正完成奖励。末尾只保存 Env 实际交付的 observation 到 `last_delivered_observation.npz`，并按 depth mask 生成无效像素为黑色的伪彩 preview；不使用原始缓存帧冒充策略 observation。

Train 使用 `Monitor`、`make_model` 和 `RecurrentPPO`，默认 `n_steps=128`、3 epochs。首次优化前保存 `initial_checkpoint.zip` 作为同次初始化权重快照；它只用于参数比较，不是任务能力基线。训练墙钟从建环境前开始计到 `learn` 返回，覆盖导出、Gazebo 启动、reset、模型初始化和优化；另外单列 optimizer learn 时间、reset 延迟/数量、episode task mode、rollout `env.step` 时间、有效仿真秒/墙钟秒、参数有限性、参数平方差、checkpoint 保存/重载与重载推理验证。短训练只验证代码路径和权重更新，不表示任务已经学会。

Benchmark 载入同次 train 生成的 checkpoint（单跑 benchmark 可用 `--checkpoint`），测实际 observation 上缓存/未缓存各 200 次策略推理；再做默认 128 decision 的非 paced rollout，分开记录 policy inference、`env.step`、combined latency 和 reset interruption。另做 5 秒、由 `dt` 推导 target decision rate 的 wall paced rollout，记录在 `paced_control` 下（默认 `target_hz=1/dt=20`），并统计 active/wall 实际启动 Hz、起始间隔 p95、active deadline miss，以及 reset 中断时长。Gazebo benchmark 报告从实际运行 SDF 读取的 `physics_step_size_s=0.005` 和 `real_time_factor=0`；physics iteration rate 不用作 decision FPS。

所有输出写在 `--output` 下：`smoke_summary.json`、实际 observation/preview、`train/initial_checkpoint.zip`、`train/checkpoint.zip`、`train/train_summary.json`、Monitor episode log、`benchmark_summary.json` 和 `all_summary.json`。异常会写入 `failure.json`，环境在 `finally` 中关闭。训练不启动 GUI；Runner/Env 的传感器 fresh count 和 age 有数据时会一并写入统计。环境 step rate、policy decision FPS 与 Gazebo physics Hz 是不同指标，应分别解释。植物检测与压苗奖励/统计沿用共享的 FieldSim 几何代理；Gazebo 中叶片和茎是可穿越 visual，没有柔性植物物理，本短训也不验证实机识别或标定，不能据此声称策略已具备任务能力。对抗场景生成器尚未实现。

## 无监看的短跑对照

以下两行训练使用相同配置和 seed 7，各自从同 seed 初始化；性能数据来自无 GUI、无 monitor 的 512 步训练与 128 decision benchmark。两行 benchmark 使用同一个 Gazebo 训练 checkpoint；FieldSim 一行是将该 Gazebo checkpoint 载入 FieldSim 环境测量，不是 FieldSim 自己训练出的 checkpoint。

| Runtime | Train wall / throughput | 128-step combined latency (mean / p95) | Combined decision rate | 5 s paced target / actual | Deadline misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gazebo Harmonic | 26.87 s / 19.06 steps/s | 13.76 / 23.45 ms | 72.69 decisions/s | 20 / 20.000 Hz | 0 |
| FieldSim | 5.20 s / 98.40 steps/s | 4.67 / 17.31 ms | 214.35 decisions/s | 20 / 20.000 Hz | 0 |

Gazebo runtime SDF 使用 0.005 s physics step 和 `real_time_factor=0`（unpaced）。训练墙钟含环境启动、导出和 reset；两个 backend 的 reset 与传感开销不同。Benchmark 的 latency 只统计 policy inference 和 `env.step`，reset 单独记录。训练实际 reset mode 为 search 与 goto 各一次。对照是每个 backend 一次短跑，只有描述性结果，不是重复试验或置信区间；512 步仅确认优化、有限且变化的权重与 checkpoint 可执行，不证明任务能力。主机为 WSL2 / Ubuntu 24.04、Intel Core Ultra 9 275HX（20 个逻辑 CPU）；策略使用 CPU、Torch 1 线程。主机检测到 RTX 5070 Ti Laptop GPU，但未测 GPU 利用率。完整数字和四份输入 summary 的绝对路径见 `results/gazebo_field_training/closed_loop_comparison.json`。

## 浏览器检验保存的模型

双击根目录 `启动Gazebo浏览器.cmd`，或者运行 `scripts/field_gazebo_web.ps1 -Checkpoint <权重的Windows路径>`，打开 `http://127.0.0.1:8768/`。省略 checkpoint 时，只会自动选择本项目中最新且重载验证通过的 **Gazebo** 训练权重，不会使用 FieldSim 自检权重。没有权重时会清楚报错；手动调试可通过 Python 入口的 `--manual` 启动。

页面默认加载策略并暂停，进入到点模式，提供一条约3米的畅通基础测试路线；点击“继续”观察，也可点击地图改变目标、切换搜索或手动驾驶。默认20秒 episode 与本轮短训练一致，Python 入口可用 `--max-time` 调整。页面显示实际加载模型的训练步数、目标距离、压苗、碰撞、累计奖励与结束原因。规则“自动演示”仍是单独的控制选项，选择它不代表模型行为。

深度预览使用策略已收到的同一帧Z深度和有效mask；无效像素为黑色。雷达点来自实际Gazebo回波，物体类别未知时仍显示有效端点。俯视地图的完整植株/地形属于调试真值，不输入策略。首次加载和重建Gazebo场景需等待数秒；旧8767轻量服务不受影响。训练监视与交互验收共用默认8768端口，需等训练监视关闭后启动交互验收。

固定新种子测试通过已有运行环境执行，例如在WSL项目目录运行：

```bash
bash scripts/gazebo_python.sh -m simulation.evaluate_field_gazebo \
  --checkpoint results/gazebo_field_training/headless_gazebo_train_20261017_155524_284/checkpoint.zip \
  --output results/gazebo_training/my_new_evaluation \
  --seeds 10007,10008,10009 --seconds 10
```

输出目录必须是新目录。每个seed分别执行搜索和到点，核对两者场景几何哈希一致，保存轨迹、最终状态、策略观测和`evaluation.json`。到点先检验短距离畅通路线，记录初末距离、到点成功、压苗与碰撞；这个测试不能单独证明障碍绕行能力或整块农田搜索能力。

2026-10-17 的实际评估在 `results/gazebo_training/evaluation_20261017_10007_10009_01/evaluation.json`：六集各运行10仿真秒，到点0/3，平均目标距离3.000→3.128米；搜索见到5株、标记0/293个目标，另有一次误报；压苗和岩石碰撞均为0。短训练模型未通过基础到点验收，低损伤也不能证明已学会避障。早期状态文件的雷达/深度origin XY使用了Gazebo中心坐标；已在同目录另存`final_state_field_coordinates.json`供可视化，保留原始文件与全部评估指标。当前实时后端已修正坐标变换，范围、深度图和策略输入数值不受这次显示修正影响。

## 小批课程实验与现成训练组件

课程入口 `simulation.improve_field_policy` 继续调用 SB3-Contrib `RecurrentPPO`、SB3 `DummyVecEnv` 或 `SubprocVecEnv`（默认 `dummy`）、原生模型保存/加载与 callback；另支持 `--device cpu|cuda|auto`（默认 `cpu`）及非负 `--distance-cost`（默认0，仅训练奖励按 `r' = r - coefficient × 当前目标距离 × dt` 变换）。自定义部分负责农田起点/目标采样、固定验证种子和安全指标。没有另写强化学习优化器、物理引擎或手工动作教师。参考 [RecurrentPPO官方接口](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html) 与 [SB3自定义环境/归一化建议](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)。

输入/动作维度保持原样。训练默认 `goal_gain=12.5`，该参数只在已有特征提取器内放大相对目标和到点容差；参数随checkpoint保存，旧模型默认gain1，行为不随源码升级而悄悄改变。课程先用无植物平地、随机起点/朝向及近目标，之后才扩大角度和加入田地；原始400植物、地形和石块场景用于独立验证。任务采样可查询场景真值，策略观测仍不包含地图或物体ID。

```powershell
.\.venv-field\Scripts\python.exe -m simulation.improve_field_policy --stage easy --steps 16384 --chunk-steps 8192 --wall-seconds 240
```

`--checkpoint` 续训保留原checkpoint的特征/算法超参，日志分别记录请求值与生效值。每个chunk保留最新权重，同时按固定validation seeds 20007–20009的安全到点率/进展保留best；测试种子10007–10009不参与这一步选优。额外的新测试种子应在候选模型选定后使用。评估器现在支持 `--backend fieldsim|gazebo --mode goto|search|mixed --episode-budget 20 --seconds 20`；episode budget与实际rollout长度分开，避免时间状态归一化不一致。

初期16384步课程和后续32768步续训均没有到点成功。较好的验证候选平均末距约2.25米，后续则退化为强转向，已停止并保留较好的checkpoint；这些结果不应描述成已经学会导航。规则控制诊断在相同Gazebo安全路线用5.2仿真秒到点，说明动作链条可用，但它不是学习模型成绩。

性能改动只优化现有实现：缓存同一姿态的履带支撑计算，并预计算地形射线的正弦相位。地形求交步长/二分精度不变；随机射线及岩石/量程边界比较要求类别相同、距离差不超过1毫米。局部微基准不能替代整环境吞吐。另试用了Gazebo现成`/server_control`退出服务：空场景成功，完整田地的传感器销毁却更慢并出现异常退出，因此生产默认已恢复原清理路径，未作为优化上线。

较早探索使用SB3已有gSDE（`use_sde=True, sde_sample_freq=32, log_std_init=-1, squash_output=True`）。同一easy课程新增16384步时，较好的chunk平均验证末距约2.18米，随后退化，固定3米任务仍0/3成功。更短的0.45–1.05米、±10°课程用Gymnasium `TimeLimit(160)`限制到8秒，原`Config.max_time=20`及到点/停车判据保持不变：首chunk训练成功1/49，后两个chunk均0/52，原田地验证仍0/3。已停止这两次实验，未将最佳进展当成导航成功。上一轮近目标实验保存的latest checkpoint为24576步，停止前另外7168步未保存，不计入保存模型的步数；详见各自`training_summary.json`。

这些较早失败实验保存在`results/field_policy_improvement/`，包括原始summary、best和latest checkpoint；best按验证集选出，不能把验证集当成最终独立测试。此前一个候选在前后0.6米目标的同姿态反事实输入中只产生很小动作变化；该诊断提示那一候选的目标条件控制不足，不是任务成绩。

较新的 FieldSim 课程实验 `results/field_policy_improvement/easy_distancecost_gain100_20261017_164842/` 请求32768步，在240s墙钟上限内实际完成16384步；使用4个 `DummyVecEnv`、CPU、`goal_gain=100`、`distance_cost=1`，目标0.45–1.05m、±10°、`TimeLimit(160)`。同一10个近目标任务由初始权重0/10到latest checkpoint 10/10，无损伤／碰撞，平均49.3步结束。该checkpoint在未加distance-cost的400株、默认地形、20s、3m固定seed验证仍0/3，到点距离由3.00m平均降至1.62m。结果见 `near_goal_comparison.json` 与 `latest_complete_validation.json`；这是 FieldSim 结果，不是Gazebo随机spawn验证，也不证明避障或完整田间能力。对抗场景生成器尚未实现。

## GPU 渲染、浏览器与训练设备

当前 8768 实时服务的 WSL Gazebo 进程（PID 1621）环境明确为 GALLIUM_DRIVER=d3d12、MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA；同一时刻 nvidia-smi 将 PID 1621 列为 NVIDIA RTX 5070 Ti 上的图形进程。WSL 的 glxinfo -B 也报告 D3D12 (NVIDIA GeForce RTX 5070 Ti Laptop GPU)。因此当前 Gazebo 的传感器渲染走独显 D3D12 路径；利用率是瞬时值，不能据单次 0% 判断没有渲染。

项目已有的选择机制在 scripts/env.sh：检测 WSL /dev/dxg、显示会话与 NVIDIA 可用性后，为 Mesa D3D12 选择 NVIDIA，并在加速探测通过时设置 GALLIUM_DRIVER=d3d12。scripts/gazebo_python.sh 会加载该环境。Windows 上的 Codex/Edge WebView 是独立进程；本次 GPU 引擎计数器发现它们活跃于同一 Windows LUID，但该 LUID 尚未映射到 Intel 或 NVIDIA，不能用浏览器的任务管理器条目推断 Gazebo 渲染适配器。策略创建、加载和训练默认选择 Torch CPU；这与 Gazebo 传感器图形渲染设备是两回事。

### Windows 策略 CPU/CUDA 配对测量（2026-10-17）

在同一 Windows `.venv-field` 进程中测量，Torch 2.8.0+cu128 检测到 RTX 5070 Ti Laptop GPU；CPU Torch 线程数固定为1。策略为341,901参数的 `RecurrentPPO`，来自 `goal_gain=12.5` 的 easy 短课程。实际观测形状为深度 `2×160×88`、雷达 `2×400`、检测 `16×10`、状态 `44`。每组先 warmup 4 次，再计时 100 次；CUDA 每次调用前后同步，计时含 NumPy 到设备张量转换和动作输出物化，不含环境观测生成。

| 测量 | CPU | CUDA | 结果 |
|---|---:|---:|---|
| 单环境 `predict`，平均延迟 / p95 | 0.608 / 0.861 ms | 1.478 / 2.073 ms | CUDA 平均延迟为 CPU 的 2.43 倍 |
| batch 4 `predict`，平均延迟 / p95 | 1.149 / 1.745 ms | 1.936 / 2.641 ms | CUDA 平均延迟为 CPU 的 1.68 倍 |
| `learn` 配对1，1024步 | 2.252 s | 18.381 s | CUDA 首次运行含冷启动，不能当作稳态值 |
| `learn` 配对2，1024步 | 2.545 s | 2.631 s | CUDA 慢约3.4%，接近持平 |

两组 `learn` 都从 SHA-256 为 `c10cfbca…` 的同一初始权重、seed 7 和同一配置开始，顺序为 CPU、CUDA、CUDA、CPU；随机策略运行后的轨迹可能分叉。学习计时包括环境 reset/step、采样、CPU/GPU 张量传输和更新，排除环境构造及权重加载。该测量使用4个串行 `DummyVecEnv`，且 easy 场景无植株、无地形；记录的环境 step/reset 时间占单次 `learn` 的约2.3%–20.3%，不能代表400株完整田地或 Gazebo 的环境耗时占比。较重的多环境并行收益仍需单独测量。

单环境与 batch 4 的策略推理在这台机器上均以 CPU 更快，因此策略默认仍用 CPU。`make_model` / `PolicyRunner` 已支持显式传入 `device`，可由 SB3 直接选择 CUDA；本组短课程结果不表示 Gazebo 渲染、物理步进或完整训练已因 CUDA 提速。原始数据见 `results/policy_device_benchmark_20261017_164208/` 下的 `predict_summary.json`、`predict_batch4_summary.json` 和 `learn_summary.json`。
