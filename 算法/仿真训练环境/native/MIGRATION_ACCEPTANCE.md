# RTX Spark N1X 原生训练迁移与验收

> 当前状态记录（2026-10-25）：Native V2 协议已于当天冻结，SHA256 为 83db5a6e512d7d7244bc47ddf2059720c757504270e9bbf9129d0c7d8f92f87e。V6 的 1/3/64-world 物理前置门和同身份 resume / strict-eval 链路 smoke 已通过；正式 300 轮 trial 仍在运行，尚未通过最终训练门。
>
> 正式 run native_v2_formal_v6_live_20261025_224005 于北京时间 22:40:05 启动。23:06:51 记录的是 CPU viewer 切换，不是训练迭代采样；本说明不写动态 iteration，亦不把短基线当作正式 run 的吞吐。

## 当前身份、环境与固定配方

目标机器为 Windows 11 ARM64 / RTX Spark N1X；训练在 Ubuntu 24.04 ARM64 WSL2、ARM64 Python、PyTorch CUDA 中运行，不启动 Isaac Sim/Kit。Windows ARM64 原生 CUDA Python 尚未实测。

| 项目 | 身份 |
|---|---|
| 协议 | PROTOCOL_NATIVE_V2.md，protocol SHA256 83db5a6e512d7d7244bc47ddf2059720c757504270e9bbf9129d0c7d8f92f87e |
| task identity | native_task_contract_v2，SHA256 209016904867c97633eb006cbb8f52fc75d95f82d74463e7f1d05ad49ecbc1b2 |
| 物理 | Newton MuJoCo Warp V6，revision newton_mujoco_warp_v6_deterministic_identity |
| 启动源码 | commit 7d8a167f；raw source manifest SHA256 6026336cad5b961e2fe95885d6010781ae75e59e792d1715988bda45d35e8938 |

测量环境使用 Python 3.12.3、torch 2.14.1+cu130、NumPy 2.5.4、Warp 1.18.0、Newton 1.7.0、MuJoCo 3.15.0、mujoco-warp 3.15.0、VQF 2.1.2。环境位于 /opt/venv_mujoco315_smoke；requirements-native-v2.txt 记录已观察版本。它是复用原有包的隔离 overlay，尚未证明从空环境安装可完整重建；旧 /opt/venv 与 MuJoCo 3.17.0 属于 XPBD 历史环境，不可混作 V2。

正式配方固定为 64 env × 32 steps × 300 iterations（614,400 samples），LSTM-512 / PPO，seed 0、threads 8、1 ms × 50、20 Hz 策略、fast_stop_v2、五路真实观测、Newton MuJoCo Warp、普通 eager GPU、NOT_GUARANTEED 数值模式。CUDA Graph 请求和实际启用均为 false。正式训练从头开始，不续训旧 XPBD checkpoint。任务和物理以冻结协议为准，不通过性能验收修改奖励或任务。

正式训练与评估入口：

~~~bash
cd /mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native
PYTHON=/opt/venv_mujoco315_smoke/bin/python bash train/run_native_v2_trial.sh <unique-tag> \
  --live-state-file /mnt/c/pioneer/agtc-native-acceptance-backup/native_v2_live/state.json \
  --live-state-interval 5

PYTHON=/opt/venv_mujoco315_smoke/bin/python bash train/run_eval_v2.sh <checkpoint.pt> <new-output-dir> pilot
~~~

runner 校验依赖、协议 SHA 与物理参数，固定正式配方，只允许额外传递 live-state viewer 参数，并在新的带时间戳目录中保存 run。评估默认 pilot 四个开发 seed；正式训练/评估规则见冻结协议与 train/README.md。

## 已通过的前置门与链路 smoke

- V6 1、3、64 worlds 均通过冻结前置物理门。每项 20 policy steps / 1,000 policy physics substeps，另有诊断开销；登记的 zero、fixed、ramped 控制，逐 world 读回与 partial-reset 隔离均在通过记录中。三个 world 数量的证据在 [verified-inputs README](acceptance/20261025_v2_freeze_validation/README.md)。
- 同身份续训 smoke 从 iteration 7 到 8，Adam step 从 28 到 32；resume 后 task SHA 不变。单环境评估报告 checkpoint_contract_validation=strict_match，并匹配 task SHA。见 [resume summary](acceptance/20261025_mjw_quiet_resume/run/summary.json) 和 [strict evaluation](acceptance/20261025_mjw_quiet_eval/evaluation.json)。该链路是 checkpoint / 评估完整性 smoke，不是导航能力结果。
- 当前正式 run 仍须完成完整 300 轮并满足冻结协议：exit 0、metrics 迭代完整且身份一致、每轮样本数正确、非有限/capacity overflow 为零、solver event 计数完整、checkpoint policy 与 optimizer 张量有限，并完成规定的后续 checkpoint 验证。未完成 run 不记正式通过。

## 实时状态可视化的范围

正式 run 的 CPU viewer consumer 使用该训练进程输出的原始状态、目标与每环境任务时间，每 5 个 policy steps 发布一次；静态 19-body primitive 与 local transform 来自本次冻结 CPU 模型导出，地面按实际 flat plane 绘制。旧 Isaac USD 缺少两个云台和两个前导轮共 4 个 body 的几何，旧 viewer 方块 fallback 不再用于 Native V2；它不再作为车辆或地面的真值来源。

当前 viewer 不显示 LiDAR/depth/detection 图像，也没有声明 CPU 导出几何与 GPU shape arrays 逐字节相等。它用于观察训练状态，不替代物理前置门、训练数值审计或能力评估。正式长测包含 viewer、Git 和 CPU 导出活动的实际运行负载；其最终 wall 与吞吐不得用 isolated short benchmark 代替。

## 独立短基线（非正式 300 轮成绩）

两组均为普通 eager、64 × 32、seed 0、task SHA 209016…；每个独立进程运行 7 轮，前 2 轮 warmup，后 5 轮共 10,240 measured samples。短测源、日志与原始计时保留在各自 acceptance 目录，不能作为长测或导航能力结论。

| Warning-enabled baseline | rep01 | rep02 | 加权 |
|---|---:|---:|---:|
| Iteration window samples/s | 112.036 | 89.540 | 99.533 |
| 全部 14,336 samples / 完整进程 wall samples/s | 84.178 | 78.313 | 81.139 |

来源 commit fce4ba90ab3f5bce15cf869cad9623334087579a，source manifest 5641d9ae5bc77c2d533a3b504fe01a5798c374e5b494565cedba84c4d284fabf。两次 exit 0，非有限 reset/action、capacity overflow、outer iteration-limit 均为 0；line-search-limit 事件仍保留报告。原始结果：[rep01](acceptance/20261025_mjw_eager_rep01/benchmark_result.json)、[rep02](acceptance/20261025_mjw_eager_rep02/benchmark_result.json)。

| Quiet baseline | rep01 | rep02 | 加权 |
|---|---:|---:|---:|
| Iteration window samples/s | 170.946 | 175.633 | 173.258 |
| 全部 14,336 samples / 完整进程 wall samples/s | 136.085 | 143.581 | 139.733 |

来源 commit 8903e4e91fddfce63875ae91ddc6ba765f2dadbf，source manifest 57b0d3bffe311f7633474726d2a816d9d4fa9d899bdedda1743d7bbdb686f861。两次源运行前后 raw manifest 稳定，task SHA 相同；每次 7 轮共 716,800 world-substeps，fresh line-search-limit 计数分别为 76,971 和 76,974。quiet 只改变 warning 输出处理，不改变物理 revision 或 task SHA。两组速率差异是观察结果，不能单独归因于日志静音。原始结果：[quiet rep01](acceptance/20261025_mjw_quiet_rep01/benchmark_result.json)、[quiet rep02](acceptance/20261025_mjw_quiet_rep02/benchmark_result.json)。

加权值使用总样本除总耗时，不平均各轮速率。iteration window 只含 rollout 与 PPO 更新时间；完整进程 wall 还含启动、初始化、检查和保存。测量使用主机计时，不是 kernel time。上述 isolated baseline 没有本次正式长测的 viewer / Git / CPU 几何导出组合负载，不得用于替代 formal throughput。

## 保真、能力与 Git 边界

代理车辆、flat-plane 接触与解析传感器不等于完整 Isaac 场景或实车。LiDAR/depth/detections 的 RTX 遮挡、薄叶边缘、材质域差尚未逐像素验收；当前 viewer 也不呈现传感器图像。吞吐和训练链路通过不代表导航、地形/植物碰撞或实车通过；capability_metrics_allowed=false 保持不变，尚无 16-seed 配对能力或实车验收结论。

冻结 V1/V1.1、历史源快照、模型、日志与原始成绩按原字节保留。source manifest 只描述列出的输入文件，不能解释成整个共享工作区干净。提交与推送由总协调按 PR 约定处理；本说明不改写冻结协议。

## 历史证据：Newton XPBD 与早期候选

以下数据属于不同于当前 V2 的旧 Newton XPBD / 原型，不可和 V2 合并为同引擎性能、物理等价或能力结果；raw artifacts 保留在链接目录。

### XPBD 修复与短测

V3 曾修正 1 ms 子步接触生成、world/joint/coordinate 映射、partial reset 隔离、设备位置与摩擦身份。旧空接触缓冲路径会跳过刚体接触；旧 world 读取/控制映射与动态 shape 局部默认材料亦有问题。此前 Newton XPBD v3 另用 task identity f04ab2db16e4bcfd5c5c0ade1fe3eb39687ea20c2f77440d3e9fccc71d76a329。两个短子进程的后置 warmup iteration window 分别为 278.538 与 167.671 samples/s，完整 launcher 分别为 196.598 与 142.085 samples/s；样本量有限，波动原因未定位。原始记录：[v3 rep01](acceptance/20261025_v3_rep01/benchmark_result.json)、[rep02](acceptance/20261025_v3_rep02/benchmark_result.json)。

同一 XPBD 源码的正式 300 轮 trial 在北京时间 18:58:07、iter 58 / step 21 出现 env11 非有限位置/速度/轮速，训练按零异常门 exit 3，formal_eligible=false，只完成 57 轮，完整 wrapper wall 666.72 s。最后 checkpoint_40 的 policy 与 optimizer 张量有限不代表物理有效。目录 native_v1_1_fixed_physics_v3_20261025_184052_20261025_184704 与异常记录保留，不能扩大异常预算或把失败前速率叫作正式成绩。

旧 pilot06 原始 exit 0、manifest formal_eligible=true，但 iter185 存在 NaN 与约 −108 m 高度，异常计数被 world0 读取广播放大；其 330.2 samples/s、7.7 倍和 4-seed 结果均拒绝，不能归因于策略或奖励。pilot01 实际使用 simplified backend，后期 exit 1、无完成摘要，同样不是正式成绩。两组原 run 与 checkpoint 保留并另存 rejected audit。

XPBD 中还发现过 Newton 执行器限制与预期不同、底盘惯量尺寸公式错误；这两项属于旧后端审计，不能移植成当前 V2 物理结论。

### 早期 MuJoCo-Warp 原型与性能模式探针

旧外部 v4b GPU 原型 3 world × 120 policy steps 曾报告 finite、capacity 与 reset 隔离通过，但缺启动时完整源码冻结，且不是当前 V6 final preregistered gate；原始记录与限制见 [candidate archive](acceptance/20261025_mjw_candidate/README.md)。

旧探针 LS_ITERATIONS 曾记录 4999/4999/4699 个 flag-positive 子步，但相关位是 sticky OR、未逐步清零，不能视为真实逐步计数。当前 V6 已改为子步清零并统计 fresh solver event。CUDA Graph 对照超出状态容差，普通 eager 重复本身也分叉，因此不能把分叉单独归因于 Graph。旧 9.78 倍是隔离物理调用计时，不是训练端到端提速。RUN_TO_RUN 初始化因内存压力被 TERM，未产生 policy 数据，不是 benchmark。

历史 Isaac 32-env / 5 ms 记录约 42.65 samples/s，早期占位 M3b 约 328.2 samples/s；任务、通道、并行数及物理保真不同，均不是当前严格对照。