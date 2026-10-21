# Isaac Lab 项目状态

2026-10-20训练方法方案：新增独立 `train_isaac_mastery.py`/`run_isaac_mastery.sh`，保留PPO+LSTM和原奖励，接入终止步奖励分项/四维动作/PPO优化诊断、按独立开发探针连续达标晋级、真实旧任务与plants0→2→8→16采样、固定32场景逐类安全选模，以及单变量GAE入口。代码验收最终CPU325 passed/16 skipped/8 warnings、compileall及bash-n真实0；另1项真实Newton专项通过（只验终止奖励与自动reset，非完整native或学习验收）。本任务4项改动已收口，当前批次4/5，最近正式整理仍为第十四批。

固定课表、能力课程、λ=.99三份v2协议准备时已冻结47源并通过preflight，见 `results/project_history/training_method_solutions_20261020/prepared_comparisons.json`。尚未初始化模型或启动正式训练，不能称已提高到点/避苗能力。首份旧准备目录的JSON tuple/list预检失败已保留并修复；v2未覆盖它。当前另一聊天继续处理random_rough、被动前导向轮及车身稳定性；本聊天已交出env.py/newton_drive.py/learning.py写入和后续GPU验证，保留对方新增可选轨迹/稳定性字段。共享源码继续变化会使既有v2预检拒绝，双方收口后须用新输出目录重新prepare。使用命令与各方案的实现/未实现边界见使用说明“训练方法改进入口”，定时任务保持PAUSED。

本项目继续使用本地 `external/IsaacLab`、Newton MJWarp 和 SB3-Contrib RecurrentPPO。同侧轮联动与 1 ms 步长已接入并通过正式回归，drive 合同升为 `independent_left_right_track_speed_coupled_pi_v7`；最终全14 native通过，CPU244 passed/14 skipped，compileall/39个bash检查均exit0；第十三批整理已完成。新初始化训练 `fresh_coupled_tracks_20261020` 于2026-10-20 12:35:58（北京时间）以真实exit0完成196608步/29个检查点，final SHA-256 `6b4c85884a51c2e23f557eb98bc04ea63905fc8f20d4379715fb9fe72b75cdc5`。协议 SHA `2735ca85917643e49e27088957d00b1576f37902ce9664203253ae698181d5b6`、正式0步 baseline SHA `beda8fde92236b558433a148245094caeefb88d7c23b9ea73ed94c4cd596f1d5`；42个训练源与冻结快照匹配，final CPU检查通过。free和field各4个相同初态开发种子均为成功0/4→1/4、0损伤、0碰撞，平均末距分别1.685283→1.523645 m和1.835561→1.801870 m。每组仍有3/4超时；这是小样本开发改进而非盲测，不足以推断同侧联动为单一因果或具备整田能力。评估细节见该trial的 `comparison_summary.json` 和四份配对报告。

整理前完整历史已按原字节保存到 [第十三批文档归档](results/project_history/batch13_before_coupled_tracks_20261020/ISAAC_LAB.md)，同目录 manifest 记录 SHA-256。旧模型、协议、源码快照和原始成绩没有移动或改写。日常命令与训练流程见 [使用说明](使用说明.md)。

## 当前设计

2026-10-20追加删除辅助目录：二维页左侧分区导航与移动抽屉整块移除并清掉专属代码，实际网络分区、层名、全部60节点/71线和右侧张量详情保留。画布向左扩大226px，两页重复介绍同步精简；viewer专项25 passed、Node及py_compile均0。当前viewer载入相同final196608，暂停0秒，证据 `results/isaac_field_viewer/remove_sidebar_20261020/`。当前viewer.py、network_2d.py/html及第三方复用说明相对旧训练快照有记录的预期差异，旧实验文件不改写。

2026-10-20追加UI修改：连线/交叉弧不再随选层或悬停亮暗，主图保持全局统一色标，详情单张预览默认独立自动±maxabs且可切回全局；激活、weight和bias各按当前张量显示独立色条。专项CPU17 passed、compileall/Node语法均0，实际浏览器验证71条线样式不变、60节点、主图图片/色标不受详情开关影响且无console错误。仅当前network_2d.py/html相对已结束训练的42源快照发生预期变化，旧快照/模型/协议/成绩保持；新viewer载入同一final196608模型，当前暂停0秒。证据 `results/isaac_field_viewer/detail_local_scale_20261020/`。

- 物理：30 kg 轻量代理，车身盒体、左右各七个半径 45 mm 的圆柱轮及双轴云台；外宽 470 mm、履带中心距 410 mm。每侧以轮 0 为主轮，六个从轮通过 Newton 公开 mimic/equality 接口满足同向同速关系。每环境十二条关系，作用于本 world，不连接不同环境。
- 数值设置：物理步长 0.001 s、decimation 50、策略周期 0.05 s；椭圆摩擦锥、impratio=1、原 Newton 接触源。mu=0.8、实际 kf=1000、原材料和接触几何不变。
- 执行器：原来的14个显式 P/I 轮执行器及各2 Nm native clamp保持；P=0.04、侧积分Ki=0.1，仍是未标定仿真假设。联动不等于实车每侧单电机的完整传动模型，也没有额外车体速度伺服或手写转向力。
- 记录：drive_config增加按环境数量归一的同侧联动描述和真实物理/材料快照；一次构建读取并缓存，viewer刷新不逐帧读回GPU材料。训练和评估记录训练drive与实际drive，旧metadata不重贴。
- 神经网络：sensor-minimal-v6，LiDAR1×400、深度1×160×88、检测16×8、state33、feedback37；融合192→LSTM128→连续4维直接左右履带/云台动作。本轮不改网络、观测、奖励或课程。
- 界面：固定深色、完整二维网络展开、连续曲线、无整体视图或网络三维；车辆三维保留。实际激活/参数共用完整范围自适应Turbo色标。已核对并替换旧后端部署v7，8769/8770 HTTP200，GUI实际载入final 196608步模型，运行仿真时间0→6.5秒后暂停；drive/observation/action合同一致，60卡片0重叠、viewBox稳定、console errors为 `[]`。最终raw激活CPU核验真实exit0：60/60采集数组形状正确且有限，50变化/10不变（空检测支路与固定动作std），77/77参数相同；证据及截图见 `results/isaac_field_viewer/coupled_tracks_20261020/runtime_final_196608/`。

## 本轮已获得的诊断证据

详细记录见 [转向与接触诊断](results/isaac_field_diagnosis/steering_contact_repair_20261020/README.md)。

| 检查 | 结果及边界 |
|---|---|
| mu=.4/.8/1.2同初态八动作 | 增大mu反而减弱转向；不支持单纯摩擦不足。降低mu的回摆略大，不能据此选实车系数 |
| 原滚轮轴与被动DOF | 实际碰撞圆柱轴与铰轴一致，damping/frictionloss/armature全0；未发现简单轴向/被动阻尼错误 |
| cone/impratio、kf、解析plane | 未单独消除转向弱；失败或未改善结果保留，不作为生产修复 |
| 同侧强联动 | 12条关系真实进入Newton、host和MJWarp；同侧轮速和位置残差满足预设门槛 |
| 步长 | 2ms与1ms yaw差超过原门槛，故拒绝2ms；1ms与0.5ms差.007969rad/s≤.018448，通过该动作的yaw收敛检查，不外推所有量 |
| 同1ms八动作配对 | 两组各64000个物理样本、1280个策略样本，真实exit0；mimic8/8残差及原运动/制动门槛通过，原实际力矩预算保持 |

同1ms配对中，低pivot末1s yaw约 +.0704→+.1813、−.0725→−.1641 rad/s；高pivot约−.3795→−.6139、+.3854→+.6422 rad/s。直行/倒车yaw仍接近0。候选跨case最坏 qd 差 p95=.037625rad/s、max=.129288rad/s，policy q差max=.000519rad；实际effort max=1.29885Nm≤2。制动末.3s最大|v|=.0007675m/s、|w|=.003965rad/s，反向回摆积分仍约.0615m。这是固定动作的仿真验证，不是导航成绩或实车标定。

0.5ms物理已完整18000子步，但外层后处理路径错误导致真实exit1；独立CPU恢复验收exit0，raw与失败记录保留。诊断summary曾被追加/移除验收字段并重序列化，没有事前字节副本，不能声称其与初写字节完全相同；详见该目录sidecar。

## 最新完成的历史训练

`fresh_side_pi_20261020` 在PI-v6/5ms/独立滚轮物理下完成196608步、29检查点，2026-10-20 08:43:35北京时间真实exit0。协议SHA为 `0a356055c2ff08e13133ebfe12c71e90c63612a8efce7ea0db8b1187aafd10b8`；final SHA为 `f5be877d728037436bcfcca9b65df462de41dd1846f760473374f287f5a40735`。free与field成功均0/4→0/4，损伤/碰撞均0，均为no_improvement；末距变化不能称能力提升。该trial快照和成绩仍是原物理下的历史，当前源码改变后不再声称live源码与其41个快照一致。

新的 `fresh_coupled_tracks_20261020` 已独立prepare/launch，沿用四阶段196608步、正式0步baseline、每8192步检查点、42源/协议冻结和开发配对评估，不续旧权重。实验于2026-10-20 12:35:58（北京时间）真实exit0完成196608步，共29个检查点；协议SHA `2735ca85917643e49e27088957d00b1576f37902ce9664203253ae698181d5b6`，正式0步baseline SHA `beda8fde92236b558433a148245094caeefb88d7c23b9ea73ed94c4cd596f1d5`，final SHA `6b4c85884a51c2e23f557eb98bc04ea63905fc8f20d4379715fb9fe72b75cdc5`。42个冻结源与快照一致，final CPU验证通过（CPU-only、无GPU）。free配对0/4→1/4、末距1.685283→1.523645 m；field配对0/4→1/4、末距1.835561→1.801870 m；两组均0损伤/0碰撞，各有3/4超时。仅能称小样本开发改进；820xx、870xx是已观察开发数据，不是盲测，830xx仍未观察/未评估；不据此推断同侧联动为单一因果或整田能力。最终浏览器模型/UI及raw激活核验均已通过，原值/截图/真实CPU退出证据见 `results/isaac_field_viewer/coupled_tracks_20261020/runtime_final_196608/`。

## 续训入口与当前配置

`scripts/continue_isaac_foundation.py` 只接受上述已验证父实验，校验协议、final checkpoint、CPU/Adam 报告和允许的四个 UI 源差异。它把 policy 权重及 Adam 状态载入一个新的独立 trial，并重新初始化环境、RNG、LSTM 状态及 rollout buffer；这不是逐位恢复原运行进程。当前选用 60 秒回合，输出目录为 `results/isaac_field_improvement/continue_coupled_tracks_20261020`：

```bash
bash scripts/isaaclab_python.sh scripts/continue_isaac_foundation.py --prepare \
  --output results/isaac_field_improvement/continue_coupled_tracks_20261020 --max-time 60
bash scripts/isaaclab_python.sh scripts/continue_isaac_foundation.py --preflight \
  --output results/isaac_field_improvement/continue_coupled_tracks_20261020
bash scripts/isaaclab_python.sh scripts/continue_isaac_foundation.py --launch \
  --output results/isaac_field_improvement/continue_coupled_tracks_20261020 --device cuda:0
```

60 秒配置会同时改变回合上限与 `episode_time_ratio` 策略输入。该 trial 会在相同 60 秒设置下重新评估父 final baseline 和新 final，因此历史 20 秒成绩不能与这次结果混算。评估仍使用已观察开发种子；安全门槛通过前不进入 830xx 独立验证。

## 尚未完成与评估边界

- 完整作物搜索、避苗课程、粗糙地形/石块导航、对抗训练和整田可靠性；现有0株基础操控课程不能称为已经学会田间作业。新训练每组仅4个已观察开发种子，仍有3/4超时，不构成盲测或整田导航验证。
- 真实摩擦、土壤、传动、IMU噪声、安装外参、电流/RPM的标定，及FOC/HAL、串口/CAN实机联调。

安全评估必须在同物理、同初态下配对；无损成功增加，且损伤回合数、平均损伤、碰撞均不恶化才通过。820xx、870xx是开发数据，830xx继续保留未观察。不做训练速度/FPS基准，不把配置20Hz写成实测帧率。
