# 转向弱：只读物理与构造审计（2026-10-20）

范围：核对 wheel proxy 的 USD 几何/轴、当前 Newton/MJWarp 接触配置、已有接触与动力学诊断，以及适用的官方示例。未启动仿真或 GPU；未改生产源码、模型、协议、权重或成绩。本报告只记录已有证据和可证伪的后续对照，不据名义差速 yaw 目标建议调参。

## 已证实

- 坐标与关节声明互相一致：项目采用 Z-up、车辆前向 X、左右沿 Y。滚轮为半径 45 mm、宽 60 mm 的 Y 轴圆柱，revolute joint 轴也是 Y，joint local rotations 为 identity；左右滚轮中心位于 y=±0.205 m，轮轴沿车宽方向。静态 USD 输出的 upAxis、单位和各轮位置与源配置一致。依据：isaac_field/assets.py:182–219,222–307,357–394，已生成 results/isaac_field/assets/field_robot.usda。
- 滚轮实心圆柱的声明惯量计算正确：spin 轴 Y 为 0.000253125 kg·m²，另外两轴为 0.0002015625 kg·m²（assets.py:123–135）。旧 skid_response/model_dynamics.json 记录的 26.4 kg 底盘和对角惯量 (0.2475,1.4575,1.606) kg·m² 与 USD 声明值相符，但它属于旧诊断合同，不能代替当前模型导入后的逐轮惯量/armature 快照。
- 当前物理配置明确设为 dt=0.005 s、每个策略步 10 个物理步、use_mujoco_contacts=False、implicitfast、40 iterations、Newton shape margin 2 mm、gap 5 mm、mu 0.8（isaac_field/env.py:69–79）。NewtonShapeCfg 其余未覆盖的 ke/kd 是 2500 N/m 和 100 N·s/m（external/IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py:106–146）；solver cfg 默认 cone 为 pyramidal、impratio 为 1（mjwarp_manager_cfg.py:87–90）。项目 USD 未发现 PhysicsMaterialAPI 或 MuJoCo collider 属性。当前 schema 对未写入的 mjc:condim 说明 Newton 默认 3。最新 Newton 文档进一步说明：Newton contacts + SolverMuJoCo 会按 shape 的 solref_mode 决定 force-space gains、原始 MJCF solref 或 MJCF implicit default；force-space 路径按两个 shape 的 ke/kd、solimp dmax 和 body inverse weight 写入 per-contact solref，pyramidal 且 condim>1 时还带 μ、impratio、condim 系数；正值 solref 在 MuJoCo-Warp 的 refsafe 下按 max(timeconst, 2·dt) 生效。kf→solreffriction 仅适用于 elliptic cone + Newton contacts；当前配置是 pyramidal，且项目未设 kf。见官方 [Newton MuJoCo solver 文档](https://newton-physics.github.io/newton/latest/solvers/mujoco.html)。现有探针没有 dump 运行时各 shape/contact 的 dim、solref_mode/solref/solimp、friction 结果或两 shape mixing；而运行环境安装版可能早于当前文档版本，所以这些机制只能先指导要检查的字段，不能当作当前已观测事实。
- 当前单例接触探针报告末秒平均总法向力 293.111 N，约为 30 kg 重力的 99.6%；这排除了“整车总体几乎没接地”这一解释，但不代表 14 个轮子在每个子步都持续承载。轮载不均且有零值样本。两侧平均 wheel qd 为 +9.555/−9.320 rad/s，目标为约 ±9.259 rad/s；最大实际 applied effort 为 0.896 Nm，2 Nm 限幅未触发。与此同时 body yaw 只有 −0.07255 rad/s，约为名义 −2.0325 rad/s 的 3.6%，8 秒累计 yaw 变化 −0.6513 rad。依据：results/isaac_field_diagnosis/side_pi_physics_gap_20261020/contact_probe_metrics.json 和 contact_probe_analysis.md。
- 因为轮速接近目标而车身 yaw 很弱，现有结果支持“驱动关节在 spin，但差速运动没有充分转成车体 yaw”；它不支持“电机执行器撞上 2 Nm 限幅”。父任务已核对旧 ground_contact 的 wheel-body spin、joint qd 和有限差分一致，降低了简单 rad/deg 或观测轴读错的可能。P 增益与单轮惯量之比 Kp·dt/I ≈ 0.79；这不是单独证明数值稳定的证据，5 ms 步长与接触耦合仍可能重要。
- 几何长度语义未定为错误：轮中心纵向端点间距为 0.595 m；加上端部半径后的轴向 AABB 长度约 0.685 m。若 0.595 m 表示滚轮接地点/中心 span，该几何可以自洽；必须先有设计定义才能把包络差异认定为资产 bug。
- 场景源代码用 USD triangle mesh 作为地面碰撞，并声明 approximation=none（isaac_field/scene_assets.py:8–22）。当前 use_mujoco_contacts=False 走 Newton collision pipeline；最新 Newton 文档说明该路径可对无 SDF 的 triangle mesh 做 live BVH 距离查询，而 MuJoCo 内部 contact 路径只支持凸 mesh，非凸 mesh 会被 MuJoCo compiler convex-hull。由于平地 A/B 两条路都可以用同一平面接触，这项比较适合平地诊断；不要把 internal MuJoCo 路径未经验证地全局用于非凸粗糙 terrain。文档：[Newton collisions](https://newton-physics.github.io/newton/latest/concepts/collisions.html)、[Newton MuJoCo solver](https://newton-physics.github.io/newton/latest/solvers/mujoco.html)。运行时是否仍采用预期的 mesh representation，现有探针未输出。机器人侧是 14 个离散独立旋转的圆柱，没有连续履带带面；离散 proxy 缺失履带纵向/横向剪切与连续接触分布，是合理的候选模型限制，但还未被直接验证。

## 证据边界与待验证假说

现有 sensor 报告轮体的 aggregate world-frame normal/friction vectors，而 Newton contacts 接口探针只保存 pair IDs 和 normals，没把 per-pair force/contact point 连起来。因此，当前数据不能把摩擦合力分解到单个地面接触，也不能做接触力矩对 yaw 响应的闭合检查。contact_transfer_probe_summary.json 明确列出这个限制；contact_probe_analysis.md 也说明 solver→sensor 的完整力坐标变换链尚未独立验证。

历史 skid_response/friction_sweep.json 中，较低 mu 的旧合同诊断出现更大的差速 yaw；旧 cone_sweep.json 没有显示换 cone 足以解决弱转向。这些数据与高侧向 scrub/接触正则化值得查的方向相容，但测试合同、轮距和驱动设置不同，低 mu 也未标定，不能据此调参数或宣称当前根因。较可能的候选顺序是：接触源/正则化与步长；离散轮 proxy 的侧向 scrub/缺少连续履带；再核对实际碰撞几何和惯量传递。当前证据不足以在这些原因中定因。

## 最小可证伪对照

1. 先完成当前准备中的平地接触路径 A/B：同一当前资产、初态、命令、驱动和 5 ms timestep，只切换 Newton collision pipeline 与 MJWarp internal contacts。它比较的是接触生成/导入路径及其 solref 映射整体，不是单独证明某一参数。对比 paired reset、两侧 qd、轮体 spin 轴、body yaw、法向支撑与 contact 数，并记录每个 wheel body 的实际 mass/inertia/COM、joint armature。若 yaw 明显变化，再单独检查实际 shape friction、condim、solref_mode/solref/solimp、最终接触 friction rows 和 runtime mesh type；只看 config 不够。本项仅对平地诊断有效，不能把 use_mujoco_contacts=True 全局套到非凸粗糙 terrain。若接触路径切换显著恢复 yaw，优先追究导入与 solref/contact mapping；若相近，接触路径本身就不太像单一根因。
2. 在结果允许时固定表现更可信的接触路径与其他所有参数，比较 5、2、1 ms 物理步长，保持 50 ms 策略步长及相同初态/命令。预先比较 qd、yaw、法向支撑和接触数的收敛；MuJoCo-Warp 正格式 solref 的 refsafe floor 随 dt 从 10 ms、4 ms 到 2 ms 变化。若较细步长下 yaw 大幅变化而 wheel qd、支撑量稳定，则支持积分/接触耦合问题；若几乎不变，则削弱该假说。不要同时改 mu、cone、solref 或 drive gain。
3. 若两项仍显示轮子 spin 而底盘 yaw 弱，下一次诊断才需要逐 pair 输出 world contact point/force 与实际摩擦参数，核对世界坐标力矩和车体角动量变化；这能区分“侧向摩擦确实阻止车体差速转动”与“接触/传感导出不闭合”。

## 现成示例适用性

没有找到本地可直接替换的 Isaac Lab tracked-wheel proxy。官方 Isaac Sim 的 differential controller/wheeled robot 示例主要说明普通两轮速度控制接口；Newton 的 robot_omniwheel 是全向轮机制，二者都不能直接证明此 14 圆柱侧带 proxy 的接触或转向正确。适用来源：[Isaac Sim Differential Drive Tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/controllers/tutorial_differential_drive.html)、[Newton MuJoCo solver](https://newton-physics.github.io/newton/latest/solvers/mujoco.html)、[Newton collision concepts](https://github.com/newton-physics/newton/blob/main/docs/concepts/collisions.rst)、[Newton omniwheel example](https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_omniwheel.py)。示例可用于查接口与接触参数语义，不是当前履带模型的 ground-truth benchmark。


