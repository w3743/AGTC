# Newton MJWarp 回合随机化接口审查

日期：2026-10-17  
范围：只读核对项目和本地 Isaac Lab / Isaac Lab Newton 源码。未修改源码、配置或协议，未启动仿真、GPU、测试或 benchmark；未做实车标定。

## 结论

现有 `IsaacFieldEnv` 可复用 Isaac Lab `DirectRLEnv` 的 reset 事件管理器，在 episode reset 时更新轮关节驱动 damping（本项目的速度伺服 gain）和轮碰撞 shape 的 Newton 摩擦系数。Newton articulation 还提供公开的实时 joint effort limit 写入接口，但没有找到现成的 effort-limit 随机化事件。当前静态地面以 `AssetBaseCfg` 加 USD mesh 导入，不能直接作为官方刚体材质事件的目标，所以现成接口不能独立随机地面摩擦。

建议的最小接法是使用 reset-mode EventManager：对 `robot` 的轮关节使用官方 `randomize_actuator_gains` 的 damping 参数，对轮链接 body 使用官方 Newton `randomize_rigid_body_material`。如果之后必须随机驱动力矩上限，再添加一个只负责采样并调用公开 effort-limit writer 的小 reset 事件。保持地面摩擦固定，明确记录当前变化的是轮侧 shape 材质。

这只是源码支持路径，不代表当前项目的 `MJWarpSolverCfg(use_mujoco_contacts=False)` 已完成集成验证或实车标定。

## 项目当前配置

- `isaac_field/env.py` 将 `drive_velocity_gain`（当前默认 0.5）写入轮关节 `ImplicitActuatorCfg.damping`，`stiffness=0`；`drive_effort_limit`（当前默认 2.0）作为 `joint_effort_limit`。
- Newton 的 `default_shape_cfg.mu` 当前设为 0.8。`isaac_field/scene_assets.py` 给 `/Terrain/Ground` 创建碰撞 mesh，但没有给该地面 mesh 单独绑定摩擦材质。
- `Field` 由 `AssetBaseCfg` 导入为静态资产。轮子则是 `robot` articulation 的独立 wheel links 和 wheel joints，可通过 articulation selector 选中。
- `FieldEnvCfg` 当前未配置 `events`。Isaac Lab 的 `DirectRLEnvCfg.events` 是可选项；父类在 reset 事件阶段应用事件。本项目 `_reset_idx` 开头调用 `super()._reset_idx(env_ids)`，因此 reset-mode 事件可复用该入口。

## 官方接口与限制

### 轮关节驱动增益

`isaaclab.envs.mdp.events.randomize_actuator_gains` 支持按环境随机化 stiffness/damping。对 implicit actuator，它走 `Articulation.write_joint_stiffness_to_sim_index` 或 `write_joint_damping_to_sim_index`；Newton articulation 实现了这些写接口。因此当前轮关节的 `damping` 可作为速度伺服 gain 的回合随机化入口，selector 只需选 wheel joints。

该事件只随机 stiffness/damping，不随机 effort limit。项目只改配置字段 `drive_velocity_gain` 而不更新已构建的 live joint 属性，不会完成运行时随机化。

源码证据：

- `external/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py`：`randomize_actuator_gains`。
- `external/IsaacLab/source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`：Newton joint stiffness/damping 写入实现。
- `external/IsaacLab/source/isaaclab_newton/test/assets/test_newton_actuators_newton.py`：`TestRandomizeActuatorGainsViaEventsNewton` 检查 Newton actuator gains 经事件更新。

### 轮侧摩擦

通用 `randomize_rigid_body_material` 根据活动 physics backend 分派到 Newton 实现。Newton 实现读取/更新 shape 的 `shape_material_mu`，按 body selector 选 articulation shapes，并通知 `ModelFlags.SHAPE_PROPERTIES`。Newton 使用单一摩擦系数；`dynamic_friction_range`、`num_buckets` 和 `make_consistent` 在该后端被忽略。selector 只指向 `robot` 的 wheel links 时，可以随机轮侧 shape 摩擦，而不改变底盘、云台或其他刚体 shape。

这个官方材质事件只接受 `RigidObject` 或 `Articulation`。当前 `Field` 是 `AssetBaseCfg` 静态 USD mesh，不能直接作为它的目标。`NewtonShapeCfg.mu` 是模型构造期默认值，修改 cfg 不等于按回合更新已创建的 shape buffer。

源码证据：

- `external/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py`：公共事件与 backend dispatch。
- `external/IsaacLab/source/isaaclab_newton/isaaclab_newton/envs/mdp/events.py`：Newton `randomize_rigid_body_material`，包括对 Kamino 的共享材质特例；当前项目使用 MJWarp。
- `external/IsaacLab/source/isaaclab_newton/test/envs/test_events.py`：Newton 材质 body selection 测试。
- `isaac_field/env.py`：`Field` 资产类型和 Newton 默认 shape friction 配置。
- `isaac_field/scene_assets.py`：地面碰撞 mesh 作者代码。

### 力矩上限

没有找到 Isaac Lab 现成的随机化 effort-limit EventTerm。`randomize_joint_parameters` 覆盖 joint friction、armature 和 position limits；`randomize_actuator_gains` 覆盖 stiffness/damping，不覆盖 `joint_effort_limit`。

Newton articulation 有公开 `write_joint_effort_limit_to_sim_index(limits, joint_ids, env_ids)`。如需按 episode 随机化，可由一个小 reset event 依据保存的标称值生成每环境限值并调用该 API；应从标称值采样，避免连续 episode 累乘。此写入会通知 `ModelFlags.JOINT_DOF_PROPERTIES`。这是可用的底层公开接口，不是现成的随机化事件。

源码证据：

- `external/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py`：`randomize_joint_parameters` 和 `randomize_actuator_gains` 参数范围。
- `external/IsaacLab/source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`：Newton `write_joint_effort_limit_to_sim_index` 实现。

## 验证边界

- 本次仅审查了本地源码，没有在当前环境创建事件项或检查运行时 buffer 变化。
- Isaac Lab Newton 有相关事件和 actuator 单测，但本次没有运行这些测试；这些源码证据不能替代对项目具体 14 轮 articulation、当前 MJWarp collision-pipeline 配置的原生回归。
- 轮 shape 摩擦随机化不等同于独立随机化地面材质；实际接触行为还需要当前 Newton 配置下的原生接触验证。
- 摩擦、增益和力矩取值范围都未由实车测量或标定，本审查不提供推荐数值范围。
