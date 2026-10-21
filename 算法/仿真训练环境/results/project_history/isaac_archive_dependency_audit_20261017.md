# Isaac 归档依赖与运行状态核查（2026-10-17）

本次为只读审计；未停止进程、未改源码，也未运行测试。唯一新增项为本审计记录。

## 结论

- 当前训练和 viewer 都在运行。两者使用同一个 WSL Ubuntu-24.04 环境 `/home/wangj/robotics-lab/venv/isaaclab-field`，工作目录为本项目。归档/移动项目源码或运行目录会影响正在运行的任务。
- **不得归档当前 `external/IsaacLab` checkout、Isaac 源码、活动 venv、`results/isaac_field_improvement/.../resume_8192`、viewer 输出/其加载的 checkpoint，或活动的 `.cache/uv`、`.cache/warp`。** 当前 venv 中 Isaac Lab 组件是从项目内 `external/IsaacLab/source/...` editable 安装的；`direct_url.json` 直接指向该工作树。
- Isaac 运行路径没有导入旧顶层 `field_sim`、`simulation`、`config`、`training` 或 `uwb_plugin` 的 Python 模块。`field_sim` 仅在旧 checkpoint 元数据迁移说明/映射及其测试数据中出现；不需要保留该旧包来运行当前 Isaac 模型。
- 旧 Gazebo/ROS 工具确实引用 `simulation`、`training`、`config`、`uwb_plugin`、`.venv-field` 和 `.runtime-gazebo`。如果归档这些目录，应将它们与各自旧启动脚本作为一组处理；它们不属于当前 Isaac 活动依赖，但也不是项目中完全无引用的文件。

## 活动进程与运行环境

由 Windows 进程表、WSL 进程/cwd/命令行、监听端口和 `/proc` 环境变量核对：

| 任务 | Windows WSL 包装 PID | WSL 实际 Python PID | 当前用途 |
|---|---:|---:|---|
| foundation 恢复训练 | 8424 | 299 | `scripts/train_isaac_foundation.py --launch --output results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive/resume_8192 --device cuda:0`；运行进程的解释器为 `/home/wangj/robotics-lab/venv/isaaclab-field/bin/python`，cwd 为本项目。 |
| viewer | 17740 | 670 | `-m isaac_field viewer`；同一解释器和 cwd，监听 `127.0.0.1:8768`。载入 `results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive/resume_8192/stage_2_steer30/checkpoints/field_policy_steer30_49152_steps.zip`，输出目录为 `results/isaac_field_viewer/chinese_reset_fixed_49152_20261017`。 |

训练进程命令显式指定 `cuda:0`。本次 `nvidia-smi --query-compute-apps` 查询未能返回进程映射，因此这里只记录命令配置，不据此断言 GPU 分配。运行时环境变量经进程环境直接读取：两者的 `UV_PROJECT_ENVIRONMENT` 都是 `/home/wangj/robotics-lab/venv/isaaclab-field`，`UV_CACHE_DIR` 和 `WARP_CACHE_PATH` 分别是项目内 `.cache/uv`、`.cache/warp`。该 venv 的 `bin/python` 解析到系统 `/usr/bin/python3.12`，但包和 editable 安装都在此 venv 中，不能以系统 Python 替代。

当前环境的 `site-packages/isaaclab*.dist-info/direct_url.json` 将 `isaaclab`、`isaaclab_newton`、`isaaclab_visualizers` 等标记为 editable，并指向项目 `external/IsaacLab/source/...`。例如 Isaac 模块从 [env.py](../../isaac_field/env.py#L17) 导入 `isaaclab` 与 `isaaclab_newton`。安装器在 [install_isaaclab.sh](../../scripts/install_isaaclab.sh#L5) 固定 checkout 路径和上游 commit `5870804e4f6aee57bdef68d59f67d328b990d1e1`，并以 editable 方式安装本项目（同文件 L61）；因此当前 `external/IsaacLab` 必须留在原路径。`external/` 目前只含这个 `IsaacLab` 目录。

## Isaac 入口和依赖引用

- [pyproject.toml](../../pyproject.toml#L5) 定义项目包为 `isaac-field`；L9–16 要求 Python 3.12、依赖 `isaaclab-dev[sb3,viser]` 与 `sb3-contrib==2.9.0`，并注册 `isaaclab.tasks` 入口点 `isaac_field`。
- [启动IsaacLab.cmd](../../启动IsaacLab.cmd#L12) 委托给 `scripts/isaaclab_python.sh -m isaac_field viewer`；[安装IsaacLab.cmd](../../安装IsaacLab.cmd#L12) 委托给 `scripts/install_isaaclab.sh`。二者没有调用旧 Gazebo 环境。
- [isaaclab_python.sh](../../scripts/isaaclab_python.sh#L4) 设置项目根；L7–10 选择项目外的默认 venv，并设置 `.cache/uv`、`.cache/warp`；L18 从该 venv 启动 Python。[install_isaaclab.sh](../../scripts/install_isaaclab.sh#L41)–61 在 `external/IsaacLab` 的固定 checkout 同步依赖，再安装本项目。
- Isaac 源码与测试中没有 `field_sim`、`uwb_plugin`、`simulation`、`config`、`training` 模块导入，也没有 `.venv-field` 或 Gazebo runtime 路径导入。Isaac 源文件里的 “simulation/config/training” 多为仿真概念或配置字段，并非这些旧目录依赖。
- 精确的 `field_sim` 引用是兼容层文字：[assets.py](../../isaac_field/assets.py#L446) 明确表示不导入它；[policy.py](../../isaac_field/policy.py#L440) 的 `convert_legacy_checkpoint` 通过 `custom_objects` 将旧 extractor 元数据映射到当前 `SensorEncoder`，实现本身未导入旧包。对应测试 [test_isaac_field_policy.py](../../tests/test_isaac_field_policy.py#L72) 只把旧类名作为 checkpoint 元数据字符串。
- `scripts/compare_isaac_exploration.py` 的 frozen runtime 清单包含 `scripts/isaaclab_python.sh`（L12）；恢复/训练脚本依赖 `isaac_field` 和这些 Isaac 对照脚本，未依赖旧顶层训练目录。

## 旧目录的引用边界

下表是项目源码中的命中位置，不表示该旧子系统适合直接删除：

| 路径 | Isaac 依赖判断 | 项目其他引用 / 处理边界 |
|---|---|---|
| `field_sim/` | 当前 Isaac 不导入；只有旧 checkpoint 兼容映射和测试字符串引用。 | 可把旧实现本身归档；保留 `isaac_field.policy.convert_legacy_checkpoint` 才能继续迁移旧权重。 |
| `simulation/` | 当前 Isaac 不用该顶层目录。 | 旧启动和脚本引用，例如 `scripts/field_gazebo.sh:13`、`scripts/launch.sh:8–24`、`scripts/build_gazebo_control.sh:17`、`scripts/generate_world.py:6`。须与旧 Gazebo/其他仿真入口成组归档。 |
| `config/` | 当前 Isaac 安装不读取它。 | 旧 `scripts/install.sh:31,55–56`、`scripts/verify.py:41,68`、`scripts/package.py:10–17` 引用；旧安装/验证/打包如要保留则需保留。 |
| `training/` | 当前 Isaac 使用包内 `isaac_field/training.py`，不导入顶层 `training/`。 | `scripts/launch.sh:23` 和 `scripts/package.py:15` 引用旧 `training/robot_learning.py`。 |
| `uwb_plugin/` | 当前 Isaac 不构建或加载它。 | 旧 `scripts/install.sh:61` 和 `scripts/package.py:15` 引用。 |
| `.venv-field/` | 当前训练/viewer 均不使用。 | 旧 `scripts/field_gazebo.ps1:28` 构造其中 Windows Python 的路径。 |
| `.runtime-gazebo/` | 当前 Isaac 不使用。 | 旧 `scripts/build_gazebo_control.sh:12`、`scripts/field_gazebo_web.sh:5`、`scripts/field_train_gazebo.sh:17`、`scripts/gazebo_python.sh:5` 引用。 |
| `.runtime-gazebo-wheels/` | Isaac 代码/入口中无引用。 | 当前已扫描的脚本中未找到字面路径引用；归档前仍应将其视为旧运行环境产物核对，而非 Isaac 必需项。 |
| `dist/` | Isaac 源码和入口中无目录路径引用。 | 当前活动训练/viewer 不从此目录加载内容；无需为 Isaac 保留。 |
| `.cache/` | 活动进程实际使用 `.cache/uv` 与 `.cache/warp`；两个 Isaac shell 脚本也默认设置这些路径。 | 保留活动子目录；本次未据此判断 `.cache` 内其他缓存是否必要。 |
| `external/` | 必需；当前只有 `external/IsaacLab`，且 venv 中有指向该 checkout 的 editable 安装元数据。 | 不可在活动训练/viewer期间移动或归档。 |

## 可复用的 CPU 验证命令

此处是历史记录，不是本次运行结果。第九批前的最新完整 CPU 记录为 `results/project_history/cleanup_and_validation_20261017_batch8.log`：

```bash
unset RUN_ISAAC_PHYSICS
bash scripts/isaaclab_python.sh -m pytest -q tests
```

历史结果为 **117 passed、8 skipped，exit 0**；8 项跳过的是需 `RUN_ISAAC_PHYSICS=1` 的原生物理测试。静态/语法验证的历史命令也有记录：

```bash
bash scripts/isaaclab_python.sh -m compileall -q isaac_field scripts tests
bash -n scripts/install_isaaclab.sh scripts/isaaclab_python.sh
```

对应日志为 `results/project_history/cleanup_and_validation_20261017_batch8.md/.log`；第七批历史结果 `101 passed、8 skipped`，不覆盖较新的第八批结果。
