# 仿真驱动限制与FOC等效换算

本记录只读审计了 `isaac_field/env.py`、`drive_control.py`、`task.py`、`assets.py` 和准备中的 random-rough protocol。没有启动仿真、pytest、GPU 或训练；`random_global_rough_coupled_guides_20261020` 仍是 prepared、0 timestep。

## 仿真里实际限什么

- `task.py` 将 15 km/h 转为 `max_speed=4.1667 m/s`，并将 `acceleration=1.5 m/s²` 作为可配置的速度命令斜坡。`env.py` 先将左右动作缩放为目标履带速度，再按 `acceleration * step_dt` 限制每个策略步的命令变化，最后用 45 mm 支撑轮半径换算关节目标角速度。因此 15 km/h 是策略可请求的速度上限，不是车体实际速度硬裁剪或速度保证。
- `env.py` 给每个 `.*_wheel_joint` 执行器设置 `actuator_effort_limit=2 Nm`，并在同一控制器中用 `2 Nm` 限制 PI 积分前馈；Newton native 执行器的每轮输出夹限是最终安全边界。当前 14 个支撑轮驱动关节各有这一限幅。导向轮不匹配这个驱动关节命名，且无独立电机、PID 或附加驱动扭矩。
- `joint_velocity_limit=110 rad/s` 也写在驱动执行器配置中；本次只确认该配置值，未断言 solver 一定以硬夹限方式执行它。按 15 km/h 和 45 mm proxy 半径换算出的目标角速度为 92.59 rad/s，低于该配置值。
- `drive_control.py` 的条件积分防饱和检查 `P_estimate + integral`。这里的 `P_estimate` 是比例控制项 `kp * speed_error`，不是电功率估算或瓦特限值。`task.py` 还保留 `angular_acceleration=4.0` 配置值，但当前 Newton 履带速度控制路径使用的是线加速度斜坡；本记录不把该角加速度字段当成已生效的驱动限制。

当前所审代码和 protocol 未发现总电功率、母线电流、电压、电池、热降额/过温、再生制动或随转速变化的电机转矩限制。环境反馈中的 `motor_iq_a` 是用 Newton 限幅后的代理轮关节力矩，经 `r_sprocket / (r_proxy * reduction * efficiency * Kt)` 换算得到的仿真等效信号，不是实测相电流，也没有对应的电流控制限值。温度反馈固定写入 25°C，不是热模型或热保护。

## 在所列假设下的算术换算

假设 14 个代理支撑轮都同时输出各自的 2 Nm 限幅、每侧 7 轮、proxy 半径 45 mm、驱动链轮半径 30 mm、减速比 4:1、效率 1、`Kt=0.035 Nm/A`，并以 15 km/h 目标速度计算：

| 量 | 每侧 | 两侧合计 |
|---|---:|---:|
| 代理轮轴力矩和 | 14.0 Nm | 28.0 Nm |
| 由代理轮力矩换算的理想切向推力 | 311.1 N | 622.2 N |
| 等效链轮输出力矩 | 9.333 Nm | 18.667 Nm |
| 等效电机轴力矩 | 2.333 Nm | 4.667 Nm |
| 等效 `Iq = τ_motor / Kt` | 66.67 A/电机 | 2 台各 66.67 A |
| 满速时的代理轮机械功率算术值 | 1.296 kW | 2.593 kW（14 轮） |

速度口径须区分：按 45 mm 代理轮换算，关节目标为 `v / r_proxy = 92.59 rad/s = 884.19 rpm`；按 30 mm 驱动链轮换算，链轮为 `138.89 rad/s = 1326.29 rpm`，4:1 电机轴为 `555.56 rad/s = 5305.16 rpm`。`env.py` 的 `drive_config` 以链轮半径和减速比记录后一项电机转速。

力矩换算为：每侧代理轮力矩和 `7 × 2 = 14 Nm`；推力 `F = 14 / 0.045 = 311.1 N`；链轮输出力矩 `F × 0.03 = 9.333 Nm`；电机轴力矩 `9.333 / (4 × 1) = 2.333 Nm`；等效电流 `2.333 / 0.035 = 66.67 A`。机械功率为 `Σ(τ_proxy × v/r_proxy)`，每侧 1.296 kW，14 轮总和 2.593 kW；以等效电机轴 `τ × ω` 算得相同值，因为假设效率为 1。

这些是同时达到力矩与速度目标时的**机械算术上界**，不是仿真总电功率限额，也不是电池输入功率、连续额定值、土壤可用牵引力或实测电流。它不含实际传动效率、转矩-转速曲线、控制带宽、电压下垂、相电流定义、接触滑移、热状态或电池约束。`Kt`、效率和代理轮每轮 2 Nm 限幅均不能当成 N5065 实测/额定值；66.67 A 也不能解读为卖家标称的 80 A。

## 文件与行号依据

行号对应本次只读审计时的 live source；prepared protocol 的值另列在 JSON 中。

- `isaac_field/task.py:73-76`：速度、线加速度和遗留角加速度配置。
- `isaac_field/env.py:80,93-94`：每轮力矩限幅、等效 `Kt` 与效率假设默认值。
- `isaac_field/env.py:208-214`：驱动执行器力矩与 110 rad/s 速度配置。
- `isaac_field/env.py:487-506`：动作缩放、速度命令斜坡、proxy 半径到关节目标速度、控制输出。
- `isaac_field/drive_control.py:48-69,73-131`：比例估算项、条件积分 anti-windup 和每侧积分前馈限幅。
- `isaac_field/env.py:543-570,581-585`：post-clamp proxy effort 换算等效 Iq/RPM；温度反馈固定为 25°C。
- `isaac_field/env.py:761-803`：等效每侧电机力矩、满速电机转速及单位/非实测元数据。
- `isaac_field/assets.py:65-81`：30 kg 几何、45 mm proxy 半径、30 mm 链轮半径、4:1 减速比、每侧 7 轮。
- `results/isaac_field_improvement/random_global_rough_coupled_guides_20261020/protocol.json:181-183,241-245`：prepared trial 中的最高命令速度、线加速度与几何/减速参数。protocol 的 `config` 没有单独冻结 `drive_effort_limit`、`Kt` 或效率数值；这些默认值由其 source snapshot 锁定。
- `results/isaac_field_improvement/random_global_rough_coupled_guides_20261020/run_status.json`：当前 `prepared`、prepare exit 0、`num_timesteps=0`；protocol SHA 为 `5de27ea7b7e1714b8822ec8fe11f0b53e6f08d194577b253d376913a00405980`。
