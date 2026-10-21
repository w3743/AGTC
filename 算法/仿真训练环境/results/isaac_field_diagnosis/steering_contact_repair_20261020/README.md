# 转向接触诊断阶段记录

本页汇总2026-10-20的隔离诊断及随后正式实现。已将同侧联动、1ms积分与椭圆摩擦锥接入生产；完整14项native和全CPU均通过。新初始化训练现已完成，free/field各4个同初态开发种子均由0/4提升至1/4成功，且两组都保持0损伤、0碰撞；末距和限制详见下文。这是小样本开发改进，不是盲测或整田导航结果，不能据此归因于同侧联动单一因素。没有改旧资产、旧训练协议、模型或历史成绩；实车标定仍未完成。

## 可靠结论

**“摩擦系数太小”不受当前对照支持。** 三组使用相同 5 ms / pyramidal / `impratio=1` / `kf=1000`、同一初态和八动作响应矩阵，且真实退出码均为 0。低速左/右原地转向末 1 s 平均 yaw rate 绝对值，µ=.4/.8/1.2 分别约为 `.237/.237`、`.107/.098`、`.082/.073 rad/s`；高速原地转向的累计 yaw 也随 µ 增大而变小（µ=.4 约 `+1.576/−1.232 rad`，µ=.8 `+.462/−.506 rad`，µ=1.2 `+.360/−.350 rad`）。低 µ 在本模型中更容易转，高 µ 更弱；因此不能把“继续增大 µ”称作修复，也不能把这组数值当作实车地面参数。具体矩阵和退出记录见 [`staged_response/README.md`](staged_response/README.md) 与 [`staged_response/runs/`](staged_response/runs/)。

**修复前模型是 14 个独立滚轮转动自由度（每侧 7 个），原始运动链没有同侧带速联动。** v2 诊断及当前生产实现为每侧以 wheel 0 为主轮、另外 6 轮各加一条 `q_follower=q_master` 等式，共 12 条；等式用公开 builder schema 配置 `solref=[.004,1]`、`solimp=[.999,.9999,.001,.5,2]`。为了隔离机械关系，原 14 个 P/I 轮执行器及各自 2 Nm 上限均保留，因此这不是单侧电机加连续履带的完整模型。v2 单动作 2 ms 运行中，末 1 s 原生 qd 残差 p95/max 为 `.02633/.07052 rad/s`，策略采样 q 差最大 `.000381 rad`，均通过原门槛；详见 [`same_side_mimic_v2/README.md`](same_side_mimic_v2/README.md) 及其 summary/exit 文件。

同侧联动的 2 ms 八动作 baseline 与 mimic 矩阵均真实 exit 0，各捕获 `32,000/32,000` 原生子步，八个动作的初态哈希匹配；mimic 的全部八个动作均通过残差和 2 Nm 力矩门槛。配对核验的真实 CPU exit 0 见 [`paired_comparison_cpu_attempt3.exit.json`](staged_response/same_side_mimic_motion_v1/paired_comparison_cpu_attempt3.exit.json)，字段和逐动作比较在 [`paired_comparison.json`](staged_response/same_side_mimic_motion_v1/paired_comparison.json)。它证明该诊断约束可工作，不证明实际履带动力学已还原或转向问题已经修好。其余原始数据见 [`staged_response/same_side_mimic_motion_v1/README.md`](staged_response/same_side_mimic_motion_v1/README.md) 及其 `runs/` 子目录。

**候选步长收敛判定：2 ms 未过，1 ms 与 0.5 ms 按原阈值通过。** 同侧联动 v2、µ=.8、椭圆摩擦锥、`impratio=1`、`kf=1000`、PI/动作/初态不变，初态 SHA-256 为 `7d7cd17756b8eee3eedc39828b8a2e90537820f208daf8cd611a40d657e8e8af`，固定比较合同 SHA-256 为 `33b4298fa1bd57f5c114d5a17d28ba0c171a697cc8ff42d95559233501142b05`。

| physics dt | 末 1 s yaw rate (rad/s) | 对照与判定 |
|---:|---:|---|
| 2 ms | −0.171675 | 与 1 ms 差 `.020778`，超过原门槛 `.019245`，未通过收敛判定；[原始 summary](same_side_mimic_v2/contact_probe_runs_v2/dt-2ms__cone-elliptic__impratio-1__mu-0.8__kf-1000__equality-softening-v2__attempt1/same_side_mimic_v2_summary.json)。 |
| 1 ms | −0.192453 | 与 0.5 ms 差 `.007969`，通过原门槛 `.018448`；原生 qd p95/max `.02633/.07052`、策略 q 差最大 `.000381 rad`，残差门槛通过；[原始 summary](same_side_mimic_v2_dt1ms/contact_probe_runs_dt1ms/dt-1ms__cone-elliptic__impratio-1__mu-0.8__kf-1000__equality-softening-v2__attempt2/same_side_mimic_v2_dt1ms_summary.json)。 |
| 0.5 ms | −0.184484 | 原生 qd p95/max `.01895/.05211`、策略 q 差最大 `.000275 rad`，残差门槛通过。物理段完整捕获 `18,000/18,000` 子步；见[离线接受结果](same_side_mimic_v2_dt0p5ms/contact_probe_runs_dt0p5ms/dt-0p5ms__cone-elliptic__impratio-1__mu-0.8__kf-1000__equality-softening-v2__attempt1/postprocess_repaired_acceptance.json)。 |

因此 1 ms 只是本单动作测试下的步长收敛候选。0.5 ms 的 motion runner 外层真实 exit 为 1，原因是 post-run 接受检查误指向 baseline summary；raw 与物理采样完整，原始退出码见 [motion `.exit.json`](same_side_mimic_v2_dt0p5ms/contact_probe_runs_dt0p5ms/dt-0p5ms__cone-elliptic__impratio-1__mu-0.8__kf-1000__equality-softening-v2__attempt1.exit.json)。之后仅 CPU 离线重算真实 exit 0，接受判定通过，见 [CPU postprocess `.exit.json`](same_side_mimic_v2_dt0p5ms/contact_probe_runs_dt0p5ms/dt-0p5ms__cone-elliptic__impratio-1__mu-0.8__kf-1000__equality-softening-v2__attempt1/postprocess_repaired_acceptance.exit.json)。其 summary 经诊断后处理重新序列化，缺少独立的处理前字节副本，故不声称原 summary 字节完全未变；恢复说明与保留副本见[接受结果 sidecar](same_side_mimic_v2_dt0p5ms/contact_probe_runs_dt0p5ms/dt-0p5ms__cone-elliptic__impratio-1__mu-0.8__kf-1000__equality-softening-v2__attempt1/postprocess_repaired_acceptance.json)。

## 尚未定位的候选

| 检查项 | 已观察结果 | 结论边界 |
|---|---|---|
| 椭圆摩擦锥 / `impratio` | `impratio=10/100` 运行完成；`100` 的末 1 s yaw rate 为 −`.06970 rad/s`，未见转向改善。首个 `impratio=1` 尝试在运动前因 MJWarp `impratio_invsqrt` 数组形状检查失败；不得当作物理结果。 | 未支持把锥型/该参数调整称为修复。v3 的 runtime 字段和失败尝试见 [`contact_probe_runs_v3/`](contact_probe_runs_v3/)。 |
| `kf=10000` | 15 个实际轮/地接触形状均核验为 10000；2/1/0.5 ms 末 1 s yaw rate 分别约 −`.1054/−.0980/−.0866 rad/s`。 | 增大接触刚度没有消除弱转向；不同 dt 的这些值不能单独作为 `kf` 因果对照。见 [`contact_probe_runs_v5/`](contact_probe_runs_v5/)。 |
| 平地几何替换 | flat mesh 换成 native plane 的单因素 case 已运行。 | 未改善；几何替换不是已确认修复。见 [`contact_ground_plane_v6/README.md`](contact_ground_plane_v6/README.md) 与其 summary/exit。 |
| 接触材料、力方向、滚轮轴 | 运行时轮/地材料核验 µ=.8、condim=3、`solref=[.02,1]`、`solimp=[.9,.95,.001,.5,2]`。runtime cylinder 几何轴经 quaternion 映射后与关节轴平行（绝对点积 1）；22 个 DOF 的 damping、frictionloss、armature 均为 0。采样承载接触中前向滑移大于横向滑移，切向摩擦到达 `µFn` 边界，`Ft·vt` 为负。 | 当前采样未发现切向力坐标/符号反用证据；已排除检查到的轴向转换不一致和隐含关节阻尼。稀疏接触快照不能排除完整 solver/积分行为，也不能验证实车转向。详见 [`contact_axis_telemetry_v1/README.md`](contact_axis_telemetry_v1/README.md) 与 [`contact_probe_runs_v4/`](contact_probe_runs_v4/)。 |

## 正式验收与后续训练

**C 的 1 ms 同侧联动八动作配对矩阵已完成，两个 case 均真实 exit 0。** Baseline/mimic 各采集 `64,000/64,000` 原生子步与 `1,280` policy rows，八个动作初态匹配；mimic 八动作残差门槛均通过，最坏 native qd p95/max 为 `.037625/.129288 rad/s`，最大 policy q 差 `.000519 rad`，effort 峰值 `1.29885 Nm`。低速 pivot yaw rate 从 `+.0704/−.0725` 变为 `+.1813/−.1641 rad/s`；高速 pivot 从 `−.3795/+.3854` 变为 `−.6139/+.6422 rad/s`。末段制动最大 `|v|=.0007675 m/s`、`|yaw rate|=.003965 rad/s`；直行 arc 制动反向速度积分约 `.0615/.0603 m`，此回摆保留记录。结果支持继续验收同侧联动候选，但仍不是完整连续履带模型或生产修复证明。Baseline 与 mimic 的摘要/raw、各自 `.exit.json` 在 [`staged_response/same_side_mimic_motion_1ms_v1/runs/`](staged_response/same_side_mimic_motion_1ms_v1/runs/)；之前的 2 ms 配对 CPU 比较保持在 [`staged_response/same_side_mimic_motion_v1/paired_comparison.json`](staged_response/same_side_mimic_motion_v1/paired_comparison.json)。

**C 的 runner/fixtures CPU 检查已 55 passed**（见 [`cpu_v7_migration/pytest_final.log`](cpu_v7_migration/pytest_final.log) 与 [`pytest.exit.json`](cpu_v7_migration/pytest.exit.json)）。本目录全量 CPU attempt1 另为 244 passed、14 skipped、8 warnings，compileall 和 39 个 Bash 脚本语法检查均 exit 0；源码哈希核验见 [`cpu_validation/attempt1/validation_hash_manifest.json`](cpu_validation/attempt1/validation_hash_manifest.json)，实际 WSL 退出记录见 [`cpu_validation/attempt1/wsl_process_exit.json`](cpu_validation/attempt1/wsl_process_exit.json)。

正式完整native回归已完成：14 passed/55 warnings，LF runner的pytest、bash和WSL均真实exit0，见 [最终日志](production_v7_validation/v7_native_full_suite_attempt1.log) 和 [退出记录](production_v7_validation/v7_native_full_suite_attempt1.exit.json)。新增用例验证两world无串联、原力矩预算、另一world局部reset前后状态精确不变，以及按官方reset传感器初始化流程后正常env.step。root已核对42源并完成第十三批整理。原始采样、真实退出、测试失败尝试和旧协议均保留。

独立新初始化实验 `results/isaac_field_improvement/fresh_coupled_tracks_20261020/` 于2026-10-20 12:35:58（北京时间）完成196608步、29个检查点，真实训练exit0。final `stage_04_turn90.zip` SHA-256 为 `6b4c85884a51c2e23f557eb98bc04ea63905fc8f20d4379715fb9fe72b75cdc5`；协议SHA `2735ca85917643e49e27088957d00b1576f37902ce9664203253ae698181d5b6`、正式0步baseline SHA `beda8fde92236b558433a148245094caeefb88d7c23b9ea73ed94c4cd596f1d5`，42个训练源/快照匹配，final CPU检查通过。Runner的 `comparison_summary.json` 确认free与field配对均valid、各4个同初态种子且无配对不匹配：free成功0/4→1/4，平均末距1.685283142→1.523644775 m；field成功0/4→1/4，平均末距1.835561316→1.801870071 m；两组均0损伤、0碰撞，但各3/4仍超时。故结果只支持当前候选在小样本已观察开发集上的有限改进，不是盲测，不证明完整田间作业能力，也不识别同侧联动的单独因果贡献；830xx未观察、未评估。GUI已载入final 196608步模型并从仿真0运行至6.5秒后暂停，drive/observation/action合同一致，60卡片无重叠、viewBox稳定、console errors为 `[]`；最终raw激活CPU检查真实exit0：60/60采集数组形状正确且有限，50变化/10不变（空检测支路和固定动作std），77/77参数相同；原值及截图见 `results/isaac_field_viewer/coupled_tracks_20261020/runtime_final_196608/`。历史采样、真实退出及原始评估报告保留在trial目录。
