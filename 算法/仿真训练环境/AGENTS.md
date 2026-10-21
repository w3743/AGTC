# 项目协作规则

## 复用与协作

- 完成任务前先查找项目已有实现及现成、开源解决方案，以最小复杂度完成任务；优先复用训练库、仿真器和标准接口，避免另写同类基础设施。
- 查找、机械性修改、计算、测试、验证等低推理需求任务，尽可能交给子代理，使用低额度模型 `gpt-6-luna`、`max` 推理档。新建此类子代理使用 `fork_turns="none"`，提供明确范围和必要上下文。
- 勤于向用户提问，优先确认会影响目标、架构或验收的关键问题；已明确的要求直接推进。

## 每五项改动整理一次

- 每完成 **5项独立的功能改动或问题修复**，必须先对项目进行一次整理，再开始下一项改动。
- 同一功能涉及的实现、测试与文档算一项；不得按文件或代码行重复计数，也不得把无关功能合并为一项来绕过整理。
- 整理包括：清理确认无用或重复的实现，统一受影响的接口、命名与目录组织，同步说明文档和运行命令，并执行必要的测试、静态检查与验证。
- 性能相关结论保持相同测量口径；原始实验记录与需要的模型保留，整理不得覆盖或改写实验成绩。
- 整理围绕已修改范围开展，不扩大成无关重构。
- 主代理统一记录改动计数和最近整理结果；子代理报告独立改动，避免并行时重复记数。更新计数记录本身不另算一项。

## 当前批次与运行状态


- 当前课程trial为 `results/isaac_field_improvement/fast_stop512_mastery_20261021_attempt2`；动态进展集中在 `ISAAC_LAB.md` TRAIN-01与trial原始run_status/decisions/probes，不在本文件滚动复制。运行中不得改47项冻结训练源、protocol、模型或成绩。两自动跟进项保持PAUSED；实际预览与后继采样均须等课程自然完成、真实exit 0且无并行GPU仿真后串行启动，不能用定时任务或替代轮询绕过暂停。
- 第十七批5/5项功能已完成并整理：①LASSC生产消费桥与模型回放，②快停512可追溯续训，③mastery独立fast_stop_512课程，④配对过程量评估，⑤暂停策略地图点选预览。全套WSL CPU pytest在显示修复前真实exit 0（428 passed、22 skipped、8 warnings）；MDX显示修复后预览专项18 passed，py_compile与--check均exit 0。课程冻结47源与protocol保持一致；此前报告、模型、试验成绩未改。CPU-only Viser 1.0.30浏览器交互已通过（证据截图`results/project_history/fast_stop_objective_20261020/preview_map_cpu_verified.png`），但这不等于原生策略物理验收；训练自然完成且真实exit 0前不启动实际策略预览。第十八批计数1/5：新增非冻结的课程全场过程采样入口，复用既有轨迹adapter；验收与运行边界统一见ISAAC_LAB.md TRAIN-01。训练及只读审计不另计功能。
- 2026-10-21恢复前只读复核（其后当前状态见本节首条）：`wsl -l -v` 显示Ubuntu-24.04为Stopped，未启动发行版，因此下条2026-10-20 SIGSTOP/Tl+仅为历史现场，不能承诺今日仍保留原进程、RAM或显存，也不得对旧PID直接SIGCONT。57344步完整磁盘checkpoint实算SHA仍为 `db473f5e78038c066226f61cb6a5ee02ea71615823f28e519366cb0809ecbb8a`，ZIP含Adam、与manifest一致，45/45 live/snapshot源码均匹配原protocol。最后完整诊断65024步，超出磁盘7680步不构成可恢复检查点。没有final正常退出证据、未恢复训练或运行GPU；原automation保持PAUSED。用户再次确认0.4m位置范围及严格停稳。TRAIN-01离线差异/对照/资源/验收与下一依赖统一写入现有 `ISAAC_LAB.md`，不另建方案体系，不改冻结训练文件；该早期核查时点计数为第十七批0/5。
- 2026-10-20历史暂停记录（适用于当时trial及PID）：用户当时明确要求“暂停训练”。2026-10-20 18:15:36北京时间，核实本轮fast-stop Python336098实际argv后已发SIGSTOP，实际ps确认状态 `Tl+`（进程暂停、没有退出），launcher336095仍等待。训练现场保留内存，显存仍分配；未终止进程、未另起训练或评估、不自动恢复。最近完整磁盘checkpoint57344步、SHA `db473f5e78038c066226f61cb6a5ee02ea71615823f28e519366cb0809ecbb8a` 与manifest/实算一致，ZIP完整且含Adam。原run_status仍running不感知OS暂停，以 `results/project_history/fast_stop_objective_20261020/pause_20261020/pause_record.json` / pause_audit.json及实际ps为准。用户未来明确“继续/恢复”才可核实际PID/argv/T状态和45源SHA后发SIGCONT恢复同进程；不要重新launch或覆盖协议。两个独立viewer未操作。原生automation仍PAUSED，功能计数仍0/5。
- fast-stop首8192检查点于18:04:03生成；CPU-only独立审计真实exit0，15检查全true，含模型/rollout buffer gamma1、严格停车及稳定奖励系数0、actor/critic512×1、3065801独立参数、45源/协议/模型SHA稳定、权重及Adam有限。37项Adam step均300，65个state张量/3021076元素相对正式0步更新（state含共享别名，不能等同独立参数计数）；plants0检测支路不称已学习。baseline SHA `3366b22d4b2bb76f2688dcd3dd43b2b3447996cf52f9e23e5713523e7d2190d0`，8192 SHA `49280e36e068049ba7e811355311c0b228f718878ac330ec8fd8508f856ae933`。证据 `results/project_history/fast_stop_objective_20261020/first_checkpoint_cpu.json` 及log/exit_code。该8192步记录时仍在训练；滚动训练0/4不是开发评估，未宣称效果提升。
- fast-stop新512实验已于2026-10-20 18:01:53北京时间正式启动：WSL Ubuntu-24.04 Python336098、launcher336095、exec37814仅作启动记录；后续核真实argv/日志/检查点/退出。新trial/protocol SHA见下条，45源冻结；正式0步baseline已保存，actual actor/critic512×1、policy独立参数3065801，新权重与Adam。gamma1/稳定权重0/严格停车合同已在协议冻结；诊断JSONL已出现真实PPO更新、reward逐步对账matched、稳定惩罚为0；截至该历史记录时仍在训练，不能称最终完成或能力提升。用户关闭定时任务的偏好保持，automation仍PAUSED。
- fast_stop_v1第5项已完成并整理：CPU377 passed/20 skipped/8 warnings、compileall/Bash/源码稳定性全部真实0；原生Newton20 passed/61 warnings真实exit0，含真实XY/full角速度、局部reset与官方SB3 timeout无bootstrap/终态快照。清除重复终态缓存，复用既有diagnostic_step_infos/episode_results；奖励常量单一来源，旧profile兼容。12历史模型/协议和45旧源码快照不变，live仅4预期源变化。详见 `results/project_history/cleanup_and_validation_20261020_batch16.md`。第十六批5/5完成并已整理；该快照时第十七批为0/5。新实验prepare/preflight真实0，独立目录 `results/isaac_field_improvement/random_global_rough_fast_stop_lstm512_20261020`，45源冻结，protocol SHA `3ca58fcc05b9d7ac34798f836704c91b5ec608d62f7424c3897ed291451a620c`；当时准备正式启动，不混旧512成绩。原生automation保持暂停。
- 用户确认继续按“最短时间到点停稳”调整。旧512 trial `random_global_rough_lstm512_20261020` 已于17:42:43经核argv后停止；SIGINT未停后台子进程，随后SIGTERM，父shell真实Python exit143，原Python335230/launcher335227实际已不存在。最后完整40960 checkpoint SHA `67bcb7b73dd5f22c1b01e33a1a0e86cf8f8ed82c25587088ceffa907c1df33d5` CPU-only 13项核验真实exit0，原protocol、baseline及所有checkpoint保留；raw run_status仍running是被终止后的陈旧值，以真实退出及 `results/project_history/fast_stop_objective_20261020/previous_stop_outcome.json` 为准。该历史时点正在实现独立fast_stop_v1新奖励/严格停车配置，尚未启动新训练；本项完成后为本批第5项，需先完成第十六批整理再开始后续功能，当前已完成计数仍4/5。定时任务仍暂停。
- 512实验首8192步检查点已于17:36:10生成；CPU-only独立核验实际exit0、13检查全true，45源live/snapshot和协议稳定，baseline SHA `9fbad2d40f53aec3ab4763f20890abd49578d6b62c3d9bf01d96b697e3af8675`，8192 SHA `c363f75703adbd740dc2b43a1c81065a889ad3cb8cae601c6e716964e02ae817`。实际3065801参数、actor/critic512×1，37项Adam step均289且有限；较0baseline有65个state张量/3028603元素改变（含alias，非独立参数计数），检测分支plants0不称已学会。见 `results/project_history/lstm512_experiment_20261020/first_checkpoint_cpu.json`。本轮仍运行；8192滚动日志0/5不是开发评测，未声称效果改善。原5份历史协议/模型hash核验未变。
- 2026-10-20 17:33:58北京时间，按用户“先跑512维，可以开始实验”正式启动独立 `results/isaac_field_improvement/random_global_rough_lstm512_20261020/`。WSL Ubuntu-24.04 Python335230、launcher见trial launcher.pid、exec50333仅为启动记录，后续必须核实际argv/进程/日志。45源冻结，protocol SHA `010486bd7ceeb2ed121a1a1b765b6901e7732208496a9c699cc89da53fdad954`；actor/critic各单层LSTM512、融合192，实际policy独立参数3065801，正式0步baseline已保存，initialization.json核实新初始化、没有旧权重迁移。沿用random_rough/4env/69009/196608/60s/植物0岩石0/同侧联动前导向轮/稳定奖励，原2Nm/轮与指令斜坡不变，未将硬件审计候选值写成实车边界。尚无学习提升结论，不构成512与128因果对照。用户后续询问奖励函数，未在运行中更改协议/奖励；automation保持PAUSED。
- 本项增加prepare专用 `--lstm-hidden-size`、协议/真实模型维度校验、初始化参数记录及可选宽度透传，计数为第十六批第4项，当前4/5。CPU首次2新增测试缺sys导入而失败（357 passed），修正导入后359 passed/18 skipped/8 warnings、compileall/bash语法实际0；首次失败证据保留。prepare/preflight实际0，源冻结后不再编辑45训练源。记录入口 `results/project_history/lstm512_experiment_20261020/`。
- 用户最新保持摩擦系数0.8，并要求查真实FOC驱动文件判断电驱能力边界。本轮为只读审计，结果入口 `results/hardware_limits/foc_20261020/README.md`；未启动新的rough训练（仍prepared/0步），未改硬件、45冻结源或协议。当前FOC软件仅主机codec/会话/反馈测试，未发现可构建G431完整FOC固件；单板1.8kW输入/1秒与±100A/5000A/s均为选型或模型候选，不是已资格连续能力，卖家80A口径不明。当前BOM主桥为IPT015N10N5、0.5mΩ/5W分流器，不能套旧IRLB/STL主桥筛查。现有仿真仅目标限速15km/h、指令斜坡1.5m/s²、14轮各2Nm，无总电输入功率/母线电流/热/回馈约束；按Kt=.035/效率1每侧2Nm×7轮等效电机轴2.333Nm、Iq66.67A，属于未标定换算。若后续据硬件审计改电驱源，必须另建trial，不能覆盖已准备协议。审计/计算不计新增功能；该只读审计时点为3/5，最新计数与启动状态以上文512实验为准。
- 最新计数：第十七批5/5已完成并整理，第十八批1/5（课程全场过程采样入口）。第十七批五项及验收见本节上方与ISAAC_LAB.md TRAIN-01；历史训练日志和成绩保留。
- 用户最新要求：导向轮转速与同侧负重轮绑定，完成后捋顺逻辑、整理文件、适当归档。此前已允许与同目录聊天“评估并优化项目训练方法”协调；该聊天已交出env.py/newton_drive.py/learning.py写入，方法功能保留，当前不再改共享Python或启GPU。
- 当前启用FrontGuideSpec默认rotation_coupling=same_side_track_speed。每world14条Newton公开mimic/equality（12负重轮+2guide），guide按R_main/R_guide跟随本侧wheel0，默认45mm/45mm=1:1；无新增actuator/PID/扭矩，原14驱动轮每轮2Nm保持。左右guide各.25kg从车体扣除，总30kg；x=.42m、比支撑轴抬高.05m、宽.06m。None模式旧资产/hash/12条关系保持，free仅诊断。新几何仍是离散轮代理，无连续履带斜面/张紧机构，不推定真实翻越高度。
- 接口由FrontGuideSpec统一资产→cfg普通dict→环境恢复只读规格→Newton world hook→drive metadata→fresh合同。注册hook仅构建期且按joint_world隔离。实际Newton/MuJoCo joint/DOF需用官方映射，不假设全局索引或裸关节名相同。host eq_type=JOINT时eq_objtype=0仍正常；多world的solver模板neq不是全局关系条数。
- 当前native attempt8专项3 passed/39 warnings、attempt9原环境14 passed/55 warnings，实际exit均0；位置残差最坏4.96e-5rad、速度残差p95 .01286/max .01617rad/s，通过原门槛。粗糙reset、30kg、actuator0、力矩限幅、多环境隔离、真实body角速度和部分reset统计均已验。记录 results/isaac_field_diagnosis/front_guide_coupled_20261020/README.md，早期失败attempt保留。
- BodyStabilityConfig仅奖励/统计用真值，不加策略观测：每策略步dt×[2×超35°总倾角²+.02×bodyωX/Y平方]，35°为试验软阈值，原1.2rad硬终止不变。root world角速度旋转到body；不能拿云台gyro代替。None不加惩罚，奖励分项和总reward对账；记录策略步峰值/RMS、超阈时长与分开终止原因，非物理子步极值。
- 当前random_rough：100×100m、plants0/rocks0、地形幅值预算.35m/波长4–10m、60s、4env/seed69009、新随机初始化196608步。目标/起点/朝向全场独立采样，不限附近；每run固定一份地形，不是每回合重生成。rough_a/b用地形80003/80004和开发seed87001–4/82001–4，必须对同实际USD路径/SHA/bytes、真实地形seed、同初态做0baseline/final配对，姿态summary缺失/非有限拒绝。新output results/isaac_field_improvement/random_global_rough_coupled_guides_20261020，prepare/preflight/外层均实际0，45源live/snapshot匹配，protocol SHA 5de27ea7b7e1714b8822ec8fe11f0b53e6f08d194577b253d376913a00405980，状态prepared/0步且未创建baseline或正式训练。
- 当前完整CPU整理349 passed/18 skipped/8 warnings、compileall/3项目bash-n以及7项额外静态检查均0，80源码/测试SHA前后完全相同。首次outer文件名误写导致exit1，inner已0；第二次outer/inner均真实0，原失败保留，见 results/project_history/front_guide_coupled_cleanup_20261020/read_only_audit_attempt2.json。不改测试门槛、不因wrapper问题覆盖第一次证据。
- 用户授权归档已执行：6个自由轮阶段旧目录、130文件/2554465字节原样移动到 归档/front_guide_free_rotation_20261020，逐文件SHA/bytes全部一致，实际exit0；原路径映射见该目录archive_manifest_before.json。旧自由轮台阶未跨越、旧rough prepare成功/preflight失败和未启动旧viewer包装器都保留。正式历史模型/协议/源码快照/成绩及资产缓存、共享运行时未移动或改写。
- 台阶pair_attempt3 free/coupled/analyzer/launcher均真实0，同源码/同地形SHA/同初态固定动作200步有限；guide-step快照187/200、guide输出0、驱动effort最大.9334Nm。60mm台阶没有越过，两模式root最大x约2.2538/2.2535<前缘2.6m，不称翻越提升；最坏guide速度残差.0921rad/s。attempt1/2探针映射/eq_objtype审计失败原样保留。新8781已GUI检查轮组，手动.03/.03运行至5.45s后暂停归零/reset0，console0 error/1上游warn，无checkpoint；metadata14关系/guide2、60s/粗糙plants0/rocks0一致。证据 results/isaac_field_viewer/front_guides_coupled_rough_20261020/。旧8769/8770未替换，不当作新几何或393216策略。
- 前一轮60s续训实际完成393216步、29checkpoint、2026-10-20 14:44:06北京时间exit0；protocol SHA fe54ce6233e10705a9bbc393784eeca8cbe4d459ac467af1b2fd3656fb4baa18，final SHA 9777af6addd30427026675c0fb9d577c555fd288294bc6129a80b53b1d633a70。CPU全12检查通过，37项Adam step14806、65张量更新、12检测张量plants0未改。重评60s free成功1/4→1/4、field1/4→0/4，零损伤/碰撞但no_improvement，不能称已会可靠导航。父fresh196608 final SHA 6b4c85884a51c2e23f557eb98bc04ea63905fc8f20d4379715fb9fe72b75cdc5，旧20s报告不混算。
- 独立训练方法入口train_isaac_mastery.py保留奖励/PPO诊断、连续达标课程、真实旧任务on-policy采样、安全选模与显式GAE；三份method_v2协议按旧47源准备，当前源已变，必须新目录重prepare，不能覆盖旧protocol或先称可直接启动。未新增特权critic/势函数塑形，未证明GAE唯一根因。
- 用户明确关闭的是原周期训练automation，保持PAUSED且不恢复；本轮一次性预览heartbeat当前也PAUSED（详见本节首条），不把原任务恢复或扩大为常驻训练监控。每次进程核验真实argv/日志/检查点/退出，旧PID只是历史，句柄失效不据此重启。60s改变episode_time_ratio，纯延时对照需另固定原时间输入。
- 本轮完整整理入口 results/project_history/front_guide_coupled_cleanup_20261020/README.md；整理前AGENTS/ISAAC_LAB/使用说明原字节和SHA在同目录docs_before/manifest.json。旧运行期状态、历史PID和逐批失败细节以该备份及各原trial为准，不再在当前状态堆叠互相矛盾的“当前/运行中”。

## 任务边界与验收

- 后端仅发展Isaac Lab + Newton MJWarp，复用本地external和SB3-Contrib；模型自己学，连续四维动作，无Nav2或手写规划器替代策略。
- 不做训练速度/FPS基准。必要的物理正确性、学习效果、PC负载和UI核验仍做；逻辑20Hz配置不冒充实测帧率。
- 优先不压苗，允许更慢或超时。历史/通用默认到点为0.4m位置范围、低速hold0.3s；用户最新fast_stop_v1实验为位置≤0.4m、真实水平XY速度<0.05m/s、完整车体角速度模长<0.1rad/s持续0.5s，优先成功并尽量缩短用时，不要求终态朝向。不同成功判据的报告不得混算。
- 安全验收必须同配置/同初态配对：clean成功增加，且损伤episode数、平均损伤、碰撞不恶化；不能把成功/损伤权衡称能力提升。旧权重可在当前物理重评，但不同规则/几何/驱动/观测下的旧报告不混算，不重贴metadata。
- 810xx/820xx/870xx是已观察开发数据，830xx保留未观察，只有开发安全门槛通过后才做独立验证。滚动训练窗口不是held-out；CPU固定观测反事实不是车辆轨迹。
- 完整植物搜索、粗糙地形/石块、对抗训练及整田可靠导航未达成，不以局部通过或训练跑通标记总体完成。当前基础课程plants0，不称已经学会避苗。
- 每次训练保留协议、源码哈希/快照、配置、种子、检查点、依赖/资产及真实退出。观察超时不是进程结束；句柄失效须核实际进程/端口，不能只看状态文件猜测或重启。
- 旧后端69项批量删除曾被自动审批拒绝，未执行或绕过。用户后续授权移动，已原样移至 `归档/legacy_before_isaac_20261017/`；不能把历史拒绝视为删除授权。共享运行时/upstream/Isaac模型和记录保留；原路径按archive_manifest.json与SHA查阅。

## 实车事实与代理限制

- 30kg；车身0.8×0.3m，300mm不含履带；整车最外宽470mm；每侧履带宽60mm，中心距410mm，接地595mm，最高15km/h。
- 两台BDUAV N5065 270KV，4:1减速，主动轮直径60mm；硬地和农田均能原地转向。转一圈用时暂缺，不反复索取、不称已标定。
- 轻量代理是盒体、七对圆柱轮和双轴云台，不导入高精CAD。45mm代理轮半径不同于30mm主动轮；P.04、Ki.1、2Nm/轮、Kt.035和效率1均未标定。14个轮执行器+同侧约束仍不是两台实车电机的完整模型；不额外施加车体速度/转向伺服。
- 用户已批准IMU/驱动反馈/协议及硬件修改，并授权自主推进；此前“需批准/尚未实施”属历史，不再阻塞。ATK-IMU901为10轴、随云台偏航和俯仰一起动；云台两轴实测角度/速率、左右驱动电流/机械RPM均可回读，驱动器自研文件在项目内。云台IMU不直接等于车体IMU，不用动作指令冒充实测速率。
- 已有Python telemetry解码/组装与FOC C主机96checks/0fail，但未完成实机串口/CAN/HAL联调；不能把仿真等效Iq/RPM称实测或接触力。硬件路径 `电力/FOC驱动模块/软件/`；厂家手册镜像 `docs/hardware/atk_imu901/ATK-IMU901_User_Guide_v1.2.pdf` 封面标V1.0，未核实与实物/最新固件一致。各通道单位、噪声、电流/安装外参仍需实机核实。
- 云台中心离地.4m、俯仰−25°..60°，雷达/相机一起俯仰。X2雷达360°/8m/10Hz/400束；策略与预览同88×160轴向Z深度。气压/海拔不入策略，磁场保留，无重复Euler/温度输入。
- 玉米约1m，行距.8m、株距.3m，允许健康/干叶混合/遮挡；UWB多基站覆盖。识别相机/单点测距与云台同动，须对准目标再测。
- 损伤默认stem-contact-v2：履带压茎基或车身撞主茎；叶片擦碰及旧9–15cm保护圈不算。履带595×60mm带和主茎垂直相交，车身当前Z/pitch/roll扩张AABB；水平扫掠是离散2.5D近似，不称精确接触/弯折。旧配置缺metric用root-protection-v1保持复现。
- 固定植物root未证实每株增加6个q/qd；有位姿/mocap/几何/BVH成本，未测不猜收益，不据此重写bodyless。

## UI与关键证据

- 用户选定bbycroft LLM Visualization风格，external/llm-viz MIT实际复用范围记在 `isaac_field/visualization/THIRD_PARTY_NOTICES.md`。固定深色，无主题切换；只保留完整二维网络展开，无整体视图/网络三维，车辆三维保留，连续曲线连线。
- 色标为整帧全部采集层及绑定参数最大绝对值±全范围、0同色，扩大立即/新frame收缩最多5%，暂停重复GET/选层不重定标；reset/bind清历史。完整值域无百分位截断。不能仅用PNG变色判断raw激活变化。
- reset先确定最终start/yaw/goal再完整校验，推进RNG重试；不验证后覆盖端点。证据 `validated_reset_repair/`。生产reset采用H/V矩形连通验证，0.1m网格可能错过狭窄锚点；Scene.radius导航保护区与stem_radius损伤语义不同，尚未修复，不断言策略无法驶入行间。
- Newton BVH包括VISIBLE与COLLIDE_SHAPES，视觉叶片也能测距；历史空命中因最近株31.54m。近株1.5m LiDAR/单点实测1.488780m，不再传播植物透明/射线只查碰撞的已纠正误判。
- 旧53248/safe118784/Gaussian-gSDE/GAE对照及转向偏置见文档归档和原results，未通过整体不压苗优先门槛。目标通道正确、梯度非零，未发现训练cache陈旧；不称GAE是唯一根因。
- WSL必须显式Ubuntu-24.04。localhost/NAT UTF16提示曾用/bin/true实证actual exit0，不把warning当失败；后台检查真实exit和进程。PowerShell会展开双引号里的$rc，状态采集优先保存bash文件再执行。不要擅自重启WSL/驱动/电脑。
- 详细命令与流程见使用说明.md，当前项目状态见ISAAC_LAB.md；更早历史在 `results/project_history/AGENTS_before_20261017_batch8_cleanup.md` 及本批文档归档。历史PID/源码计数/“当前”只属于相应时点，不覆盖最新记录。



