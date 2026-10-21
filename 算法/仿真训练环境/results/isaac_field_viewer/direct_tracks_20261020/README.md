# 左右履带独立控制与纯神经网络图 · 2026-10-20

四维归一化动作依次为左履带速度、右履带速度、云台偏航角速度、云台俯仰角速度，范围均为 `[-1,1]`。前两维各自乘最大履带速度，逐侧加速度、速度伺服增益和力矩上限继续生效；云台执行逻辑保持。移除原来的前进/差速混合及两侧共同缩放。

动作合同为 `direct-track-speed-v1`，驱动接口为 `independent_left_right_track_speed_v3`，观测为 `sensor-depth-direct-tracks-v4`。状态仍43维，其中上一动作字段13–16同步新语义。旧四维checkpoint在SB3加载前因动作合同不匹配被拒绝，不改写旧metadata或转换旧权重。编码器、融合160维、LSTM128和训练课程保持。

用户确认“只显示神经网络中的”：二维图只保留输入、编码、融合、LSTM、策略/价值网络及四维动作。删除 `actuator.tracks`、`actuator.wheels`、`actuator.gimbal` 和 `motion.actual` 图节点、导出张量及其色标统计。实际速度仍在车辆面板，物理后端仍执行必要机械换算。保留统一全光谱自适应色标。

## 验证

- CPU最终192 passed、10 skipped、1既有NumPy极端float32标准差溢出warning，exit0。首次完整CPU因旧foundation硬编码19/20源文件数失败，现按实际完整清单验证，失败日志保留，最终回归通过。旧协议和实验记录没有改写。
- Newton四项原生检查通过：官方SB3 wrapper精确深度值、同速前进/制动、左右反向原地转向、左右独立/限幅/加速度/14轮映射及云台隔离。首次前三项通过，第四项因float32 v/r与float64期望约1e-7相对误差失败；按float32精度修正轮速断言并仅重跑失败项通过。两个日志均保留。
- compileall、两个真实bash脚本、HTML内嵌JavaScript语法通过。实际PolicyInspector图57节点，全部parents存在；snapshot和导出层集合一致，导出JSON带机器可读action_config。详见graph_audit.json与ui_validation.log。
- 8769车辆页和8770二维页实际浏览器验收：四个动作字段/手动滑块正确，drive/observation/action/damage均一致。整体11卡片，图终点为左右履带＋云台。运行到7.40秒后暂停：帧149、动作前7.35秒、步后7.40秒；viewBox保持`-45 -31.139 1052 728.279`，errors为空。原始stats有8/12个采样节点改变；不把PNG颜色变化或所有元素都称为激活变化。暂停重复GET的序列/时间/色标均稳定。详见runtime_acceptance.json、layout_before/after.json和截图network_neural_only_verified.png。

## 预览与重新训练

独立UI零步预览由真实make_env、官方wrapper和make_model生成，没有learn或step；ZIP为untrained_direct_tracks_preview.zip，SHA-256 `7d39b3bf0482fd9bdbb11bc527467a93548757cfbecffa4d39042d78c5a1ae9c`。它与训练正式零步基线分开保存；页面明确标记未训练。run_viewer.sh为本轮命令，已保存启动场景和进程日志。

新trial `results/isaac_field_improvement/fresh_direct_tracks_20261020/` 已prepare并launch。协议SHA-256为 `85b9b4dd44c9b9292078d77eb1fb3f56d06725007af2679011dd97efecce4ab0`，28个源快照；从零随机初始化、4环境，near32768、steer3032768、steer6065536、turn9065536，合计196608步，随后做free/field各4组配对开发评估。不观察830xx，不做FPS/训练速度基准，不把启动或UI显示正确称为能力提高。真实退出状态由run_training.sh在进程退出后写process_exit.json；run_status.json和检查点显示进度。

启动时WSL viewer PID564/exec64049，训练PID1359/exec27447；这些是启动记录，后续须查实际进程。训练源已冻结，训练过程中不修改本轮28个源文件。

前代v2协议和最终196608步ZIP只读SHA仍为 `f32707b8fdfa8098613d766b6ce38d0edbd74bf396c2c6abea432f698b9d9e84` / `fbb2e39f741bfa812b192c9b9320f711bb82187e46dc64fc5461b17a6296cc77`；旧记录不覆盖，新成绩不与旧动作语义混算。

## 后续完成状态

上述launch与零步预览为早期记录。新trial已在2026-10-20 01:11:13北京时间真实退出0，完成196608步及free/field各4组配对开发评估。正式0步基线SHA为 `896c7776ee9ee719941cdb8ee85c665e94ffbe2ac2d373e43ccba88066ed5335`，最终模型SHA为 `342237b1b2cacfb08135535891f9aaf1d4eb5ef957a68d85168bfe2f43e21329`。free clean成功0/4→1/4，损伤/碰撞未增加；field仍0/4→0/4、平均末距1.797→2.811m，不能宣称总体导航能力提升。

正式viewer随后已载入final模型，并按用户要求去掉整体视图、保留57层完整二维展开，布局/实时验收见 [完整展开记录](../expanded_layout_20261020/README.md)。原28源快照保留且匹配协议，正式布局采用后仅HTML为清单记录的预期差异、其余27项一致。初次completion_validation与部署补充addendum同时保留，不把Copy-Item继承的LastWriteTime当作部署时间。IMU和电机负载反馈尚未接入，须先按用户要求报告计划并获得执行确认。
