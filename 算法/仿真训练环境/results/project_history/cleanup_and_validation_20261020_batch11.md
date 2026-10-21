# 第十一批整理与验证记录

日期：2026-10-20。整理前本批计数为4/5，本轮全局自适应色标为第5项。整理只覆盖本批变更和紧邻的训练结果记录。

## 本批五项

1. 深度编码器保留16×8×6空间张量并投影到64维。
2. LiDAR改为环形Conv1d两层2→16→32，池化到50并投影32维；融合维度160，驾驶LSTM维度128。
3. 增加独立新初始化训练入口、冻结协议与源码快照、零步基线、检查点和开发种子配对评估。
4. 修复单通道深度与Isaac Lab官方Sb3VecEnvWrapper的兼容：Box声明[-1,1]，实际深度仍为[0,1]且不重映射，无效值为0。
5. 二维模型页采用对称全局自适应色标：全部有限激活与已绑定参数的最大绝对值决定完整值域；超出立即扩展，新帧回落最多5%，重复同帧读取不收缩，reset/bind清历史，拒绝数值手动scale请求，Turbo保持不变。

## 验证与行为证据

功能回归日志记录176 passed、9 skipped、1 warning；隔离staging回归12 passed、1 warning。warning来自极端float32数值测试触发既有numpy标准差平方溢出；范围JSON仍是安全值。整理静态检查包括compileall、两个bash脚本语法检查和Windows Node解析live HTML内嵌JS，结果见 [静态验证日志](../isaac_field_viewer/adaptive_global_color_20261020/static_validation.log)。

真实浏览器验收使用196,608步最终checkpoint。初始/参数显示范围相同为±2.146289825439453；仿真14.75秒时扩大到±17.186372756958008，19.90秒暂停帧为±21.128650665283203，重复GET帧399范围稳定；reset后时间和范围都回到初始。相同的0值跨state与fusion图像RGB均为[164,252,60]。errors为空，overview为12卡片、0重叠、0滑块，最终停在fusion160。完整JSON及PNG保存在 adaptive_global_color_20261020 证据目录。

## 训练结果与范围

fresh v2真实完成196,608步，exit 0；协议SHA-256为 f32707b8fdfa8098613d766b6ce38d0edbd74bf396c2c6abea432f698b9d9e84，final stage_04_turn90.zip SHA-256为 fbb2e39f741bfa812b192c9b9320f711bb82187e46dc64fc5461b17a6296cc77。有效配对开发比较free与field各4个seed，二者均为no_improvement。没有据此声称提升或盲测结果；830xx没有观察。

## 清理审计

本轮整理仅同步使用说明、ISAAC_LAB及历史批次记录；未改生产代码或测试。代码搜索未发现旧手动色标控件或第二套自适应范围实现；manual_scale仅是拒绝数值scale请求的负向测试变量。保留staging、manifest、训练日志、模型和浏览器验收证据。

协议冻结列表的27个快照SHA全部匹配；当前25个文件与快照一致，只有viewer专用network_2d.py及network_2d.html具有部署manifest所列预期差异，且在训练exit 0后部署。最终checkpoint SHA与manifest一致。工作目录没有.git元数据，无法获取git status。此次没有重跑CPU全量/GPU、启动新训练或开展FPS基准。