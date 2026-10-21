# Coupled PI-v7 查看器部署

当前最终部署已完成：GUI载入 `fresh_coupled_tracks_20261020/stage_04_turn90.zip`，显示196608步，drive/observation/action及损伤合同一致。重置到0秒后实际运行至6.50秒并暂停；seq2/input0→seq132/input6.45/output6.50，车辆位姿实际变化。前后60张卡片0重叠、viewBox不变、两个页面console warn/error均为空。

最终原值CPU核验真实exit0：60/60采集数组有限且形状正确，50变化/10不变（空检测支路及固定动作std），77/77参数相同。导出的四份JSON/NPZ副本与原文件SHA一致，原API帧未被覆盖。详见 [最终模型运行核验](runtime_final_196608/README.md)，截图为 `runtime_final_196608/network_final_196608.png` 和 `runtime_final_196608/vehicle_final_196608.png`。训练和同物理开发配对已真实exit0；free/field各4种子成功均0/4→1/4，损伤/碰撞均0，仍各3/4超时，仅小样本开发改进。

以下为首次部署及8192步验收历史，保留作复现依据。

2026-10-20核对旧viewer PID198426的命令后，仅停止该进程；用新训练正式0步baseline启动v7后端，再通过GUI载入首8192步checkpoint。新进程部署时WSL PID298144/exec88604，端口8769车辆、8770二维网络均HTTP200；这些PID为历史记录，后续检查须核实实际进程。

启动命令见 `run_viewer_baseline.sh`，日志为 `viewer_baseline.log`。启动argv仍指baseline，当前GUI已载入8192步模型，不能仅凭argv判断当前权重。新训练自身继续运行，查看器在17.35秒暂停。

实际浏览器验收：drive/observation/action合同一致，车辆位置和朝向变化；seq2/0秒到seq349/输入17.30秒/步后17.35秒；60张卡片0重叠、viewBox固定、console errors为空。运行前后原值比较：60/60采集层有限且形状正确，50变化，空检测分支及固定动作std等10项不变；77/77参数完全相同。

详细raw、CPU退出及说明见 [runtime_8192](runtime_8192/README.md)，浏览器记录为 `runtime_8192/browser_acceptance.json`，截图为 `runtime_8192/vehicle_8192.png` 和 `runtime_8192/network_8192.png`。本次为UI开发回放，不是正式导航成绩。最终训练完成及配对评估仍待后续自动跟进。
