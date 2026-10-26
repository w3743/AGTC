# v3正式试验失败证据

trial为`native_v1_1_fixed_physics_v3_20261025_184052_20261025_184704`，源码提交c94553e2；64×32、1ms/50、seed0、LSTM512、fast_stop_v2、五路real。第58轮rollout索引21（第22个动作）发生env11物理非有限状态，严格门exit3、formal_eligible=false，仅57轮完整。666.72s为完整wrapper墙钟，不将其失败前速度当作正式成绩。

`copy_manifest.json`记录166份复制文件，原`SHA256SUMS.txt`的165项逐项校验通过；这个README在归档后新增，因此不在原清单内。`source_snapshot/`、run_snapshot及checkpoint40均是原字节。checkpoint40策略/优化器finite不代表物理通过；失败时latest.pt另留原run，不把它伪装为完成58轮。

**命令记录勘误：** `actual_cli_args.txt`只是原runner的`cli_args.txt`副本；其名称不代表逐进程argv捕获。旧runner把`-u`错记在`train_wsl.py`后，真正执行是`python -u train_wsl.py`。原捕获文件不改写；该记录错误已在后续19b49ec1修正。完整wrapper/训练输出及launch/monitor元数据分别保留。

`failure.json`内`step=645`取各环境episode计数的最大值，不是全局policy step或env11自身年龄。用于定位的确定记录是iter58/rollout索引21/env11；0-based全局policy调用为1845。原始failure含JSON NaN，是失败证据，不能当作有效有限数据。

该run没有保存失败前模拟状态或RNG；checkpoint不能精确重放NaN。接下来的外部诊断将单独保存prestate/action，不覆盖本目录。
