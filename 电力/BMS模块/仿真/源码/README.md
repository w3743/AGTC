# BMS 仿真源码

本目录包含用于纸面计算、参数筛查或只读源审计的Python脚本。脚本结果和输入分别见同级`结果/`、`输入/`；所有模型均须按脚本说明理解，不能替代固件、PCB、台架或整车验证。

## 直接文件

| 文件 | 用途 |
|---|---|
| [行进充电能量筛查.py](行进充电能量筛查.py) | 对母线电容和行进充电能量预算进行一阶筛查；脚本说明明确其不是电路验证。 |
| [verify_charge_limit_16a.py](verify_charge_limit_16a.py) | 对9S4P净充电上限进行可复算参数筛查；不是固件或整车验证。 |
| [replay_16a_with_static_p1.py](replay_16a_with_static_p1.py) | 将16A参考模型与已保存P1进行只读重放，不导出ENET。 |
| [verify_hlk_implementation.py](verify_hlk_implementation.py) | 重算Hi-Link供电候选的简化算术/RC检查；不模拟真实开关变换器。 |
| [verify_precharge_layout.py](verify_precharge_layout.py) | 重现BMS预充Layout闭合相关的纸面计算。 |
| [verify_protection_reduction.py](verify_protection_reduction.py) | 绑定当前BMS原生P1进行只读保护精简审计；不验证BQ寄存器写入、固件、物理保护时序或整车放行。 |
