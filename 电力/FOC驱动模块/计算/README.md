# FOC 计算

本目录保存FOC功率级、保护路径和功率MOSFET的可复算纸面计算。脚本与相邻JSON/Markdown成对维护；其条件假设、范围和限制以对应报告为准，不表示器件完成定型或实物热资格。

## 直接文件

| 文件 | 用途 |
|---|---|
| [1秒脉冲能力核算.md](1秒脉冲能力核算.md) | 说明功率级1秒条件核算的输入、假设和限制。 |
| [calculate_foc_power.py](calculate_foc_power.py) | 执行第一轮FOC逆变器功率级计算，生成同目录[foc_power_results.json](foc_power_results.json)。 |
| [foc_power_results.json](foc_power_results.json) | 保存上述功率/电流、MOSFET候选、母线和分流器计算结果。 |
| [protection_path_screen.md](protection_path_screen.md) | 解释FOC保护路径纸面筛查的范围和结论边界。 |
| [protection_path_screen.py](protection_path_screen.py) | 重算保护裕量与失效路径筛查，并写出对应JSON；脚本不修改原理图或BOM。 |
| [protection_path_screen.json](protection_path_screen.json) | 保存保护路径纸面筛查的结构化输入、结果和限制。 |
| [selected_ipt015_thermal_screen.md](selected_ipt015_thermal_screen.md) | 说明当前选用IPT015N10N5六管桥的条件热筛查。 |
| [selected_ipt015_thermal_screen.py](selected_ipt015_thermal_screen.py) | 复算IPT015N10N5桥臂的条件性损耗/温升筛查，并生成同名JSON结果。 |
| [selected_ipt015_thermal_screen.json](selected_ipt015_thermal_screen.json) | 保存该MOSFET条件热筛查的输入、公式、结果和限制。 |
