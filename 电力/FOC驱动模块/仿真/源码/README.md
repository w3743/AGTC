# FOC 仿真源码

本目录包含母线能量、功率级开关行为、BOV关断链、电流环/5V启动等模型和结果整理/绘图脚本。脚本说明各自的输入和使用限制；相邻`结果/`保存输出或原始仿真数据，不能据模型结果推断目标固件、完整FOC闭环、电机SOA或硬件验收。

## 直接文件

| 文件 | 用途 |
|---|---|
| [simulate_dc_link_1s.py](simulate_dc_link_1s.py) | 对1秒母线/制动斩波事件进行平均值能量筛查，不生成开关波形。 |
| [simulate_dc_link_ngspice.py](simulate_dc_link_ngspice.py) | 运行独立的1秒ngspice母线/斩波网络交叉检查。 |
| [simulate_power_stage.py](simulate_power_stage.py) | 生成并运行ngspice FOC功率级开关行为模型；不是器件厂商宏模型或闭环固件验证。 |
| [simulate_bov_bjt.py](simulate_bov_bjt.py) | 通过ngspice检查BUSOV_N至BC857C/BC847C再到BREAK_N的行为链路；不是完整角落保证。 |
| [simulate_current_loop_and_5v.py](simulate_current_loop_and_5v.py) | 运行20 kHz dq电流环与LM5008A 5V启动条件筛查。 |
| [plot_dc_link_1s.py](plot_dc_link_1s.py) | 绘制Python平均值1秒再生母线配置的曲线。 |
| [plot_switch_raw_waveforms.py](plot_switch_raw_waveforms.py) | 读取并绘制已保存的ngspice二进制raw波形，不重新运行电路。 |
| [summarize_switch_cases.py](summarize_switch_cases.py) | 汇总已保存的ngspice开关工况结果，生成便于核查的对照数据。 |
