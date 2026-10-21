"""同步当前风扇/电感/REG1设计文字，不覆盖原始过程证据。"""
from pathlib import Path
B=Path(__file__).resolve().parents[1]
topology=('控制复用`EL357N(C)(TA)-G`光耦：`FET_RELEASE`经160Ω点亮LED，LED阴极接B0；'
 '`Q_FAN_STOP`的E接B0、C接LED阳极，PA15经470Ω驱动B，B经100kΩ下拉至B0。'
 'PA15高时旁路LED停止风扇，低/高阻撤销停止请求。光耦C由`FAN_VIN`经100kΩ供电、E驱动EN，'
 'EN经10kΩ下拉至PACK−。移除两颗旧跨地集电极电阻，风扇页仍为23件。'
 'B0/PACK−保持分流器边界，控制没有导电跨地支路，覆盖分流器开路时旧EN负压风险；'
 '全温CTR/暗电流和停机波形仍待验收。')
for rel in ['README.md','TODO.md','原理图准备/控制与接口设计.md','原理图准备/编程交接.md','原理图准备/Layout交接.md','原理图准备/需求与系统边界.md','检查/原理图构建与回读结论.md','检查/需求覆盖与待验收.md']:
 p=B/rel;t=p.read_text(encoding='utf-8-sig')
 t=t.replace('PA15经10 kΩ','PA15经470 Ω').replace('PA15经10kΩ','PA15经470Ω')
 if rel=='README.md':
  t=t.replace('B0与PACK−仅经0.5 mΩ分流器相连，控制由限流晶体管跨域驱动EN，不硬短接两地；',topology)
  t=t.replace('仍需闭合真实电感参数、','电感已选SRP1265A-220M；仍需闭合实装电感温升、')
 if rel=='原理图准备/控制与接口设计.md':
  a=t.index('控制跨域采用限流晶体管');z=t.index('阈值可调',a);t=t[:a]+topology+t[z:]
  t=t.replace('PRE_CTRL_3V3 只供预充/比较/时序逻辑。','PRE_CTRL_3V3 供预充/比较/时序逻辑及风扇光耦LED，REG1全支路按≤30mA预算；不供MCU或风扇功率。R_BREG_FEED保持300Ω并升为2512/2W，Q_BREG热与SOA仍须核对。')
 if rel=='原理图准备/编程交接.md':
  a=t.index('控制跨域由限流晶体管');z=t.index('硬件运行条件',a);t=t[:a]+topology+t[z:]
  t+='\nREG1/PRE_CTRL_3V3总负载按≤30mA预算；PA15停止输出为推挽高，470Ω基极支路保守最大7.8mA，纳入GPIO组总电流。禁止从REG1给MCU或风扇功率供电；寄存器配置与读取确认由人类完成。\n'
 if rel=='原理图准备/Layout交接.md':
  a=t.index('B0与PACK−仅经0.5 mΩ');z=t.index('运行还受',a);t=t[:a]+topology+'光耦两侧铜皮、测试连接和EN回路保持各自电源域；'+t[z:]
  old='22µH电感使用普通可采购的功率电感，YSPI1365-220M为当前封装候选，厂商饱和电流、温升曲线与库参数仍需一致性核定。'
  t=t.replace(old,'22µH电感选Bourns SRP1265A-220M，DCR最大37mΩ、Irms9A、Isat12A（下降20%）；参数已按原厂核对，实际RMS/磁芯损耗与板上温升仍待验收。原生器件关联改为该料号封装；不得视作旧YSPI焊盘兼容。')
  t+='\nREG1前级300Ω电阻改为RC2512FK-07300RL 2512/1W；REG1总负载预算≤30mA。为Q_BREG保留独立散热铜区并核对SOA与结温；参考[风扇使能收尾](../检查/风扇使能设计收尾研究.md)，不将纸面功耗作为温升通过。\n'
 if rel=='原理图准备/需求与系统边界.md':
  t=t.replace('B0与PACK−只经0.5 mΩ分流器相连，100 A时约50 mV；风扇控制跨域仅使用限流高阻信号，不铺设跨地铜线或硬短接两地。','B0与PACK−保持0.5mΩ分流器边界，风扇EN控制通过复用EL357光耦传递，不建立导电跨地支路。')
  a=t.index('`FET_RELEASE`经100 kΩ接TPSM');z=t.index('SW_RUN OFF',a);t=t[:a]+topology+t[z:]
 if rel=='TODO.md':
  a=t.index('`FET_RELEASE`（B0参考3.3 V）经100 kΩ');z=t.index('主输出有效、FET_RELEASE高',a);t=t[:a]+topology+t[z:]
  t=t.replace('- [ ] 核定22µH电感饱和电流、温升和DCR；不直接采用有冲突的库参数。','- [x] 按原厂核定22µH电感参数并选SRP1265A-220M，纠正YSPI误把Irms5.8A当DCR5.8mΩ；实装温升仍见样机任务。')
 if rel=='检查/原理图构建与回读结论.md':
  a=t.index('风扇支路由');z=t.index('\n\n',a);t=t[:a]+'风扇支路由`PACK+`/`PACK-`独立供电，使用专用`TPSM65610`降至12V。'+topology+'以上为原理图实施记录，不代表功能验收。'+t[z:]
 if rel=='检查/需求覆盖与待验收.md':
  t=t.replace('PA15高有效`FAN_STOP`经NPN与限流电阻控制EN。`B0`与`PACK-`之间保留0.5mΩ分流器，风扇控制取消光耦；浮地均衡和外部接口隔离仍保留','PA15高有效`FAN_STOP`经470Ω驱动NPN旁路LED；复用EL357，输出侧100kΩ/10kΩ控制EN。控制无导电跨分流器连接，正常压差及开路故障均列入设计')
  t=t.replace('2. 普通电感候选YSPI1365-220M的饱和电流、DCR和温升资料尚有冲突，采购前闭合；COMP/ILIMCOMP数值为初始值，留调试位。','2. 电感规格冲突已纠正，当前选SRP1265A-220M；实际RMS、磁芯损耗、温升和物理焊盘仍待人类验收。COMP/ILIMCOMP数值为初始值，留调试位。')
 t=t.replace('1946','1954')
 p.write_text(t,encoding='utf-8')
print('updated eight current handoff documents')
