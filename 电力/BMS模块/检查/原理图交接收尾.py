"""原位维护当前设计入口；不运行ERC、仿真或PCB操作。"""
import json,csv,hashlib,zipfile
from pathlib import Path
B=Path(__file__).resolve().parents[1];S=B/'原理图准备';C=B/'检查';R=B.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

# 先保存当前文稿，读回确认，随后原位更新。
names=[B/'README.md',B/'TODO.md',S/'需求与系统边界.md',S/'Layout交接.md',C/'需求覆盖与待验收.md']
zpath=B/'归档/历史与调试.zip';archive=[]
with zipfile.ZipFile(zpath,'a',zipfile.ZIP_DEFLATED) as z:
 existing=set(z.namelist())
 for p in names:
  if not p.exists():continue
  n='共享电感替换前/入口收尾/'+p.relative_to(B).as_posix();data=p.read_bytes()
  if n in existing:
   if z.read(n)!=data:raise RuntimeError('archive collision '+n)
  else:z.writestr(n,data)
  archive.append({'source':str(p),'entry':n,'sha256':hashlib.sha256(data).hexdigest()})
with zipfile.ZipFile(zpath) as z:
 for a in archive:
  if hashlib.sha256(z.read(a['entry'])).hexdigest()!=a['sha256']:raise RuntimeError('archive readback '+a['entry'])
write(C/'原理图交接文档归档.json',archive)

p=S/'需求与系统边界.md';t=p.read_text(encoding='utf-8-sig')
t=t.replace('均衡执行架构重新选型，禁止 MA5421-ALD，不限定 LTC3300','均衡采用LM51772单只普通电感、9组选芯矩阵与4S2P缓冲；禁止MA5421-ALD')
t=t.replace('既有 LTC3300 连接仅为替换前快照','旧LTC3300及变压器已退出当前连接源和原生工程')
t=t.replace('当前研究聚焦成本可闭合的标准电感及低成本电容方案，均未冻结','当前原理图选择共享普通电感矩阵方案，性能及完整成本仍需后续证据')
t=t.replace('主动均衡执行级：架构待定','共享电感+9组选芯+缓冲')
t=t.replace('TE DTP 4位断开均衡抽头','独立TE DT 12位B键断开十根均衡抽头（与CELL键位不同）')
t=t.replace('`VOUT_5V` 的 600 mA 预算','`VOUT_5V` 的 1 A电源设计与当前负载预算')
t=t.replace('局部均衡芯片还须另算各串组电流','矩阵、偏置及输入保护漏电还须另算各串组电流')
t+='\n## 当前均衡连接与残能补充\n\n共享电感具体连接见AFE与主动均衡设计.md和共享电感编程接口.md。BAL十抽头与CELL测量线束独立；B1…B9均衡分支各设置源端5A保险，BAL_B0不熔断。缓冲器在VOUT_5V掉电后，经常闭继电器自动接入10Ω/10W阻燃泄放电阻；至少等待120s并实测BAL_BUF_P对BAL_N低于1V。时间不能代替测量；BUS、驱动电容及超容残能须分别处理。低温ESR升高时降流或停止，均衡能力不作为全温度实测保证。\n'
p.write_text(t,encoding='utf-8')

p=S/'Layout交接.md';t=p.read_text(encoding='utf-8-sig').replace('BMS-9S4P.eprj2','BMS-9S4P.epro2')
t=t.replace('10mΩ峰流和27mΩ缓冲平均电流分流器使用2512低感载体，四端Kelvin布线；采购值必须是10/27mΩ，不能购买库模板中的12/33mΩ。','10mΩ峰流分流器为RL2512FK-070R01L，27mΩ缓冲平均分流器为RL2512FK-070R027L；均为实际值2512器件，两端器件采用独立Kelvin采样走线。')
t=t.replace('LM51772/ISO1640查价时无现货，采购前应核实交期。','人民币销售页面与国际USD页面的库存口径不同，采购前按检查/共享电感器件证据.md核实人民币结算、库存和交期；全套成本未闭合。')
t+='\n## 新增泄放与5V预算\n\nBAL_DRAIN的继电器1/4为5V线圈、5为COM、3为NC、2为NO不接；线圈续流SS14为1K/2A。NC失电闭合后将缓冲正端经10Ω/10W SQP10AJB-10R接BAL_N。电阻须按厂家高温降额和脉冲散热布置；缓冲并非关机即无残压。隔离触点与B0线圈之间的隔离边界不得被布线破坏。5V偏置电源预算及继电器负载见检查/5V电源预算.md；主5V采用TPSM65610SVCGR 1A，浮地偏置采用B0512S-2WR3，静态矩阵偏置保留B0512S-1WR3。额度来自设计估算，须按实际温升及负载核定。原理图使用通用R/C封装载体的项目只定义阻值、电容值和初始封装，采购前按额定电压、功率、精度和DC偏压选定真实料号。\n'
p.write_text(t,encoding='utf-8')

(B/'README.md').write_text('''# 9S4P BMS 当前设计

本轮已将共享普通电感均衡方案写入嘉立创EDA原生原理图。PCB布局布线、机械尺寸和固件由人类完成。

## 当前入口

- [当前工作工程](eda工程文件/BMS-9S4P.eprj2) · [官方导出交付](eda工程文件/BMS-9S4P.epro2)
- [布局交接](原理图准备/Layout交接.md)
- [需求与系统边界](原理图准备/需求与系统边界.md)
- [AFE与主动均衡](原理图准备/AFE与主动均衡设计.md)
- [编程接口与可调参数](原理图准备/共享电感编程接口.md)
- [BOM](原理图准备/BOM.csv) · [连接表](原理图准备/连接表.csv)
- [当前交接结论](检查/需求覆盖与待验收.md) · [TODO](TODO.md)

## 已采用的设计

9S4P EVE 30P，BQ76952独立测量与保护；60A连续、100A/1s、6A日常充电、16A净充电为设计输入。超限需求由既有超容双向DC/DC补充，能力不足时限负载。

主动均衡使用LM51772、一只22µH普通功率电感、九组选芯矩阵和8只10F电容组成4S2P缓冲。每次只选择一组，通过取能、切换、送能实现任意两组转移。MA5421-ALD禁止使用，旧LTC3300及定制变压器退出当前工程。独立硬件保护、空窗、心跳和故障锁存保留，参数提供调整空间。

ON由本地自检与预充建立输出，供上位机启动，不等待CAN授权。日常DPDT OFF关断输出、管理与均衡，保留BQ保护监测；长期仓储/检修采用MAIN/CELL/BAL物理维护隔离。缓冲在5V失电后自动泄放，至少等待120s并测量残压。各板保留自身测温，浮地均衡区另设NTC。

## 实施边界

可以开始layout准备、封装核对与布置。原生施工保存不等于ERC通过、性能验证或生产放行。本轮没有执行ERC、仿真或实物测试。仍需闭合真实电感参数、双向补偿与负向限流、缓冲ESR、主回路热/短路/预充、全部封装和全套报价。均衡成本目标200元可合理浮动，完整人民币报价未齐，不宣称已达标。

历史与旧设计保存在[历史与调试.zip](归档/历史与调试.zip)，当前入口以上述文件为准。
''',encoding='utf-8')

(B/'TODO.md').write_text('''# BMS 当前任务清单

## 本轮电气设计与原理图交接

- [x] 确认9S4P、60A/100A脉冲和6A/16A充电设计输入。
- [x] 保留BQ76952、本地短路/过流保护、独立预充与CAN状态接口。
- [x] ON由本地自检预充供上位机启动，无CAN开机授权。
- [x] 两级关机：日常DPDT OFF保留保护，MAIN/CELL/BAL作仓储与维护物理断开。
- [x] 选择共享普通电感、9组选芯、4S2P缓冲；移除禁止器件与旧均衡执行级。
- [x] 加入隔离控制、单组选通、切换空窗、心跳、故障锁存、温度及四组缓冲过压停止。
- [x] 加入关机自动缓冲泄放支路，记录等待与实测残压要求。
- [x] 留出电流、频率、补偿与算法调整空间；PCB及固件交由人类完成。
- [x] 同步现行原生工程、连接源、BOM、连接表与布局交接。

## layout、采购和生产前仍需闭合

- [ ] 人类核定全部实际封装、焊盘映射和板内/板外标记；通用R/C载体选真实采购料。
- [ ] 人类执行当前原理图ERC并处理真实问题；旧ERC记录不适用于新工程。
- [ ] 核定22µH电感饱和电流、温升和DCR；不直接采用有冲突的库参数。
- [ ] 闭合LM51772双向补偿、负向限流精度、预偏置启动、真实ESR及切换空窗。
- [ ] 完整人民币报价、实际单套用量、MOQ多购和库存交期；对200元目标记录浮动依据。
- [ ] 人类完成PCB布局、布线、热设计和装配审查。
- [ ] 人类完成固件：BQ保护、自检/预充、CAN遥测、选芯/缓冲控制、故障恢复和参数边界。

## 样机与装车前

- [ ] 全周期净转移：任意组包含C1↔C9，50mV≥0.5A、100mV约1A；记录共同电压、温度、效率和调度停顿。
- [ ] 逐组缓冲过压、NTC开短路、失电/心跳故障、门极与电感电流和残压。
- [ ] 5V高温总负载与供电启动、隔离模块最小负载、继电器释放和泄放电阻热态。
- [ ] 60A连续/100A脉冲、短路关断、保险配合、预充正常与故障脉冲。
- [ ] 日常OFF逐抽头与全包耗电、USB/SWD反灌、维护断接次序和仓储漏电。
- [ ] 驱动母线、超容、电机回馈与整车关断残能分别验收。

本清单的勾选表示设计交接完成，未证明电气、生产或实物验收通过。
''',encoding='utf-8')

p=R/'README.md';t=p.read_text(encoding='utf-8-sig');t=t.replace('[需求覆盖与待验收](BMS模块/检查/需求覆盖与待验收.md)','[共享电感原理图交接](BMS模块/检查/需求覆盖与待验收.md)');p.write_text(t,encoding='utf-8')
print('current entry documents updated; archived',len(archive))


