"""Read current sources, cross-check BOM, and bind current evidence hashes."""
from pathlib import Path
import csv,json,hashlib,datetime,importlib.util
P=Path(__file__).resolve().parent;M=P.parent;R=M.parent
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('audit',P/'只读官方网表核对.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
spec=importlib.util.spec_from_file_location('counting',P/'更新当前BOM计数.py');counting=importlib.util.module_from_spec(spec);spec.loader.exec_module(counting)
expected=load(P/'当前施工输入.json');pcb={x['ref']:x for x in expected if x['addIntoPcb']}
bom=list(csv.DictReader((M/'原理图准备/BOM.csv').open(encoding='utf-8-sig',newline='')));byref={x['位号']:x for x in bom}
assert len(byref)==len(bom)
errors=[]
for ref,x in pcb.items():
 if ref not in byref:errors.append([ref,'missing BOM']);continue
 row=byref[ref]
 if audit.canonical_mpn(row['MPN'])!=audit.canonical_mpn(x['mpn']):errors.append([ref,'MPN',row['MPN'],x['mpn']])
 if float(row['每套用量'])!=1:errors.append([ref,'PCB quantity',row['每套用量']])
retired=['U_PSU','C_PSU_BOOT','C_PSU_FF','R_PSU_FB_TOP','R_PSU_FB_BOT','L_PSU','D_PSU','R_PSU_EN','R_IF_RUN_PD','R_FAN_EN_PD']
for ref in retired:
 if ref in byref:errors.append([ref,'retired still in BOM'])
assert not errors,errors
assert len(pcb)==216
mpns={counting.normalize(x['mpn']) for x in pcb.values()};assert len(mpns)==53,len(mpns)
net=load(P/'官方网表连接核对.json');assert net['summary']['pinDifferenceCount']==0
for record in load(P/'剩余验证/原生发布校验.json'):
 assert sha(M/record['file'])==record['sha256'],record['file']
library=load(P/'库同步校验.json');assert library['exitCode']==0 and library['result']['pass'] is True
drc=load(P/'DRC逐项核对.json')['currentOfficialResult']
assert drc['fatal']==0 and drc['errors']==0 and drc['warnings']==8 and not drc['strictPass']
fan=load(M/'仿真/结果/温控调速落实仿真.json');hlk=load(M/'仿真/结果/HLK_implementation_model.json')
startup=load(M/'仿真/结果/fan_foldback_envelope_20261015_result.json')
assert len(startup['measurements'])==3
assert startup['measurements'][0]['time_to_4p999v_s']>0.0013
protection=load(P/'保护精简落实审查.json');solar=load(P/'行进充电与回收能量筛查.json')
files=[M/'项目要求.md',M/'原理图准备/BOM.csv',P/'当前施工输入.json',P/'BMS-9S4P.enet',P/'官方网表连接核对.json',P/'温控调速落实验证.json',P/'海凌科电源落实验证.json',P/'保护精简落实审查.json',P/'行进充电与回收能量筛查.json',P/'行进充电与回收支持审查.json',P/'当前原理图导航审阅.pdf',P/'当前原理图导航审阅.json',P/'当前BOM计数证据.json',R/'元件封装库/资产索引.json']
files += [P/'库同步校验.json',P/'风扇限流料号归并权衡.json',P/'DRC逐项核对.json',P/'16A净充电上限验证.json',P/'剩余验证/原生发布校验.json',P/'剩余验证/发布归档记录.json']
files += [P/'3.3k基极电阻510Ω归并评估.md',P/'风扇SGM与HLK动态纸面验证_20261015.md',P/'剩余验证/工作副本归档.json']
files += [M/'README.md',M/'TODO.md',P/'当前设计状态.md',P/'DRC逐项核对.md',P/'16A净充电上限验证.md',Path(__file__)]
files += [f for folder in ['源码','输入','结果'] for f in (M/'仿真'/folder).glob('*') if f.is_file() and any(t in f.name for t in ['16A','charge_limit_16a','bal_base_resistor','fan_urb_startup','fan_foldback'])]
for f in files:
 assert f.is_file(),f
files+=list((M/'eda工程文件').rglob('*'))
files+=[f for d in ['源码','输入','结果'] for f in (M/'仿真'/d).glob('*') if f.is_file() and ('implementation' in f.name.lower() or '温控调速落实' in f.name or '行进充电' in f.name or '保护精简' in f.name)]
report={'date':datetime.datetime.now().astimezone().isoformat(),'status':'原理图实现与受限模型完成；非投板或实物验收','counts':{'pcbBefore':213,'mpnBefore':60,'pcbNow':len(pcb),'mpnNow':len(mpns),'CAD':len(expected),'pins':723,'bomRows':len(bom),'totalAssemblyQty':sum(float(x['每套用量']) for x in bom)},'checks':{'bomVersusIndependentExpected':'PASS','savedReopenedNetlist':net['summary'],'fanModel':fan['test_summary'],'hlkModel':hlk['summary'],'protectionReport':'保护精简落实审查.json','strictDRC':{'pass':False,'warningCount':8,'individualClassification':True,'closedUnusedOutputWarnings':2},'librarySync':'同步全部.py + --check成功后见库同步校验.json'},'retiredRefs':retired,'solar':{'nameplateW':30,'architecture':'独立MPPT→公共BUS→BMS主FET与分流器→电池','regenPowerDuration':'未确定','noLossCurrentRangeA':[round(30/37.8,3),round(30/22.5,3)],'CCL16AIsDesignInputNotCertified':True,'noAbsorptionTimeTo45Vms':solar['results']},'evidence':[{'path':str(f.relative_to(R)).replace('\\','/'),'sha256':sha(f)} for f in files if f.is_file()]}
report['checks']['fanStartupBehavioralSpice']={'status':'OPEN for hardware; behavioral transient completed','measurements':startup['measurements'],'notModeled':startup['model_assumptions']['not_modeled']}
charge=load(P/'16A净充电上限验证.json')
report['checks']['chargeLimit16A']={'modelStatus':charge['model_status'],'releaseStatus':charge['release_status'],'report':'16A净充电上限验证.md'}
(P/'本轮落实与验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
text='''# 本轮BMS落实与验证

2026-10-15。已更新唯一当前原生工程、连接源、BOM和项目要求。当前板内 **216颗、53种精确MPN**，本轮前213颗、60种：增加3颗，减少7种。采购配套另计。CAD含一个板外DPDT载体共217个位号、723个引脚。

## 实际变更与保护取舍

- 采用海凌科URB4805YMD-10WR3，移除自设计降压芯片及专属外围。最低负载复用7颗510Ω，并联标称72.86Ω，避免新增75Ω料号；不是取消厂家最低负载要求。
- 一个板外DPDT分别控制PRE_CTRL_3V3许可和模块CTRL：ON浮空开启、STORE接B0关闭。合并SW_RUN与PRE_CTRL_3V3并删除重复下拉R_IF_RUN_PD，保留R_PRE_CTRL_PD。普通OFF不等于电池隔离或<10µA仓储验收。
- U_AND2复用SN74LVC1G11DCKR，第三输入接VCC，实际更换SC70-6符号/封装；8颗100nF统一已有100V、X7R、0603型号，物理去耦保留。
- 四线5V风扇采用25kHz开漏PWM和3.3V转速反馈；PA6/P2-14、PA1/P2-17已连接，PA15释放。SGM支路持续使能并删除R_FAN_EN_PD，复位PWM释放时请求全速。
- 保留BQ电池保护、独立预充超时、故障锁存/POR、风扇支路限流、浮动均衡控制光耦。它们分别覆盖电芯故障、预充失败、MCU/失电默认态、支路线缆故障和浮动电位控制，未找到可删除且保留对应故障处置的完整替代。详见[保护审查](保护精简落实审查.json)。

另评估了将风扇6.8kΩ限流电阻归并到现有5.1kΩ，可再省1种。按手册档位反比估算，最低折返电流约0.725A，可覆盖0.45A启动假设与电容充电之和；但模块异常限流总压力将升至约1.838A，相对2A只剩约0.162A，且启动峰值与该阻值的限流边界未获保证。本轮保留6.8kΩ及约0.527A算术余量，不以缩窄故障余量换取这一种料号。见[取舍计算](风扇限流料号归并权衡.json)。

再次核查的两个3.3kΩ可候选改用已有510Ω，理论不增件即可降到52种。GPIO电流上界约7.15mA、REG1规划约16.05mA的算术预算可行；但串联光耦驱动的MMBT5551饱和/全温边界及低线BREG调节尚未闭合，故未写入正式BOM。原3.3kΩ也保留待验证状态，不能因未更换而认为通过。C_REG18改用已有高耐压4.7µF仍缺有效电容依据；预充4.7kΩ复用0603功耗不足；100pF100V降为50V不满足现有浪涌筛查。当前53种是本轮验证范围内的保留结果，不宣称全局最少。

## 本轮验证

| 项目 | 结果 | 证据与限制 |
|---|---|---|
| 保存重开官方网表 | PASS | 217位号、723针网、MPN、入板标志0差异；NC标志单独回读。是实现一致性核对 |
| BOM与独立施工输入 | PASS | 216个入板器件逐项比对精确MPN及用量，10个退出位号均不在现行BOM |
| 风扇策略 | PASS（模型） | 温度曲线、迟滞、反相、故障锁存回归及假设RC模型；未实现固件或证明散热能力 |
| 电源静态预算 | PASS（算术包络） | 连续规划0.673A、限流压力1.473A，相对于模块标称2A；基础0.3A是规划输入 |
| 风扇启动 | OPEN | 假设0.45A与47µF/1.3ms充电叠加约0.631A，高于约0.544A折返下限。ngspice分段电流包络：47µF充至4.999V约2.23ms，22µF约1.04ms；1.3ms仅参考，未证明真实风扇能否启动 |
| 海凌科动态及热 | OPEN | 厂家明确不支持热插拔，应用须避免直接带电插拔；输入有效电容、环路、CTRL失效/回灌及60°C热性能待验。行为电路不等于模块开关模型 |
| 严格DRC | FAIL（未闭合） | 用户日志10条已分类；U_SEQ未用6/8针改NC，保存重开复跑剩8条：板外预充连接边界1项、供应商属性7组 |
| 共用库同步 | 见库同步校验.json | 正式工程/BOM修改后运行同步和只读校验 |

模型源码、参数和结果分别位于`../仿真/源码`、`../仿真/输入`、`../仿真/结果`；本轮JSON汇总绑定SHA256。原理图PDF由官方导出重框并加入217个位号书签，已检查本轮电源、与门和风扇新增标注，非全板制造审查。

风扇电源详细边界见[动态纸面验证](风扇SGM与HLK动态纸面验证_20261015.md)。海凌科与其他厂家存在同名模块，采用海凌科手册约束，不将其他厂家数据当作目标器件保证。

## 太阳能与回收

用户确认30W太阳能、独立MPPT板接公共BUS，重点回收。电池侧电流须经主CHG FET和分流器，不能旁路接B0。光伏无损上限仅约0.79–1.33A；按效率0.8/0.9/1.0与三个电池电压建模，并与500/1000/2000W回馈组合筛查。27个无负载算例中24个超过16A设计输入；16A没有全温、均流和误差余量认证。

回收和太阳能共用一个净充电额度，不能各自使用完整CCL。电池净充电上界16A，超额回收由双向DC/DC转入超容；超容已满、失效或能力不足须源端减小回馈并由制动系统承接。满电、低温、故障或CCL=0时，需要停止外部注入并安排剩余能量去向。作为时间尺度筛查，假设60mF直接母线电容、初始37.8V、无电池/超容/负载吸收，剩余1kW仅约17.9ms升至45V；45V仅筛查点，不是可放行工作点。CAN超时不能单独承担这个瞬态保护。

16A模型已覆盖分流器容差/温漂、测量误差、支路不均流敏感性和源端超调。OCC code3为当前设计候选，实际写入/读回未验证；6mV阈值的早/晚动作角约7.92/17.47A，候选延时9.9ms，只作后备切断。正常CCL须兼顾OCC早动作角：10.5A算例虽满足给定电芯/测量边界，却不保证与code3兼容。另留0.5A OCC规划余量的条件峰值界限约7.42A，连同源端超调和测量角反算命令约6.1A；这些均依赖未验证的均流、测量与动态假设，不能作为默认额定。本轮未修改OCC。详见16A净充电上限验证.md。

BMS接口约束已落文档；最大回馈功率/时间、超容可吸收能力、驱动停注入及制动协调未冻结，整车回收功能尚未验收。未改动MPPT、FOC或中央DC/DC工程。

## 后续必要闭合

1. 固件实现PWM/FG状态机、BQ五路NTC参数与读回、CCL/DCL分配及过期处理。
2. 样板验证风扇启动、模块动态/热/EMC、60A与100A热和关断；本轮没有样板。
3. ETA3000的9S级联厂家歧义、TVS对下游60V器件浪涌余量、采购MPN与CAD载体差异及报价/MOQ仍待闭合。

当前资料可用于继续设计与人工审查，尚不能作为投板、采购或整车放行依据。

过程工作副本已逐文件SHA256读回核对后归档清理，正式五文件未改变。既有历史ZIP有一条旧记录的本地头/中央目录路径斜杠不一致，整体testzip不能报通过；本轮新增前缀可独立读取且哈希一致，未修复该旧条目。详见剩余验证/工作副本归档.json。
'''
(P/'本轮落实与验证.md').write_text(text,encoding='utf8')
print(json.dumps({'counts':report['counts'],'bomCheck':'PASS','evidenceFiles':len(report['evidence'])},ensure_ascii=False))
