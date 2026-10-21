"""Write current handoff reports only after release and verification."""
from pathlib import Path
import csv,datetime,hashlib,json,zipfile
P=Path(__file__).resolve().parent;M=P.parent;R=M.parent
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    net=load(P/'官方网表连接核对.json')['summary']
    assert net['actualRefCount']==213 and net['actualPinCount']==711 and net['pinDifferenceCount']==0
    assert net['mpnMismatchCount']==0 and net['pcbFlagMismatchCount']==0
    assert load(P/'库同步校验.json')['result']['pass']
    rows=list(csv.DictReader((M/'原理图准备/BOM.csv').open(encoding='utf-8-sig',newline='')))
    refs={r['位号'] for r in rows};assert not refs & {'U_FAN_SW','R_FAN_ILIM','C_FAN_IN1','C_FAN_OUT1'}
    carrier=P/'载体封装与BOM差异复核_20261015.md'
    raw=carrier.read_bytes();key=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_旧载体审查加注/检查/'+carrier.name
    with zipfile.ZipFile(M/'归档/历史与调试.zip','a',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(key,raw);assert hashlib.sha256(z.read(key)).hexdigest()==sha(carrier)
    carrier.write_text('> 历史载体审查：当前正式212件/51种已使用海凌科模块，R_PSU_FB_TOP、L_PSU及U_FAN_SW均已删除；下文对应旧降压和风扇开关的问题不再是现行布局门禁。其余问题须以当前网表和[Layout前置审查](Layout前置审查.md)复核，不能整体沿用旧判定。\n\n'+raw.decode('utf-8-sig'),encoding='utf8')
    layout='''# BMS人工Layout交接

2026-10-15：当前正式原生与BOM为**212颗板内器件、51种归一化采购MPN**，另有1个板外DPDT的CAD载体，共213位号/711针。代理工作停在此处；PCB创建、布局布线与固件由人类完成。本文件允许开始Layout，不构成投板、全功率或整车运行放行。

当前入口：[原生工程](../eda工程文件/BMS-9S4P.eprj3)、[BOM](BOM.csv)、[器件映射](../检查/Layout器件映射.json)、[审阅图](../检查/当前原理图导航审阅.pdf)。eprj3为描述入口，实际页面为`eda工程文件/sch/Schematic1/P1.esch2`；当前官方导出包为同目录`BMS-9S4P.epro2`。保存重开逐针核对0差异，DRC 0错误/8警告。

## 本轮落实与保留理由

海凌科URB4805YMD-10WR3替代旧5V降压；本轮进一步删除U_FAN_SW、R_FAN_ILIM和其两颗47µF，J_FAN pin1由FAN_5V改为VOUT_5V。相比上版216/53减少4颗、2种；相比本阶段起点213/60减少1颗、9种。四串电阻候选已撤回。

两颗AT3H74的8个通道全部用于8级主动均衡，不能利用不存在的空闲通道代替VO617A。TPL5111保留独立预充硬超时，TPS3809/双触发器保留独立供电域复位和锁存；两颗同型号与门分属主许可和QPRE硬窗口。六颗主MOS保持每bank三并联，两个bank均减到两颗将使纸面导通损耗增加50%。详见[高价值器件定案](../检查/高价值器件定案.md)。

## 布局时必须落实

| 区域 | 约束与理由 |
|---|---|
| 主功率与分流器 | 60A连续/100A 1s仅为设计输入。三并联MOS各自栅阻、走线对称，校核铜排、过孔、端子和散热；分流器四线Kelvin，采样线远离门极及回流。B0与PACK−禁止另一路连通，否则绕过电流采样。 |
| 预充 | QPRE_SW接板外8.2Ω预充电阻，当前单针警告属于此接口边界。电阻、开关和线束须校核约0.51kJ最坏故障脉冲；正常60mF母线约42.9J充电能量。硬超时约2.87s不是连续耐受额定。预充完成需新鲜电压采样且ΔV≤1V；不能只到时闭合主FET。 |
| 5V模块 | 25.4×25.4×11mm；厂家图以底视/实际标签为准：1=输入−、2=输入+、3=输出+、4=NP、5=输出地、6=CTRL。输入18–75V覆盖本包，输出地接B0。检查引脚孔径、底部净空、通风和线束插拔；不直接带电插拔模块输入。输入/输出电容靠近对应引脚，7颗510Ω最低负载分散散热。不能把厂家整套EMC推荐外围无条件全部增加。 |
| 风扇 | J_FAN 1=VOUT_5V（风扇红2脚），2=B0（黑1脚）；J_FAN_CTRL 1=PWM蓝4脚，2=FG黄3脚。PA6/P2-14=PWM，PA1/P2-17=FG，PA15/P1-15=NC。开漏25kHz，1k栅阻/100k下拉/FG 10k上拉3.3V保留。风扇正负支路含端子、过孔和线束按≥4A持续校核，另验短路脉冲；4A不是模块峰值钳位保证。PWM/FG不得充当功率回线。 |
| 浪涌与隔离 | BMS SMCJ45A回路短且宽，不将TVS作为制动耗能器；72.7V数据表钳位与75V直连采样电阻余量仅2.3V，须结合寄生与脉冲能量复核。FOC本地TVS按其板约束。隔离CAN两侧参考地保持清楚，不旁路分流器。 |
| 均衡与测量 | 8颗ETA3000按B0至TAP_B9相邻级连接；电感/开关回路紧凑，采样与NTC远离功率/最低负载热源。9S拓扑已核对，低单体电压及逐级EN启停仍为首板运行门禁。 |
| 器件落位 | U_AND2为SC70-6，不能沿用旧SOT23-5。VO617A-2采购与-4载体同几何不等于电气可替代。D_CAN_TVS精确载体、BAT46W极性、FCX495TA数字脚序和连接器实体方向在对应器件落位前复核；未闭合的器件不能靠封装名直接放行。NKK板外DPDT按真实6端子，CAD载体7/8为NC壳体占位。 |

## 人工建立焊线焊盘

这些焊盘尚未创建于PCB，不增加BOM或MPN；机器表见[接口焊盘映射](../检查/Layout接口焊盘映射.json)。

| 焊盘组 | 1脚起的网络顺序 | 建议 |
|---|---|---|
| J_CELL_SERVICE | B0、TAP_B1、TAP_B2、TAP_B3、TAP_B4、TAP_B5、TAP_B6、TAP_B7、TAP_B8、TAP_B9 | 单排1×10 PTH，5.0mm节距、2.0mm成品孔、4.0mm铜盘为待机械核对建议；TAP位于源端保险下游，绝缘目标≥60Vdc。 |
| J_CELL_NTC_SERVICE | B0、BQ_TS1、BQ_DCHG、BQ_CFETOFF | 单排1×4 PTH，与抽头组分开；孔径/铜盘/节距按实际低压探头线径确定。 |

焊线应力释放、铜箔载流及间距须复核，不把手工接口表当已画好的PCB。

## 固件和首板门禁

1. BQ全部保护阈值、NTC系数/开短路、看门狗、DFETOFF的BOTHOFF配置必须写入并读回。许可只在完整自检后发出；禁止预充错误采样使主MOS带大压差合闸。布局后校核短路SOA和关断时间。
2. 风扇计划总5V负载0.673A，连续预算有余量。启动峰值、模块打嗝/恢复不是已知波形；共享5V失效可能影响CAN和风扇。CORE也失电时验证LOCAL_OK下拉、锁存异步清除和DFETOFF默认关断；独立CORE仍供电时必须以FG故障锁存撤销本地许可。恢复5V不能自动恢复功率许可。FG向未供电风扇最多约0.33mA注入的容许性待实测/厂家确认。初次低能验证不外供CORE来掩盖失电行为。
3. 16A为电池净充电上界，非OCC恒流点。OCC code3角落约7.92–17.47A，延时约9.9ms；CCL必须低于各项误差/均流/温度/源超调约束。6.1A是条件模型例子，未资格确认前CCL=0。30W独立MPPT入母线，超额回收交超容双向DC/DC；CAN仅发布能力，本地源端须快速调节。回馈峰值/时间、超容余能未定，整车未获额定。
4. 完成海凌科和风扇启动/短路恢复、温控风量、热/EMC、均衡低压、TVS瞬态、主功率热/均流及电流校准等首板测试；补齐精确采购型号、机械图与报价，再判断投板/采购和运行放行。板内200元目标仍未证明。

严格DRC仍有8条警告：板外预充边界及供应商属性差异。7组供应商属性来自此前用户日志的类别依据，当前API本轮只给聚合数，不声称旧位号列表原样有效。不得删采购属性只为消警告。[前置审查](../检查/Layout前置审查.md)区分局部落位条件和后续验证；[当前报告](../检查/本轮落实与验证.md)保存本轮证据。
'''
    (M/'原理图准备/Layout交接.md').write_text(layout,encoding='utf8')
    fan=load(M/'仿真/结果/fan_direct_supply_result.json')
    charge=load(P/'16A净充电上限验证.json')
    files=['eda工程文件/BMS-9S4P.eprj3','eda工程文件/BMS-9S4P.epro2','eda工程文件/sch/Schematic1/P1.esch2','原理图准备/BOM.csv','检查/当前施工输入.json','检查/BMS-9S4P.enet','检查/Layout器件映射.json','检查/当前原理图导航审阅.pdf','仿真/结果/fan_direct_supply_result.json','检查/16A净充电上限验证.json']
    report={'date':datetime.datetime.now().astimezone().isoformat(),'status':'已到人工Layout交接并停止；非制造、额定或实物验收','counts':{'pcbBefore':216,'mpnBefore':53,'pcbNow':212,'mpnNow':51,'CAD':213,'pins':711,'bomRows':len(rows),'totalAssemblyQty':sum(float(r['每套用量']) for r in rows)},'checks':{'savedReopenedNetlist':net,'strictDRC':{'pass':False,'errors':0,'warnings':8,'layoutWithDocumentedExceptions':True},'librarySync':load(P/'库同步校验.json'),'fanModel':'仿真/结果/fan_direct_supply_result.json','chargeModel':'16A净充电上限验证.json','pdf':load(P/'当前原理图导航审阅.json')},'changed':['U_FAN_SW/R_FAN_ILIM/C_FAN_IN1/C_FAN_OUT1 removed','J_FAN.1 FAN_5V -> VOUT_5V'],'boundaries':['无样板；参数化模型不等于模块开关模型或实物验收','Layout由人类完成，尚无PCB','风扇共享5V故障行为/温升与EMC、功率SOA、16A及整车回馈资格未通过'],'evidence':[{'path':rel,'sha256':sha(M/rel)} for rel in files]}
    (P/'本轮落实与验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (P/'本轮落实与验证.md').write_text('''# BMS本轮落实与验证

2026-10-15：**到人工Layout交接并停止。** 正式原生、BOM、连接输入和共用库已同步。板内212颗/51种归一化采购MPN；板外DPDT另有1个CAD载体，故CAD共213位号/711针。总BOM含装配/线束配套，不能与板内计数混用。

| 项目 | 本轮判定 | 证据与边界 |
|---|---|---|
| 精简落实 | PASS | 官方API删除风扇SGM2593、6.8kΩ与两颗47µF；J_FAN.1改VOUT_5V。相比上版少4颗、2种；其它高价值块保留理由见高价值器件定案。 |
| 官方保存重开 | PASS | 213位号/711针，独立预期逐针、MPN和入板标志0差异；见官方网表连接核对.json。 |
| 原生/BOM恢复性 | PASS | 发布前替换文件已入历史与调试.zip并SHA256读回；见Layout收尾/发布归档记录.json及交接文档归档记录.json。未声称旧ZIP每个历史条目全部通过testzip。 |
| 共用库同步/只读校验 | PASS | 当前BOM/原生同步后--check，见库同步校验.json。供货、精确载体及机械/电气资格另行保留。 |
| 海凌科和风扇连续预算 | 模型PASS | 22.5–37.8V输入覆盖；最低负载最差64.55mA>60mA；85°C模块降额1.2A下0.673A预算尚余0.527A。详见风扇直供验证.md。 |
| 温控/故障状态机 | 模型PASS | 温度曲线、迟滞、踢转、无效温度全速、FG故障锁存及5V恢复不自动重新许可；依赖固件实现。未验证真实风扇启动峰值或海凌科打嗝动态。 |
| 16A净充电与回馈 | 条件模型完成 | 16A上界、OCC误差及延时、4P分流和CCL余量、30W光伏与母线能量筛查；初始CCL=0。超容DC/DC响应与真实回馈功率仍未资格确认。 |
| 审阅图 | 见图形核对记录 | 当前官方PDF重框，213位号书签；局部供电、直供、PWM、门控与NC截图目视复核，见当前原理图导航审阅.json。 |
| 严格DRC | FAIL（有警告） | 0错误、8警告；预充板外接口与供应商属性差异有记录。允许按例外开始Layout，不等于ERC全绿或投板放行。 |
| Layout入口 | 可开始 | 按原理图准备/Layout交接.md分区落实功率、散热、Kelvin、模块和线束约束；局部器件在精确机械/极性核验前不得冻结落位。 |
| 实物/整车/额定/成本 | 未通过 | 无样板，无PCB及固件；60A/100A、16A、全温、EMC、回收峰值/持续时间、200元成本未获正式验收。 |

当前模型：`仿真/源码/verify_fan_direct_supply.py`（输入与当前结果同名）、`verify_charge_limit_16a.py`、`行进充电能量筛查.py`。旧SGM启动/折返SPICE仅属退出拓扑的历史，不沿用其PASS/FAIL为当前电路结论。本轮验证文件哈希和程序输出见同名JSON。

交接入口：[Layout交接](../原理图准备/Layout交接.md)、[前置审查](Layout前置审查.md)、[风扇直供](风扇直供验证.md)、[16A验证](16A净充电上限验证.md)。PCB创建、布局布线及固件留给人类，代理在此停止。
''',encoding='utf8')
    print(json.dumps({'layout':'ready for human placement','counts':report['counts'],'report':'本轮落实与验证.md'},ensure_ascii=False))

if __name__=='__main__': main()

