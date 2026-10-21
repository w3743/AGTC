"""Refresh current handoff documents after the official native/BOM publication."""
from pathlib import Path
from collections import Counter
import csv, datetime, hashlib, importlib.util, json, zipfile
P=Path(__file__).resolve().parent; M=P.parent; R=M.parent; O=P/'Layout收尾'
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dump=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    expected=load(P/'当前施工输入.json'); pcb=[x for x in expected if x['addIntoPcb']]
    assert len(pcb)==212 and len(expected)==213
    net=load(P/'官方网表连接核对.json'); assert net['summary']['pinDifferenceCount']==0
    assert net['summary']['actualPinCount']==711
    assert load(O/'保存重开器件核对.json')['drc']==[{'type':'warn','count':8}]
    # Preserve every replaced current document, including its full byte content.
    rels=['README.md','TODO.md','项目要求.md','原理图准备/Layout交接.md','原理图准备/控制与接口设计.md',
          '检查/当前设计状态.md','检查/本轮落实与验证.md','检查/本轮落实与验证.json',
          '检查/DRC逐项核对.json','检查/DRC逐项核对.md','检查/Layout接口焊盘映射.json',
          '检查/温控调速落实验证.json','检查/海凌科电源落实验证.json','检查/保护精简落实审查.json',
          '检查/保护料号复核_清单.json','检查/保护料号复核_功率.md','检查/浪涌电压域闭合.md',
          '检查/ETA3000当前选型依据.md','检查/高价值器件定案.md']
    key=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_Layout交接文档刷新前'
    records=[]
    with zipfile.ZipFile(M/'归档/历史与调试.zip','a',compression=zipfile.ZIP_DEFLATED) as z:
        for rel in rels:
            p=M/rel
            if p.is_file():
                data=p.read_bytes(); entry=key+'/'+rel; z.writestr(entry,data)
                assert hashlib.sha256(z.read(entry)).hexdigest()==sha(p)
                records.append({'path':rel,'entry':entry,'sha256':sha(p)})
    dump(O/'交接文档归档记录.json',{'readbackSHA256Verified':True,'records':records})
    (M/'README.md').write_text('''# 9S BMS模块

当前停在**人工Layout交接点**：板内212颗、51种归一化采购MPN；原理图含1个板外DPDT载体，共213位号/711针。相比本阶段最初213颗/60种，减少1颗、9种；相比上一版216颗/53种，减少4颗、2种。PCB布局布线及固件由人类完成，本轮未创建PCB。

- 先读[项目要求](项目要求.md)和[Layout交接](原理图准备/Layout交接.md)。
- 唯一当前原生工程：[BMS-9S4P.eprj3](eda工程文件/BMS-9S4P.eprj3)；[BOM](原理图准备/BOM.csv)、[连接输入](检查/当前施工输入.json)、[器件映射](检查/Layout器件映射.json)、[当前报告](检查/本轮落实与验证.md)已同步。
- [审阅PDF](检查/当前原理图导航审阅.pdf)由当前官方导出重框并加入位号书签，原生工程仍是CAD入口。
- 海凌科5V已落实；风扇改为5V直供，保留25kHz开漏PWM、FG和温控模型。移除SGM2593及其限流电阻和两颗电容。
- 太阳能30W由独立MPPT接公共母线；电池净充电上界16A，超额回收交超容双向DC/DC。整车回收功率、能量及快响应资格尚未验证。

DRC为0错误、8条警告，严格DRC未全绿；布局例外、采购载体差异和首板验证边界见交接。当前允许开始人工Layout，不代表投板、采购、60A/100A额定、整车运行或实物验收通过。旧过程材料见`归档/历史与调试.zip`，当前模型入口见本轮报告。
''',encoding='utf8')
    status='''# BMS当前设计状态

2026-10-15：**已到人工Layout交接阶段，本轮在此停止。** 当前正式原生与BOM为212颗/51种MPN，213个CAD位号/711针；保存重开后的预期网表、MPN及入板标志一致。PCB未创建，固件未实现，无样板。

本轮删除U_FAN_SW、R_FAN_ILIM、C_FAN_IN1、C_FAN_OUT1；J_FAN pin1改接VOUT_5V。温控PWM/FG、主AFE、预充硬超时、故障锁存、隔离CAN、主动均衡与六颗主MOS保留。既有四串电阻候选已撤回，不计入现状。

| 核查 | 判定 | 边界 |
|---|---|---|
| 官方API修改、保存重开、逐针核对 | PASS | 213位号、711针、0差异，非实物验收 |
| BOM/连接源/映射/共用库 | 见本轮报告 | 按当前文件核对和同步 |
| 风扇直供/温控、电源、充电与回收模型 | 见当前模型 | 受显式输入约束，非模块开关模型/风扇电机模型 |
| 严格DRC | FAIL（仍有警告） | 0错误、8警告；板外预充边界及供应商属性差异，不改采购身份来消警告 |
| 人工Layout入口 | 可开始 | 按Layout交接约束做功率、散热、Kelvin及接口设计 |
| 投板/全功率/整车/实物验收 | 未通过 | 需完成布局与固件、首板验证、精确料号机械及报价核验 |

风扇直供取消支路选择性限流。短路时海凌科模块可能自恢复打嗝，并使共享5V掉电；模块OCP 110–200%是动作范围，不是4A峰值钳位保证。CORE_3V3也掉电时，BMS_LOCAL_OK下拉、异步清锁存、DFETOFF上拉的默认关断链成立于已读回BOTHOFF配置；独立CORE仍存活时必须由FG故障锁存撤销本地许可，BQ独立温保作为后备。首轮带载前要完成这些固件与配置门禁。

16A是电池净充电允许上界，不是OCC恒流点或全温保证；OCC code3误差早/晚角约7.92/17.47A。6.1A只是带均流、采样和0.5A源超调假设的CCL模型例子，不是默认额定。配置/校准/源响应未验证前CCL=0，不允许回馈进入电池。

详见[Layout交接](../原理图准备/Layout交接.md)、[本轮报告](本轮落实与验证.md)、[Layout前置审查](Layout前置审查.md)、[风扇直供验证](风扇直供验证.md)。
'''
    (P/'当前设计状态.md').write_text(status,encoding='utf8')
    (M/'TODO.md').write_text('''# BMS交接后待办

本轮已完成原理图精简与Layout输入交接，代理停在此处。下列由后续人工Layout、固件和首板阶段处理，不是再次申请已授权工作。

1. 人工创建PCB并检查精确MPN焊盘/朝向；建立电芯/NTC焊线焊盘、板外预充电阻接口，避免B0与PACK−旁路分流器。
2. 按60A连续/100A1s输入完成铜排、MOS均流、分流器Kelvin、热设计与短路SOA；未完成前仅受控低能调试。
3. 固件实现BQ全部保护/NTC系数/开短路/看门狗/DFETOFF BOTHOFF配置和读回；预充新鲜采样许可；风扇PWM/FG锁存、欠压/复位默认态；未通过自检不发功率许可。
4. 风扇与共享5V启动/短路/打嗝/恢复、独立CORE供电和FG未供电注入、海凌科输入滤波/热/EMC实测。禁止模块输入直接带电插拔。故障清除不能自动重合主FET。
5. 依布局寄生参数验证TVS瞬态及各电压域余量；验证ETA3000九串端部、分级EN/UV窗口与默认关闭。详细资料歧义仍保留，不能称厂家认证。
6. 完成电流校准、4P支路分配和温度降额、每个源本地CCL响应；电池净充电≤16A但初始CCL=0，合格后逐步放开。确定回馈功率/持续时间、超容功率和剩余能量；超容不足时源端减回馈并由制动承接。
7. 核实VO617A-2采购/同几何-4载体、外置DPDT真实端子、端子配对及线束；补报价/MOQ并重新核算板内200元目标。当前报价不齐，未证明达标。
8. 布局复核和首板全部必要测试通过后再决定制造及整车放行；严格DRC8警告保留分类，不称全绿。
''',encoding='utf8')
    p=M/'项目要求.md'; s=p.read_text(encoding='utf8')
    s=s.replace('本轮修改前基线：正式原生工程为 213 件/60 种归一化 MPN；此前58、52等为不同候选口径；本轮落实后216/53见第8节，不能混用。四组电阻改为六段串联虽然曾获同意，现按用户最新的总件数/种类平衡要求重新评估，尚未落实正式工程。','当前Layout交接基线为212件/51种归一化MPN（CAD另含1个板外开关载体，共213位号/711针）。本阶段初始213/60，上次正式216/53；58、52及四串电阻是历史或撤回候选，不能作为现状。')
    s+='\n## Layout交接收尾（2026-10-15）\n\n已通过官方API删除风扇SGM2593、独有6.8kΩ及两颗47µF，J_FAN直接接海凌科5V；板内212颗/51种。保留温控PWM/FG。当前到人工Layout入口停止，PCB与固件由人类完成。静态预算、敏感性时域与状态机模型有边界，风扇启动峰值、短路峰值/恢复、热/EMC、16A资格及整车回收仍须后续验证；不把模型当实物验收。当前判定与证据以检查/本轮落实与验证.md及原理图准备/Layout交接.md为准。\n'
    p.write_text(s,encoding='utf8')
    p=M/'原理图准备/控制与接口设计.md'; s=p.read_text(encoding='utf8')
    s=s.replace('SGM2593 EN直接接VOUT_5V，保留6.8kΩ支路限流，删除其MCU使能及重复下拉。J_FAN：1=FAN_5V','J_FAN直接由海凌科VOUT_5V供电；U_FAN_SW、R_FAN_ILIM及两颗专属47µF已删除。J_FAN：1=VOUT_5V')
    s=s.replace('SGM2593 EN接5V，PWM栅极下拉使复位时开漏释放、请求全速','VOUT_5V直供，PWM栅极下拉使复位时开漏释放、请求全速')
    start=s.find('风扇启动约束：SGM2593')
    if start>=0:
        end=s.find('\n',start)
        if end<0:end=len(s)
        s=s[:start]+'风扇直供约束：计划总负载约0.673A，模块额定2A；不再使用SGM折返限流模型。启动峰值未获厂家上限，模块OCP也不是瞬态电流上界。风扇供回线按≥4A持续校核并另验短路脉冲，PWM/FG回线不能承载故障回流。CORE独立存活时必须靠FG故障锁存撤销BMS_LOCAL_OK；未实现前不允许带载运行。详见检查/风扇直供验证.md。'+s[end:]
    s=s.replace('限流压力总预算约1.473A；启动电流缺少厂家上限保证，启动峰值叠加电容充电的模型在部分折返限流角落余量不足，必须首板测5V/风扇电流/FG，暂不认定启动通过。详见检查/温控调速落实验证.json与检查/海凌科电源落实验证.json。','直供启动/短路/模块打嗝须按检查/风扇直供验证.md中的受限模型核对；旧SGM限流压力及折返模型不再对应当前拓扑。首板测5V、风扇电流、FG与主FET许可，未证明启动/故障恢复实物通过。')
    s=s.replace('需物理OFF/ON复位。','本轮策略采用物理OFF/ON并重新自检后复位（工程选择，非用户固定要求）。')
    p.write_text(s,encoding='utf8')
    # Rebind hand-created PCB pad requirements to the actual native snapshot.
    p=P/'Layout接口焊盘映射.json'; d=load(p); d['baseline'].update({'component_count':212,'cad_component_count':213,'normalized_mpn_count':51,'netlist_sha256':sha(P/'BMS-9S4P.enet'),'native_p1_sha256':sha(M/'eda工程文件/sch/Schematic1/P1.esch2'),'project_uuid':'0fadff9daf47efab052134ae7df95c6e1e9996df8154a5e9d94f5e47ad92aec4'}); dump(p,d)
    p=P/'DRC逐项核对.json'; d=load(p); d['currentP1SHA256']=sha(M/'eda工程文件/sch/Schematic1/P1.esch2');d['currentOfficialResult'].update({'evidence':'Layout收尾/保存重开器件核对.json','CAD':213,'pins':711,'netlistSHA256':sha(P/'BMS-9S4P.enet')}); d['boundary']='当前官方API保存重开复跑仍8条warning、0错误；用户旧日志用于类别参考，不声称旧日志中每条位号列表全部仍是当前列表。已删除风扇四件；7组供应商属性差异不通过删除采购属性掩盖。严格DRC未通过，允许按列明例外开始人工Layout，不是制造或实物验收。';dump(p,d)
    p=P/'DRC逐项核对.md'; s=p.read_text(encoding='utf8');p.write_text('> 当前Layout基线：212件/51种；官方保存重开0错误、8警告，证据为`Layout收尾/保存重开器件核对.json`。以下用户旧日志的位号明细是历史分类依据，已删除风扇四件，不能作为当前完整警告对象清单。严格DRC未全绿。\n\n'+s,encoding='utf8')
    # Keep old analysis available but explicitly stop it masquerading as current topology.
    for name in ['温控调速落实验证.json','海凌科电源落实验证.json','保护精简落实审查.json','保护料号复核_清单.json']:
        p=P/name;d=load(p);d['currentLayoutOverride']={'status':'正式212件/51种；旧SGM相关字段仅历史','currentReport':'本轮落实与验证.md','currentFanEvidence':'风扇直供验证.md','removedRefs':['U_FAN_SW','R_FAN_ILIM','C_FAN_IN1','C_FAN_OUT1'],'fanPowerNet':'VOUT_5V','date':'2026-10-15'};dump(p,d)
    for name in ['保护料号复核_功率.md','浪涌电压域闭合.md','ETA3000当前选型依据.md','高价值器件定案.md']:
        p=P/name;s=p.read_text(encoding='utf8');p.write_text('> 当前Layout交接为212件/51种；海凌科5V直供风扇，SGM及旧自设计降压已退出。下文旧基线/旧拓扑数值须结合当前[Layout前置审查](Layout前置审查.md)、[风扇直供验证](风扇直供验证.md)及[本轮报告](本轮落实与验证.md)读取，不作为已退出器件的现行约束。\n\n'+s,encoding='utf8')
    print(json.dumps({'updatedDocuments':len(rels),'archiveReadback':'PASS','counts':[212,51,213,711]},ensure_ascii=False))

if __name__=='__main__': main()
