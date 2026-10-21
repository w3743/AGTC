"""Reconcile current handoff documents after the approved source simplification."""
from pathlib import Path
import datetime, hashlib, json, re, zipfile

module = Path(__file__).resolve().parents[1]
names = ['AFE与主动均衡设计.md', '共享电感编程接口.md', '控制与接口设计.md',
         '编程交接.md', '需求与系统边界.md', 'Layout交接.md']
paths = [module/'原理图准备'/n for n in names]
paths += [module/'README.md', module/'TODO.md', module/'检查/极简性价比决策.md']
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
with zipfile.ZipFile(module/'归档/历史与调试.zip', 'a', compression=zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        raw = p.read_bytes()
        key = '温度精简交接前_'+stamp+'/'+p.relative_to(module).as_posix()
        z.writestr(key, raw)
        if hashlib.sha256(z.read(key)).digest() != hashlib.sha256(raw).digest():
            raise RuntimeError('Archive readback mismatch: '+str(p))

current = ('当前温度配置为11只B0域NTC：9组电芯、MOS热点、BMS板上缓冲电容热点。'
           '`NTC_BOARD`兼顾板温与缓冲温度，MCU通过原ADC支路读取；'
           '同一传感节点接已有B0域TLV1704两个空余通道，过冷、过热、开短路均进入现有均衡故障锁存。'
           '硬件名义冷停约10.3°C、热停约54.5°C，元件角落的单Beta估算约9.5–11°C、53.3–55.6°C，尚非实测门限。'
           '软件均衡许可同时要求9组电芯及缓冲热点15–40°C。'
           '浮地专用NTC及其网络已删除；浮地PG/FAULT_ARMED仍本地硬停且无独立READY遥测。'
           'FAULT传感器位图bit12保留并置0；I²C读回与电流观察不能证明浮地硬件许可。'
           '核心板改用STM32G431内部温度传感器作自身芯片温度参考，需人类固件使用TS_CAL1/TS_CAL2和VREF校准；'
           '芯片温度不等于PCB所有热点，P2-12/PC4留NC备用。'
           '细节见[第二轮减件审视](../检查/均衡控制第二轮减件审视.md)，原生工程尚待同步。')

for p in paths[:6]:
    s = re.sub(r'\r+\n', '\n', p.read_text(encoding='utf-8-sig'))
    lines = s.splitlines()
    out = []
    for line in lines:
        # Replace obsolete dedicated float-temperature paragraphs, not unrelated safety text.
        if ('浮地`NTC_BAL_FLOAT`' in line or line.startswith('浮地NTC_BAL_FLOAT')
                or line.startswith('- `NTC_BAL_FLOAT`只接浮地硬件')):
            line = current
        elif line.startswith('| 浮地缓冲电容热点 |'):
            line = '| BMS板上缓冲电容热点 | `NTC_BOARD`，MCU ADC与B0硬件温窗共用 | 均衡硬停止与板温记录 | 不接BAL_N；名义冷/热停10.3/54.5°C，软件许可15–40°C |'
        elif line.startswith('原B0域12个测点采用'):
            line = '现B0域11只NTC采用Vishay `NTCLE100E3103GB0`（10kΩ，R25±2%，B25/85=3977K±0.75%）。九组电芯各一只，另测MOS热点和BMS缓冲热点。核心板用MCU内部温感参考自身芯片温度。单点不能代表每颗电芯或整板所有热点；传感器须电气绝缘，安装位置和热延迟待首板确认。'
        elif line.startswith('| WeAct CORE 板温关断 |'):
            line = '| WeAct CORE芯片温度关断参考 | 60°C候选 | 默认值需结合内部传感器校准与核心板温升确认；采样无效即OFF，不能当作板面60°C门限 |'
        elif line.startswith('- 定义12个B0域温度测点'):
            line = '- 温度通道共12个：11只B0域NTC加核心板MCU内部温感参考；不再设置浮地NTC。'
        line = line.replace('| P2-12 | PC4，ADC1_IN5 | NTC_CORE_ADC | WeAct 核心板自身温度 |',
                            '| P2-12 | PC4，ADC1_IN5 | NC | 备用；核心板自身温度改用MCU内部传感器参考 |')
        line = line.replace('12个B0域温度测点', '11个B0域NTC测点及MCU内部温感参考')
        line = line.replace('确认第13个`NTC_BAL_FLOAT`由浮地硬件保护监视；独立READY遥测已删除，该温度不接MCU ADC，本地PG/FAULT_ARMED也无单独MCU状态反馈。',
                            '确认NTC_BOARD缓冲热点读数有效且其硬件温窗未触发；本地PG/FAULT_ARMED无单独MCU状态反馈。')
        line = line.replace('全部 12 个 B0 域 NTC 测点有效', '全部11个B0域NTC及内部温感参考有效')
        line = line.replace('浮地均衡NTC不输出原始温度，独立READY遥测已删除', '缓冲温度由NTC_BOARD读取；独立READY遥测已删除')
        line = line.replace('NTC_BAL_FLOAT只有浮地硬件保护状态，没有MCU数值温度', '缓冲热点由NTC_BOARD提供数值温度和独立硬停')
        line = line.replace('`NTC_BAL_FLOAT`没有MCU温度值', '缓冲热点由NTC_BOARD读取；CORE温度为MCU芯片参考')
        line = line.replace('`NTC_BAL_FLOAT`只有浮地硬件保护状态，没有MCU数值读数，不得虚构温度值', 'NTC_BOARD缓冲热点及内部MCU温度可遥测，内部温度不能虚构为整板温度')
        line = line.replace('温度与温度探头开短路检测。', '；温度与探头开短路硬停改由B0域共用NTC_BOARD完成。')
        line = line.replace('**MCU ADC 六路外围：**对 PA1–PA4、PB0、PC4 每路', '**MCU ADC五路外围：**对PA1–PA4、PB0每路')
        line = line.replace('CORE_3V3 — R_PU 10.0 kΩ 1%', 'CORE_3V3 — R_PU 10.0 kΩ 0.1%')
        line = line.replace('`RC0603FR-0710KL`（10 kΩ）', '`RT0603BRD0710KL`（10 kΩ、0.1%）')
        line = line.replace('其余四组电芯温度及两个 PCB 自身温度依赖 MCU ADC，共六路。MCU 挂死时这六路没有独立的 BQ 温度保护',
                            '其余四组电芯温度和NTC_BOARD经MCU ADC，共五路；内部温感参考另采样。MCU挂死时四组电芯温度没有独立BQ通道；NTC_BOARD均衡温窗另由B0比较器硬停')
        line = line.replace('将两脚映射为 NTC_CORE/NTC_BOARD', '将PB0映射为NTC_BOARD；PC4留NC备用')
        line = line.replace('MAIN/CORE板温', 'BMS板温/MOS温度或CORE芯片温度参考')
        line = line.replace('MAIN与CORE板温', 'BMS板温、MOS温度及CORE芯片温度参考')
        out.append(line)
    s = '\n'.join(out)+'\n'
    if p.name == 'Layout交接.md':
        s += '\n## 当前温度复用变更\n\n'+current+'\n'
    s = re.sub(r'\n{3,}', '\n\n', s)
    p.write_text(s, encoding='utf-8')

evidence = json.loads((module/'检查/当前BOM计数证据.json').read_text(encoding='utf-8-sig'))
c=evidence['counts']; qty=c['perSetTotalQty']; refs=c['uniqueRefs']; models=c['normalizedSelectedMpnTypesExcludingCoreBoard']
if qty != 469:
    raise RuntimeError('Unexpected count; review before updating current entry')
count=f'{qty}个实物件、{refs}个采购位号、{models}种归一化已选型号（不含MCU核心板；含核心板{models+1}种）'
readme = f'''# 9S4P BMS 当前入口

2026-10-14：当前处于电气设计精简阶段，尚未释放layout。目标为一块BMS物理PCB、一张原理图；MCU核心板只画排母。人类负责PCB与固件。

## 当前文件

- [TODO](TODO.md)
- [BOM](原理图准备/BOM.csv) · [连接表](原理图准备/连接表.csv)
- [需求与系统边界](原理图准备/需求与系统边界.md)
- [编程交接](原理图准备/编程交接.md) · [Layout交接](原理图准备/Layout交接.md)
- [性价比决策](检查/极简性价比决策.md) · [计数证据](检查/当前BOM计数证据.json)
- 计划原生入口：`eda工程文件/BMS-9S4P/BMS-9S4P.eprj3`，尚待生成并读回；旧native不是当前连接证明。

## 当前设计与数量

9S4P EVE 30P；60A持续、100A/1s、6A日常充电、16A最大净充电为设计输入。BQ76952负责独立电芯保护。ON本地自检与预充建立输出，不等待CAN。日常OFF保留保护；仓储维护需主回路及所有电芯取电支路物理断开。

当前源/BOM为**{count}**，包括板内外配套与线束。相对535件已退出66件；本轮温度复用、比较器合并和核心板温感改用内部参考实际退出12件。FAN_EXT型号仍待机械选定，完整人民币采购成本尚未证明。

主动均衡当前源保留LM51772、22µH电感、选芯矩阵、8只10F电容4S2P缓冲。均衡缓冲与车辆超容是不同回路。任意两组50mV净接收≥0.5A、100mV约1A仍为比较基准，当前纸面方案尚未证明完整周期达标。MA5421-ALD禁用；MP2643相邻方案未采用。

温度配置为11只B0域NTC加MCU内部温感参考。NTC_BOARD置于缓冲热点，兼顾BMS板温、数值遥测和独立均衡硬停；MOS热点另保留NTC_MOS。核心板内部温感反映芯片温度，不能等同每个PCB热点。软件均衡要求9组电芯及缓冲热点15–40°C，硬件窗名义约10.3–54.5°C。所有阈值仍需安装校准。

只保留散热，取消加热；5V风扇由BMS本地控制。塑料箱通风、防溅、风量和60A温升需人类机械/首板确认；铝箔只能作可选均热层。主5V国产降压替代、抽头熔丝和预充负载联锁仍在有限收敛，不把候选计作已采用或已节省成本。

## 交付边界

本轮修改的是连接源和交接材料，没有运行ERC、DRC、仿真或实物测试。预充下游负载、完整均衡性能与采购成本仍未闭合。源和数量冻结、先向用户报告后再生成eprj3；人类随后layout。历史材料保存在[归档](归档/历史与调试.zip)。
'''
(module/'README.md').write_text(readme, encoding='utf-8')

todo = f'''# BMS 当前 TODO

日期：2026-10-14。目标按用户要求压低整套成本、实际器件数与采购复杂度。保护与两级关机保留；原任意两组50mV≥0.5A、100mV约1A净均衡指标保留为比较基准，任何速度折中必须明确说明。

## 已完成

- [x] AFE与均衡共用保险后的九路抽头，删除重复线束；主功率正端独立，不走抽头保险。
- [x] 删除6件独立READY遥测链，本地PG与EN硬停保留。
- [x] TI定时器触发修正：pin2=RUN健康心跳，pin10=CS_N换接空窗，不增加数量。
- [x] 保留4S2P均衡缓冲，退出未闭合的3S/2S减容尝试。
- [x] NTC_BOARD复用数值测温与均衡温窗，B0比较器空闲通道复用；四bank保护归并到两颗浮地TLV1704。核心板改内部温感参考，删除原NTC支路。
- [x] 从实际源重新合并BOM/连接表：**{count}**；相对535件净减少66件。本轮没有测试或原生工程生成。

## 持续完成

1. [ ] 闭合带负载预充：优先复用现有LOAD_POWER_VALID，处理FOC内置Buck自动启动负载，区分SAFE电源与Buck输入功耗；禁止用额外复杂电路堆叠掩盖问题。
2. [ ] 主5V按整套成本比较国产SGM61720及现TPSM65620；复用现监督器，独立PG先保留，USB外供时反灌不明的PG合并不实施。
3. [ ] 九路源端抽头保险与座取得直流额定、尺寸配合和大陆小批报价；国产替代满足37.8V以上分断边界后才改源。
4. [ ] 均衡高价项优先处理18只QXV255；2A开关不能代替2.9A通路。MP2643完整约204–216件，远端性能、断抽头及完整成本未闭合，暂不替换。
5. [ ] 冻结实际连接源，报告全量型号/实物数与完整成本及待询价项，再生成并读回单页eprj3。PCB和编程由人类完成。

完整成本尚未证明，FAN_EXT型号待定，当前未释放layout。资料入口见[README](README.md)。
'''
(module/'TODO.md').write_text(todo,encoding='utf-8')

p=module/'检查/极简性价比决策.md'
s=re.sub(r'\r+\n','\n',p.read_text(encoding='utf-8-sig'))
s=s.replace('目前最少9件候选（两块FOC合计），未计板外分支配套', '现报告为新增13件候选（BMS3件、两块FOC共10件），未实施；优先复用现有许可接口减少重复光耦')
s=s.replace('当前主均衡方案仍为220件，完整整套仍为481件。上述有限筛选没有新增可立即合并的减件或已证实节省金额；这不是已达到性价比最优的结论。',
            f'当前主均衡方案212件，完整整套{qty}件。温度和比较器复用已实际净减12件；没有据此宣布已达到性价比最优或采购成本目标。')
s=s.replace('当前481件/144种（不含外部MCU核心板）是已合并基准', f'当前{qty}件/{models}种（不含外部MCU核心板）是已合并基准')
s += f'\n## 本轮已落实的复用\n\n当前为{count}。温度与比较器复用净减12件，四个bank OVP、总缓冲OV、单体UV/OV、过流及锁存均保留。NTC_BOARD移至缓冲热点；核心板内部温感仅作芯片温度参考。主5V优先按整套成本比较SGM61720，允许增加便宜外围换取昂贵模块退出；现监督器复用，USB外供反灌未闭合的PG共享暂不实施。以上为源变更及候选决策，非实物验收。\n'
p.write_text(s,encoding='utf-8')
print(count+'; source documents reconciled; archive readback SHA256 MATCH')
