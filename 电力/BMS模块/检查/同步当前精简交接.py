"""Update current handoff prose from current inventory; no electrical tests."""
from pathlib import Path
import json, csv, datetime, hashlib, zipfile

M = Path(__file__).resolve().parents[1]
C = M / '检查'
P = M / '原理图准备'
counts = json.loads((C / '当前BOM计数证据.json').read_text(encoding='utf-8-sig'))
n = counts['counts']
qty, refs = n['perSetTotalQty'], n['uniqueRefs']
types = n['normalizedSelectedMpnTypesExcludingCoreBoard']
rows = list(csv.DictReader((P / 'BOM.csv').open(encoding='utf-8-sig')))
allparts = {r['ref']:r for f in P.glob('*连接.json') if '候选' not in f.name for r in json.loads(f.read_text(encoding='utf-8-sig')) if isinstance(r,dict) and 'ref' in r}
assert sum(r['ref'].startswith('C_BAL_BUFFER') for r in allparts.values()) == 8
assert allparts['U_BAL_M01_P']['mpn'] == 'QXV255-CuH-S'
stage = '单页eprj3尚未完成原生读回，layout入口未释放。'
record = C / '单页eprj3施工记录.json'
if record.exists():
    native = json.loads(record.read_text(encoding='utf-8-sig'))
    if native.get('status') == 'created' and native.get('bomCounts',{}).get('bomSha256') == counts['bomSha256']:
        stage = '单页eprj3已完成原生结构读回，可交人类开始原型layout；电气性能仍须首板确认。'
summary = f'完整板内外{qty}件、{refs}个采购位号、{types}种已选采购型号（不含外置MCU）；外置MCU另1种，外接风扇另1种待选。'
fbtop = allparts['R_BAL_FB_TOP']['mpn']
fbextra = allparts['R_BAL_FB_EXTRA']['mpn']
fuse = allparts['F_BAL_TAP_B1']['mpn']
fan = allparts['U_FAN_SW']['mpn']

def save(rel, text):
    p = M / rel
    if p.exists():
        b = p.read_bytes()
        key = '当前交接同步_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '/' + rel.replace('\\','/')
        with zipfile.ZipFile(M/'归档/历史与调试.zip','a',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(key,b)
            assert hashlib.sha256(z.read(key)).digest() == hashlib.sha256(b).digest()
    p.write_text(text.strip()+'\n',encoding='utf-8')

save('检查/当前BOM边界清单.md', f'''# 当前BOM边界清单

2026-10-14。{summary} 当前数量以[计数证据](当前BOM计数证据.json)和[完整BOM](../原理图准备/BOM.csv)为准，含线束触点、保险座、配套插头及外置件；不把采购位号数当作实物数量。

{stage}

## 已收敛的选型

| 模块 | 当前选型及数量 |
|---|---|
| 本地电芯保护 | BQ7695202PFBR×1，9S采样，TS2悬空，NTC_CELL5复用CFETOFF |
| 主开关/分流器 | IPT015N10N5ATMA1×6；PSG4NTEBL500F×1，四端Kelvin |
| 5V与风扇 | TPSM65620SVCGR×1，{fan}×1，风扇5V连续≤0.5A/启动≤0.8A |
| 选芯 | QXV255-CuH-S×18，ULN2003LVDR×2，XL74HC4514D×1，矩阵33件 |
| 均衡功率级 | LM51772RHAR×1，WSF60N06×4，SRP1265A-220M×1，普通电感 |
| 缓冲及检测 | FH2R7L106M-N1025×8，4S2P；ACS724LLCTR-05AB-T×1，放在选中电芯端 |
| 浮地辅助 | B0512S-2WR3×1，620Ω最低负载；独立矩阵12V支路已删除 |
| 维护 | CELL/BAL独立抽头；BAL每根B1…B9使用{fuse}，主功率及全部采样/均衡支路可物理断开 |
| 外置MCU | WeAct Normal STM32G431CBU6，原理图只保留两组排母 |

## 采购与性能边界

均衡约200元目标允许浮动，完整人民币报价、MOQ和交期未闭合。风扇机械尺寸和风量由人类选定。完整周期净电流见[周期计算](QXV255_4S2P周期净流计算.md)：效率、持续SSR阻抗和温升是设计输入，不能作为实测额定。板载贴装数仍取决于热点探头及外置功率电阻位置，不给出虚假的固定贴装数量。

本轮未进行ERC、DRC、电路仿真或实物验收。''')

save('原理图准备/AFE与主动均衡设计.md', f'''# 9S监测与共享电感主动均衡

2026-10-14。BQ76952监测保护；一套LM51772四开关双向功率级、22µH普通电感、国产光继电器选芯及4S2P缓冲。MA5421-ALD禁止使用，旧LTC3300与定制变压器连接已退出。

## 监测和维护

B0…B9供AFE，BAL_B0…BAL_B9供均衡，使用独立抽头线束及保险。板上不得短接两域B0或绕过主分流器。日常OFF保留AFE保护；长期仓储需要MAIN/CELL/BAL物理断开，不能把面板OFF当作仓储隔离。ON由本地自检与预充建立输出，不等待CAN授权。

BQ TS2脚22悬空；NTC_CELL5接CFETOFF脚29及滤波，CFETOFF配置0x92FA=0x07；DFETOFF仍配置0x92FB=0x42，承担本地总关断。具体寄存器和阈值以编程交接为准。

## 选芯与能量转移

九组每组两颗QXV255-CuH-S，功率脚4/6双向连接，3/5不接。正端公共节点BAL_CELL_POS经过Hall到BAL_P，负端到BAL_N。每组两输入LED串联，经220Ω/1206与ULN2003LV驱动；XL74HC4514D正逻辑译码，INH默认上拉全关，地址9…15空档。矩阵33件，原独立12V门极偏置4件删除；浮地B0512S-2WR3保留。

选择源组充缓冲、停止转换并确认零流、换目标组反向放缓冲；任意两组复用同一功率级，不经相邻八级传递。30ms硬件空窗、10ms稳定等待、独立心跳和故障锁存保留。QXV导通1s内Ron表值不是最长导通时间，周期分段只作为敏感度分析，不能用短暂停顿宣称获得热保证。

## 功率级和缓冲

LM51772脚36 VIN=BAL_P，脚40 BIAS=独立12V；脚18 VOUT/20 ISNSP=BAL_VOUT_PWR，脚19 ISNSN=BAL_BUF_P。27mΩ分流器位于缓冲侧，Hall位于电芯侧，两者电流口径不同。I²C经ISO1640隔离、地址0x6A，初始100kHz。10mΩ峰值取样、22µH及200kHz为初始设计值，COMP/ILIMCOMP保留人工调整空间。

四组串联、每组两颗10F/2.7V电容，共8颗，等效4–6F；初始总ESR预算0.13Ω。每bank静态均压和独立过压保护保留，3S2P因容量容差过压撤回。当前反馈料号：上臂{fbtop}，阶段并联{fbextra}，下臂100kΩ。空缓冲首次建立电压时ISNS共模不足5V，不套用平均电流精度，由峰值限制及软件监控完成初次建立。

正常电芯阶段目标2.9A；3V参与时≤1A降额。Hall独立OCP采用30.1k/100k门限，名义±3.36A，误差预算约2.98–3.85A；最高值仅故障瞬态压力，不能用作持续许可。普通停止先关闭CONV_EN并回读、保持选芯、确认零流再换址；OCP先停EN并锁存，避免带电流立即切选芯。失电和其他硬故障的本地储能、体二极管和TVS续流仍需首板确认。

## 指标与交接

默认静置、全部组≥3.8V、9组温度15–40°C；任意两组含C1↔C9，50mV净接收≥0.5A、100mV约1A。可配置至3V降额参与，2.5V电芯最低边界不等于正常均衡窗口。所有指标按含换接等待的完整周期平均接收电流计算。详见[周期计算](../检查/QXV255_4S2P周期净流计算.md)。持续SSR阻抗、实际效率、ESR、双向补偿、限流误差、瞬态及温升须首板确认，不能以计算替代验收。

每板保留自测温；浮地NTC只提供硬件状态，没有MCU数值遥测。取消加热，只保留风冷，BMS控制风扇。{summary} {stage}

原厂依据：[LM51772](https://www.ti.com/lit/ds/symlink/lm51772.pdf)、[QXV255-CuH-S](https://qxmicro.com/upfiles/products/202505/ZBKM0912HBU7.pdf)、[ACS724](https://www.allegromicro.com/-/media/files/datasheets/acs724-datasheet.pdf)。''')

save('检查/需求覆盖与待验收.md', f'''# BMS当前设计交接结论

2026-10-14。{summary} {stage} PCB与固件由人类完成。

| 需求 | 当前电路设计 | 人类后续确认 |
|---|---|---|
| 9S4P、60A/100A 1s、6A/16A净充电 | AFE、本地保护、主FET及分流器按这些设计输入选择 | 铜厚、温升、脉冲、短路和保险配合；尚非实测额定 |
| 超容补充 | 外部双向DC/DC协调，本地硬保护 | 整车瞬态和回馈，不计为母线直连电容 |
| 无CAN开机授权 | 本地自检、预充及合闸供上位机启动 | 冷启动、预偏置、超时、故障恢复 |
| 两级关机 | 日常OFF保留AFE；仓储物理断开MAIN/CELL/BAL | 逐抽头漏电、反灌、隔离和残能 |
| 每板测温、取消加热 | 主板/MCU核心板/浮地保护与9组电芯测温 | 安装、开短路、校准、热滞后 |
| 任意组主动均衡 | 单电感、18国产SSR、4S2P；保留原净转移指标 | 按周期计算边界确认持续阻抗、效率、ESR、全周期净电流 |
| 必要硬保护 | 默认关闭、互斥空窗、独立心跳、锁存、单bank/总压/温度/OCP | Gate、电流、残压、失电及恢复；旧ERC不代表当前通过 |
| 5V风扇由BMS控制 | 主5V/2A共享，{fan}限流与默认下拉 | 风扇5V≤0.5A/启动≤0.8A；5V总限流压力约1.87A，需确认热与瞬态 |
| 单物理板、单张图、MCU例外 | MCU只画P1/P2排母，外置/机械采购项保留页内注记 | 原生读回和人工layout |
| 缓冲残能 | 失电常闭继电器、10Ω/10W | ≥120s后实测BUF<1V，另测BUS残压 |
| 均衡约200元可浮动 | 已保留完整采购数量 | 全人民币报价与MOQ未闭合，不宣称预算达标 |

当前源与[BOM](../原理图准备/BOM.csv)、[Layout交接](../原理图准备/Layout交接.md)是电气设计依据。符号、封装关联和数量读回仅为工程结构证据，不代表ERC、焊盘、热设计或实物验收。''')

print(summary,stage)
