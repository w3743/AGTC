"""据实际原生保存和导出结果维护当前证据；不执行电气测试。"""
import json, hashlib, zipfile, csv, sys
from pathlib import Path
B=Path(__file__).resolve().parents[1]; C=B/'检查'; S=B/'原理图准备'
def read(n): return json.loads((C/n).read_text(encoding='utf-8-sig'))
native=read('当前原生器件清单.json'); export=read('当前官方工程导出.json')
parts=[p for page in native for p in page['parts']]
nofp=[p['ref'] for p in parts if p['pcb'] and (not p['footprint'] or not p['footprint'].get('uuid'))]
excluded=[p['ref'] for p in parts if not p['pcb']]
with (S/'BOM.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
if '--refresh-current' in sys.argv:
 # 仅更新本轮统计与导出证据，保留已经人工收敛的当前设计文字。
 import re
 empty=[r for r in rows if not r.get('MPN','').strip()]
 (C/'原生工程保存证据.md').write_text(f'''# 当前原生工程保存证据

日期：{export['date']}。嘉立创EDA官方API保存后读取器件清单，并导出当前工程。

- 当前导出：[BMS-9S4P.epro2](../eda工程文件/BMS-9S4P.epro2)，{export['size']:,} bytes。
- SHA256：`{export['sha256']}`。
- 本地工作工程：[BMS-9S4P.eprj2](../eda工程文件/BMS-9S4P.eprj2)，大小与SHA见[导出记录](当前官方工程导出.json)。
- 原生器件清单：{len(native)}页、{len(parts)}个part符号；外部开关拆成两个符号单元，不能按符号数直接采购。
- 入PCB且无关联封装：{len(nofp)}；不入PCB符号：{len(excluded)}。关联封装不等于焊盘已审核。
- BOM：{len(rows)}行；采购状态与位号见当前BOM。

本轮修改：[Layout原生修改保存记录](Layout原生修改保存记录.json)。全部当前器件见[原生器件清单](当前原生器件清单.json)。

以上是保存、数据读取及导出记录，未执行ERC、DRC、仿真或实物测试，不宣称电气或生产验收通过。本轮已完成原型layout交接；性能与生产验收边界见[简洁与性能收敛](均衡简洁方案收敛.md)。
''',encoding='utf-8')
 p=C/'需求覆盖与待验收.md'
 t=p.read_text(encoding='utf-8-sig')
 t=re.sub(r'日期：2026-10-\d{2}',f"日期：{export['date']}",t)
 t=re.sub(r'当前原生工程共\d+页、\d+个part符号',f'当前原生工程共{len(native)}页、{len(parts)}个part符号',t)
 p.write_text(t,encoding='utf-8')
 print('refreshed current metrics',len(native),len(parts),len(rows),'pcb without footprint',len(nofp))
 raise SystemExit(0)
old=C/'原生工程保存证据.md'
if old.exists() and '--refresh-current' not in sys.argv:
 data=old.read_bytes(); key='共享电感替换前/入口收尾/检查/原生工程保存证据.md'
 with zipfile.ZipFile(B/'归档/历史与调试.zip','a',zipfile.ZIP_DEFLATED) as z:
  if key not in z.namelist(): z.writestr(key,data)
  elif z.read(key)!=data: raise RuntimeError('archive collision')
 with zipfile.ZipFile(B/'归档/历史与调试.zip') as z:
  if hashlib.sha256(z.read(key)).digest()!=hashlib.sha256(data).digest(): raise RuntimeError('archive mismatch')
old.write_text(f'''# 当前原生工程保存证据

日期：2026-10-12。通过嘉立创EDA官方API建立、接线及保存当前页面，并使用官方工程导出API保存当前交付。

- 当前文件：[BMS-9S4P.epro2](../eda工程文件/BMS-9S4P.epro2)
- 文件大小：{export['size']:,} bytes；SHA256：`{export['sha256']}`。
- 本地工作工程：[BMS-9S4P.eprj2](../eda工程文件/BMS-9S4P.eprj2)，与官方可携带导出各保留一份；大小与SHA另见官方工程导出JSON的workingProject字段。
- 原生回读：{len(native)}页、{len(parts)}个part符号；外部开关拆成两个单元，不能据符号数直接计算采购件数。
- 入PCB且无封装的符号：{len(nofp)}，清单：{', '.join(nofp) or '无'}。
- 不入PCB的符号：{len(excluded)}，逐项用途与真实物料见Layout交接及当前原生器件清单，外置件保持真实交付边界。
- 当前BOM：{len(rows)}个行项目；采购数量与板外线束另见BOM数量列。

页面读取、完整引脚核对、保存和文件导出是本轮施工证据。本轮未执行ERC、DRC、仿真或实物测试；旧工程ERC与网表结果已归档，不适用于当前共享电感工程。关联封装存在也不等于焊盘尺寸或pad映射已核准。

证据：[当前原生器件清单](当前原生器件清单.json)、[官方工程导出](当前官方工程导出.json)、[共享电感施工记录](共享电感原生施工证据.json)、[旧均衡页面退出](旧均衡页面退出证据.json)。施工记录保留过程失败，判断当前状态须取每组最后一次成功结果。
''',encoding='utf-8')
(C/'需求覆盖与待验收.md').write_text(f'''# BMS 当前设计交接结论

日期：2026-10-12。当前原生工程共{len(native)}页、{len(parts)}个part符号，已通过官方API保存并导出。PCB与固件由人类完成；可以开始layout准备和实物封装核定，尚未生产放行。

## 当前设计覆盖

| 需求 | 当前实现 | 后续证据 |
|---|---|---|
| 9S4P EVE30P | B0…B9采样，BQ76952本地保护 | 实际电芯配组、焊接、抽头顺序及电芯批次 |
| 60A连续、100A/1s；6A日常、16A净充电 | 主FET、分流器、保险、保护阈值和充放电限制按设计输入保留 | 热、短路、脉冲、保险配合与充电器硬限流；这些电流尚非实测额定 |
| 超限由超容补充 | 经既有双向DC/DC协调，能力不足降负载 | 整车瞬态、回馈能量与故障边界；隔着DC/DC不计为母线直连电容 |
| ON本地启动，无CAN授权 | 本地自检、预充和合闸；供上位机启动 | 冷启动、母线预偏置、超时和故障恢复 |
| 两级关机 | DPDT日常OFF；MAIN/CELL/BAL维护物理断开；BQ日常保护保留 | 逐抽头静耗、USB/SWD反灌、物理断开及残能测量 |
| 每板自测温 | 12个原测点及浮地缓冲NTC；核心板和主板独立测点 | 探头安装、开短路、校准和热点响应 |
| 任意组主动均衡 | LM51772、单只22µH普通电感、9组选芯矩阵、4S2P缓冲；MA5421-ALD禁止，旧LTC/变压器退出 | 50mV≥0.5A、100mV约1A须测全周期接收平均电流，含C1↔C9及切换停顿 |
| 均衡默认关闭 | 单组选通、硬件空窗、心跳、故障锁存、独立电压/温度保护 | 失电、复位、异常选芯、门极及电感电流故障注入 |
| 辅助供电 | TPSM65610SVCGR主5V/1A；浮地B0512S-2WR3；矩阵B0512S-1WR3 | 实际负载、最小负载、输入瞬态、偏压有效电容及高温降额 |
| 缓冲残能 | 失电常闭继电器接10Ω/10W泄放 | ≥120s后实测低于1V；继电器释放与电阻热态，时间不能代替测量 |
| 约200元均衡目标，可浮动 | BOM与报价缺口已列明 | 全部板上/板外人民币报价、MOQ、交期；尚不能证明达标 |

## layout交接的具体限制

1. 主功率MOS、分流器、TVS及外置功率件的入板标志与实物封装须由人类确定；不得给外置件虚假封装。当前入PCB且无关联封装的符号数为{len(nofp)}。
2. 普通电感候选YSPI1365-220M的饱和电流、DCR和温升资料尚有冲突，采购前闭合；COMP/ILIMCOMP数值为初始值，留调试位。
3. 双向补偿、负向限流精度、预偏置启动、缓冲温度/ESR与真实切换暂态仍须验证。均衡软件默认15–55°C且ESR≤0.20Ω；低温停用或降额，不承诺全温度均衡额定。
4. 通用R/C载体规定了值和初始封装，真实采购料还需按耐压、功率、精度和DC偏压选定。封装关联回读不等于pad映射验收。
5. 人类完成当前ERC、PCB、固件、台架与装车验证；旧ERC及旧模型结果不代表当前工程通过。

完整边界见[Layout交接](../原理图准备/Layout交接.md)、[功率与预充](../原理图准备/功率与预充设计.md)、[均衡设计](../原理图准备/AFE与主动均衡设计.md)、[5V预算](5V电源预算.md)和[器件证据](共享电感器件证据.md)。
''',encoding='utf-8')
print('current handoff evidence written',len(native),len(parts),len(rows),'pcb without footprint',len(nofp))
