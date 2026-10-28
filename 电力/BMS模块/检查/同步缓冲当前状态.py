"""Reconcile current design notes with the connection source; no circuit tests."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import zipfile

module = Path(__file__).resolve().parents[1]
source = module / '原理图准备/共享电感控制保护连接.json'
rows = json.loads(source.read_text(encoding='utf-8-sig'))
refs = {r['ref']: r for r in rows}
caps = [r for r in rows if re.fullmatch(r'C_BAL_BUFFER[1-4][AB]', r['ref'])]
if len(caps) != 8 or refs['C_BAL_BUFFER4A']['pins'] != {'1': 'BAL_BUF_P', '2': 'BAL_CAP3'}:
    raise RuntimeError('Source is not the expected four-bank buffer')
sha = hashlib.sha256(source.read_bytes()).hexdigest().upper()
paths = [module / '原理图准备/共享电感编程接口.md',
         module / '检查/共享电感缓冲电容证据.md',
         module / '检查/共享电感控制实际减件方案.md']
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
old = {}
with zipfile.ZipFile(module / '归档/历史与调试.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
    for p in paths:
        raw = p.read_bytes()
        entry = '缓冲现行说明纠正_' + stamp + '/' + p.relative_to(module).as_posix()
        archive.writestr(entry, raw)
        if hashlib.sha256(archive.read(entry)).digest() != hashlib.sha256(raw).digest():
            raise RuntimeError('Archive readback mismatch')
        old[p.name] = re.sub(r'\r+\n', '\n', raw.decode('utf-8-sig'))

s = old['共享电感编程接口.md']
start = s.index('## 2026-10-14 缓冲拓扑更新')
end = s.index('## 主控接口', start)
s = s[:start] + '''## 2026-10-14 当前缓冲拓扑：4S2P

现行连接源为八只FH2R7L106M-N1025，四组串联、每组两只并联；保留BAL_CAP3节点、四只330Ω均压电阻及四组独立硬件过压检测。3S2P减件曾尝试，因容量不平衡角落先触发单bank过压而撤回，不能按六只电容编程或采购。低/高缓冲端点仍为5.381322–6.230000V，27mΩ平均电流取样、独立Hall OCP和两次换接空窗保持。现行准备源尚未同步目标单页原生工程，完整周期净转移电流尚未验收；见[当前缓冲证据](../检查/共享电感缓冲电容证据.md)。

''' + s[end:]
paths[0].write_text(s, encoding='utf-8')

s = old['共享电感缓冲电容证据.md']
marker = '## 历史4S2P结论（不代表当前连接源）'
tail = s[s.index(marker):].replace(marker, '## 先前4S2P研究数据：仅作追溯', 1)
front = f'''# 共享电感缓冲电容证据（当前4S2P）

同步日期：2026-10-14。以当前连接源读回为准；此次修正现行说明，不改电路、BOM或原生工程，不运行测试或仿真。

## 当前结构与边界

- C_BAL_BUFFER1A/B…4A/B共八只FH2R7L106M-N1025，四个串联bank、每bank两只并联；节点依次为BAL_N、BAL_CAP1、BAL_CAP2、BAL_CAP3、BAL_BUF_P。
- 每bank一只330Ω均压电阻，共四只；四bank独立过压分压及比较器通道保留。3S2P尝试已撤回，不能计为现行减件。
- 标称等效5F，按单只容量±20%折算4–6F；按先前署名规格书的初始DC ESR最大65mΩ/只，四串双并联折算总0.13Ω。这些是器件数据折算，不是装机测量。
- 低/高反馈目标5.381322–6.230000V；27mΩ平均取样名义1.85185A，Hall独立硬停止名义±2.4185A，均为缓冲侧电流。不能当作接收电芯的全周期净转移电流。
- 3S2P在一弱bank16F、其余两bank24F时，忽略ESR的6.23V总压分配已使弱bank约2.67V；原逐bank2.50/2.55V保护会提前停止。撤回该减件，保留四bank，不增加数小时预均压启动过程。
- 现行源共{len(rows)}条器件连接；SHA256：`{sha}`。原生单页工程尚未生成，均衡净电流、全路径阻抗与热性能仍未验收。

## 先前模型的适用限制

下文保留先前4S2P资料与条件模型供追溯。恢复相同bank数量不表示先前结果已覆盖现行QXV矩阵、共用抽头熔丝/座/线阻和切换时序。既有“任意两组50mV净接收≥0.5A、100mV约1A”仍是比较基准；不得用模型中的缓冲侧电流或未覆盖现行全路径的数值宣布满足。

---

'''
paths[1].write_text(front + tail, encoding='utf-8')

paths[2].write_text(f'''# 共享电感控制与缓冲减件落实记录

日期：2026-10-14。本页只列当前状态；3S2P尝试的详细原记录已保存于归档/历史与调试.zip。

## 已实施与已撤回

| 动作 | 当前状态 | 数量影响 |
|---|---|---:|
| 4S2P改为3S2P | 已撤回，恢复八只电容、四只均压电阻及第4bank保护网络 | 不计减件 |
| 删除独立BAL_READY遥测链 | 已合并；PB2/P2-7留NC，本地PG/FAULT_ARMED仍禁止EN | −6件 |

当前共享均衡为矩阵33件、功率级41件、控制保护143件、泄放3件，共220件；完整整套BOM为481件、458采购位号、144种型号（不含外部MCU核心板）。

## 保留理由与证据边界

- 3S2P在16F/24F/24F容量角落的弱bank已接近额定上限并超过原保护阈值；恢复4S2P，避免额外预均压启动流程。
- READY删除的是MCU独立遥测链，未删除本地PG/故障许可硬关断。I²C回读与Hall观察不能代替已删除的独立PG/armed状态证明。
- 浮地测温、独立过流、选中电芯UV/OV、I²C超时、换接空窗及故障恢复硬件仍在。是否存在进一步重复，由本轮完整保护路径审视决定；不因器件数量高而直接删除。
- 当前控制保护源{len(rows)}项，SHA256：`{sha}`。现行缓冲结构见[缓冲证据](共享电感缓冲电容证据.md)，编程接口见[接口合同](../原理图准备/共享电感编程接口.md)。
- 源文件及数量更新不等于原生原理图同步、电气/热验收或成本达标；本次没有运行测试、仿真、ERC或DRC。
''', encoding='utf-8')
for p in paths:
    print(p.name + ': archive SHA256 MATCH; current 4S2P note reconciled')
