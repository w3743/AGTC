"""Synchronize current inventory notes; not an electrical test."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import zipfile

module = Path(__file__).resolve().parents[1]
check = module / '检查'
counts = json.loads((check / '当前BOM计数证据.json').read_text(encoding='utf-8-sig'))['counts']
if counts['perSetTotalQty'] != 481 or counts['uniqueRefs'] != 458:
    raise RuntimeError('Current inventory differs from this update')

names = ['极简性价比决策.md', '先进拓扑精简比较.md', '均衡简洁方案收敛.md', 'Layout成本与保险精简.md']
prefix = '20261014_cost_priority_ready_merged_' + datetime.datetime.now().strftime('%H%M%S')
with zipfile.ZipFile(module / '归档' / '历史与调试.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in names:
        path = check / name
        before = path.read_bytes()
        entry = f'{prefix}/BMS模块/检查/{name}'
        archive.writestr(entry, before)
        if hashlib.sha256(archive.read(entry)).digest() != hashlib.sha256(before).digest():
            raise RuntimeError(f'Archive mismatch: {name}')
        s = re.sub(r'\r+\n', '\n', before.decode('utf-8-sig'))
        if name != '极简性价比决策.md':
            s = s.replace('完整实物基准为487件、144种采购型号', '完整实物基准为481件、144种采购型号', 1)
        else:
            s = s.replace('旧共享均衡226件；', '当前共享均衡220件（取消READY前226件）；')
            s = s.replace('仅少10–22件，', '相对当前只少4–16件，')
            s = s.replace('因此MP2643即使完整只减10–22件，', '因此MP2643即使相对当前完整只减4–16件，')
            s = s.replace('本轮决定取消独立READY遥测光耦链6件，PB2排母脚留备用；', '本轮已合并取消独立READY遥测光耦链6件，PB2/P2-7留NC备用；')
            s = s.replace('以合并后的实际源计数为准。', '现BOM为481实物件、458采购位号、144种型号（不含MCU核心板），相对535件累计退出54件。')
            s = s.replace('缓冲电容的1S/2S简化仍是计算候选：1S2.7V低于LM51772 ISNS推荐2.8V；2S需重算每bank过压裕量、平均/独立限流、换接能量及净均衡速度。未闭合前保留4S2P，不计入减件。', '缓冲电容保留4S2P：1S2.7V低于LM51772 ISNS推荐2.8V；2S约4.4–4.7V窗口会增加缓冲侧电流，且低于ISNS精度的5–55V表征范围，还需容量不平衡、ESR与纹波/UV/OV角落闭合。当前不实施1S/2S，不计入减件。')
            s = s.replace('当前487件/144种', '当前481件/144种')
        path.write_text(s, encoding='utf-8')
        print(f'{name}: archived SHA256 MATCH; current note synchronized')
