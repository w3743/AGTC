"""Refresh current status text, preserving old bytes in the module archive."""
from pathlib import Path
import datetime, hashlib, json, zipfile

module = Path(__file__).resolve().parents[1]
updates = {
    '检查/当前BOM边界清单.md': [('615行、615个唯一位号、659个实物件', '598行、598个唯一位号、642个实物件'), ('1,849条', '1,813条'), ('数据状态已包含共用5V风扇支路。', '已包含共用5V风扇、PACK取样简化、NTC_CELL5复用AFE及3S2P缓冲；候选SSR矩阵尚未计入。')],
    'README.md': [('8只10F电容组成4S2P缓冲', '6只10F电容组成3S2P缓冲'), ('当前BOM为615个位号、659个实物件', '当前BOM为598个位号、642个实物件'), ('当前风扇源与BOM已同步；原生工程尚未读回这两项变更', 'PACK取样简化、电芯5测温复用AFE和3S2P另净减17件，累计较681件基准减少39件；连接源与BOM已同步，原生工程尚未同步这些变更')],
    '检查/需求覆盖与待验收.md': [('日期：2026-10-13。当前原生工程共44页、605个part符号，已通过官方API实施、保存并导出。当前状态是原型layout入口与文件交接；尚未执行ERC/DRC、仿真、实测、封装验收或成本验收，也未生产放行。PCB与固件由人类完成。', '日期：2026-10-14。当前连接源持续精简，BOM为642件、598个位号、148种已选型号（不含外置MCU），风扇另待定。旧44页原生工程不代表当前连接；单页eprj3尚未生成并读回，layout入口未释放。PCB与固件由人类完成；本轮未执行ERC/DRC、仿真或实测。'), ('九组选芯矩阵、4S2P缓冲', '九组选芯矩阵、3S2P缓冲'), ('## 本轮layout入口定稿', '## 当前工程同步状态'), ('新增B0域双向过流锁存、Hall改缓冲侧、复用HCT08空门、反馈支路2.7MΩ、12V双向TVS、两侧47µF/35V本地续流储能；当前BOM和原生工程原位更新。设计取舍见[均衡简洁定稿](均衡简洁方案收敛.md)。可以开始原型layout；没有执行ERC、PCB、固件或实测，报价/供货和生产放行仍未闭合。', '当前源保留双向过流锁存、空窗、看门狗、逐bank/总压和温度保护。国产SSR矩阵及选芯端共用Hall仍属待落实候选；旧Hall缓冲侧方案仅描述当前未替换源。最终选型/数量先报告，再生成单页eprj3并读回。此前“可以开始原型layout”的结论已撤回。')],
}
key = '当前数量与入口澄清_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
changed = []
with zipfile.ZipFile(module/'归档/历史与调试.zip', 'a', compression=zipfile.ZIP_DEFLATED) as archive:
    for rel, edits in updates.items():
        path = module/rel
        old = path.read_bytes()
        body = old.decode('utf-8-sig')
        for before, after in edits:
            if before not in body and after in body:
                continue
            if before not in body:
                if before == '九组选芯矩阵、4S2P缓冲':
                    before = '9组选芯矩阵、4S2P缓冲'
                    after = '9组选芯矩阵、3S2P缓冲'
                if before not in body:
                    raise RuntimeError(f'expected current text missing: {rel}: {before[:40]}')
            body = body.replace(before, after)
        name = key+'/'+rel
        archive.writestr(name, old)
        if hashlib.sha256(archive.read(name)).digest() != hashlib.sha256(old).digest():
            raise RuntimeError('archive readback mismatch')
        path.write_text(body, encoding='utf-8')
        changed.append(rel)
print(json.dumps({'changed': changed, 'archivePrefix':key}, ensure_ascii=False))
