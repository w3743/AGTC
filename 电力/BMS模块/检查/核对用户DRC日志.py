"""Read-only reconciliation of the user's official DRC export with current ENET."""
from pathlib import Path
import hashlib, json, re, shutil

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
SOURCE = Path(r'C:\Users\wangj\Downloads\schDrcLog_2026-10-15.txt')
DEST = HERE / '剩余验证/用户导出DRC日志.txt'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
DEST.parent.mkdir(exist_ok=True)
shutil.copyfile(SOURCE, DEST)
assert sha(SOURCE) == sha(DEST)
text = DEST.read_text(encoding='utf-8-sig')
net_path = HERE / '剩余验证/用户日志对应基线.enet'
enet = json.loads(net_path.read_text(encoding='utf-8-sig'))
by_ref = {v['props']['Designator']: v for v in enet['components'].values()}
warnings = [s for s in text.splitlines() if '[警告]' in s]
assert len(warnings) == 10
single = []
for line in warnings:
    m = re.search(r'导线 (\S+) (\S+) 是单网络', line)
    if not m: continue
    name = m[1]
    pins = [{'ref': ref, 'pin': pin} for ref, v in by_ref.items()
            for pin, p in v['pinInfoMap'].items() if p.get('net') == name]
    assert len(pins) == 1, (name, pins)
    single.append({'net': name, 'wireId': m[2], 'CADpins': pins})
assert {s['net'] for s in single} == {'MAIN_HOLD_N', 'PRE_USED_N', 'QPRE_SW'}
for s in single:
    s['disposition'] = {
        'MAIN_HOLD_N': '基线U_SEQ pin6=/Q1；现已删除未用短线与端口，设置NC，保存重开后该警告消除。',
        'PRE_USED_N': '基线U_SEQ pin8=/Q2；现已删除未用短线与端口，设置NC，保存重开后该警告消除。',
        'QPRE_SW': 'Q_PRE2 pin2漏极接板外R_PRE pin1；R_PRE为外置TE CJP300J8R2J、8.2Ω。CAD未包含外置功率电阻，因此板内单引脚属边界提示；线束/实际接线仍须验收。',
    }[s['net']]
power = json.loads((MODULE/'原理图准备/功率元件连接.json').read_text(encoding='utf-8-sig'))
rpre = next(x for x in power if x['ref'] == 'R_PRE')
assert rpre['pins']['1'] == 'QPRE_SW'
mismatch = []
for line in warnings:
    if '属性与供应商编号不匹配' not in line: continue
    refs = re.findall(r'([A-Za-z][A-Za-z0-9_]*)\(\$1I\d+\)', line)
    assert refs and all(r in by_ref for r in refs), refs
    mismatch.append(refs)
assert len(mismatch) == 7
affected = sorted({r for rs in mismatch for r in rs})
empty_value = re.findall(r'元件 (\S+) 的属性“值”内容为空', text)
info_count = sum('[信息]' in l for l in text.splitlines())
current = json.loads((HERE/'剩余验证/保存重开器件核对.json').read_text(encoding='utf8'))
assert current['drc'] == [{'type':'warn','count':8}]
seq=next(x for x in current['rows'] if x['ref']=='U_SEQ')
assert all(next(p for p in seq['pins'] if p['number']==n)['noConnected'] for n in ['6','8'])
result = {
    'source': {'path': str(SOURCE), 'sha256': sha(SOURCE), 'copiedLog': str(DEST.relative_to(MODULE)), 'copySHA256': sha(DEST)},
    'logBaselineNetlistSHA256': sha(net_path), 'nativeEprj3SHA256': sha(MODULE/'eda工程文件/BMS-9S4P.eprj3'),
    'currentP1SHA256': sha(MODULE/'eda工程文件/sch/Schematic1/P1.esch2'),
    'currentOfficialResult': {'fatal':0,'errors':0,'warnings':8,'strictPass':False,'closedWarnings':['MAIN_HOLD_N','PRE_USED_N'],'remaining':'QPRE_SW板外边界1项与供应商属性7组'},
    'officialResult': {'fatal': 0, 'errors': 0, 'warnings': 10, 'information': 230, 'strictPass': False},
    'checks': {'all10WarningsClassified': True, 'threeSingleNetsMatchCurrentNetlist': True, 'externalPrechargeConnectionDeclared': True, 'supplierWarningRefsExistInCurrentNetlist': True},
    'singlePinWarnings': single,
    'supplierAttributeWarningGroups': mismatch,
    'supplierAffectedRefs': affected,
    'supplierDisposition': '七组共%d个位号：当前ENET保留采购MPN和供应商编号，同时使用CAD载体及自定义Name/属性。警告不能直接当成开短路，也不能全部当成无害；自动标准化可能覆盖已修订的引脚/封装，未执行。已有针网一致性核对只证明实现与连接源一致；实体封装、受控厂家资料和46项载体身份差异仍按共用库逐项验证。' % len(affected),
    'information': {'logInformationLinesIncludingStartEnd': info_count, 'emptyValueCount': len(empty_value), 'emptyValueRefs': empty_value, 'disposition': '值为空/含下划线位号为信息级；Name和精确MPN已存在。未为清空日志而批量改位号或替换库器件。'},
    'boundary': '用户日志无原生文件SHA；以217位号和警告对象与保留的基线ENET吻合建立关联，非加密证明日志由该确切字节版本生成。本轮NC修订后官方API保存重开复跑8条warning。eprj3是描述符，必须同时核对P1文件SHA；没有样板，不等于电气或制造验收。',
}
(HERE/'DRC逐项核对.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
lines = ['# DRC逐项核对', '', '用户导出于2026-10-15 14:15:35：**致命0、错误0、警告10、信息230**。本轮修订后官方API保存重开：**0错误、8条警告**，严格DRC仍未通过。', '', '| 原日志警告 | 当前处置 |', '|---|---|']
for s in single: lines.append('| '+s['net']+' | '+s['disposition']+' |')
lines += ['', result['supplierDisposition'], '', 'Value为空共%d个位号；非标准位号等为信息级。精确型号和Name不为空并不能代替封装验收。'%len(empty_value), '', result['boundary'], '', '完整警告对象与SHA256见同名JSON；原始导出已按SHA256读回核对保存在`剩余验证/用户导出DRC日志.txt`。']
(HERE/'DRC逐项核对.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
print(json.dumps({'warnings': len(warnings), 'singlePin': len(single), 'supplierGroups': len(mismatch), 'supplierRefs': len(affected), 'emptyValues': len(empty_value), 'status': 'classified, strict DRC remains false'}, ensure_ascii=False))
