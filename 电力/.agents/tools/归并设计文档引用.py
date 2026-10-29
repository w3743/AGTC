"""迁移已归并设计文档的当前Markdown链接；--apply前检查四板均只剩两份正文。

只清理本脚本显式列出的共享旧报告，不删除模块文件、原生、BOM或冻结证据。
"""
from pathlib import Path
from urllib.parse import unquote
import argparse
import hashlib
import json
import os
import re

POWER = Path(__file__).resolve().parents[2]
ROOT = POWER.parent
LEDGER = POWER / '检查/设计文档归并映射.json'
MODULES = ('BMS模块', 'FOC驱动模块', '稳压模块', '水泵驱动模块')
OLD_GLOBAL = {
 '检查/本轮全电路审查/BMS审查.md': ('BMS模块/设计细节.md', 'audit-current'),
 '检查/本轮全电路审查/BMS精简整改候选.md': ('BMS模块/设计细节.md', 'dfetoff-candidate'),
 '检查/统一审查/BMS/BMS全原理图逐针与故障链复核.md': ('BMS模块/设计细节.md', 'pin-chain-review'),
 '检查/统一审查/BMS/README.md': ('BMS模块/设计细节.md', 'evidence-archive'),
 '检查/本轮全电路审查/FOC审查.md': ('FOC驱动模块/设计细节.md', ''),
 '检查/本轮全电路审查/FOC水泵及能量接口精简候选.md': ('FOC驱动模块/设计细节.md', ''),
 '检查/本轮全电路审查/稳压及水泵审查.md': ('稳压模块/设计细节.md', ''),
 '检查/本轮全电路审查/审查结论.md': ('检查/README.md', ''),
 '检查/统一审查/审查结论.md': ('检查/README.md', ''),
 '检查/统一审查/电驱原理可实现性审查.md': ('FOC驱动模块/设计细节.md', ''),
 '检查/统一审查/稳压_SC隔离_A_B_只读比较.md': ('稳压模块/设计细节.md', ''),
 '检查/统一审查/README.md': ('检查/README.md', ''),
}

def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def frozen(p):
    rel = p.relative_to(POWER).as_posix()
    return any(s in rel for s in ('/source_snapshot/', '/ngspice-47-portable/', '/.git/', '/node_modules/', '/.agents/skills/'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    migration = {}
    def add(old, new, anchor='', **metadata):
        old, new = Path(old).resolve(), Path(new).resolve()
        old.relative_to(POWER.resolve())
        new.relative_to(POWER.resolve())
        migration[str(old)] = {'target': str(new), 'anchor': anchor, **metadata}

    bms_reg = POWER / '检查/设计文档归并映射_BMS稳压.json'
    if bms_reg.exists():
        for old, spec in load(bms_reg).get('mapping', {}).items():
            add(old, spec['target'], spec.get('anchor', ''),
                disposition=spec.get('content_absorbed', 'see_module_map'))
    foc_map = POWER / 'FOC驱动模块/检查/文档归并映射.json'
    if foc_map.exists():
        for item in load(foc_map).get('mapping', []):
            target = '设计需求.md' if item['old_markdown'] == 'FOC驱动设计构想.md' else '设计细节.md'
            add(POWER/'FOC驱动模块'/item['old_markdown'], POWER/'FOC驱动模块'/target,
                disposition=item.get('new_targets', []), source_sha256=item.get('source_sha256'))
    pump_map = POWER / '水泵驱动模块/检查/文档归并映射.json'
    if pump_map.exists():
        for item in load(pump_map).get('mappings', []):
            target = '设计需求.md' if item['source_path'] == '项目要求.md' else '设计细节.md'
            add(POWER/'水泵驱动模块'/item['source_path'], POWER/'水泵驱动模块'/target,
                disposition=item.get('targets', []), source_sha256=item.get('source_sha256_before_delete'))

    prior = load(LEDGER).get('mapping', {}) if LEDGER.exists() else {}
    for old, (new, anchor) in OLD_GLOBAL.items():
        p = (POWER/old).resolve()
        add(p, POWER/new, anchor, scope='shared_board_report',
            source_sha256=digest(p) if p.exists() else prior.get(str(p), {}).get('source_sha256'))
    readiness = {}
    for name in MODULES:
        folder = POWER/name
        docs = sorted(p for p in folder.rglob('*.md') if not frozen(p))
        expected = sorted(folder/n for n in ('设计需求.md', '设计细节.md'))
        readiness[name] = docs == expected
    if not all(readiness.values()):
        print(json.dumps({'status':'WAIT_FOR_MODULE_CONSOLIDATION', 'ready': readiness}, ensure_ascii=False))
        return 2
    assert all(Path(spec['target']).is_file() for spec in migration.values()), 'missing migration target'
    for spec in migration.values():
        if spec.get('anchor'):
            assert f'id="{spec["anchor"]}"' in Path(spec['target']).read_text('utf-8'), spec

    normalized = {os.path.normcase(k): v for k,v in migration.items()}
    removed_set = {(POWER/p).resolve() for p in OLD_GLOBAL}
    rewrites = []
    if args.apply:
        for p in POWER.rglob('*.md'):
            if frozen(p) or p.resolve() in removed_set or any(s in p.as_posix() for s in ('/参考资料/EDA/', '/参考资料/EDA官方工具/')):
                continue
            before = p.read_text(encoding='utf-8-sig')
            count = 0
            def replace(match):
                nonlocal count
                raw = match.group(2).strip().strip('<>')
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', raw) or raw.startswith('#'):
                    return match.group(0)
                raw = re.sub(r':\d+(?:[–-]\d+)?$', '', unquote(raw.split('#',1)[0]))
                source = Path(raw)
                source = source.resolve() if source.is_absolute() else (p.parent/source).resolve()
                spec = normalized.get(os.path.normcase(str(source)))
                if not spec:
                    return match.group(0)
                target = os.path.relpath(spec['target'], p.parent).replace('\\','/')
                if spec.get('anchor'):
                    target += '#'+spec['anchor']
                if ' ' in target:
                    target = '<'+target+'>'
                count += 1
                return '['+match.group(1)+']('+target+')'
            after = re.sub(r'\[([^\]]*)\]\(([^\n)]+)\)', replace, before)
            if after != before:
                assert p.read_text(encoding='utf-8-sig') == before, 'concurrent document edit: '+str(p)
                p.write_text(after, encoding='utf-8')
                rewrites.append({'path':str(p.relative_to(ROOT)), 'links':count})
        deleted = []
        for p in sorted(removed_set):
            # Exact file-only deletion; target was checked against named global list and workspace.
            p.relative_to((POWER/'检查').resolve())
            if p.is_file():
                assert digest(p) == migration[str(p)]['source_sha256'], 'concurrent source edit: '+str(p)
                p.unlink()
                deleted.append(str(p.relative_to(ROOT)))
        result = {'schemaVersion':1, 'date':'2026-10-29', 'status':'CURRENT_MARKDOWN_REFERENCES_MIGRATED',
            'mapping':migration, 'rewritten_files':rewrites, 'shared_reports_removed':deleted,
            'module_maps':[str(p.relative_to(ROOT)) for p in (bms_reg,foc_map,pump_map)],
            'frozen_evidence_or_third_party_modified':False, 'native_or_bom_modified':False}
        LEDGER.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'status':result['status'], 'mapping_entries':len(migration),
                          'rewritten_files':len(rewrites),'shared_reports_removed':len(deleted)}, ensure_ascii=False))
    else:
        print(json.dumps({'status':'READY', 'mapping_entries':len(migration),'ready':readiness}, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
