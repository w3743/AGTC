"""只读比对活动Markdown的本地链接；旧断链单列，避免归档引入新断链。"""
from pathlib import Path
from urllib.parse import unquote
import json,re,sys
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'.agents/cleanup_link_baseline.json'

def broken():
    rows=[]
    docs=list(ROOT.glob('*.md'))
    for board in ['BMS模块','FOC驱动模块','稳压模块','元件封装库']:
        docs.extend((ROOT/board).rglob('*.md'))
    for p in docs:
        rel=p.relative_to(ROOT).as_posix()
        if '/仿真/输入/' in rel or any(x in p.parts for x in ['归档','.git','参考资料','厂商资料','共享仓库','原生片段','tmp']):continue
        for m in re.finditer(r'\[[^\]\n]+\]\(([^)\n]+)\)',p.read_text(encoding='utf-8-sig')):
            raw=m.group(1);target=unquote(raw.strip('<>')).split('#')[0]
            if not target or '://' in target or target.startswith('#'):continue
            q=Path(target) if Path(target).is_absolute() else p.parent/target
            try:q.resolve().relative_to(ROOT)
            except (ValueError,OSError):continue
            if not q.exists():rows.append({'document':rel,'target':raw})
    return rows

if sys.argv[1]=='baseline':
    rows=broken();BASE.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('原有断链',len(rows))
else:
    old={(x['document'],x['target']) for x in json.loads(BASE.read_text(encoding='utf-8'))}
    rows=broken();new=[x for x in rows if (x['document'],x['target']) not in old]
    record=ROOT/'归档/目录整理.json';d=json.loads(record.read_text(encoding='utf-8'))
    d['markdown_link_check']={'new_broken':new,'preexisting_broken_remaining':rows,'status':'PASS' if not new else 'HOLD'}
    record.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'new_broken':new,'preexisting_remaining':len(rows)},ensure_ascii=False,indent=2))
    raise SystemExit(bool(new))
