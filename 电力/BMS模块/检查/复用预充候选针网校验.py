"""Compare official candidate ENET against the frozen full pin contract."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];d=root/'检查/复用预充原生候选'
c=json.loads((root/'原理图准备/复用预充合并施工合同.json').read_text(encoding='utf8'))
raw=(d/'候选重开.enet').read_bytes();net=json.loads(raw.decode('utf-8-sig'))
actual={};issues=[]
for item in net['components'].values():
    p=item['props'];ref=p['Designator']
    if ref in actual:issues.append({'duplicate':ref})
    actual[ref]=item
want=c['mergedPinNets'];missing=sorted(set(want)-set(actual));extra=sorted(set(actual)-set(want))
if missing:issues.append({'missing':missing})
if extra:issues.append({'extra':extra})
for ref in sorted(set(want)&set(actual)):
    expected={str(k):(v or '') for k,v in want[ref]['pins'].items()}
    got={str(k):(v.get('net') or '') for k,v in actual[ref]['pinInfoMap'].items()}
    if expected!=got:issues.append({'ref':ref,'expected':expected,'actual':got})
result={'status':'PASS_STATIC_PINNET_ONLY' if not issues else 'FAIL','candidate_sha256':hashlib.sha256(raw).hexdigest(),'expected_count':len(want),'actual_count':len(actual),'issues':issues,'scope':'ENET pins only; no procurement, native PCB footprint, DRC, drawing, or electrical qualification.'}
(d/'候选重开_针网差异.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(result['status'],len(issues),'issues')
if issues:raise SystemExit(1)
