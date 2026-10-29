"""将官方CLI只读查询的器件身份落实到目标位号；不修改EDA工程。"""
from pathlib import Path
import csv, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    bp=ROOT/'原理图准备/选型BOM.csv'
    pp=ROOT/'检查/接线实施/首次只读预检.json'
    dp=ROOT/'检查/接线实施/采用库属性.json'
    bom={r['Reference']:r for r in csv.DictReader(bp.open(encoding='utf-8-sig'))}
    actual={c['ref']:c for c in read(pp)['inventory']['actualComponents']}
    candidates=read(dp)['results'];rows=[]
    contract=read(ROOT/'原理图准备/最终逐针连接合同.json')
    for c in contract['components']:
        ref=c['ref'];b=bom[ref];old=actual.get(ref)
        exact=[d for d in candidates if ref in d['refs'] and d['mpn']==b['Manufacturer part']
               and (not b['Supplier part'] or d['detail']['property'].get('supplierId')==b['Supplier part'])]
        row={'ref':ref,'procurement_mpn':b['Manufacturer part'],'procurement_manufacturer':b['Manufacturer'],
             'supplier_part':b['Supplier part'],'convert_to_PCB':b['Convert to PCB'],
             'expected_pins':sorted(str(p['number']) for p in c['pins']),
             'native_applied':False}
        if exact:
            assert len(exact)==1,(ref,len(exact))
            d=exact[0];a=d['detail']['association']
            row.update(device_uuid=d['uuid'],library_uuid=d['libraryUuid'],
                       symbol=a.get('symbol'),footprint=a.get('footprint'),
                       binding='OFFICIAL_LIBRARY_EXACT_MPN_SUPPLIER',
                       geometry_pin_verification='REQUIRED_AT_PLACEMENT')
        elif old and ''.join((old.get('manufacturerPart') or '').split())==''.join(b['Manufacturer part'].split()) and (
                not b['Supplier part'] or old.get('supplierPart')==b['Supplier part']):
            assert row['expected_pins']==sorted(str(p['number']) for p in old['pins']),ref
            row.update(device_uuid=old['componentUuid'],library_uuid=old['componentLibraryUuid'],
                       binding='REUSE_EXISTING_NATIVE_DEVICE',geometry_pin_verification='EXISTING_PIN_NUMBERS_MATCH')
            if ref=='U_AUX1':
                row['note']='固定HI-LINK海凌科12V采购身份；保留现有自定义6脚载体，不换成金升阳等同名模块。'
        elif ref=='R_DUMP_EXT':
            carrier=next(d for d in candidates if d['detail']['property'].get('supplierId')=='C115324')
            row.update(device_uuid=carrier['uuid'],library_uuid=carrier['libraryUuid'],
                       binding='EXTERNAL_RESISTOR_SYMBOL_CARRIER_ONLY',geometry_pin_verification='TWO_PIN_CARRIER_MATCH',
                       note='仅复用已解析风华10k库项的两脚电阻符号，避免本地载体UUID远端重取失败；覆盖全部采购和数值属性为板外单只0.82Ω±5%，Convert to PCB=No。不继承载体功率或采购料号。')
        else:raise AssertionError(('UNRESOLVED',ref,b['Manufacturer part']))
        rows.append(row)
    assert len(rows)==170
    counts={key:sum(x['binding']==key for x in rows) for key in sorted({x['binding'] for x in rows})}
    report={'status':'ALL_TARGET_DEVICE_IDENTITIES_BOUND_NATIVE_NOT_APPLIED','date':'2026-10-29',
            'client_version':'4.1.93','counts':counts,'component_count':len(rows),
            'unresolved_device_identities':[],
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [bp,pp,dp]},
            'notes':['库项身份已唯一解析；新增/替换符号放置时仍须核对实际脚号集合。',
                     '没有执行原生元件替换、接线、保存或DRC。'], 'components':rows}
    (ROOT/'原理图准备/实际元件库绑定.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(counts,ensure_ascii=False))
if __name__=='__main__':main()
