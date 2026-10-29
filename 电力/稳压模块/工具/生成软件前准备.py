"""逐针合同 -> 离线网表/网络表/库绑定/变更表。无需嘉立创客户端，不改原生工程。"""
from pathlib import Path
import csv, hashlib, json
from collections import defaultdict
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
POWER=ROOT.parent
def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    contract_path=ROOT/'原理图准备/最终逐针连接合同.json'
    contract=load(contract_path)
    bom_path=ROOT/'原理图准备/选型BOM.csv'
    bom={r['Reference']:r for r in csv.DictReader(bom_path.open(encoding='utf-8-sig'))}
    baseline_path=ROOT/'检查/接线前网表基线.enet'
    baseline=load(baseline_path if baseline_path.exists() else ROOT/'检查/当前官方网表.enet')
    old={c['props']['Designator']:c for c in baseline['components'].values() if c['props'].get('Designator')}
    components=contract['components']
    assert len({c['ref'] for c in components})==len(components)
    assert {c['ref'] for c in components}==set(bom)-{'COREBOARD_WEACT'}
    nets=defaultdict(list); pinrows=[]; changes=[]; bindings=[]; enet_components={}
    library=load(POWER/'元件封装库/索引/元件总表.json')
    live_path=ROOT/'原理图准备/实际元件库绑定.json'
    live={r['ref']:r for r in load(live_path)['components']} if live_path.exists() else {}
    for c in components:
        ref=c['ref']; b=bom[ref]
        c['manufacturer_part']=b['Manufacturer part'];c['supplier_part']=b['Supplier part']
        c['manufacturer']=b['Manufacturer'];c['footprint']=b['Footprint'];c['scope']=b['Scope']
        assert len({str(p['number']) for p in c['pins']})==len(c['pins']),ref
        assert not any(p.get('net') in ['NC','N/C'] for p in c['pins']),ref
        pinmap={}
        for p in c['pins']:
            number=str(p['number']);net=p.get('net')
            assert p['connection'] in ['NET','NC'],(ref,number)
            assert (bool(net) and p['connection']=='NET') or (net is None and p['connection']=='NC'),(ref,number,net)
            assert net not in ['BUS_BR','PERMIT','PMOS_GATE','PMOS_CTRL','TEST','TEST_FB','DUMP_IN'],(ref,number,net)
            row={'Reference':ref,'Pin':number,'Pin name':p.get('name',''),'Net':net or '',
                 'Connection':p['connection'],'MPN':b['Manufacturer part'],'Scope':b['Scope']}
            pinrows.append(row)
            if net: nets[net].append({'ref':ref,'pin':number})
            prev=old.get(ref,{}).get('pinInfoMap',{}).get(number)
            previous=prev.get('net') or None if prev else None
            if previous!=net or prev is None:
                changes.append({'Reference':ref,'Pin':number,'Before':previous or '', 'After':net or '',
                                'Action':'ADD_PIN' if prev is None else 'REWIRE' if net else 'NC'})
            pinmap[number]={'name':p.get('name',''),'number':number,'net':net or '',
                            'pinClass':'','pinDifferentialPairClass':'','props':{'Pin Number':number}}
        matches=[i for i in library['instances'] if i.get('mpn')==b['Manufacturer part'] and
                 (not b['Supplier part'] or i.get('supplier_part')==b['Supplier part'])]
        distinct={ (i.get('device_id'),i.get('symbol_id'),i.get('footprint_id')) for i in matches }
        bindings.append({'Reference':ref,'Manufacturer':b['Manufacturer'],'MPN':b['Manufacturer part'],
                         'Supplier part':b['Supplier part'],'Selected footprint':b['Footprint'],
                         'Existing library matches':len(distinct),
                         'Device UUID candidates':';'.join(sorted({i.get('device_id','') for i in matches if i.get('device_id')})),
                         'Status':'EXISTING_EXACT_IDENTITY_GEOMETRY_RECHECK' if matches else 'RESOLVE_BY_EXACT_ID_IN_EDA',
                         'Note':'精确MPN和供应代码命中仅表示库来源存在，不代表软件已导入/实物验收'})
        if b['Supplier part']=='C2846678' and not matches:
            bindings[-1].update({'Existing library matches':1,'Device UUID candidates':'e4b0ce5ab33f6cbe',
                                 'Status':'CACHED_EXACT_DEVICE_NOT_CURRENT_INSTANCE',
                                 'Note':'原生片段DEVICE/0aec00e132e0afd3e33b351acbdf3b47f51ee09da7144f29cb7c2f1a3839a551.jsonl精确命中；软件阶段核入库'})
        if ref in live:
            item=live[ref]
            assert item['procurement_mpn']==b['Manufacturer part'] and item['supplier_part']==b['Supplier part'],ref
            bindings[-1].update({'Existing library matches':1,'Device UUID candidates':item['device_uuid'],
                                 'Status':item['binding'],
                                 'Note':item.get('note','实际库身份已解析；放置后验脚、接线、官方导出和DRC仍待执行。')})
        props={'Designator':ref,'Name':b['Value or candidate'],'Manufacturer':b['Manufacturer'],
               'Manufacturer Part':b['Manufacturer part'],'Supplier':b['Supplier'],
               'Supplier Part':b['Supplier part'],'Selected Footprint':b['Footprint'],
               'Add into BOM':'yes','Convert to PCB':b['Convert to PCB']}
        enet_components[ref]={'props':props,'pinInfoMap':pinmap}
    # 不把设计网表冒充官方导出；软件阶段通过实际位号映射生成真实unique-id索引。
    design_enet={'version':'2.0.0','projectId':baseline['projectId'],
                 'components':enet_components,'designRule':baseline.get('designRule',{}),
                 'differentialPair':{},'netClass':{},'equalLengthNetGroup':{},'differentialPairGroups':{},
                 'preparationStatus':'DESIGN_TARGET_NOT_OFFICIAL_EXPORT_NOT_NATIVE_GEOMETRY',
                 'componentKeyRule':'reference designators; map to live unique IDs before SCH_Netlist.setNetlist'}
    save(ROOT/'原理图准备/目标连接网表.enet',design_enet)
    save(ROOT/'原理图准备/目标网络成员.json',dict(sorted(nets.items())))
    write_csv(ROOT/'原理图准备/最终逐针连接.csv',pinrows,list(pinrows[0]))
    write_csv(ROOT/'原理图准备/软件接线变更.csv',changes,['Reference','Pin','Before','After','Action'])
    write_csv(ROOT/'原理图准备/软件元件库绑定.csv',bindings,list(bindings[0]))
    # 独立XML交换视图，方便不用EDA也能查阅网络成员。
    xml=ET.Element('export',version='D');xs=ET.SubElement(xml,'design')
    ET.SubElement(xs,'source').text='DESIGN TARGET; NOT OFFICIAL EDA EXPORT'
    xc=ET.SubElement(xml,'components')
    for c in components:
        item=ET.SubElement(xc,'comp',ref=c['ref'])
        ET.SubElement(item,'value').text=c['manufacturer_part'] or bom[c['ref']]['Value or candidate']
    xn=ET.SubElement(xml,'nets')
    for n,(net,pins) in enumerate(sorted(nets.items()),1):
        item=ET.SubElement(xn,'net',code=str(n),name=net)
        for p in pins: ET.SubElement(item,'node',ref=p['ref'],pin=p['pin'])
    ET.indent(xml,space='  ')
    ET.ElementTree(xml).write(ROOT/'原理图准备/目标连接网表.xml',encoding='utf-8',xml_declaration=True)

    byref={c['ref']:{str(p['number']):p['net'] for p in c['pins']} for c in components}
    def require(ref,pin,net): assert byref[ref][str(pin)]==net,(ref,pin,byref[ref][str(pin)],net)
    require('J_CORE_P1',5,'DRV_FAULT_N');require('J_CORE_P1',11,'PA9_TIM1_CH2')
    require('J_CORE_P1',12,'DRV_FAULT_N');require('J_CORE_P1',7,'SC_ADC_A');require('J_CORE_P1',8,'SC_ADC_B')
    for pin in [10,11,20]: require('J_CORE_P2',pin,'IMON')
    for pin in [14,18]: require('J_CORE_P2',pin,'DRV_FAULT_N')
    require('U_ISO_DRV',1,'12V_DRV');require('U_ISO_DRV',2,'GND_SYS');require('U_ISO_DRV',11,'GND_SYS')
    require('U_DUMP_DRV',5,'DUMP_OFF');require('U_DUMP_DRV',6,'12V_DRV')
    require('U_SC_OV',3,'BUS_OV_DIV');require('U_SC_OV',6,'DRV_FAULT_N')
    require('C_SC_BULK',1,'SC_ISO_STORAGE');require('C_SC_BULK2',1,'SC_ISO_STORAGE')
    require('C_SC_HF',2,'GND_SYS');require('C_BUS_FILM3',1,'SC_FUSED')
    for ref,a,k in [('D_SC_LO','GND_SYS','SC_FUSED'),('D_SC_HI','SC_FUSED','BUS_EXT'),('D_DUMP_FW','DUMP_SW','BUS_EXT')]:
        require(ref,1,a);require(ref,3,a);require(ref,2,k)
    assert 'SC_MINUS' in nets and 'GND_SYS' in nets
    assert byref['R_SC_SHUNT']['1']=='GND_SYS' and byref['R_SC_SHUNT']['2']=='SC_MINUS'
    for bank in [1,2]:
        for p in [3,6,8,11]: require(f'U_CELL_MUX{bank}',p,'SC_ADC_A' if bank==1 else 'SC_ADC_B')
    require('U_SIGNAL_ISO',7,'GND_SYS');require('U_SIGNAL_ISO',14,'3V3_CORE')
    # 所有非板外边界网络至少有两个连接端点；禁止单端悬空遗漏。
    boundary={'BUS_SOURCE'}
    singles={net:pins for net,pins in nets.items() if len(pins)<2 and net not in boundary}
    assert not singles,singles
    # 所有NC独立保存，XML/ENET均不建立“NC”公用网络。
    assert 'NC' not in nets
    contract['prepared_material_source_sha256']=sha(bom_path)
    contract['preparation_note']='逐针连接不变；实际厂家、MPN、供应代码与封装由当前选型BOM归一更新；原生工程尚未应用。'
    save(contract_path,contract)
    save(ROOT/'原理图准备/软件接线输入.json',{'contract':{'components':components},
          'target_page_uuid':'9546dca0bc8bce7f','selected_bom_sha256':sha(bom_path),
          'target_enet_sha256':sha(ROOT/'原理图准备/目标连接网表.enet')})
    report={'status':'OFFLINE_CONNECTION_CHECKS_PASS_SOFTWARE_IMPORT_NOT_EXECUTED',
            'component_count':len(components),'pin_count':len(pinrows),'network_count':len(nets),
            'connected_pin_count':sum(len(p) for p in nets.values()),
            'NC_pin_count':sum(r['Connection']=='NC' for r in pinrows),
            'changed_or_added_pins':len(changes),'single_endpoint_nets_except_boundary':singles,
            'checks':['all_selected_instances_except_coreboard_assembly_accounted','unique_pin_numbers',
                      'each_pin_NET_or_individual_NC','legacy_net_names_absent','core_headers_mapped',
                      'fault_line_all_sources_same_net','SC_ground_not_merged','LTC_UCC_TMUX_pinout',
                      '35V_caps_storage_side','diode_polarities','no_internal_single_endpoint_net'],
            'official_export':False,'official_DRC':False,'native_modified':False,
            'inputs':{'contract':sha(contract_path),'selected_BOM':sha(bom_path)},
            'software_actions_remaining':['resolve_exact_library_items','place_or_replace_parts','apply_target_connections',
                                          'save_reopen','export_official_netlist','compare_each_pin','official_DRC','visual_review']}
    save(ROOT/'检查/软件前连接检查.json',report)
    print(json.dumps({k:report[k] for k in ['status','component_count','pin_count','network_count','NC_pin_count','changed_or_added_pins']},ensure_ascii=False))

if __name__=='__main__': main()
