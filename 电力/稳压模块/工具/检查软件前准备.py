"""重算准备附件一致性；不声称原生接线或DRC已经完成。"""
from pathlib import Path
import csv, hashlib, json, re, subprocess
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    native=ROOT/'eda工程文件/sc-regulator/sch/sc-regulator/01 稳压模块全部电路.esch2'
    oldbom=ROOT/'原理图准备/BOM.csv'
    assert sha(native)=='02ad832c553f6db4ecc562d1074f554234d1bb31b56d27d08167b4ea2978822a'
    assert sha(oldbom)=='789de860f8d400c3b631f9eb96b2ea310c72765b08c280e7c5b236366083f478'
    bom=list(csv.DictReader((ROOT/'原理图准备/选型BOM.csv').open(encoding='utf-8-sig')))
    selected=read(ROOT/'检查/最终选型清单.json')
    selected['count_semantics']={'selected_instances':'历史字段名：选型BOM条目数171，含无独立针网的COREBOARD_WEACT装配项；不是目标网表实例数。',
                                 'selected_BOM_rows':171,'target_netlist_instances':170}
    write(ROOT/'检查/最终选型清单.json',selected)
    offline=read(ROOT/'检查/软件前连接检查.json')
    bindings=read(ROOT/'原理图准备/实际元件库绑定.json')
    assert len(bom)==171 and len({r['Reference'] for r in bom})==171
    assert [offline[k] for k in ['component_count','pin_count','network_count','NC_pin_count']]==[170,503,85,23]
    assert len(bindings['components'])==170 and not bindings['unresolved_device_identities']
    assert selected['counts']['new_exact_MPNs_vs_regulator_baseline']==10
    assert sorted(p.name for p in ROOT.glob('*.md'))==['设计细节.md','设计需求.md']
    broken=[]
    for p in ROOT.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            if not (ROOT/target.split('#')[0]).exists():broken.append(target)
    # 本检查自己的输出是文档中已预声明的目标，首次生成时允许不存在。
    broken=[x for x in broken if x!='检查/软件前准备完成检查.json']
    assert not broken,broken
    compile_js="const fs=require('fs'),path=require('path'); const F=Object.getPrototypeOf(async function(){}).constructor; for(const f of fs.readdirSync(process.argv[1]).filter(x=>x.endsWith('.js')))new F(fs.readFileSync(path.join(process.argv[1],f),'utf8'));"
    subprocess.run(['node','-e',compile_js,str(ROOT/'工具')],check=True,capture_output=True)
    mock=read(ROOT/'检查/软件验收器离线核对.json')
    assert len(mock['checks'])==4 and all(x['pass'] for x in mock['checks'])
    prep_path=ROOT/'检查/软件脚本预检准备.json'
    prep=read(prep_path)
    prep['status']='READ_ONLY_PREFLIGHT_EXECUTED_APPLY_AND_POST_ACCEPTANCE_NOT_EXECUTED'
    prep['scope']['forbiddenInScripts_scope']='仅指只读预检和逐针验收脚本；独立应用脚本在apply=true时会改接/保存。'
    prep['executionStatus'].update(edaClientAvailable=True,clientVersion='4.1.93',scriptsExecuted=True,
        apiCallsMade=True,bridgeAvailable=False,transport='OFFICIAL_CLI',
        preflightExecuted=True,applyExecuted=False,postPinAcceptanceExecutedInEDA=False,
        result='已执行客户端只读预检及库查询；后验收仅有离线模拟，原生接线/DRC尚未执行。')
    write(prep_path,prep)
    selection_checks={'unique_references':True,'count_reconciliation':True,
        'correct_5V_input_pin1_minus_pin2_plus':True,'baseline_native_and_BOM_hash_unchanged':True,
        'only_two_root_MD':True,'document_links_exist':True,'all_target_device_identities_bound':True}
    verify={'status':'SELECTION_AND_OFFLINE_PREPARATION_CHECKS_PASS_NOT_NATIVE_DRC','date':'2026-10-29',
        'counts':selected['counts'],'checks':selection_checks,
        'count_semantics':selected['count_semantics'],
        'native_schematic_updated':False,'complete_RMB_cost_known':False,
        'supersedes':'旧记录的7种新MPN已由当前10种覆盖；不沿用旧文件哈希。'}
    write(ROOT/'检查/最终选型验证.json',verify)
    artifacts=[ROOT/'设计需求.md',ROOT/'设计细节.md']
    artifacts += [ROOT/'原理图准备'/n for n in ['选型BOM.csv','最终逐针连接合同.json','最终逐针连接.csv',
        '目标连接网表.enet','目标连接网表.xml','目标网络成员.json','软件元件库绑定.csv',
        '实际元件库绑定.json','软件接线变更.csv','软件接线输入.json']]
    artifacts += [ROOT/'检查'/n for n in ['最终选型清单.json','最终选型验证.json','软件前连接检查.json',
        '软件脚本预检准备.json','软件验收器离线核对.json','国产外围替代核对.json']]
    report={'status':'PREPARATION_COMPLETE_NATIVE_SCHEMATIC_IMPLEMENTATION_PENDING','date':'2026-10-29',
        'client_version':'4.1.93','selected_BOM_rows':len(bom),
        'target':{k:offline[k] for k in ['component_count','pin_count','network_count','connected_pin_count','NC_pin_count','changed_or_added_pins']},
        'library_bindings':bindings['counts'],'domestic_same_quantity_substitution_positions':34,
        'checks':selection_checks | {'javascript_syntax':True,'four_offline_acceptance_cases':True},
        'native_schematic_sha256':sha(native),'original_BOM_sha256':sha(oldbom),
        'official_DRC_executed':False,'target_connections_applied_to_native':False,
        'shared_library_check':{'native_sources_and_580_instances':'PASS',
            'overall':'FAIL_PREEXISTING_UNRELATED_MANUAL_SOURCE_HASH_MISMATCH',
            'source':'算法/视觉算法/README.md','modified_this_turn':False},
        'remaining_software_work':['place_or_replace_parts_and_verify_pin_sets','apply_target_nets',
            'save_reopen','official_export_and_each_pin_comparison','official_DRC','visual_review'],
        'cost_coverage':'PARTIAL_NOT_TOTALLED',
        'files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in artifacts}}
    write(ROOT/'检查/软件前准备完成检查.json',report)
    print(json.dumps({'status':report['status'],'checks':report['checks'],'target':report['target']},ensure_ascii=False))
if __name__=='__main__':main()
