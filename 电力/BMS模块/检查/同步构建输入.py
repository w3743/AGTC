"""把现行连接源合并到施工清单；库替代由映射脚本另行审查。"""
import json,pathlib,copy
B=pathlib.Path(__file__).resolve().parents[1]; C=B/'检查'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
m=read(C/'原理图构建清单.json'); old={e['ref']:e for e in m}
src={e['ref']:e for n in ['AFE元件连接.json','功率元件连接.json','接口元件连接.json'] for e in read(B/'原理图准备'/n)}
maintenance={e['ref']:e for e in read(B/'原理图准备'/'维护线束元件连接.json')}
src['J_CELL_SERVICE']=maintenance['J_CELL_SERVICE']
if 'R_BMS_LOCAL_OK_PD' not in old:
    old['R_BMS_LOCAL_OK_PD']=copy.deepcopy(old['R_HOST_ALLOW_PD']);old['R_BMS_LOCAL_OK_PD']['ref']='R_BMS_LOCAL_OK_PD'
res=next(e for e in m if e['ref']=='R_WDT_PD1'); cap=next(e for e in m if e['ref']=='C_REG1')
cache={g['key']:g['items'] for n in ['Layout新增库检索.json','Layout端口库检索.json'] for g in read(C/n)['groups']}
def template(e,i):
    ref=e['ref']
    if ref.startswith('R_'): x=copy.deepcopy(res)
    elif ref.startswith('C_'): x=copy.deepcopy(cap)
    elif ref.startswith('U_WDT_SUP'): x={'libUuid':cache['TLV803EB46DCKR'][0]['uuid'],'libType':'3'}
    elif ref=='Q_LOAD_POWER_VALID_DRV': x={'libUuid':e['libUuid'],'libType':'3'}
    elif ref=='U_LOAD_POWER_VALID': x={'libUuid':'39f487c726af47ce80aa9154f06c7bd5','libType':'3'}
    elif ref=='J_LOAD_POWER_VALID': x={'libUuid':cache['1757242'][0]['uuid'],'libType':'3'}
    elif ref in ['J_BAL_SERVICE','J_CELL_SERVICE']:
        item=cache['XH-4A'][1] if ref=='J_BAL_SERVICE' else cache['XH-10A'][0]
        x={'libUuid':item['symbolUuid'],'libType':'2','addIntoBom':False,'addIntoPcb':False,'libraryNote':'Logical port symbol only. Actual custom solder-PTH footprint required; do not use XH connector footprint.'}
    else: raise ValueError(ref)
    x.update(ref=ref,x=300+(i%3)*650,y=180+(i//3)*420,pageGroup='AFE_LOCAL_UV' if 'WDT_' in ref else 'IF_LOAD_POWER_VALID' if 'LOAD_' in ref else 'SERVICE_PORTS')
    return x
out=[];i=0
for ref,e in src.items():
    if ref=='SW_RUN_STORE':continue
    x=copy.deepcopy(old[ref]) if ref in old else template(e,i)
    if ref not in old:i+=1
    for k in ['mpn','pins','note','addIntoBom']:
        if k in e:x[k]=e[k]
    if ref in ['J_BAT_HV','J_PACK_HV']:x['pins']['3']=None
    if ref=='RSHUNT':p=e['pins'];x['pins']={'1':p['I_BAT'],'2':p['I_LOAD'],'3':p['V_BAT'],'4':p['V_LOAD']}
    if x.get('value') and (ref not in old or ref.startswith('T_BAL')):x.pop('value',None)
    out.append(x)
switch=src['SW_RUN_STORE'];s=[]
for idx,suffix in enumerate(['A','B']):
    x={'ref':'SW_RUN_STORE_'+suffix,'mpn':switch['mpn'],'libUuid':'2ed628d31bda4994801901491c727b50','libType':'2','pins':{str(n):switch['pins'][str(n)] for n in range(1+3*idx,4+3*idx)},'pinRemap':{'1':str(2+3*idx),'2':str(3+3*idx),'3':str(1+3*idx)},'x':350+650*idx,'y':250,'pageGroup':'SW_MANUAL','addIntoBom':idx==0,'addIntoPcb':False,'note':switch['note'],'value':'DPDT manual switch '+suffix}
    s.append(x)
write(C/'原理图构建清单.json',out)
write(C/'开关构建清单.json',s)
print('manifest',len(out),'switch units',len(s))
