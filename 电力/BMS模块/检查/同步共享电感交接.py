"""原位同步新均衡接口和当前工程入口，不运行电气测试。"""
import json,copy,csv
from pathlib import Path
B=Path(__file__).resolve().parents[1];C=B/'检查';S=B/'原理图准备'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
m=read(C/'原理图构建清单.json')
removed={e['ref'] for e in m if e['pageGroup'].startswith('AFE_LTC') or e['pageGroup']=='AFE_LOCAL_UV'}
removed.update(['U_LEVEL','C_LEVEL_A','C_LEVEL_B','R_LEVEL_OE_PD','R_LTC_CSBI_PU','R_LTC_SCK_PD','R_LTC_SDI_PD','R_LTC_SDO_PU','R_BAL_RUN_PD'])
for n in ['AFE元件连接.json','接口元件连接.json']:
 es=[e for e in read(S/n) if e['ref'] not in removed]
 for e in es:
  if e['ref']=='J_CORE_P2':
   e['pins']['7']='BAL_READY_N';e['pins']['14']='BAL_CURRENT_ADC'
   e['note']+=' Shared-inductor update: P2-7=PB2/BAL_READY_N; P2-14=PA6/BAL_CURRENT_ADC. P2-8 remains NC (VREF+).'
 write(S/n,es)
maint={e['ref']:e for e in read(S/'维护线束元件连接.json')}
interface=read(S/'接口元件连接.json')
interface=[e for e in interface if e['ref'] not in ['J_BAL_SERVICE','J_CELL_SERVICE']]
interface.extend(copy.deepcopy(maint[r]) for r in ['J_BAL_SERVICE','J_CELL_SERVICE'])
write(S/'接口元件连接.json',interface)
src={e['ref']:e for n in ['AFE元件连接.json','功率元件连接.json','接口元件连接.json'] for e in read(S/n)}
mapdoc=read(C/'Layout器件映射.json');mapping=mapdoc['components'] if isinstance(mapdoc,dict) else mapdoc
cell=copy.deepcopy(next(e for e in mapping if e['ref']=='J_CELL_SERVICE'))
for es in [m,mapping]:
 es[:]=[e for e in es if e['ref'] not in removed]
 for e in es:
  if e['ref'] in src:
   for k in ['pins','mpn','note','value']:
    if k in src[e['ref']]:e[k]=src[e['ref']][k]
  if e['ref']=='J_BAL_SERVICE':
   e.update(libUuid=cell['libUuid'],libraryUuid=cell.get('libraryUuid'),libType='2',nativeLibUuid=cell['libUuid'],addIntoPcb=False,addIntoBom=False,carrierReason='Logical ten-tap port. Human creates ten solder PTH pads; not an XH device.')
 if es is m:write(C/'原理图构建清单.json',es)
 else:write(C/'Layout器件映射.json',mapdoc)
write(C/'共享电感退出器件.json',sorted(removed))
print('removed legacy refs',len(removed),'retained manifest',len(m))
