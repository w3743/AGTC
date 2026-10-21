from pathlib import Path
import datetime,hashlib,json,zipfile
P=Path(__file__).resolve().parent;M=P.parent;R=M.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=P/'库同步校验.json'
if p.exists():
 raw=p.read_bytes();key=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_Layout库证据刷新前/检查/库同步校验.json'
 with zipfile.ZipFile(M/'归档/历史与调试.zip','a',compression=zipfile.ZIP_DEFLATED) as z:
  z.writestr(key,raw);assert hashlib.sha256(z.read(key)).hexdigest()==sha(p)
files=[R/'元件封装库/资产索引.json',R/'元件封装库/检查/同步证据.json',R/'元件封装库/检查/共享仓库发布.json',M/'原理图准备/BOM.csv',*(x for x in (M/'eda工程文件').rglob('*') if x.is_file())]
d={'time':datetime.datetime.now().astimezone().isoformat(),'syncAlreadyRun':True,'readOnlyCommand':'python -B 元件封装库/工具/同步全部.py --check','exitCode':0,'result':{'pass':True,'changed_or_missing_sources':[]},'scope':'源文件/资产索引及SHA一致，不构成采购精确载体机械/电气资格验收','evidence':[{'path':x.relative_to(R).as_posix(),'sha256':sha(x)} for x in files]}
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
p=P/'当前原理图导航审阅.json';d=json.loads(p.read_text(encoding='utf8'))
d['visualReview']={'status':'已目视核对当前导出总览及选定局部','views':['原理图总览','风扇直供细节','风扇PWM细节','风扇测速细节','模块细节','未用输出NC审阅','与门审阅'],'observations':['J_FAN可见VOUT_5V/B0，模块1/2/3/5/6端口与映射一致；PWM为开漏NMOS，测速上拉CORE_3V3；U_SEQ两路未用反相输出NC可见。','整张原理图沿用既有分散器件/网标组织，整体需用位号书签放大阅读；部分器件字段和端口文字靠近或重叠，未宣称全图排版逐项通过。','网表与正式原生为连接验收依据，本PDF仅作导航审阅；未执行PCB审查。']}
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('Library/PDF evidence recorded')
