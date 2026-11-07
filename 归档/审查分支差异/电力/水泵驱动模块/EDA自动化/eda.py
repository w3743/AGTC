"""Official EasyEDA CLI adapter, scoped to this pump project."""
import argparse,subprocess,json,sys,base64
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CLI=r'D:\lceda-pro\lceda-pro.exe'
PROJECT_FILE=ROOT/'eda工程文件'/'水泵驱动'/'水泵驱动.eprj3'
_SESSION_OVERRIDE=None
_RESOLVED_SESSION=None
def cli(*args):
 p=subprocess.run([CLI,*args],capture_output=True,timeout=80)
 raw=p.stdout.decode('utf-8-sig',errors='replace').strip()
 try:r=json.loads(raw)
 except Exception:raise RuntimeError(raw[:1000]+p.stderr.decode('utf8',errors='replace')[:500])
 if not r.get('ok'):raise RuntimeError(str(r))
 return r.get('value')
def _target_sheet_uuids():
 if not PROJECT_FILE.is_file():raise RuntimeError(f'水泵目标工程文件不存在：{PROJECT_FILE}')
 try:project=json.loads(PROJECT_FILE.read_text(encoding='utf-8-sig'))
 except Exception as e:raise RuntimeError(f'无法读取水泵目标工程清单：{PROJECT_FILE} ({e})')
 sheets=project.get('profile',{}).get('sheets',{})
 ids={str(v.get('uuid') or k) for k,v in sheets.items() if isinstance(v,dict)}
 if not ids:raise RuntimeError(f'水泵目标工程没有可验证的原理图页 UUID：{PROJECT_FILE}')
 return ids
def resolve_session(explicit=None):
 """Select only a live CLI session whose open project contains this project's sheet UUID."""
 sessions=cli('session','list')
 if not isinstance(sessions,list):raise RuntimeError('官方 CLI session list 未返回会话列表，拒绝选择 EDA 会话。')
 if explicit:
  candidates=[s for s in sessions if str(s.get('sessionId'))==str(explicit)]
  if not candidates:
   ids=', '.join(str(s.get('sessionId')) for s in sessions if s.get('status')=='alive') or '无'
   raise RuntimeError(f'指定的 --session {explicit} 不在当前 CLI 会话中。当前活动会话：{ids}')
 else:
  candidates=[s for s in sessions if s.get('status')=='alive']
 if not candidates:
  raise RuntimeError('没有活动的 EDA 会话。请先打开水泵工程，或使用 --session <会话ID> 指定会话。')
 target_ids=_target_sheet_uuids()
 code="const p=await eda.dmt_Project.getCurrentProjectInfo();const ids=[];for(const b of (p?.data||[])){for(const q of (b?.schematic?.page||[])){if(q?.uuid)ids.push(String(q.uuid));}}return {name:p?.name??null,uuid:p?.uuid??null,pageUuids:ids};"
 matches=[];errors=[]
 for s in candidates:
  sid=str(s.get('sessionId'))
  try:info=cli('invoke','--session',sid,'--ext-uuid','eda','--timeout','60000','--code',code)
  except Exception as e:
   errors.append(f'{sid}: {e}')
   continue
  if isinstance(info,dict) and target_ids.intersection(map(str,info.get('pageUuids') or [])):
   matches.append((sid,info))
 if len(matches)==1:return matches[0][0]
 if len(matches)>1:
  ids=', '.join(sid for sid,_ in matches)
  raise RuntimeError(f'发现多个会话打开了水泵目标工程（{ids}）。请通过 --session <会话ID> 明确指定。')
 target=', '.join(sorted(target_ids))
 if explicit:
  raise RuntimeError(f'指定会话 {explicit} 未能验证为水泵工程（目标原理图页 UUID：{target}），拒绝执行。')
 detail=(' 会话检查失败：'+'；'.join(errors[:3])) if errors else ''
 raise RuntimeError(f'没有活动会话匹配水泵目标工程（原理图页 UUID：{target}）。请用官方 CLI 打开 {PROJECT_FILE} 后重试，或通过 --session <会话ID> 指定并验证目标会话。{detail}')
def invoke(code,session=None):
 global _RESOLVED_SESSION
 if session is not None:
  sid=resolve_session(session)
 elif _SESSION_OVERRIDE is not None:
  sid=resolve_session(_SESSION_OVERRIDE)
 elif _RESOLVED_SESSION is not None:
  sid=_RESOLVED_SESSION
 else:
  sid=resolve_session()
  _RESOLVED_SESSION=sid
 return cli('invoke','--session',sid,'--ext-uuid','eda','--timeout','60000','--code',code)
def put(name,obj):
 raw=base64.b64encode(json.dumps(obj,ensure_ascii=False).encode()).decode()
 invoke(f'globalThis.{name}="";return true;')
 for i in range(0,len(raw),16000):invoke(f'globalThis.{name}+={json.dumps(raw[i:i+16000])};return true;')
 invoke(f'globalThis.{name}=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(globalThis.{name}),c=>c.charCodeAt(0))));return true;')
def save(name,obj):
 path=ROOT/'检查'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description='按水泵工程原理图页 UUID 验证官方 EDA CLI 会话后运行脚本。')
 parser.add_argument('--session',help='显式指定 CLI 会话 ID；指定后仍会验证该会话确属水泵目标工程。')
 parser.add_argument('args',nargs='*',help='脚本路径与输出名，或 doc <API类名> [方法名]')
 ns=parser.parse_args()
 _SESSION_OVERRIDE=ns.session
 if not ns.args:parser.error('请提供脚本路径和输出名，或 doc <API类名> [方法名]。')
 if ns.args[0]=='doc':
  if len(ns.args)<2:parser.error('doc 模式需要 API 类名。')
  cls=ns.args[1];method=ns.args[2] if len(ns.args)>2 else None
  r=cli('doc','api','--class-name',cls,*(['--method-name',method] if method else []));save('API参考/'+cls+('-'+method if method else '')+'.json',r);print(json.dumps(r,ensure_ascii=True))
 else:
  if len(ns.args)<2:parser.error('脚本模式需要脚本路径和输出名。')
  script=Path(ns.args[0]);
  if not script.is_file():parser.error(f'脚本文件不存在：{script}')
  r=invoke(script.read_text(encoding='utf8'));save(ns.args[1],r);print(json.dumps(r,ensure_ascii=True)[:5000])
