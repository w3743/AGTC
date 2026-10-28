import sys,json,pathlib
from 原生更新 import invoke
code=sys.stdin.read()
value=invoke(code)
if len(sys.argv)>2:
    pathlib.Path(sys.argv[2]).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',sys.argv[2])
else: print(json.dumps(value,ensure_ascii=True,indent=2))
