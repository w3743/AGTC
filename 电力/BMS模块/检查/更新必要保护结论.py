"""Refresh current prose after the archived official publication, not electrical release."""
raise SystemExit('已撤回：四串电阻候选未发布，正式基线仍216/53。不得刷新为219/52。')
from pathlib import Path
import hashlib, json
P=Path(__file__).resolve().parent; M=P.parent; O=P/'必要保护精简'
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (O/'发布归档记录.json').exists()
expected=load(P/'当前施工输入.json')
assert len(expected)==220 and sum(x['addIntoPcb'] for x in expected)==219
intro='风扇限流电阻已用现有3.3kΩ、100Ω各两颗串联，保持6.8kΩ，不上调限流值；板内219颗、52种精确MPN。C_BAT2保留为TI参考电路中的BAT高频旁路，不作为多余保护删除。保护逐项理由见检查/保护料号复核_逻辑.md与检查/保护料号复核_功率.md。'
for path in [M/'README.md',M/'项目要求.md',P/'当前设计状态.md']:
 text=path.read_text(encoding='utf8')
 text=text.replace('216颗、53种','219颗、52种').replace('216颗/53种','219颗/52种')
 text=text.replace('本轮落实后216/53','本轮落实后219/52').replace('较本轮前减少7种、增加3颗','较213/60基线减少8种、增加6颗')
 text=text.replace('板内增加3颗、减少7种','板内增加6颗、减少8种')
 text=text.replace('217个CAD位号、723项','220个CAD位号、729项').replace('216个入板器件+1个板外DPDT','219个入板器件+1个板外DPDT')
 if intro not in text:text=text.split('\n',1)[0]+'\n\n'+intro+'\n'+text.split('\n',1)[1]
 path.write_text(text,encoding='utf8')
path=M/'TODO.md';text=path.read_text(encoding='utf8')
text+='\n本轮已完成：6.8kΩ风扇限流电阻归并为四颗已有料号；219板内器件/52种MPN。预充POR/锁存/独立超时保留的具体电源域与故障依据见保护料号复核报告。若继续合并锁存与BQ功能，须先完成独立关断、默认态及故障恢复的替代设计；不把物理OFF-ON重试策略当成用户固定要求。\n'
path.write_text(text,encoding='utf8')
d=load(P/'DRC逐项核对.json');snap=load(O/'保存重开器件核对.json')
assert snap['drc']==[{'type':'warn','count':8}]
d['currentP1SHA256']=sha(M/'eda工程文件/sch/Schematic1/P1.esch2')
d['currentNetlistSHA256']=sha(O/'final_reopened.enet')
d['currentOfficialResult']['snapshot']='必要保护精简/保存重开器件核对.json'
d['currentOfficialResult']['classificationBoundary']='本轮官方API重新计数0错误/8警告；旧日志逐项分类保留作为基线，API未返回新的详细警告清单，不将数量相同当成逐项重验。'
(P/'DRC逐项核对.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
path=P/'DRC逐项核对.md';text=path.read_text(encoding='utf8')
text+='\n必要保护精简后官方保存重开复跑：0错误、8警告。最新计数证据见必要保护精简/保存重开器件核对.json；API本次返回数量，原日志分类为基线，不宣称已重新导出逐条清单或严格DRC通过。\n';path.write_text(text,encoding='utf8')
print(json.dumps({'pcb':219,'mpn':52,'CAD':220,'pins':729,'strictDRC':False},ensure_ascii=False))
