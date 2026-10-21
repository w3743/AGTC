"""Refresh current entry documents after NC publication and evidence checks."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;M=P.parent
replacements={
 M/'README.md':[('DRC为0错误、10条警告已分类','DRC为0错误、8条警告（原日志10条，2条未用输出已改NC）')],
 P/'当前设计状态.md':[('0错误、10条warning已分类','0错误、8条warning已分类'),('3条单针网络、7组属性差异','原3条单针网络已关闭2条，剩板外预充1条及7组属性差异')],
 M/'TODO.md':[('DRC已分类：0错误、10警告；保留未用反相输出和板外预充电阻边界说明','DRC已分类：0错误、8警告；U_SEQ未用6/8脚已改NC，剩板外预充电阻边界说明')],
 M/'项目要求.md':[('DRC聚合10条warning尚无逐条闭合','DRC原10条已分类；未用反相输出6/8针改NC后保存重开剩8条（1条板外预充边界、7组供应商属性差异），严格DRC未闭合')]
}
for path,pairs in replacements.items():
 text=path.read_text(encoding='utf-8-sig')
 for old,new in pairs:
  assert old in text,(path,old);text=text.replace(old,new)
 path.write_text(text,encoding='utf8')
path=P/'当前原理图导航审阅.json';data=json.loads(path.read_text(encoding='utf8'))
data['visualReview']='已查看本轮U_SEQ局部官方PDF：6/8针NC叉标可见，原反相输出端口已移除；217位号各出现一次并有书签。其它区域沿用前轮审阅，不构成全板制造验收。'
data['publishedNativeHashes']=json.loads((P/'剩余验证/原生发布校验.json').read_text(encoding='utf8'))
path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
print('current entry documents updated')
