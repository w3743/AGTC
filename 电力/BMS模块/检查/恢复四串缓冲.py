"""Restore the four-bank buffer after capacitor tolerance design review."""
from pathlib import Path
import datetime, hashlib, json, zipfile

m = Path(__file__).resolve().parents[1]
deleted = {'C_BAL_BUFFER4A','C_BAL_BUFFER4B','R_BAL_BUFFER4_BLEED','R_BAL_BUF_OV_T','R_BAL_BUF_OV_B','R_BAL_BUF_OV_LOW','R_BAL_BUF_OV_REF','R_BAL_BUF_OV_GND'}
prefix = '四串缓冲恢复前_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
paths = ['原理图准备/共享电感控制保护连接.json','检查/大量精简退出器件.json','原理图准备/BOM.csv','原理图准备/连接表.csv','检查/本轮直接减件落实.json']
with zipfile.ZipFile(m/'归档/历史与调试.zip','a',compression=zipfile.ZIP_DEFLATED) as z:
    entries = [n for n in z.namelist() if '20261014_3S2P减件前/' in n and n.endswith('共享电感控制保护连接.json')]
    if len(entries) != 1:
        raise RuntimeError('unique original control source required')
    original = z.read(entries[0])
    data = json.loads(original)
    if len(data) != 153 or not deleted.issubset({p['ref'] for p in data}):
        raise RuntimeError('original four bank source incomplete')
    for rel in paths:
        p=m/rel; old=p.read_bytes(); name=prefix+'/'+rel
        z.writestr(name,old)
        if hashlib.sha256(z.read(name)).digest()!=hashlib.sha256(old).digest():
            raise RuntimeError('archive mismatch')
    (m/paths[0]).write_bytes(original)
    p=m/paths[1]; exits=json.loads(p.read_text(encoding='utf-8-sig'))
    p.write_text(json.dumps([r for r in exits if r not in deleted],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=m/paths[-1]; record=json.loads(p.read_text(encoding='utf-8-sig'))
    record['bufferDecision']={'current':'4S2P','reverted':'3S2P','reason':'16F weakest bank with two 24F banks reaches 2.67V at 6.23V total before ESR; incompatible with current 2.50/2.55V bank OVP. Restore eight parts, no new prebalance startup procedure.','netReductionVersus681':31}
    p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'archivePrefix':prefix,'restoredCount':8,'sourceCount':len(data)},ensure_ascii=False))
