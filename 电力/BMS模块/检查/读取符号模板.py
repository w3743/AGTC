import sqlite3, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
c = sqlite3.connect(f'file:{(root / "eda工程文件/BMS-9S4P.eprj2").as_posix()}?mode=ro', uri=True)
rows = c.execute("select uuid,title,dataStr from components").fetchall()
for uid,title,src in rows:
    if title in ['WSKW0612L5000FEA','D','Zener'] or 'WSKW' in title:
        (root/'检查'/f'模板-{title}.txt').write_text(src, encoding='utf-8')
        print(title, src[:6000])
print([(uid,title,len(src)) for uid,title,src in rows])
for name,sql in c.execute("select name,sql from sqlite_master where type='table'"):
    if name in ['documents','db_paths','schematics','projects','data','sheets']:
        print(name,sql,c.execute('select * from '+name+' limit 2').fetchall().__str__()[:1600])
