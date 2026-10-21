"""Reframe official PDF (renderer caps original page), preserve vectors, add ref navigation."""
import fitz,json,hashlib,re,sys
from pathlib import Path
P=Path(__file__).resolve().parent;O=P/(sys.argv[1] if len(sys.argv)>1 else '电源调速精简落实')
raw=O/'BMS_current_raw.pdf';d=fitz.open(raw);p=d[0];stream=d.xref_stream(p.get_contents()[0])
m=re.search(rb'0\.00 14400\.00 ([0-9.]+) -([0-9.]+) re',stream);assert m
w,h=map(float,m.groups());p.set_mediabox(fitz.Rect(0,14400-h,w,14400))
refs=set(json.loads((O/'final_reopened.enet').read_text(encoding='utf8'))['components']);hits={ref:[] for ref in refs}
for b in p.get_text('dict')['blocks']:
 for line in b.get('lines',[]):
  s=''.join(x['text'] for x in line['spans']).strip()
  if s in hits:hits[s].append(fitz.Rect(line['bbox']))
assert all(len(v)==1 for v in hits.values()),{k:len(v) for k,v in hits.items() if len(v)!=1}
out=fitz.open();q=out.new_page(width=w*.5,height=h*.5);q.show_pdf_page(q.rect,d,0)
out.set_toc([[1,ref,1,{'kind':fitz.LINK_GOTO,'page':0,'to':fitz.Point(hits[ref][0].x0*.5,hits[ref][0].y0*.5),'zoom':8}] for ref in sorted(refs)])
out.set_metadata({'title':'BMS当前原理图审阅','subject':'官方PDF矢量重框与位号书签；不构成ERC或实物验收','creator':'EasyEDA official PDF + PyMuPDF reframe'})
path=P/'当前原理图导航审阅.pdf';out.save(path,garbage=4,deflate=True);out.close()
f=fitz.open(path);assert len(f.get_toc())==len(refs)
checks={'sourceOfficialPDF':str(raw.relative_to(P)),'sourceSha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'outputSha256':hashlib.sha256(path.read_bytes()).hexdigest(),'refCount':len(refs),'allRefsExactlyOnce':True,'bookmarkCount':len(f.get_toc()),'nativeModifiedByPDFPostprocess':False,'pageSizePt':list(f[0].rect),'visualReview':'pending'}
# Full overview plus local circuit crops for review.
f[0].get_pixmap(matrix=fitz.Matrix(.12,.12),alpha=False).save(O/'原理图总览.png')
for name,selected in [('电源',['PSU5_EXT','SW_RUN_STORE','R_PSU_MINLOAD1','R_PSU_MINLOAD7']),('风扇直供',['J_FAN']),('调速',['Q_FAN_PWM','R_FAN_PWM_GATE','R_FAN_PWM_PD','R_FAN_TACH_PU','J_FAN_CTRL']),('与门',['U_AND2','C_AND2']),('未用输出NC',['U_SEQ'])]:
 boxes=[hits[r][0] for r in selected];rect=fitz.Rect(min(b.x0 for b in boxes)*.5-120,min(b.y0 for b in boxes)*.5-120,max(b.x1 for b in boxes)*.5+180,max(b.y1 for b in boxes)*.5+150);rect=rect & f[0].rect
 scale=4 if name=='未用输出NC' else 1
 f[0].get_pixmap(matrix=fitz.Matrix(scale,scale),clip=rect,alpha=False).save(O/(name+'审阅.png'))
(P/'当前原理图导航审阅.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8');print(checks)
for name,ref in [('风扇直供细节','J_FAN'),('风扇PWM细节','Q_FAN_PWM'),('风扇测速细节','R_FAN_TACH_PU'),('模块细节','PSU5_EXT')]:
 b=hits[ref][0];rect=fitz.Rect(b.x0*.5-35,b.y0*.5-40,b.x1*.5+70,b.y1*.5+45)&f[0].rect
 f[0].get_pixmap(matrix=fitz.Matrix(8,8),clip=rect,alpha=False).save(O/(name+'.png'))
