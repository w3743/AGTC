from eda import *
p=ROOT/'EDA自动化/连接.js';s=p.read_text(encoding='utf-8-sig').replace("p.ref==='D_5V'||",'');p.write_text(s,encoding='utf8')
print(invoke("const cs=await eda.sch_PrimitiveComponent.getAll();for(const r of ['C_5V_IN','C_5V_OUT']){const c=cs.find(c=>c.getState_Designator()===r);await eda.sch_PrimitiveComponent.modify(c.getState_PrimitiveId(),{rotation:180});}return true;"))
