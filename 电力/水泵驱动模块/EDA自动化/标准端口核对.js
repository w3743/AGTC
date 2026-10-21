const cs=await eda.sch_PrimitiveComponent.getAll(),ws=await eda.sch_PrimitiveWire.getAll(),out=[];
function onLine(x,y,line){const ls=Array.isArray(line[0])?line:[line];for(const l of ls)for(let i=2;i<l.length;i+=2){const [x1,y1,x2,y2]=[l[i-2],l[i-1],l[i],l[i+1]];const cross=(x-x1)*(y2-y1)-(y-y1)*(x2-x1);if(Math.abs(cross)<1e-6&&x>=Math.min(x1,x2)-1e-6&&x<=Math.max(x1,x2)+1e-6&&y>=Math.min(y1,y2)-1e-6&&y<=Math.max(y1,y2)+1e-6)return true;}return false;}
for(const c of cs.filter(c=>c.getState_ComponentType()==='netport')){
 const id=c.getState_PrimitiveId(),x=c.getState_X(),y=c.getState_Y();const attrs=(await eda.sch_PrimitiveAttribute.getAll(id)).filter(a=>a.getState_ParentPrimitiveId()===id);const name=attrs.find(a=>a.getState_Key()==='Name'),desc=attrs.find(a=>a.getState_Key()==='Description');
 const same=ws.filter(w=>w.getState_Net().toUpperCase()===name.getState_Value().toUpperCase()&&onLine(x,y,w.getState_Line()));out.push({id,net:name.getState_Value(),symbol:desc?.getState_Value(),x,y,nameVisible:name.getState_ValueVisible(),sameNetAnchor:same.length>0});
}return {standardPortCount:out.length,allAnchored:out.every(v=>v.sameNetAnchor),allNamesVisible:out.every(v=>v.nameVisible),types:out.reduce((a,v)=>(a[v.symbol]=(a[v.symbol]||0)+1,a),{}),ports:out};

