async function open(uuid){
 await eda.dmt_EditorControl.openDocument(uuid);
 await new Promise(r=>setTimeout(r,1800));
 if((await eda.dmt_SelectControl.getCurrentDocumentInfo())?.uuid!==uuid)throw Error('page '+uuid);
}
async function cut(wires,components){
 const w=new Set((await eda.sch_PrimitiveWire.getAll()).map(x=>x.getState_PrimitiveId()));
 const c=new Set((await eda.sch_PrimitiveComponent.getAll(undefined,false)).map(x=>x.getState_PrimitiveId()));
 if(wires.some(x=>!w.has(x))||components.some(x=>!c.has(x)))throw Error('cut does not match source');
 if(wires.length&&!await eda.sch_PrimitiveWire.delete(wires))throw Error('cut wires');
 if(components.length&&!await eda.sch_PrimitiveComponent.delete(components))throw Error('cut components');
}
async function wire(net,...lines){
 for(const line of lines)if(!await eda.sch_PrimitiveWire.create(line,net))throw Error('wire '+net);
}
async function port(net,x,y,a){
 if(!await eda.sch_PrimitiveComponent.createNetPort('BI',net,x,y,a,false))throw Error('port '+net);
}
async function save(){
 const ok=await Promise.race([eda.sch_Document.save(),new Promise(r=>setTimeout(()=>r(false),10000))]);
 if(!ok)throw Error('save');
}
const result=[];
if(STAGE==='power'){
 await open('28c1f3740a9b8e04');
 await cut(['e50e414e071f91de','98160eb65cefd46d','46cbe84e31f36cf2'],['db57330e506b0efc','7ef4204f9cdf4daa','711fd32ac0ea3b94']);
 await wire('PACK_GUARD_FUSED_POS',[250,500,350,500,480,500,690,500],[350,500,350,475],[480,500,480,475]);
 await port('PACK_GUARD_FUSED_POS',250,500,180);
 await wire('B0_AUX_EXTERNAL',[350,345,480,345,660,345,660,410,690,410],[350,435,350,345],[480,435,480,345]);
 await port('B0_AUX_EXTERNAL',350,345,270);
 await wire('12V_PROT',[870,500,1050,500,1190,500,1350,500],[1050,500,1050,475],[1190,500,1190,475]);
 await port('12V_PROT',1350,500,0);
 await eda.sch_PrimitiveText.create(200,635,'ISOLATED REVIEW / U_AUX1 AND INPUT CAPACITORS IN BATTERY ENCLOSURE',0,null,null,13);
 await eda.sch_PrimitiveText.create(200,605,'Service disconnect and fuse upstream; only 12V_PROT and its secondary return enter regulator J_GUARD_12V',0,null,null,12);
 await save();result.push('auxiliary');
}
if(STAGE==='drive'){
 await open('28c1f3740a9b8e02');
 await cut(['80f777dd0ebe3734','c557be857950255b','8a3f18c459d87e1f','528ac68bcfc3bffe','bd0ce2b1928a104c',
  '98c2eb6d59af83eb','1fe531081bb8e2d9','a3ff85e468048b43','5932da67dd6d7934','9dc2d15a9a5f13d5','d610ed442a32b922','4671229f7ddc03fe','cefb489f905173de','bf547aa8942fc1e4'],
 ['48a35ae9523e287e','e6da973496e8c35a','f53c1395539401d8','5ff98c8fa66e6339','b1e0590d4e2da930',
  '3e63ac379b5a1102','c86273ec11c7bb30','44f17eb9fbbe3d9b','96b13e267813cbba','1ce9dea8c9ed3465','e4fdfc17b2874eab',
  'da151d8ed3bb9fc8','42e4c6335e8e9a00']);
 await wire('DRV_FAULT_N',[1930,1600,2070,1600,2070,1550]);
 await wire('12V_DRV',[2130,1900,2130,1945]);await port('12V_DRV',2130,1945,90);
 await wire('12V_PROT',[970,400,970,460,970,480],[970,460,1100,460]);await port('12V_PROT',970,480,90);
 await wire('12V_PROT',[2080,700,2080,710]);await port('12V_PROT',2080,710,90);
 await wire('12V_PROT',[1720,450,1720,460]);await port('12V_PROT',1720,460,90);
 await wire('12V_PROT',[2360,880,2360,970,2360,990],[2360,970,2460,970,2550,970,2550,940]);await port('12V_PROT',2360,990,90);
 await wire('GUARD_OK',[2260,780,2180,780]);await port('GUARD_OK',2180,780,180);
 await eda.sch_PrimitiveText.create(2180,1660,'SC isolation drive and async kill moved to protection page',0,null,null,13);
 await save();result.push('driver');
}
if(STAGE==='gate'){
 await open('9546dca0bc8bce7f');
 const old=(await eda.sch_PrimitiveComponent.getAll(undefined,false)).find(c=>c.getState_Designator()==='R_ISO_TGUP');
 if(!old||old.getState_PrimitiveId()!=='9f5a8e272003aaf7')throw Error('original R_ISO_TGUP');
 let c=await eda.sch_PrimitiveComponent.create({libraryUuid:'0819f05c4eef4c71ace90d822a990e87',uuid:'91ed62e3d3584de481255b4ba4b37f6f'},1420,730,undefined,0,false,true,true);
 if(!c)throw Error('create 10 ohm');
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 if(!pins.some(p=>p.getState_X()===1400&&p.getState_Y()===730)||!pins.some(p=>p.getState_X()===1440&&p.getState_Y()===730))throw Error('resistor footprint at pin anchors');
 if(!await eda.sch_PrimitiveComponent.delete(old))throw Error('delete old 10k');
 if(!await eda.sch_PrimitiveComponent.modify(c,{designator:'R_ISO_TGUP',name:'RC0603FR-0710RL'}))throw Error('set resistor ref');
 await save();result.push('gate resistor');
}
return {result};