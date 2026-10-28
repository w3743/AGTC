$ErrorActionPreference='Stop'
$edaCli='D:\lceda-pro\lceda-pro.exe';$edaSession='d89c487f-db54-4a5e-a326-d10278f67ad2'
$manifest=@(Get-Content -LiteralPath 'BMS模块\检查\原理图构建清单.json' -Raw|ConvertFrom-Json)
$bmsMap=Get-Content -LiteralPath 'BMS模块\检查\原理图页面映射.json' -Raw|ConvertFrom-Json
$bmsMap|Add-Member -NotePropertyName 'SW_MANUAL' -NotePropertyValue '6d2dd82fa41b09c5'
$switch=@(Get-Content -LiteralPath 'BMS模块\检查\开关构建清单.json' -Raw|ConvertFrom-Json)
$reports=@()
foreach($group in $bmsMap.PSObject.Properties){
 $entries=if($group.Name -eq 'SW_MANUAL'){$switch}else{@($manifest|Where-Object pageGroup -eq $group.Name)}
 $payload=ConvertTo-Json -InputObject @($entries) -Depth 30 -Compress
 $code=@'
const entries=__ENTRIES__;await eda.dmt_EditorControl.openDocument('__PAGE__');
await eda.dmt_Schematic.modifySchematicPageName('__PAGE__','__GROUP__');
await eda.dmt_Schematic.modifySchematicPageTitleBlock(false);
const deleted=[];
for(const c of await eda.sch_PrimitiveComponent.getAll()){
 if(c.getState_ComponentType()!=='part')continue;
 const e=entries.find(e=>e.ref===c.getState_Designator());
 if(!e){deleted.push(c.getState_Designator());await eda.sch_PrimitiveComponent.delete(c.getState_PrimitiveId());continue;}
 await eda.sch_PrimitiveComponent.modify(c.getState_PrimitiveId(),{name:e.value??e.mpn,manufacturerId:e.mpn??'',addIntoPcb:false,addIntoBom:e.addIntoBom??true,otherProperty:{'Design note':e.note??'','Electrical manifest':e.ref}});
}
await eda.sch_Document.save();
const cs=await eda.sch_PrimitiveComponent.getAll();
const wires=(await eda.sch_PrimitiveWire.getAll()).map(w=>({net:w.getState_Net(),line:w.getState_Line()}));
const ports=[];for(const c of cs.filter(c=>c.getState_ComponentType()==='netport')){for(const p of await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId()))ports.push({name:c.getState_Name(),x:p.x,y:p.y});}
const errors=[],parts=[];
const same=(a,b)=>Math.abs(a-b)<.001;
for(const c of cs.filter(c=>c.getState_ComponentType()==='part')){
 const ref=c.getState_Designator();const e=entries.find(e=>e.ref===ref);const ps=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());const pins=[];
 for(const p of ps){
  const expected=e?.pins[String(p.pinNumber)];
  const at=wires.filter(w=>w.line.length===4&&((same(w.line[0],p.x)&&same(w.line[1],p.y))||(same(w.line[2],p.x)&&same(w.line[3],p.y))));
  const connected=at.map(w=>{const far=(same(w.line[0],p.x)&&same(w.line[1],p.y))?[w.line[2],w.line[3]]:[w.line[0],w.line[1]];return {net:w.net,ports:ports.filter(port=>same(port.x,far[0])&&same(port.y,far[1])).map(port=>port.name)};});
  if(expected===null){if(!p.noConnected||at.length)errors.push({ref,pin:p.pinNumber,kind:'NC mismatch',noConnected:p.noConnected,connected});}
  else if(expected===undefined)errors.push({ref,pin:p.pinNumber,kind:'unexpected pin'});
  else if(p.noConnected||connected.length!==1||connected[0].net!==expected||!connected[0].ports.includes(expected))errors.push({ref,pin:p.pinNumber,kind:'wire/port mismatch',expected,noConnected:p.noConnected,connected});
  pins.push({pin:p.pinNumber,name:p.pinName,x:p.x,y:p.y,noConnected:p.noConnected,expected,connected});
 }
 const absent=Object.entries(e?.pins??{}).filter(([pin,net])=>net!==null&&!ps.some(p=>String(p.pinNumber)===pin));if(absent.length)errors.push({ref,kind:'missing live pins',absent});
 const expectedName=e?.value??e?.mpn;
 if(c.getState_Name()!==expectedName||c.getState_ManufacturerId()!==(e?.mpn??''))errors.push({ref,kind:'value/MPN mismatch',expectedName,actualName:c.getState_Name(),expectedMpn:e?.mpn,actualMpn:c.getState_ManufacturerId()});
 parts.push({ref,name:c.getState_Name(),mpn:c.getState_ManufacturerId(),pins});
}
for(const e of entries){const n=parts.filter(p=>p.ref===e.ref).length;if(n!==1)errors.push({ref:e.ref,kind:'component count',count:n});}
return {group:'__GROUP__',page:'__PAGE__',deleted,partCount:parts.length,wireCount:wires.length,portCount:ports.length,errors,parts};
'@
 $code=$code.Replace('__ENTRIES__',$payload).Replace('__PAGE__',$group.Value).Replace('__GROUP__',$group.Name)
 $r=(& $edaCli invoke --session $edaSession --ext-uuid eda --timeout 60000 --code $code|Out-String|ConvertFrom-Json)
 $reports+=$r
 [pscustomobject]@{group=$group.Name;ok=$r.ok;parts=$r.value.partCount;wires=$r.value.wireCount;errors=@($r.value.errors).Count}|ConvertTo-Json -Compress|Write-Output
 $reports|ConvertTo-Json -Depth 40|Set-Content -LiteralPath 'BMS模块\检查\原理图独立回读.json' -Encoding utf8
 if(!$r.ok){break}
}
