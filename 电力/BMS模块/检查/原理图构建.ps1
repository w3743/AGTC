param([Parameter(Mandatory=$true)][string]$Manifest,[Parameter(Mandatory=$true)][string]$Page,[int]$Start=0,[int]$Count=12)
$ErrorActionPreference='Stop'
$edaCli='D:\lceda-pro\lceda-pro.exe'
$edaSession='d89c487f-db54-4a5e-a326-d10278f67ad2'
$items=@(Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json)
$selected=@($items | Select-Object -Skip $Start -First $Count)
$payload=ConvertTo-Json -InputObject $selected -Depth 30 -Compress
$template=@'
const entries=__ENTRIES__;
const libraryUuid='0819f05c4eef4c71ace90d822a990e87';
const results=[];
await eda.dmt_Schematic.modifySchematicPageTitleBlock(false);
let existing=await eda.sch_PrimitiveComponent.getAll();
for(const p of existing.filter(c=>c.getState_ComponentType()==='netport'))await eda.sch_PrimitiveComponent.delete(p.getState_PrimitiveId());
for(const w of await eda.sch_PrimitiveWire.getAll())await eda.sch_PrimitiveWire.delete(w.getState_PrimitiveId());
for(let i=0;i<entries.length;i++){
 const e=entries[i];
 let c=existing.find(c=>c.getState_Designator()===e.ref);
 const wasExisting=Boolean(c);
 if(!c){
  const x=e.x??(240+(i%3)*330),y=e.y??(130+Math.floor(i/3)*145);
  c=await eda.sch_PrimitiveComponent.create({libraryUuid,uuid:e.libUuid,libraryType:e.libType??'3'},x,y,e.subPartName,e.rotation??0,false,true,false);
  if(!c)throw new Error('无法创建 '+e.ref+' '+e.mpn);
 }
 await eda.sch_PrimitiveComponent.modify(c.getState_PrimitiveId(),{x:e.x??(240+(i%3)*600),y:e.y??(130+Math.floor(i/3)*350),designator:e.ref,name:e.value??e.mpn??e.ref,manufacturerId:e.mpn??'',addIntoPcb:false,addIntoBom:e.addIntoBom??true,otherProperty:{'Design note':e.note??'','Electrical manifest':e.ref}});
 const pins=await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(c.getState_PrimitiveId());
 if(e.pinRemap&&!wasExisting){for(const p of pins){const n=e.pinRemap[String(p.pinNumber)];if(n!==undefined){p.pinNumber=String(n);await p.done();}}}
 const used=[];
 for(const p of pins){
  const key=String(p.pinNumber);
  if(!Object.hasOwn(e.pins,key))throw new Error(e.ref+' 存在未定义引脚 '+key+' '+p.pinName);
  const net=e.pins[key];
  if(net===null){p.noConnected=true;await p.done();used.push({pin:key,name:p.pinName,net:null});continue;}
  const r=((p.rotation%360)+360)%360;
  const d=25;
  const xx=p.x+(r===0?d:r===180?-d:0),yy=p.y+(r===90?d:r===270?-d:0);
  if(![0,90,180,270].includes(r))throw new Error('未知引脚方向 '+e.ref+' '+r);
  await eda.sch_PrimitiveWire.create([p.x,p.y,xx,yy],net);
  const port=await eda.sch_PrimitiveComponent.createNetPort('BI',net,xx,yy,r===180?0:r===0?180:r===90?270:90);
  used.push({pin:key,name:p.pinName,net,x:p.x,y:p.y});
 }
 const absent=Object.keys(e.pins).filter(k=>e.pins[k]!==null&&!pins.some(p=>String(p.pinNumber)===k));
 if(absent.length)throw new Error(e.ref+' manifest脚号不存在 '+absent.join(','));
 results.push({ref:e.ref,id:c.getState_PrimitiveId(),pins:used});
}
const saved=await eda.sch_Document.save();
return {saved,page:await eda.dmt_Schematic.getCurrentSchematicPageInfo(),components:results};
'@
$scriptText=$template.Replace('__ENTRIES__',$payload)
& $edaCli invoke --session $edaSession --ext-uuid eda --code "await eda.dmt_EditorControl.openDocument('$Page');return true;" | Out-String | Write-Output
$buildOutput=& $edaCli invoke --session $edaSession --ext-uuid eda --timeout 60000 --code $scriptText | Out-String
$buildOutput | Write-Output
$buildOutput | Set-Content -LiteralPath (Join-Path $PSScriptRoot ('构建-'+$Page+'-'+$Start+'.json')) -Encoding utf8
