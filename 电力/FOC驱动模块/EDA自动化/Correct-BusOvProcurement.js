await eda.dmt_EditorControl.openDocument('90aa6c46493624cf');
const cs=await eda.sch_PrimitiveComponent.getAll(undefined,false);
const r=cs.find(c=>c.getState_Designator()==='R_OVTOP');
const cap=cs.find(c=>c.getState_Designator()==='C_U7');
if(r?.getState_ManufacturerId()!=='RC0603FR-071ML'||cap?.getState_ManufacturerId()!=='C1608X7R1A104K080AE')throw new Error('wrong parts');
const items=await eda.lib_Device.search('RC0603FR-071ML',undefined,undefined,undefined,5,1);
const item=items.find(x=>x.manufacturerId==='RC0603FR-071ML'&&x.supplierId==='C105578');
if(!item)throw new Error('cannot confirm new 1M supplier identity');
await eda.sch_PrimitiveComponent.modify(r,{supplier:item.supplier,supplierId:item.supplierId,
  otherProperty:{...r.getState_OtherProperty(),'Supplier Part ID':item.supplierId,'LCSC Part Name':item.otherProperty['LCSC Part Name'],
    Value:'1MΩ',Tolerance:'±1%','Power(Watts)':'100mW','Voltage-Supply(Max)':'75V','Temperature Coefficient':'±100ppm/°C',
    Datasheet:item.otherProperty.Datasheet,
    'Library Carrier Original Device':'Retained original 162k R0603 carrier (old supplier C861148); fitted part is RC0603FR-071ML / C105578',
    'Footprint Carrier Disclosure':'Existing R0603 geometry retained; procurement uses the stated 1M MPN, not the original 162k library device'}});
await eda.sch_PrimitiveComponent.modify(cap,{manufacturer:'TDK',
  otherProperty:{...cap.getState_OtherProperty(),Value:'100nF','Voltage Rated':'10V','Voltage Rating':'10V',
    'Source Candidate Value / Notes':'100nF, 10V rated TDK C1608X7R1A104K080AE local bypass across DUMP_5V and GND_SYS',
    'Candidate Component Type':'BUS_OV_comparator_supply_bypass'}});
return {saved:await eda.sch_Document.save(),corrected:['R_OVTOP supplier C105578','C_U7 100nF / 10V metadata']};
