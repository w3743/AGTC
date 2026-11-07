from eda import *
r=invoke("const f=await eda.sch_ManufactureData.getExportDocumentFile('水泵驱动原理图','PDF',{theme:'Black on White',lineWidth:'Default'},'Current Schematic',{outputMethod:'Merged sheet'});const data=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(f);});return {name:f.name,size:f.size,data};")
p=ROOT/'eda工程文件/导出/当前交付/水泵驱动原理图.pdf';p.write_bytes(base64.b64decode(r['data']));print('PDF bytes',r['size'])
