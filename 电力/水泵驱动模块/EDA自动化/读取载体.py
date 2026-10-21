from eda import *
r=invoke("return await eda.lib_Device.get('f6ea2eab8eb84723a354605d95acd217','0819f05c4eef4c71ace90d822a990e87');")
save('肖特基载体结构.json',r)
print(json.dumps(r,ensure_ascii=True)[:6000])
