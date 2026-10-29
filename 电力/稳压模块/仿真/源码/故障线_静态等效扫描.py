"""DRV_FAULT_N电阻网络静态筛查；DRV内部上拉250k仅典型，内部电压作假设。"""
from pathlib import Path
import json,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[2];EXE=ROOT.parent/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
cases=[]
with tempfile.TemporaryDirectory(prefix='faultdc_') as tmp:
 d=Path(tmp)
 for mode in ('released','fault_clamped','core_unpowered_high_z','all_unpowered'):
  for core in (3.0,3.3,3.6) if mode in ('released','fault_clamped') else (0.,):
   # 未知内部上拉电压只作为假设；CORE高阻表示未建模的无源节点，不表示实际掉电芯片等效。
   internal=3.3 if mode in ('released','fault_clamped') else 5.0 if mode=='core_unpowered_high_z' else 0.0
   feed='1G' if mode=='core_unpowered_high_z' else '1u'
   clamp='Vclamp fault 0 0.4\n' if mode=='fault_clamped' else ''
   source=f'''Fault pin static resistive approximation\nVcore core 0 {core}\nRfeed core rail {feed}\nRup rail fault 1k\nRdown fault 0 10k\nRltc fault 0 1Meg\nVinternal internal 0 {internal}\nRinternal internal fault 250k\n{clamp}.control\nset noaskquit\nop\nwrdata result.txt v(fault)\nquit\n.endc\n.end\n'''
   (d/'case.cir').write_text(source,encoding='ascii');wave=d/'result.txt';wave.unlink(missing_ok=True)
   p=subprocess.run([str(EXE.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
   log=p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace')
   if p.returncode or 'Error:' in log or not wave.exists():raise RuntimeError(log[-2000:])
   values=wave.read_text().split();voltage=float(values[-1])
   expected=((core/(1000+(1e9 if mode=='core_unpowered_high_z' else 1e-6)))+internal/250000)/(1/(1000+(1e9 if mode=='core_unpowered_high_z' else 1e-6))+1/10000+1/1e6+1/250000)
   if mode=='fault_clamped':expected=.4
   assert abs(expected-voltage)<1e-5,(mode,core,expected,voltage)
   sink=((core-.4)/1000+(internal-.4)/250000-.4/10000-.4/1e6) if mode=='fault_clamped' else None
   cases.append({'mode':mode,'CORE_V':core,'assumed_internal_pullup_V':internal,'fault_voltage_V':voltage,
                 'sink_current_A_at_0_4V':sink,'DRV_input_classification':'HIGH' if voltage>=2.2 else 'LOW' if voltage<=.8 else 'UNDEFINED'})
report={'status':'RESISTIVE_DC_SENSITIVITY_NOT_FAULT_TRANSIENT','cases':cases,
 'scope':'1k上拉、10k下拉、LTC7001输入1M下拉；DRV内部250k为手册典型值，3.3V/5V内部上拉电压和CORE高阻是假设；未含GPIO漏电/ESD反灌、输出漏电、比较器时序、掉电、Break锁存、门极或功率路径。钳位0.4V是测试条件而非实际晶体管模型。'}
(ROOT/'仿真/结果/故障线_静态等效扫描.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
