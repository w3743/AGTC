from pathlib import Path
import json,shutil,subprocess,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[2];EXE=ROOT.parent/'FOC驱动模块/仿真/tools/ngspice-47-portable/Spice64/bin/ngspice_con.exe'
with tempfile.TemporaryDirectory(prefix='dumpchain_') as tmp:
 d=Path(tmp);(d/'.spiceinit').write_text('set ngbehavior=ps\n',encoding='ascii')
 shutil.copyfile(ROOT/'仿真/源码/母线过压到制动驱动_组合模型.cir',d/'case.cir')
 for package,member,dest in [('snvmc71a.zip','tlv6700.lib','tlv.lib'),('slum488.zip','UCC27511_TRANS.LIB','ucc.lib')]:
  with zipfile.ZipFile(ROOT/'仿真/输入/厂家模型'/package) as z:(d/dest).write_bytes(z.read(member))
 p=subprocess.run([str(EXE.resolve()),'-b','case.cir'],cwd=d,capture_output=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
 log=p.stdout.decode('utf-8','replace')+p.stderr.decode('utf-8','replace');wave=d/'chain_wave.txt'
 if p.returncode or 'Error:' in log or not wave.exists():raise RuntimeError(log[-2500:])
 rows=[list(map(float,l.split())) for l in wave.read_text().splitlines() if len(l.split())>=10]
 if not rows or rows[-1][0]<5.49e-3:raise RuntimeError('波形不完整')
 events=[];state=rows[0][7]>6
 for x in rows[1:]:
  new=x[7]>6
  if new!=state:events.append({'time_s':x[0],'bus_V':x[1],'TLV_OUTA_V':x[3],'DUMP_OFF_V':x[5],'gate_V':x[7],'PWM_V':x[9]});state=new
 get=lambda t:min(rows,key=lambda x:abs(x[0]-t))
 snapshots={str(t):{'bus_V':get(t)[1],'off_V':get(t)[5],'gate_V':get(t)[7],'PWM_V':get(t)[9]} for t in (.0005,.0025,.0042,.0047,.0052)}
 checks={'normal_off':snapshots['0.0005']['gate_V']<1,'hardware_on':snapshots['0.0025']['gate_V']>10,
         'hardware_release':snapshots['0.0042']['gate_V']<1,'software_on':snapshots['0.0047']['gate_V']>10,
         'software_release':snapshots['0.0052']['gate_V']<1}
 result={'status':'TWO_TI_MACROMODELS_GENERIC_NPN_APPROXIMATE_GATE','checks':checks,'events':events,'snapshots':snapshots,
  'scope':'TLV6700、UCC27511厂家模型+按图电阻，NPN采用通用模型BF100，MOS栅极用固定11.36nF；不含真实MMBT5551、IRLB4030模型、功率支路、实际固件及容差。'}
 shutil.copyfile(wave,ROOT/'仿真/结果/母线过压到制动驱动_组合波形.txt')
 (ROOT/'仿真/结果/母线过压到制动驱动_组合汇总.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False))
 if not all(checks.values()):raise SystemExit('存在未通过的逻辑检查')
