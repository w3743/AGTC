"""线性掉电与手册典型UV阈值条件时序；非DRV厂家仿真模型。"""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
cases=[]
for ramp in (.1e-6,5e-6,15e-6,100e-6,.010):
 t_uv=ramp*(12-6.7)/12;t4=ramp*(12-4)/12
 for delay,label in [(10e-6,'典型10us'),(15e-6,'最大15us')]:
  assertion=t_uv+delay
  cases.append({'ramp_s':ramp,'UV_typ_cross_s':t_uv,'VDD_4V_s':t4,'assumed_deglitch_s':delay,
   'assumed_fault_s':assertion,'fault_occurs_before_4V':assertion<t4,'timing_label':label})
out={'status':'CONDITIONAL_BEHAVIOURAL_TIMING_NOT_MANUFACTURER_TRANSIENT_MODEL',
 'assumptions':['12V线性降至0','固定采用GVDD欠压典型下降阈值6.7V，不是最坏保证值',
  '10us典型/15us最大去毛刺简化为固定延时；忽略GVDD内部动力学','4V仅对应nFAULT VOL规格的供电条件边界',
  '假设去毛刺完成时会拉低nFAULT；不推断低于4V时线电平，不模型门极放电、开关能量、LTC7001或固件'],
 'cases':cases,'interpretation':'若估算输出在4V之后，不能从给定VOL保证推断故障线有效；即使之前也不是全温设计验收。'}
(ROOT/'仿真/结果/DRV快速掉电_条件时序筛查.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(cases,ensure_ascii=False))
