"""实验驱动冒烟测试：小规模跑通并检查耗时。"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnss_sim import experiments as ex
from gnss_sim.config import ScenarioConfig

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

scn = ScenarioConfig()
t0 = time.time()
res = ex.run_case(scn, "冒烟", n_layouts=2, n_probes=3, seed=99, ppp_every=1)
print(f"总耗时 {time.time()-t0:.1f}s, 样本数 {len(res['samples'])}")

df = pd.DataFrame(res["samples"])
print("\n样本分布:")
print(df.groupby(["kind", "mode"]).size())

summ = pd.DataFrame(ex.summarize(res["samples"], "冒烟"))
cols = ["kind", "mode", "n", "h_rms", "h_cep50", "h_cep95", "v_rms", "v_p95",
        "d3_rms", "zone_err_6m", "fix_rate", "fix_wrong_rate"]
cols = [c for c in cols if c in summ.columns]
print("\n统计汇总:")
print(summ[cols].to_string(index=False,
      float_format=lambda v: f"{v:9.4f}"))
