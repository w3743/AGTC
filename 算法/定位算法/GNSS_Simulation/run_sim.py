"""农田土壤探针 GNSS 定位精度仿真 —— 主运行脚本。

用法:
    python run_sim.py            # 运行全部实验并输出到 results/

输出:
    results/*.csv   原始样本与统计汇总
    results/*.npz   绘图所需的数组
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gnss_sim import constellation as con
from gnss_sim import estimators as est
from gnss_sim import experiments as ex
from gnss_sim import models as md
from gnss_sim import geodesy as geo
from gnss_sim import metrics as mt
from gnss_sim.config import (MAIN_MODES, BaseStationConfig, ConstellationConfig,
                             IonosphereConfig, ProbeSpec, ScenarioConfig,
                             SoilConfig, T_START_UNIX)
from gnss_sim.observation import WorldModel

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

N_LAYOUTS_MAIN = 30
N_PROBES = 20
N_LAYOUTS_SENS = 15
N_PROBES_SENS = 10

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def banner(msg):
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
def experiment_main() -> pd.DataFrame:
    """主实验：三种基准站配置下的全模式精度。"""
    banner("实验 1/6  主实验：农田场景下各定位模式精度（蒙特卡洛）")
    cases = [
        ("A 本地基准站(坐标精确标定)",
         ScenarioConfig(base=BaseStationConfig(kind="local",
                                               coordinate_source="known"))),
        ("B 本地基准站(坐标为单点定位测得)",
         ScenarioConfig(base=BaseStationConfig(kind="local",
                                               coordinate_source="spp",
                                               spp_error_m=1.0))),
        ("C 单基站长基线RTK(15km外CORS)",
         ScenarioConfig(base=BaseStationConfig(kind="cors", distance_km=15.0,
                                               coordinate_source="cors"))),
    ]
    frames, summaries, rels = [], [], []
    for i, (label, scn) in enumerate(cases):
        res = ex.run_case(scn, label, n_layouts=N_LAYOUTS_MAIN, n_probes=N_PROBES,
                          seed=10, modes=MAIN_MODES, ppp_every=4)
        df = pd.DataFrame(res["samples"])
        frames.append(df)
        rows = ex.summarize(res["samples"], label)
        summaries.append(pd.DataFrame(rows))
        for m, kind in (("spp_multi", "instant"), ("sbas", "instant"),
                        ("dgnss", "instant"), ("rtk_inst", "instant"),
                        ("rtk", "static"), ("ppp", "static")):
            r = ex.relative_accuracy_by_epoch(res["samples"], label, m, kind=kind)
            if r:
                rels.append(r)
    samples = pd.concat(frames, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    samples.to_csv(OUT / "samples_main.csv", index=False)
    summary.to_csv(OUT / "summary_main.csv", index=False)
    pd.DataFrame(rels).to_csv(OUT / "relative_accuracy.csv", index=False)

    cols = ["case", "kind", "mode", "n", "h_rms", "h_cep50", "h_cep95",
            "v_rms", "v_p95", "d3_rms", "zone_err_6m", "fix_rate"]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print("\n相对定位精度（同历元探针两两差分）:")
    print(pd.DataFrame(rels).to_string(index=False,
                                       float_format=lambda v: f"{v:8.4f}"))
    return summary


# ---------------------------------------------------------------------------
def experiment_sensitivity() -> pd.DataFrame:
    """单因素敏感性分析。"""
    banner("实验 2/6  敏感性分析：土壤/天线/星座/电离层/作物")
    base = ScenarioConfig()
    variants = []

    for theta, tag in ((0.05, "干土 θ=0.05"), (0.20, "适中 θ=0.20"),
                       (0.35, "湿土 θ=0.35")):
        variants.append((f"土壤含水率: {tag}",
                         replace(base, soil=SoilConfig(moisture=theta,
                                                       roughness_m=0.02))))
    for sfu, storm, tag in ((90, False, "宁静 SFU=90"), (150, False, "中等 SFU=150"),
                            (220, False, "活跃 SFU=220"),
                            (220, True, "磁暴 SFU=220+storm")):
        variants.append((f"电离层: {tag}",
                         replace(base, iono=IonosphereConfig(sfu=sfu, storm=storm))))
    variants.append(("天线: 测量型扼流圈(贴地)",
                     replace(base, probe=ProbeSpec(mount="ground",
                                                   antenna="geodetic"))))
    variants.append(("安装: 短杆架高 0.6~1.2 m",
                     replace(base, probe=ProbeSpec(mount="mast",
                                                   antenna="lowcost"))))
    for systems, tag in ((("G",), "仅 GPS"), (("G", "C"), "GPS+BDS"),
                         (("G", "C", "E"), "GPS+BDS+Galileo"),
                         (("G", "C", "E", "R"), "GPS+BDS+GAL+GLO")):
        variants.append((f"星座: {tag}",
                         replace(base, const=ConstellationConfig(systems=systems))))
    variants.append(("低矮作物: 出苗期 15 cm",
                     replace(base, soil=SoilConfig(moisture=0.20,
                                                   canopy_height_m=0.15,
                                                   canopy_atten_db=1.5))))
    variants.append(("地表粗糙度: 刚耕翻 5 cm",
                     replace(base, soil=SoilConfig(moisture=0.20,
                                                   roughness_m=0.05))))

    rows = []
    for i, (label, scn) in enumerate(variants):
        # 所有变体共用同一批随机布局与大气实现（公共随机数法），
        # 使差异只来自被改变的那个因素
        res = ex.run_case(scn, label, n_layouts=N_LAYOUTS_SENS,
                          n_probes=N_PROBES_SENS, seed=200,
                          modes=("spp_multi", "sbas", "dgnss", "rtk", "ppp"),
                          ppp_every=2, verbose=False)
        rows += ex.summarize(res["samples"], label)
        print(f"  [{i+1}/{len(variants)}] {label}  (样本 {len(res['samples'])})",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary_sensitivity.csv", index=False)
    return df


# ---------------------------------------------------------------------------
def experiment_error_budget() -> pd.DataFrame:
    """误差预算：逐项关闭误差源，观察精度变化。"""
    banner("实验 3/6  误差预算：逐项关闭误差源")
    base = ScenarioConfig()
    items = [("全部误差源(基准)", ()),
             ("关闭 电离层残差", ("iono",)),
             ("关闭 对流层残差", ("tropo",)),
             ("关闭 地表多径", ("multipath",)),
             ("关闭 星历/钟差误差", ("ephemeris",)),
             ("关闭 观测噪声", ("noise",)),
             ("关闭 天线相位中心变化", ("pcv",))]
    rows = []
    for i, (label, dis) in enumerate(items):
        scn = replace(base, disable=dis)
        res = ex.run_case(scn, label, n_layouts=N_LAYOUTS_SENS,
                          n_probes=N_PROBES_SENS, seed=400,
                          modes=("spp_multi", "sbas", "dgnss", "rtk", "ppp"),
                          ppp_every=2, verbose=False)
        rows += ex.summarize(res["samples"], label)
        print(f"  [{i+1}/{len(items)}] {label}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary_error_budget.csv", index=False)
    return df


# ---------------------------------------------------------------------------
def experiment_occupation_time() -> pd.DataFrame:
    """观测时长对精度的影响：时段平均 SPP 与 PPP 收敛。"""
    banner("实验 4/6  观测时长：静态平均与 PPP 收敛")
    scn = ScenarioConfig()
    scn.session.duration_s = 7200.0          # 2 h
    const = con.build_constellation(scn.const.systems, scn.session.t_start_unix)
    windows_min = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120]
    rows = []
    n_probe = 6
    for li in range(n_probe):
        seed_i = scn.seed + 7000 + li
        rng = np.random.default_rng(seed_i)
        world = WorldModel(scn, const, seed=seed_i, dual_freq=False)
        world.prepare_session()
        probes = ex.make_probe_set(world, replace(scn, n_probes=1), rng)
        rx, enu = probes[0]
        appr = enu + rng.normal(0, 6.0, 3)
        # 预生成全部历元观测（双频），供各时间窗复用
        recs = [world.build_record(rx, i, dual=True)
                for i in range(len(world.times))]
        for w in windows_min:
            n_ep = max(2, int(w * 60 / scn.session.interval_s))
            sub = recs[:n_ep]
            # SPP 静态平均
            errs = []
            for k in range(0, n_ep, max(1, n_ep // 10)):
                s = est.solve_spp(sub[k], est.BROADCAST, world, appr)
                if np.all(np.isfinite(s.enu)):
                    errs.append(s.enu - enu)
            if errs:
                e = np.mean(errs, axis=0)
                rows.append(dict(mode="spp_multi(时段平均)", window_min=w,
                                 layout=li, ee=e[0], en=e[1], eu=e[2]))
            # PPP
            sol, _ = est.solve_ppp(sub, world, appr, ztd_bins=max(1, w // 15))
            if np.all(np.isfinite(sol.enu)):
                e = sol.enu - enu
                rows.append(dict(mode="ppp", window_min=w, layout=li,
                                 ee=e[0], en=e[1], eu=e[2]))
        print(f"  [{li+1}/{n_probe}] 探针 PPP/平均解算完成", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "occupation_time.csv", index=False)
    agg = []
    for (m, w), g in df.groupby(["mode", "window_min"]):
        e = g[["ee", "en", "eu"]].to_numpy()
        h = np.hypot(e[:, 0], e[:, 1])
        agg.append(dict(mode=m, window_min=w, n=len(e),
                        h_rms=float(np.sqrt(np.mean(h ** 2))),
                        h_p95=float(np.percentile(h, 95)),
                        h_median=float(np.median(h)),
                        v_rms=float(np.sqrt(np.mean(e[:, 2] ** 2))),
                        v_median=float(np.median(np.abs(e[:, 2])))))
    agg = pd.DataFrame(agg)
    agg.to_csv(OUT / "occupation_time_summary.csv", index=False)
    print(agg.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    return agg


# ---------------------------------------------------------------------------
def experiment_geometry() -> pd.DataFrame:
    """24 h 卫星几何与 DOP。"""
    banner("实验 5/6  卫星几何：24 h 可见性与 DOP")
    scn = ScenarioConfig()
    frame = geo.LocalFrame(scn.field.lat0, scn.field.lon0, scn.field.h0)
    rx = frame.enu_to_ecef([0.0, 0.0, 0.0])
    combos = [("G",), ("C",), ("E",), ("G", "C"), ("G", "C", "E"),
              ("G", "C", "E", "R")]
    times = T_START_UNIX + np.arange(0, 86400, 60.0)
    rows = []
    tracks = {}
    for systems in combos:
        c = con.build_constellation(systems, T_START_UNIX)
        az, el = c.sky_track(frame, rx, times, el_mask=0.0)
        n_vis, hd, vd, pdp = [], [], [], []
        for k in range(len(times)):
            m = el[k] >= 10.0
            d = con.dop_from_geometry(az[k][m], el[k][m])
            n_vis.append(int(np.sum(m)))
            hd.append(d["hdop"]); vd.append(d["vdop"]); pdp.append(d["pdop"])
        tag = "+".join(systems)
        rows.append(dict(systems=tag, n_vis_mean=float(np.mean(n_vis)),
                         n_vis_min=int(np.min(n_vis)),
                         hdop_mean=float(np.mean(hd)),
                         hdop_max=float(np.max(hd)),
                         vdop_mean=float(np.mean(vd)),
                         pdop_mean=float(np.mean(pdp)),
                         pdop_max=float(np.max(pdp))))
        tracks[tag] = dict(az=az, el=el)
        print(f"  {tag:12s} n_vis={np.mean(n_vis):5.1f} HDOP={np.mean(hd):5.2f} "
              f"VDOP={np.mean(vd):5.2f} PDOP={np.mean(pdp):5.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "geometry_dop.csv", index=False)
    np.savez_compressed(OUT / "geometry_tracks.npz",
                        times=times,
                        **{f"{k}_{f}": v[f] for k, v in tracks.items()
                           for f in ("az", "el")})
    return df


# ---------------------------------------------------------------------------
def experiment_multipath_soil() -> pd.DataFrame:
    """土壤含水率/天线类型/仰角对多路径的影响（物理机理展示）。"""
    banner("实验 6/6  地表多路径随土壤含水率与仰角的变化")
    rows = []
    el_grid = np.arange(10.0, 86.0, 5.0)
    for ant_name in ("lowcost", "geodetic", "patch_mast"):
        ant = md.AntennaModel(md.ANTENNA_LIBRARY[ant_name],
                              np.random.default_rng(5))
        for theta in (0.05, 0.10, 0.20, 0.30, 0.40):
            soil = SoilConfig(moisture=theta)
            mp = md.MultipathModel(soil, ant, np.random.default_rng(6))
            # 相干分量（漫反射置零，展示纯物理趋势）
            code, phase, amp = mp.errors(el_grid, 0.20, 0.1903)
            for k, el in enumerate(el_grid):
                rows.append(dict(antenna=ant_name, moisture=theta, el=el,
                                 permittivity=mp.er_real,
                                 code_mp_mm=code[k] * 1000,
                                 phase_mp_mm=phase[k] * 1000))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "multipath_vs_soil.csv", index=False)
    print(df.groupby(["antenna", "moisture"])["phase_mp_mm"]
          .agg(["min", "max"]).to_string(float_format=lambda v: f"{v:8.3f}"))
    return df


# ---------------------------------------------------------------------------
def experiment_layout_map() -> None:
    """单个代表性 layout 的探针位置与各模式误差（用于绘田块图）。"""
    banner("附加  代表性田块布局与误差分布")
    scn = ScenarioConfig()
    const = con.build_constellation(scn.const.systems, scn.session.t_start_unix)
    rng = np.random.default_rng(2025)
    world = WorldModel(scn, const, seed=2025, dual_freq=False)
    world.prepare_session()
    probes = ex.make_probe_set(world, scn, rng)
    base, base_known = ex.make_base_known(world, scn, rng)
    i_ep = 0
    base_rec = world.build_record(base, i_ep, dual=False)
    rows = []
    for pi, (rx, enu) in enumerate(probes):
        rec = world.build_record(rx, i_ep, dual=False)
        appr = enu + rng.normal(0, 6.0, 3)
        sols = {
            "spp_gps": est.solve_spp(rec, est.BROADCAST, world, appr,
                                     systems=("G",)),
            "spp_multi": est.solve_spp(rec, est.BROADCAST, world, appr),
            "sbas": est.solve_spp(rec, est.SBAS, world, appr),
            "dgnss": est.solve_dgnss(rec, base_rec, est.BROADCAST, world,
                                     base_known, appr),
        }
        # 静态 RTK
        recs_r = [world.build_record(rx, i, dual=False) for i in ex.RTK_EPOCHS]
        recs_b = [world.build_record(base, i, dual=False) for i in ex.RTK_EPOCHS]
        sols["rtk"] = est.solve_rtk_static(recs_r, recs_b, est.BROADCAST, world,
                                           base_known, appr)
        cn0_mean = float(np.nanmean(rec.cn0))
        for m, s in sols.items():
            rows.append(dict(probe=pi, mode=m, tx=enu[0], ty=enu[1],
                             ee=s.enu[0] - enu[0] if np.all(np.isfinite(s.enu)) else np.nan,
                             en=s.enu[1] - enu[1] if np.all(np.isfinite(s.enu)) else np.nan,
                             eu=s.enu[2] - enu[2] if np.all(np.isfinite(s.enu)) else np.nan,
                             n_sat=s.n_sat, cn0=cn0_mean, fixed=int(s.fixed)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "layout_map.csv", index=False)
    print(df.groupby("mode")[["ee", "en", "eu"]].apply(
        lambda g: np.sqrt(np.mean(g ** 2))).to_string(
        float_format=lambda v: f"{v:8.4f}"))


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    summary = experiment_main()
    experiment_sensitivity()
    experiment_error_budget()
    experiment_occupation_time()
    experiment_geometry()
    experiment_multipath_soil()
    experiment_layout_map()
    banner(f"全部实验完成，用时 {time.time()-t0:.1f}s，结果目录: {OUT}")


if __name__ == "__main__":
    main()
