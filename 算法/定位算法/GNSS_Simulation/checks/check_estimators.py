"""解算器自检：各定位模式在单探针上的误差量级。

运行: python checks/check_estimators.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnss_sim import constellation as con
from gnss_sim import estimators as est
from gnss_sim import models as md
from gnss_sim.config import ScenarioConfig, ProbeSpec, BaseStationConfig
from gnss_sim.observation import WorldModel


def rms_2d(err):
    return float(np.sqrt(np.mean(err[:, 0] ** 2 + err[:, 1] ** 2)))


def main():
    scn = ScenarioConfig()
    scn.session.duration_s = 1800.0
    const = con.build_constellation(scn.const.systems, scn.session.t_start_unix)
    world = WorldModel(scn, const, seed=7, dual_freq=True).prepare_session()

    spec = ProbeSpec(mount="ground", antenna="lowcost")
    ant = md.AntennaModel(md.ANTENNA_LIBRARY["lowcost"], world.rng)
    mp = md.MultipathModel(scn.soil, ant, world.rng)
    h = spec.resolved_height()
    enu_true = np.array([23.0, -17.0, h])
    rx = world.make_probe(0, enu_true, spec, ant, mp, h)

    base = world.make_base(BaseStationConfig(kind="local", coordinate_source="known"))
    base_known = base.ecef.copy()
    base_spp = base.ecef + np.array([1.2, -0.6, 0.9])       # 单点测量得到的基准站坐标

    print(f"探针真值 ENU=[{enu_true[0]:.2f}, {enu_true[1]:.2f}, {enu_true[2]:.2f}]")
    epochs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    def run(fn, label, *a, **kw):
        errs, extra = [], []
        for i in epochs:
            rec = world.build_record(rx, i)
            s = fn(rec, *a, **kw)
            if not s.ok or not np.all(np.isfinite(s.enu)):
                extra.append(np.nan)
                continue
            errs.append(s.enu - enu_true)
            extra.append(1.0 if s.fixed else 0.0)
        errs = np.array(errs)
        if len(errs) == 0:
            print(f"{label:28s} 全部失败")
            return
        e2 = np.sqrt(errs[:, 0] ** 2 + errs[:, 1] ** 2)
        print(f"{label:28s} 水平RMS={rms_2d(errs):8.4f} m  水平均值={np.mean(e2):8.4f} m  "
              f"3D RMS={np.sqrt(np.mean(np.sum(errs**2,axis=1))):8.4f} m  "
              f"固定率={np.nanmean(extra)*100:5.1f}%")
        return errs

    appr = enu_true + np.array([6.0, -4.0, 2.0])

    rec0 = world.build_record(rx, 0)
    recb0 = world.build_record(base, 0)
    print(f"\n可见卫星: 探针 {len(rec0)} 颗, 基准站 {len(recb0)} 颗, "
          f"共视 {len(np.intersect1d(rec0.idx, recb0.idx))} 颗")

    print("\n[各模式定位误差（10 个历元，30 s 间隔）]")
    run(lambda r: est.solve_spp(r, est.BROADCAST, world, appr, systems=("G",)),
        "GPS 单系统 SPP")
    run(lambda r: est.solve_spp(r, est.BROADCAST, world, appr), "多系统 SPP")
    run(lambda r: est.solve_spp(r, est.SBAS, world, appr), "SBAS 增强 SPP")
    run(lambda r: est.solve_dgnss(r, world.build_record(base, r.i_epoch),
                                  est.BROADCAST, world, base_known, appr),
        "DGNSS (基准站已知)")
    run(lambda r: est.solve_dgnss(r, world.build_record(base, r.i_epoch),
                                  est.BROADCAST, world, base_spp, appr),
        "DGNSS (基准站SPP坐标)")
    run(lambda r: est.solve_rtk(r, world.build_record(base, r.i_epoch),
                                est.BROADCAST, world, base_known, appr),
        "RTK 单历元 (基准站已知)")
    cors = world.make_base(BaseStationConfig(kind="cors", distance_km=15.0,
                                             coordinate_source="known"))
    run(lambda r: est.solve_rtk(r, world.build_record(cors, r.i_epoch),
                                est.BROADCAST, world, cors.ecef, appr),
        "RTK 单历元 (CORS 15 km)")

    print("\n[静态多历元 RTK（探针静止 10 min）]")
    n_ep = int(600 / scn.session.interval_s)
    for label, bcfg, bknown in (
            ("本地基准站(已知)", base, base_known),
            ("本地基准站(SPP标定)", base, base_spp),
            ("CORS 15 km", cors, cors.ecef)):
        rr = [world.build_record(rx, i) for i in range(n_ep)]
        rb = [world.build_record(bcfg, i) for i in range(n_ep)]
        sol = est.solve_rtk_static(rr, rb, est.BROADCAST, world, bknown, appr)
        if np.all(np.isfinite(sol.enu)):
            e = sol.enu - enu_true
            print(f"  {label:24s} 水平={np.hypot(e[0],e[1])*100:7.2f} cm  "
                  f"高程={e[2]*100:7.2f} cm  固定={sol.fixed}  "
                  f"固定数={sol.n_fixed}/{sol.n_dd}  sigma_N={sol.sigma_amb:.3f}  {sol.note}")
        else:
            print(f"  {label:24s} 失败: {sol.note}")

    # PPP：多时长静止观测
    print("\n[PPP —— 静态多历元解]")
    for win_min in (2, 5, 10, 20, 30):
        n_ep = max(2, int(win_min * 60 / scn.session.interval_s))
        recs = [world.build_record(rx, i) for i in range(n_ep)]
        sol, diag = est.solve_ppp(recs, world, appr, ztd_bins=max(1, win_min // 15))
        if np.all(np.isfinite(sol.enu)):
            e = sol.enu - enu_true
            print(f"  观测 {win_min:2d} min: 水平={np.hypot(e[0],e[1])*100:7.2f} cm  "
                  f"高程={e[2]*100:7.2f} cm  卫星={sol.n_sat}")
        else:
            print(f"  观测 {win_min:2d} min: 解算失败")


if __name__ == "__main__":
    main()
