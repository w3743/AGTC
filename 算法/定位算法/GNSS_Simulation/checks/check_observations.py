"""误差源量级自检：核对各类误差是否落在真实合理区间。

运行: python checks/check_observations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnss_sim import constellation as con
from gnss_sim import models as md
from gnss_sim import geodesy as geo
from gnss_sim.config import (ScenarioConfig, SoilConfig, IonosphereConfig,
                             ProbeSpec, T_START_UNIX)
from gnss_sim.observation import WorldModel


def stat(name, x, unit="m", scale=1.0):
    x = np.asarray(x, dtype=float) * scale
    print(f"  {name:34s} 均值={np.mean(x):9.4f}  标准差={np.std(x):8.4f} "
          f" 范围=[{np.min(x):8.4f}, {np.max(x):8.4f}] {unit}")


def main():
    scn = ScenarioConfig()
    const = con.build_constellation(scn.const.systems, scn.session.t_start_unix)
    world = WorldModel(scn, const, seed=1, dual_freq=True).prepare_session()
    print(f"历元数={len(world.times)}, 卫星数={len(const)}")

    probe_spec = ProbeSpec(mount="ground", antenna="lowcost")
    ant = md.AntennaModel(md.ANTENNA_LIBRARY["lowcost"], world.rng)
    mp = md.MultipathModel(scn.soil, ant, world.rng)
    h = probe_spec.resolved_height()
    rx = world.make_probe(0, np.array([10.0, -20.0, 0.0]), probe_spec, ant, mp, h)
    print(f"探针高度={h:.3f} m, 仰角掩蔽={rx.el_mask} deg")
    print(f"土壤含水率={scn.soil.moisture}, 介电常数={mp.er_real:.2f}-j{mp.er_imag:.2f}")

    rec = world.build_record(rx, 0)
    t = rec.t
    gps_sod = t % 86400.0
    print(f"\n历元 0: 可见卫星 {len(rec)} 颗")
    for s in np.unique(rec.systems):
        print(f"    {s}: {int(np.sum(rec.systems == s))} 颗")

    # --- 电离层 ---------------------------------------------------------
    el = None
    los = rec.sat_true - rx.ecef
    enu = los @ world.frame.rot.T
    el = np.degrees(np.arctan2(enu[:, 2], np.hypot(enu[:, 0], enu[:, 1])))
    az = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360.0
    print("\n[电离层]")
    iono_true = rec.f1.iono
    stat("真值斜距延迟 (L1)", iono_true)
    klob = md.klobuchar_delay(world.alpha, world.beta, rx.lat, rx.lon, az, el, gps_sod)
    stat("Klobuchar 广播模型", klob)
    stat("单频改正残差 |真值-模型|", np.abs(iono_true - klob))
    stat("残差/真值 比例", np.abs(iono_true - klob) / np.maximum(iono_true, 1e-9),
         unit="", scale=1.0)

    # --- 对流层 ---------------------------------------------------------
    print("\n[对流层]")
    trop_true = rec.trop_true
    trop_model = world.field_tropo.model_delay(az, el)
    stat("真值斜距延迟", trop_true)
    stat("标准模型", trop_model)
    stat("模型残差", np.abs(trop_true - trop_model))
    print(f"  天顶干延迟={world.field_tropo.zhd:.4f} m, "
          f"天顶湿延迟={world.field_tropo.zwd_series[0]:.4f} m")

    # --- 多路径 ---------------------------------------------------------
    print("\n[地表多路径]")
    lam = rec.f1.lam
    stat("码多路径", rec.f1.mp_code, unit="m")
    stat("载波相位多路径", rec.f1.mp_phase, unit="mm", scale=1000)
    stat("载波相位多路径绝对值", np.abs(rec.f1.mp_phase), unit="mm", scale=1000)
    stat("相位中心变化 PCV", rec.f1.pcv, unit="mm", scale=1000)

    # 不同含水率下的多路径
    print("\n  土壤含水率对载波多路径的影响（20 度仰角）:")
    for theta in (0.05, 0.10, 0.20, 0.30, 0.40):
        soil = SoilConfig(moisture=theta)
        mp2 = md.MultipathModel(soil, ant, np.random.default_rng(2))
        c, p, a = mp2.errors(np.array([20.0]), h, 0.1903,
                             np.array([0.5]), np.array([1.0]))
        print(f"    theta_v={theta:.2f}  er={mp2.er_real:6.2f}  "
              f"码多路径={c[0]*1000:7.3f} mm  载波多路径={p[0]*1000:7.3f} mm  "
              f"反射幅度比={a[0]-1:+.4f}")

    # --- C/N0 与噪声 ----------------------------------------------------
    print("\n[载噪比与观测噪声]")
    stat("C/N0", rec.cn0, unit="dB-Hz")
    stat("码噪声 sigma", rec.f1.sigma_code, unit="m")
    stat("相位噪声 sigma", rec.f1.sigma_phase, unit="mm", scale=1000)

    # --- 星历/钟差 ------------------------------------------------------
    print("\n[星历与卫星钟差]")
    orb_err = world.sat_err.orbit_error_ecef(rec.sat_true, t, idx=rec.idx)
    stat("广播星历轨道误差模", np.linalg.norm(orb_err, axis=-1), unit="m")
    clk_true = world.sat_err.clock_true(t)[rec.idx]
    clk_bc = world.sat_err.clock_broadcast(t)[rec.idx]
    stat("广播钟差误差", np.abs(clk_bc - clk_true), unit="m")
    # 视线方向投影
    los_hat = (rec.sat_true - rx.ecef) / np.linalg.norm(rec.sat_true - rx.ecef, axis=-1, keepdims=True)
    stat("轨道误差视线投影", np.abs(np.sum(orb_err * los_hat, axis=-1)), unit="m")

    # --- 双频消电离层 ---------------------------------------------------
    print("\n[双频消电离层组合]")
    f1 = rec.freqs["f1"].freq
    f2 = rec.freqs["f2"].freq
    a1 = f1 ** 2 / (f1 ** 2 - f2 ** 2)
    a2 = -f2 ** 2 / (f1 ** 2 - f2 ** 2)
    i_if = a1 * rec.freqs["f1"].iono + a2 * rec.freqs["f2"].iono
    stat("IF 组合残余电离层", np.abs(i_if), unit="mm", scale=1000)
    mp_if = a1 * rec.freqs["f1"].mp_phase + a2 * rec.freqs["f2"].mp_phase
    stat("IF 组合多路径", np.abs(mp_if), unit="mm", scale=1000)

    # --- 历元间一致性 ---------------------------------------------------
    print("\n[时段统计: 60 min 内可见卫星数与 DOP]")
    nv, hdop, vdop = [], [], []
    for i in range(0, len(world.times), 10):
        r = world.build_record(rx, i)
        nv.append(len(r))
        enu = (r.sat_true - rx.ecef) @ world.frame.rot.T
        e = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360.0
        e2 = np.degrees(np.arctan2(enu[:, 2], np.hypot(enu[:, 0], enu[:, 1])))
        d = con.dop_from_geometry(e, e2)
        hdop.append(d["hdop"]); vdop.append(d["vdop"])
    print(f"  可见卫星 {np.mean(nv):.1f} 颗 [{np.min(nv)},{np.max(nv)}], "
          f"HDOP={np.mean(hdop):.2f}, VDOP={np.mean(vdop):.2f}")


if __name__ == "__main__":
    main()
