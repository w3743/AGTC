"""物理模型解析极限自检。

用"模型必须在极端条件下退化到已知解析解"来检验实现是否正确，
而不是只看数值是否"看起来合理"。

运行: python checks/check_physics_limits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnss_sim import constellation as con
from gnss_sim import geodesy as geo
from gnss_sim import models as md
from gnss_sim.config import ScenarioConfig, SoilConfig
from gnss_sim.observation import WorldModel, DUAL_FREQ_HZ

FAIL = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        FAIL.append(name)
    print(f"  [{tag}] {name}  {detail}")


def main():
    print("=" * 74)
    print("1. Fresnel 反射系数的解析极限")
    print("=" * 74)
    er = 4.0
    z = np.array([90.0, 60.0, 30.0, 10.0, 1.0, 0.001])
    gco, gcr = md.fresnel_rhcp(z, er, 0.0)
    for el, a, b in zip(z, gco, gcr):
        print(f"    el={el:7.3f}°  Γ_co={a.real:+.4f}{a.imag:+.4f}j  "
              f"Γ_cross={b.real:+.4f}{b.imag:+.4f}j")
    # 正入射（天顶）：Γ_h = -Γ_v => Γ_co = 0，|Γ_cross| = |(1-√εr)/(1+√εr)|
    expect_nadir = abs((1 - np.sqrt(er)) / (1 + np.sqrt(er)))
    check("正入射 Γ_co = 0（手性完全反转）", abs(gco[0]) < 1e-9,
          f"|Γ_co|={abs(gco[0]):.2e}")
    check("正入射 |Γ_cross| = |(1-√εr)/(1+√εr)|",
          abs(abs(gcr[0]) - expect_nadir) < 1e-9,
          f"|Γ_cross|={abs(gcr[0]):.5f} vs 解析 {expect_nadir:.5f}")
    # 掠射（el→0）：Γ_h = Γ_v = -1 => Γ_co = -1，Γ_cross = 0
    check("掠射 Γ_co → 1（同极化，手性不反转）",
          abs(abs(gco[-1]) - 1.0) < 1e-3, f"|Γ_co|={abs(gco[-1]):.5f}")
    check("掠射 Γ_cross → 0", abs(gcr[-1]) < 1e-3,
          f"|Γ_cross|={abs(gcr[-1]):.2e}")
    # 单调性：反射率随仰角降低而增强（这是修正 Fresnel 仰角约定后的必然结果）
    gco_s, _ = md.fresnel_rhcp(np.array([5.0, 30.0, 85.0]), er, 0.0)
    check("同极化反射率随仰角降低单调增强",
          abs(gco_s[0]) > abs(gco_s[1]) > abs(gco_s[2]),
          f"|Γ_co|: 5°={abs(gco_s[0]):.3f} > 30°={abs(gco_s[1]):.3f} "
          f"> 85°={abs(gco_s[2]):.3f}")
    # 完美导体：任意仰角下手性都必须完全反转（用来锁定仰角约定错误）。
    # 取 εr=1e8 且仰角 ≥10°，避免 εr 有限与 el→0 的非一致极限干扰
    #（εr 有限时掠射极限的收敛条件是 √εr·sin(el) >> 1）。
    z_pec = np.array([10.0, 30.0, 60.0, 90.0])
    gco_p, gcr_p = md.fresnel_rhcp(z_pec, 1e8, 0.0)
    check("完美导体在任意仰角 Γ_co≈0 且 |Γ_cross|≈1（手性全反转）",
          np.all(np.abs(gco_p) < 0.01) and np.all(np.abs(np.abs(gcr_p) - 1) < 0.01),
          f"max|Γ_co|={np.max(np.abs(gco_p)):.4f}, "
          f"|Γ_cross|∈[{np.min(np.abs(gcr_p)):.4f}, {np.max(np.abs(gcr_p)):.4f}]")
    # 同一条水平极化曲线在低仰角必须反射更强（有耗介质）
    check("湿土反射强于干土",
          abs(md.fresnel_rhcp(20.0, 20.9, 2.0)[1]) >
          abs(md.fresnel_rhcp(20.0, 3.85, 0.9)[1]))

    print()
    print("=" * 74)
    print("2. 双射线几何：反射路径差远小于一个码片")
    print("=" * 74)
    chip = geo.C_LIGHT / 1.023e6
    for h in (0.2, 1.0):
        for el in (10.0, 45.0, 90.0):
            d = 2 * h * np.sin(np.radians(el))
            print(f"    h={h:4.1f} m, el={el:5.1f}°: Δ={d * 100:7.2f} cm "
                  f"= {d / chip:.2e} 码片 = {d / 0.1903:.2f} 个 L1 波长")
    d_max = 2 * 1.0
    check("码片延迟比 < 1% （码多径可忽略的物理依据）", d_max / chip < 0.01,
          f"最大 {d_max / chip:.2e} 码片")

    print()
    print("=" * 74)
    print("3. 载波多路径解析上界 |Δφ| <= arcsin(a)·λ/2π")
    print("=" * 74)
    soil = SoilConfig(moisture=0.20)
    ant = md.AntennaModel(md.ANTENNA_LIBRARY["lowcost"], np.random.default_rng(0))
    mp = md.MultipathModel(soil, ant, np.random.default_rng(1))
    el = np.arange(10.0, 86.0, 1.0)
    m, _, _ = mp.coherent_amplitude(el, 0.2, 0.1903)
    a = np.abs(m)
    _, ph, _ = mp.errors(el, 0.2, 0.1903)          # 相干 + 漫反射
    bound = np.arcsin(np.minimum(a, 1.0)) * 0.1903 / (2 * np.pi)
    check("相干多路径相位误差不超过解析上界",
          np.all(np.abs(ph) <= bound + 1e-6),
          f"max|Δφ|={np.max(np.abs(ph)) * 1000:.3f} mm, "
          f"min(bound)={np.min(bound) * 1000:.3f} mm")
    check("多路径幅度比 a 在 0.05~0.5（低成本天线贴地安装的合理区间）",
          0.05 < np.max(a) < 0.5, f"max a = {np.max(a):.3f}")
    check("多路径随仰角降低而增强（修正 Fresnel 后的必然趋势）",
          np.mean(a[el < 20]) > np.mean(a[el > 60]),
          f"低仰角 a={np.mean(a[el < 20]):.3f} vs 高仰角 a={np.mean(a[el > 60]):.3f}")
    # 码多路径必须落在经典包络内 |Δρ| <= a·min(τ, d)/(1+a)·chip
    code, _, _ = mp.errors(el, 0.2, 0.1903)
    chip = geo.C_LIGHT / 1.023e6
    tau_chip = 2 * 0.2 * np.sin(np.radians(el)) / chip
    env = a * np.minimum(tau_chip, 0.1) / (1.0 - a) * chip
    check("码多路径不超过经典包络 a·τ/(1-a)·chip",
          np.all(np.abs(code) <= env + 1e-9),
          f"max|码多路径|={np.max(np.abs(code)) * 1000:.2f} mm, "
          f"包络 {np.max(env) * 1000:.2f} mm")
    check("码多路径量级 < 0.2 m（贴地天线延迟远小于码片的必然结果）",
          np.max(np.abs(code)) < 0.2,
          f"RMS={np.sqrt(np.mean(code ** 2)) * 1000:.2f} mm")

    print()
    print("=" * 74)
    print("4. 对流层：Saastamoinen 典型值与仰角映射")
    print("=" * 74)
    zhd = md.saastamoinen_zhd(1013.25, 34.5, 0.0)
    zwd = md.saastamoinen_zwd(20.0, 0.5)
    print(f"    天顶干延迟 = {zhd:.4f} m (典型 2.30 m)")
    print(f"    天顶湿延迟 = {zwd:.4f} m (RH=50%, 20℃, 典型 0.10~0.15 m)")
    check("天顶干延迟在 2.25~2.35 m", 2.25 < zhd < 2.35)
    check("天顶湿延迟在 0.05~0.30 m", 0.05 < zwd < 0.30)
    m90 = geo.elev_mapping(90.0)
    m5 = geo.elev_mapping(5.0)
    check("映射函数 m(90°)≈1", abs(m90 - 1.0) < 0.02, f"m(90°)={m90:.4f}")
    check("映射函数随仰角降低而增大", m5 > m90 * 3, f"m(5°)={m5:.2f}")

    print()
    print("=" * 74)
    print("5. Klobuchar 广播模型行为")
    print("=" * 74)
    scn = ScenarioConfig()
    const = con.build_constellation(scn.const.systems, scn.session.t_start_unix)
    world = WorldModel(scn, const, seed=3, dual_freq=False).prepare_session()
    a_coef, b_coef = world.alpha, world.beta
    # 经度 113°E 处的地方时 ≈ UTC + 7.53 h
    sod_day = 23292.0      # UTC 06:28 -> 当地 14:00（峰值）
    sod_night = 66492.0    # UTC 18:28 -> 当地 02:00（夜间底值）
    for sod, tag in ((sod_day, "当地 14:00（峰值）"),
                     (sod_night, "当地 02:00（夜间）")):
        d = md.klobuchar_delay(a_coef, b_coef, 34.5, 113.0, 180.0, 30.0, sod)
        print(f"    {tag}: 30° 仰角 L1 延迟 = {d:.3f} m")
    day = md.klobuchar_delay(a_coef, b_coef, 34.5, 113.0, 180.0, 30.0, sod_day)
    night = md.klobuchar_delay(a_coef, b_coef, 34.5, 113.0, 180.0, 30.0, sod_night)
    check("白天延迟显著大于夜间（夜间退化为 5 ns 底值×F）",
          day > night * 1.5, f"day={day:.2f} m, night={night:.2f} m")
    check("夜间延迟 ≈ F(30°)×1.5 m",
          abs(night - geo.oblique_factor(30.0) * 5e-9 * geo.C_LIGHT) < 0.15,
          f"{night:.3f} m vs 解析 {geo.oblique_factor(30.0) * 5e-9 * geo.C_LIGHT:.3f} m")
    d_zen = md.klobuchar_delay(a_coef, b_coef, 34.5, 113.0, 180.0, 90.0, 21600.0)
    d_low = md.klobuchar_delay(a_coef, b_coef, 34.5, 113.0, 180.0, 10.0, 21600.0)
    check("低仰角倾斜因子放大延迟", d_low > d_zen * 2.5,
          f"10°={d_low:.2f} m vs 90°={d_zen:.2f} m")

    print()
    print("=" * 74)
    print("6. 消电离层组合系数与残余")
    print("=" * 74)
    f1, f2 = 1575.42e6, DUAL_FREQ_HZ
    c1 = f1 ** 2 / (f1 ** 2 - f2 ** 2)
    c2 = -f2 ** 2 / (f1 ** 2 - f2 ** 2)
    print(f"    c1={c1:.6f}, c2={c2:.6f}, c1+c2={c1 + c2:.6f}")
    check("c1 + c2 = 1（几何距离不放大）", abs(c1 + c2 - 1.0) < 1e-12)
    I1 = 10.0
    I2 = I1 * (f1 / f2) ** 2
    check("一阶电离层被完全消除", abs(c1 * I1 + c2 * I2) < 1e-9,
          f"残余 = {c1 * I1 + c2 * I2:.2e} m")
    check("噪声放大因子在 2.5~2.7（L1/L5 典型值）",
          2.5 < np.hypot(c1, c2) < 2.7, f"sqrt(c1²+c2²)={np.hypot(c1, c2):.3f}")

    print()
    print("=" * 74)
    print("7. 坐标与几何链：往返一致性")
    print("=" * 74)
    frame = geo.LocalFrame(34.50, 113.00, 80.0)
    enu0 = np.array([37.5, -12.25, 0.2])
    back = frame.ecef_to_enu(frame.enu_to_ecef(enu0))
    check("ENU→ECEF→ENU 往返", np.allclose(enu0, back, atol=1e-9),
          f"误差 {np.max(np.abs(enu0 - back)):.2e} m")
    # 参考点自身的大地坐标往返应精确复原
    lat0, lon0, h0 = geo.ecef_to_geodetic(frame.enu_to_ecef([0.0, 0.0, 0.0]))
    check("参考点大地坐标往返", abs(lat0 - 34.5) < 1e-9 and abs(lon0 - 113.0) < 1e-9
          and abs(h0 - 80.0) < 1e-6,
          f"lat={lat0:.9f}, lon={lon0:.9f}, h={h0:.6f}")
    # 偏移 37.5 m 东、-12.25 m 北后，纬度应相应减小
    # （34.5°N 处子午线每度约 110930 m，按 WGS84 椭球计算）
    lat1, lon1, h1 = geo.ecef_to_geodetic(frame.enu_to_ecef(enu0))
    m_per_deg = 111132.95 - 559.85 * np.cos(2 * np.radians(34.5))
    dlat_expect = -12.25 / m_per_deg
    check("偏离参考点后的大地坐标符合几何预期",
          abs((lat1 - 34.5) - dlat_expect) < 1e-7 and abs(h1 - 80.2) < 1e-3,
          f"Δlat={lat1 - 34.5:.6e} (预期 {dlat_expect:.6e}), h={h1:.4f}")
    # 天顶方向卫星：仰角应为 90°
    rx = frame.enu_to_ecef([0, 0, 0])
    up = frame.enu_to_ecef([0, 0, 20200000.0])
    az, el, rng = frame.az_el(rx, up)
    check("天顶方向仰角 = 90°", abs(el - 90.0) < 1e-6, f"el={el:.6f}°")
    # 正东方向低仰角
    east = frame.enu_to_ecef([20200000.0, 0, 0])
    az2, el2, _ = frame.az_el(rx, east)
    check("正东方向方位角 = 90°", abs(az2 - 90.0) < 1e-6, f"az={az2:.6f}°")

    print()
    print("=" * 74)
    print("8. 轨道传播：周期与 GEO 定点")
    print("=" * 74)
    cg = con.build_constellation(("G",), scn.session.t_start_unix)
    period_h = 2 * np.pi / cg.n.mean() / 3600.0
    check("GPS 轨道周期 11.9~12.0 h", 11.9 < period_h < 12.0,
          f"{period_h:.4f} h")
    cgeo = con.build_constellation(("C",), scn.session.t_start_unix)
    t = scn.session.t_start_unix + np.arange(0, 86400, 600.0)
    sat = cgeo.propagate_ecef(t)
    idx_geo = [i for i, s in enumerate(cgeo.sats) if s.kind == "GEO"]
    lat_g, lon_g, h_g = geo.ecef_to_geodetic(sat[:, idx_geo[0], :])
    check("GEO 经度漂移 < 0.5°（一天内）",
          (lon_g.max() - lon_g.min()) < 0.5,
          f"{lon_g.min():.2f}° ~ {lon_g.max():.2f}°")
    check("GEO 高度 ≈ 35786 km", abs(np.mean(h_g) / 1e3 - 35786) < 50,
          f"{np.mean(h_g)/1e3:.0f} km")
    idx_igso = [i for i, s in enumerate(cgeo.sats) if s.kind == "IGSO"]
    lat_i, lon_i, _ = geo.ecef_to_geodetic(sat[:, idx_igso[0], :])
    check("IGSO 纬度摆幅 ±55°（倾斜同步轨道“8”字）",
          abs(lat_i.max() - 55.0) < 1.0 and abs(lat_i.min() + 55.0) < 1.0,
          f"{lat_i.min():.1f}° ~ {lat_i.max():.1f}°")

    print()
    print("=" * 74)
    print("9. 条件方差序贯取整（sequential_fix）正确性")
    print("=" * 74)
    from gnss_sim.estimators import sequential_fix
    rng = np.random.default_rng(7)
    n_true = np.round(rng.normal(0, 3, 6))
    L = np.tril(rng.normal(0, 0.3, (6, 6)), -1) + np.eye(6)
    D = np.diag(rng.uniform(0.001, 0.01, 6))
    Q = L @ D @ L.T                                   # 强相关协方差
    n_f, mask, sig = sequential_fix(n_true + rng.normal(0, 0.05, 6), Q, 1.0)
    check("高精度、强相关条件下能全部正确固定",
          np.all(mask) and np.allclose(n_f[mask], n_true[mask]),
          f"固定 {int(np.sum(mask))}/6，误差 "
          f"{np.max(np.abs(n_f[mask] - n_true[mask])):.0f} 周")
    n_f2, mask2, _ = sequential_fix(n_true + rng.normal(0, 2.0, 6),
                                    Q * 100, 0.05)
    check("精度不足时不强行固定（只固定满足条件的前缀）",
          np.sum(mask2) < 6, f"固定 {int(np.sum(mask2))}/6")

    print()
    print("=" * 74)
    print("10. 观测方程量级闭环：IF 组合后的伪距应≈几何距离+钟差+对流层")
    print("=" * 74)
    from gnss_sim import estimators as est
    ant = md.AntennaModel(md.ANTENNA_LIBRARY["lowcost"], world.rng)
    mp2 = md.MultipathModel(scn.soil, ant, world.rng)
    rx = world.make_probe(0, np.array([0.0, 0.0, 0.2]), scn.probe, ant, mp2, 0.2)
    # 多历元检查：均值判系统偏差，RMS 判噪声量级
    means, rms, rms_l, res_max = [], [], [], []
    for i in range(30):
        rec = world.build_record(rx, i, dual=True)
        f1o, f2o = rec.freqs["f1"], rec.freqs["f2"]
        cc1 = f1o.freq ** 2 / (f1o.freq ** 2 - f2o.freq ** 2)
        cc2 = -f2o.freq ** 2 / (f1o.freq ** 2 - f2o.freq ** 2)
        p_if = cc1 * f1o.P + cc2 * f2o.P
        clk_sv = world.sat_err.clock_true(rec.t)[rec.idx]
        r = p_if + clk_sv - rec.rho_true - rec.clk_rx_m - rec.trop_true
        means.append(np.mean(r)); rms.append(np.sqrt(np.mean(r ** 2)))
        res_max.append(np.max(np.abs(r)))
        a1 = cc1 * f1o.L + cc2 * f2o.L
        amb = cc1 * f1o.n_true * f1o.lam + cc2 * f2o.n_true * f2o.lam
        rl = a1 + clk_sv - rec.rho_true - rec.clk_rx_m - rec.trop_true - amb
        rms_l.append(np.sqrt(np.mean(rl ** 2)))
        if i == 0:
            sig_if = float(np.mean(np.hypot(cc1, cc2) * f1o.sigma_code))
    check("IF 伪距无系统偏差（30 历元均值的均值）",
          abs(np.mean(means)) < 0.15,
          f"<mean>={np.mean(means) * 1000:+.1f} mm（历元间散布 "
          f"{np.std(means) * 1000:.0f} mm）")
    check("IF 伪距残差 RMS ≈ 理论噪声放大 2.59×sigma_code",
          abs(np.mean(rms) - sig_if) < 0.4,
          f"实测 {np.mean(rms):.3f} m vs 理论 {sig_if:.3f} m")
    check("IF 相位残差 RMS < 15 mm（仅剩多径/相位中心）",
          np.mean(rms_l) < 0.015,
          f"{np.mean(rms_l) * 1000:.2f} mm")

    print()
    print("=" * 74)
    if FAIL:
        print(f"结果: {len(FAIL)} 项未通过 -> {FAIL}")
        sys.exit(1)
    print("结果: 全部解析极限检查通过")


if __name__ == "__main__":
    main()
