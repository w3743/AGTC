"""PPP 解算内部诊断。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnss_sim import constellation as con, estimators as est, models as md, geodesy as geo
from gnss_sim.config import ScenarioConfig, ProbeSpec
from gnss_sim.observation import WorldModel

scn = ScenarioConfig()
scn.session.duration_s = 600.0
const = con.build_constellation(scn.const.systems, scn.session.t_start_unix)
world = WorldModel(scn, const, seed=7, dual_freq=True).prepare_session()
ant = md.AntennaModel(md.ANTENNA_LIBRARY["lowcost"], world.rng)
mp = md.MultipathModel(scn.soil, ant, world.rng)
enu_true = np.array([23.0, -17.0, 0.20])
rx = world.make_probe(0, enu_true, ProbeSpec(), ant, mp, 0.20)
appr = enu_true + np.array([6.0, -4.0, 2.0])

records = [world.build_record(rx, i) for i in range(21)]
x_true = world.frame.enu_to_ecef(enu_true)
x_appr = world.frame.enu_to_ecef(appr)

print("IF 组合检查（第 0 历元, 前 3 颗卫星）:")
r0 = records[0]
f1, f2 = r0.freqs["f1"], r0.freqs["f2"]
c1 = f1.freq ** 2 / (f1.freq ** 2 - f2.freq ** 2)
c2 = -f2.freq ** 2 / (f1.freq ** 2 - f2.freq ** 2)
print("  c1 =", np.round(c1[:3], 4), " c2 =", np.round(c2[:3], 4))
print("  P1 =", np.round(f1.P[:3], 3), " P2 =", np.round(f2.P[:3], 3))
print("  P_if-rho =", np.round((c1 * f1.P + c2 * f2.P - r0.rho_true)[:3], 4))
print("  L1 =", np.round(f1.L[:3], 3))
print("  lam1 =", np.round(f1.lam[:3], 5), " n1 =", f1.n_true[:3])
print("  N1*lam1 =", np.round(f1.n_true[:3] * f1.lam[:3], 3))
print("  L_if - rho =", np.round((c1 * f1.L + c2 * f2.L - r0.rho_true)[:3], 3))

sol, diag = est.solve_ppp(records, world, appr, ztd_bins=1)
print("\nsolve_ppp 结果:")
print("  估计 ENU:", np.round(sol.enu, 4), " 真值:", enu_true)
print("  误差 (cm):", np.round((sol.enu - enu_true) * 100, 3))
print("  res_rms(加权残差因子) =", sol.res_rms, " 参数数 =", diag["n_param"])
print("  估计 ZTD =", diag["ztd"], " 真值 ZTD =", world.field_tropo.zhd + world.field_tropo.zwd_series[0])
print("  形式精度 sigma_ENU (m) =", diag["sigma_enu"])
print("  估计钟差(前3):", np.round(diag["clk"][:3], 2), " 真值:", np.round(rx.clock.series[:3], 2))

# ---------------------------------------------------------------- 逐项核对
print("\n=== 第 0 历元：观测量与模型逐项核对 ===")
r = records[0]
r_sat = est.PRECISE.sat_pos(r, world)
clk_sv_m = est.PRECISE.sat_clk(r, world)
rho = np.linalg.norm(r_sat - x_true, axis=-1)
f1, f2 = r.freqs["f1"], r.freqs["f2"]
c1 = f1.freq ** 2 / (f1.freq ** 2 - f2.freq ** 2)
c2 = -f2.freq ** 2 / (f1.freq ** 2 - f2.freq ** 2)
p_if = c1 * f1.P + c2 * f2.P
l_if = c1 * f1.L + c2 * f2.L
clk_sv_true = world.sat_err.clock_true(r.t)[r.idx]
d = p_if + clk_sv_m - rho - r.clk_rx_m
print("  卫星:", r.idx[:6], " 系统:", r.systems[:6])
print("  P1 - (rho+clk_rx-clk_sv_true)  =", np.round((f1.P - (rho + r.clk_rx_m - clk_sv_true))[:6], 3))
print("  P2 - (rho+clk_rx-clk_sv_true)  =", np.round((f2.P - (rho + r.clk_rx_m - clk_sv_true))[:6], 3))
print("  I1 =", np.round(f1.iono[:6], 3), " I2 =", np.round(f2.iono[:6], 3))
print("  I1*1.7935 - I2 (应≈0) =", np.round((f1.iono * (1575.42 / 1561.098) ** 2 * (1561.098 / 1176.45) ** 2 - f2.iono)[:6], 4))
print("  IF 码 d = p_if + clk_sv - rho - clk_rx =", np.round(d[:8], 3))
print("      d 统计: mean=%.3f std=%.3f max|.|=%.3f" % (np.mean(d), np.std(d), np.max(np.abs(d))))
print("  IF 相位 l_if + clk_sv - rho - clk_rx =", np.round((l_if + clk_sv_m - rho - r.clk_rx_m)[:6], 3))
print("  真值 IF 模糊度 amb = c1*N1*lam1 + c2*N2*lam2 =",
      np.round((c1 * f1.n_true * f1.lam + c2 * f2.n_true * f2.lam)[:6], 3))
print("  各系统 P1/T/I 检查: T=", np.round(r.trop_true[:3], 3))

print("\n=== 复现最小二乘，检查矩阵条件数与残差 ===")
records = records
n_ep = len(records)
all_sats = sorted(set(int(g) for r in records for g in r.idx))
sat_pos = {g: k for k, g in enumerate(all_sats)}
n_sat = len(all_sats)
ztd_bins = 1
i_clk, i_ztd = 3, 3 + n_ep
i_amb = i_ztd + ztd_bins
n_param = i_amb + n_sat
x = x_appr
for it in range(4):
    A, Y, W = [], [], []
    for i_ep, r in enumerate(records):
        r_sat = est.PRECISE.sat_pos(r, world)
        los = r_sat - x
        rho0 = np.linalg.norm(los, axis=-1)
        los_hat = los / rho0[:, None]
        f1, f2 = r.freqs["f1"], r.freqs["f2"]
        c1 = f1.freq ** 2 / (f1.freq ** 2 - f2.freq ** 2)
        c2 = -f2.freq ** 2 / (f1.freq ** 2 - f2.freq ** 2)
        p_if = c1 * f1.P + c2 * f2.P
        l_if = c1 * f1.L + c2 * f2.L
        mf = geo.elev_mapping(est.rec_az_el(r, world)[1])
        clk_sv = est.PRECISE.sat_clk(r, world)
        n0 = np.round((l_if + clk_sv - rho0) / f1.lam)
        n = len(r.idx)
        ap = np.zeros((n, n_param)); al = np.zeros((n, n_param))
        ap[:, :3] = -los_hat; al[:, :3] = -los_hat
        ap[:, i_clk + i_ep] = 1.0; al[:, i_clk + i_ep] = 1.0
        ap[:, i_ztd] = mf; al[:, i_ztd] = mf
        for k, g in enumerate(r.idx):
            al[k, i_amb + sat_pos[int(g)]] = f1.lam[k]
        sig_p = np.sqrt((c1 * f1.sigma_code) ** 2 + (c2 * f2.sigma_code) ** 2 + 0.09)
        sig_l = np.sqrt((c1 * f1.sigma_phase) ** 2 + (c2 * f2.sigma_phase) ** 2 + 4e-6)
        A.append(ap); A.append(al)
        Y.append(p_if + clk_sv - rho0)
        Y.append(l_if + clk_sv - rho0 - n0 * f1.lam)
        W.append(1 / sig_p); W.append(1 / sig_l)
    A = np.vstack(A); Y = np.concatenate(Y); W = np.concatenate(W)
    Aw = A * W[:, None]
    sv = np.linalg.svd(Aw, compute_uv=False)
    dx, *_ = np.linalg.lstsq(Aw, Y * W, rcond=None)
    res = Aw @ dx - Y * W
    n_c = sum(len(r.idx) for r in records)
    print(f"  it{it}: 行数={len(Y)} 列数={n_param} rank={np.linalg.matrix_rank(Aw)} "
          f"sv_max={sv[0]:.3e} sv_min={sv[-1]:.3e} cond={sv[0]/sv[-1]:.3e}")
    print(f"        码残差RMS={np.sqrt(np.mean(res[:2*n_c:2]**2)):.3f} (加权) "
          f"相位残差RMS={np.sqrt(np.mean(res[1:2*n_c:2]**2)):.3f} (加权) "
          f"|dx_pos|={np.linalg.norm(dx[:3]):.4f} dx_clk0={dx[i_clk]:.2f} dx_ztd={dx[i_ztd]:.3f}")
    x = x + dx[:3]

