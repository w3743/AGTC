"""SPP 误差归因调试。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnss_sim import constellation as con
from gnss_sim import estimators as est
from gnss_sim import models as md
from gnss_sim.config import ScenarioConfig, ProbeSpec
from gnss_sim.observation import WorldModel

scn = ScenarioConfig()
const = con.build_constellation(scn.const.systems, scn.session.t_start_unix)
world = WorldModel(scn, const, seed=7, dual_freq=True).prepare_session()
ant = md.AntennaModel(md.ANTENNA_LIBRARY["lowcost"], world.rng)
mp = md.MultipathModel(scn.soil, ant, world.rng)
enu_true = np.array([23.0, -17.0, 0.20])
rx = world.make_probe(0, enu_true, ProbeSpec(), ant, mp, 0.20)
rec = world.build_record(rx, 0)
appr = enu_true + np.array([6.0, -4.0, 2.0])

svc = est.BROADCAST
r_sat = svc.sat_pos(rec, world)
clk_sv = svc.sat_clk(rec, world)
iono_m = svc.iono_model(rec, world)
trop_m = svc.tropo_model(rec, world)
p_corr = rec.f1.P + clk_sv - iono_m - trop_m

x = world.frame.enu_to_ecef(appr)
print("逐次迭代:")
for it in range(8):
    los = r_sat - x
    rho = np.linalg.norm(los, axis=-1)
    los_hat = los / rho[:, None]
    A = np.hstack([-los_hat, np.ones((len(rho), 1))])
    y = p_corr - rho
    sigma = np.sqrt(rec.f1.sigma_code ** 2 + 0.4 ** 2)
    w = 1.0 / sigma
    Aw = A * w[:, None]
    dx, *_ = np.linalg.lstsq(Aw, y * w, rcond=None)
    x = x + dx[:3]
    print(f"  it{it}: |dx_pos|={np.linalg.norm(dx[:3]):10.4f} m, dx_clk={dx[3]:12.3f} m, "
          f"y_rms={np.sqrt(np.mean(y**2)):9.3f} m")
print("估计 ENU:", world.frame.ecef_to_enu(x), " 真值:", enu_true)
print("误差:", world.frame.ecef_to_enu(x) - enu_true)

los = r_sat - x
rho = np.linalg.norm(los, axis=-1)
print("\n最终残差 (m):", np.round(p_corr - rho, 3))
print("卫星系统:", rec.systems)
print("\n各误差分量:")
print("  电离层残差 I_true-klob:", np.round(rec.f1.iono - (rec.f1.iono - (rec.f1.iono - iono_m)), 3)[:5],
      " (iono_m 即模型值)")
print("  I_true - iono_m:", np.round(rec.f1.iono - iono_m, 3))
print("  T_true - trop_m:", np.round(rec.trop_true - trop_m, 3))
orb = world.sat_err.orbit_error_ecef(rec.sat_true, rec.t, idx=rec.idx)
los_hat = los / rho[:, None]
print("  轨道误差 LOS 投影:", np.round(np.sum(orb * los_hat, axis=-1), 3))
clk_true = world.sat_err.clock_true(rec.t)[rec.idx]
clk_bc = world.sat_err.clock_broadcast(rec.t)[rec.idx]
print("  钟差误差 (bc-true):", np.round(clk_bc - clk_true, 3))
print("  接收机钟差:", rec.clk_rx_m)
print("  码多路径:", np.round(rec.f1.mp_code, 3))

print("\n逐历元 SPP（多系统）:")
x_true = world.frame.enu_to_ecef(enu_true)
for i in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]:
    r = world.build_record(rx, i)
    s = est.solve_spp(r, est.BROADCAST, world, appr)
    # 在真值位置上检查模型-观测一致性
    rs = est.BROADCAST.sat_pos(r, world)
    cs = est.BROADCAST.sat_clk(r, world)
    im = est.BROADCAST.iono_model(r, world)
    tm = est.BROADCAST.tropo_model(r, world)
    pc = r.f1.P + cs - im - tm
    d = pc - np.linalg.norm(rs - x_true, axis=-1)
    print(f"  ep{i:3d}: err={np.round(s.enu - enu_true, 2)}  "
          f"d_true: mean={np.mean(d):11.3f} std={np.std(d):7.3f}  "
          f"clk_rx={r.clk_rx_m:11.3f}  n={s.n_sat}")
