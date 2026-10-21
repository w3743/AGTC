"""GNSS 星座建模与轨道传播。

按 2025 年前后的真实星座规模构造 Walker 星座：
  * GPS      : 6 轨道面 x 5~6 星，i=55°，a=26560 km（MEO）
  * BDS-3    : 3 轨道面 x 8 星 MEO(i=55°, a=27906 km) + 3 IGSO + 3 GEO
  * Galileo  : 3 轨道面 x 9 星，i=56°，a=29600 km
  * GLONASS  : 3 轨道面 x 8 星，i=64.8°，a=25510 km

轨道传播包含 J2 长期项（升交点赤经、近地点幅角、平近点角漂移），
足以复现真实的卫星几何（DOP）时变特性。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geodesy as geo

# ----------------------------------------------------------------------------
# 系统与频率定义
# ----------------------------------------------------------------------------
SYSTEMS = {
    "G": {"name": "GPS",     "freq": 1575.42e6, "sisre": 0.9,  "clk_err": 0.9},
    "C": {"name": "BDS-3",   "freq": 1561.098e6, "sisre": 1.2, "clk_err": 1.2},
    "E": {"name": "Galileo", "freq": 1575.42e6, "sisre": 0.6,  "clk_err": 0.7},
    "R": {"name": "GLONASS", "freq": 1602.0e6,  "sisre": 1.6,  "clk_err": 1.6},
}
SYSTEM_LABEL = {k: v["name"] for k, v in SYSTEMS.items()}

# GEO / IGSO 的标称定点经度（BDS-3）
BDS_GEO_LONS = [80.0, 110.5, 140.0]
BDS_IGSO_LONS = [118.0, 118.0, 118.0]


@dataclass
class Satellite:
    prn: str
    system: str
    freq: float
    a: float          # 轨道半长轴 (m)
    e: float          # 偏心率
    inc: float        # 倾角 (rad)
    raan0: float      # t0 时刻升交点赤经 (rad)
    argp0: float      # 近地点幅角 (rad)
    m0: float         # t0 时刻平近点角 (rad)
    kind: str = "MEO"

    # 由 _finalize 填充
    n: float = 0.0
    raan_dot: float = 0.0
    argp_dot: float = 0.0
    m_dot: float = 0.0
    lam: float = 0.0   # 波长 (m)

    def __post_init__(self):
        self.lam = geo.C_LIGHT / self.freq
        p = self.a * (1.0 - self.e ** 2)
        n0 = np.sqrt(geo.MU / self.a ** 3)
        k = 1.5 * geo.J2 * (geo.WGS84_A / p) ** 2
        self.n = n0
        self.raan_dot = -k * n0 * np.cos(self.inc)
        self.argp_dot = 0.5 * k * n0 * (5.0 * np.cos(self.inc) ** 2 - 1.0)
        self.m_dot = n0 * (1.0 + k * np.sqrt(1.0 - self.e ** 2)
                           * (3.0 * np.cos(self.inc) ** 2 - 1.0) / 2.0)


def _walker(system, prefix, n_planes, sats_per_plane, a, inc, e=0.001,
            f_factor=1, raan0=0.0, m0=0.0):
    sats = []
    total = n_planes * sats_per_plane
    freq = SYSTEMS[system]["freq"]
    idx = 1
    for p in range(n_planes):
        raan = np.radians(raan0 + p * 360.0 / n_planes)
        for s in range(sats_per_plane):
            m = np.radians(
                m0 + s * 360.0 / sats_per_plane + p * f_factor * 360.0 / total)
            sats.append(Satellite(
                prn=f"{prefix}{idx:02d}", system=system, freq=freq,
                a=a, e=e, inc=np.radians(inc), raan0=raan,
                argp0=0.0, m0=m))
            idx += 1
    return sats


def build_constellation(systems=("G", "C", "E"), t0_unix=0.0):
    """构造完整星座。t0_unix 用于 GEO/IGSO 定点经度的初值对齐。"""
    sats: list[Satellite] = []
    theta0 = float(geo.gmst_rad(t0_unix))

    if "G" in systems:
        # 6 个轨道面，31 颗在轨（5,5,5,5,5,6 分布）
        per_plane = [5, 5, 5, 5, 5, 6]
        idx = 1
        for p, nsat in enumerate(per_plane):
            raan = np.radians(p * 60.0 + 5.0 * p)
            for s in range(nsat):
                m = np.radians(s * 360.0 / nsat + p * 15.0)
                sats.append(Satellite(
                    prn=f"G{idx:02d}", system="G", freq=SYSTEMS["G"]["freq"],
                    a=26560e3, e=0.008, inc=np.radians(55.0), raan0=raan,
                    argp0=np.radians(30.0 + 10.0 * s), m0=m))
                idx += 1

    if "C" in systems:
        sats += _walker("C", "C", 3, 8, 27906e3, 55.0, e=0.002,
                        f_factor=1, m0=15.0)
        # IGSO（倾斜同步轨道，星下点"8"字轨迹，与真实 BDS IGSO 一致）
        for i, lon in enumerate(BDS_IGSO_LONS):
            m0 = np.radians(lon) + theta0 + i * np.radians(120.0)
            sats.append(Satellite(
                prn=f"C{25 + i:02d}", system="C", freq=SYSTEMS["C"]["freq"],
                a=42164e3, e=0.002, inc=np.radians(55.0), raan0=0.0,
                argp0=0.0, m0=m0, kind="IGSO"))
        # GEO
        for i, lon in enumerate(BDS_GEO_LONS):
            m0 = np.radians(lon) + theta0
            sats.append(Satellite(
                prn=f"C{28 + i:02d}", system="C", freq=SYSTEMS["C"]["freq"],
                a=42164e3, e=0.0002, inc=np.radians(1.5), raan0=0.0,
                argp0=0.0, m0=m0, kind="GEO"))

    if "E" in systems:
        sats += _walker("E", "E", 3, 9, 29600e3, 56.0, e=0.0005,
                        f_factor=1, m0=10.0)

    if "R" in systems:
        sats += _walker("R", "R", 3, 8, 25510e3, 64.8, e=0.001,
                        f_factor=15, m0=0.0)

    return Constellation(sats, t0_unix)


class Constellation:
    """一组卫星及其轨道传播能力。"""

    def __init__(self, sats: list[Satellite], t0_unix: float = 0.0):
        self.sats = sats
        self.t0 = float(t0_unix)
        self.prn = np.array([s.prn for s in sats])
        self.system = np.array([s.system for s in sats])
        self.freq = np.array([s.freq for s in sats])
        self.lam = np.array([s.lam for s in sats])
        self._stack()

    def _stack(self):
        self.a = np.array([s.a for s in self.sats])
        self.e = np.array([s.e for s in self.sats])
        self.inc = np.array([s.inc for s in self.sats])
        self.raan0 = np.array([s.raan0 for s in self.sats])
        self.argp0 = np.array([s.argp0 for s in self.sats])
        self.m0 = np.array([s.m0 for s in self.sats])
        self.raan_dot = np.array([s.raan_dot for s in self.sats])
        self.argp_dot = np.array([s.argp_dot for s in self.sats])
        self.m_dot = np.array([s.m_dot for s in self.sats])
        self.n = np.array([s.n for s in self.sats])

    def __len__(self):
        return len(self.sats)

    def _eci_from_elements(self, raan, argp, m):
        """由升交点赤经/近地点幅角/平近点角（弧度，形状 (...,N)）计算 ECI 位置。"""
        ecc = geo.solve_kepler(m, self.e)
        cos_e, sin_e = np.cos(ecc), np.sin(ecc)
        a = self.a
        b = a * np.sqrt(1.0 - self.e ** 2)
        xo = a * (cos_e - self.e)
        yo = b * sin_e
        u = argp + np.arctan2(yo, xo)
        r = np.hypot(xo, yo)
        cu, su = np.cos(u), np.sin(u)
        cO, sO = np.cos(raan), np.sin(raan)
        ci, si = np.cos(self.inc), np.sin(self.inc)
        x = r * (cO * cu - sO * su * ci)
        y = r * (sO * cu + cO * su * ci)
        z = r * (su * si)
        return np.stack([x, y, z], axis=-1)

    def propagate_eci(self, dt):
        """dt: 相对 t0 的秒数（标量或 (K,)），返回 (K,N,3) ECI 位置。"""
        dt = np.atleast_1d(np.asarray(dt, dtype=float))[:, None]
        return self._eci_from_elements(
            self.raan0 + self.raan_dot * dt,
            self.argp0 + self.argp_dot * dt,
            self.m0 + self.m_dot * dt)

    def propagate_ecef(self, t_unix):
        """给定绝对 Unix 时间（标量或 (K,)），返回 (K,N,3) ECEF 位置。"""
        t = np.atleast_1d(np.asarray(t_unix, dtype=float))
        r_eci = self.propagate_eci(t - self.t0)
        theta = geo.gmst_rad(t)[:, None]
        return self._rotate_to_ecef(r_eci, theta)

    def propagate_ecef_matrix(self, t_unix):
        """逐卫星独立时刻的传播。t_unix 形状 (K,N)，返回 (K,N,3) ECEF。

        用于计算信号发射时刻的卫星位置（各星传播时延不同）。
        """
        t = np.asarray(t_unix, dtype=float)
        dt = t - self.t0
        r_eci = self._eci_from_elements(
            self.raan0 + self.raan_dot * dt,
            self.argp0 + self.argp_dot * dt,
            self.m0 + self.m_dot * dt)
        return self._rotate_to_ecef(r_eci, geo.gmst_rad(t))

    @staticmethod
    def _rotate_to_ecef(r_eci, theta):
        ct, st = np.cos(theta), np.sin(theta)
        x, y, z = r_eci[..., 0], r_eci[..., 1], r_eci[..., 2]
        return np.stack([ct * x + st * y, -st * x + ct * y, z], axis=-1)

    def visible(self, frame, rx_ecef, t_unix, el_mask=10.0):
        """返回可见卫星索引及其方位角/仰角/几何距离。"""
        sat = self.propagate_ecef(t_unix)
        if sat.ndim == 3:
            sat = sat[0]
        az, el, rng = frame.az_el(rx_ecef, sat)
        ok = el >= el_mask
        idx = np.nonzero(ok)[0]
        return idx, az[idx], el[idx], rng[idx]

    def sky_track(self, frame, rx_ecef, t_array, el_mask=10.0):
        """返回 (K,N) 的方位角、仰角数组，便于画星空图。"""
        sat = self.propagate_ecef(t_array)              # (K,N,3)
        k, n = sat.shape[0], sat.shape[1]
        rx = np.asarray(rx_ecef, dtype=float)
        los = sat - rx[None, None, :]
        enu = los @ frame.rot.T
        e, n_, u = enu[..., 0], enu[..., 1], enu[..., 2]
        el = np.degrees(np.arctan2(u, np.hypot(e, n_)))
        az = np.degrees(np.arctan2(e, n_)) % 360.0
        return az, el


def dop_from_geometry(az_deg, el_deg, weights=None):
    """由可见卫星几何计算 DOP (GDOP, PDOP, HDOP, VDOP)。

    使用单位视线向量（接收机到卫星）组成的几何矩阵。
    """
    az = np.radians(np.asarray(az_deg, dtype=float))
    el = np.radians(np.asarray(el_deg, dtype=float))
    if len(az) < 4:
        return dict(gdop=np.inf, pdop=np.inf, hdop=np.inf, vdop=np.inf, tdop=np.inf)
    # ENU 系中的单位视线向量
    u = np.sin(el)
    h = np.cos(el)
    e = h * np.sin(az)
    n = h * np.cos(az)
    a = np.column_stack([e, n, u, np.ones_like(e)])
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        a = a * w[:, None]
    q = np.linalg.inv(a.T @ a)
    gdop = np.sqrt(np.trace(q))
    pdop = np.sqrt(q[0, 0] + q[1, 1] + q[2, 2])
    hdop = np.sqrt(q[0, 0] + q[1, 1])
    vdop = np.sqrt(q[2, 2])
    tdop = np.sqrt(q[3, 3])
    return dict(gdop=gdop, pdop=pdop, hdop=hdop, vdop=vdop, tdop=tdop)
