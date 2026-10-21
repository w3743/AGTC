"""WGS84 大地测量、时间与坐标转换工具。

本模块只提供无状态的数学工具，仿真中所有坐标约定如下：
  * ECEF: 地心地固直角坐标 (m)
  * ENU : 以田块参考点（田块中心）为原点的东-北-天坐标 (m)
  * 时间: Unix 秒 (UTC, float)
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# WGS84 常数与物理常数
# ----------------------------------------------------------------------------
WGS84_A = 6378137.0                      # 长半轴 (m)
WGS84_F = 1.0 / 298.257223563            # 扁率
WGS84_B = WGS84_A * (1.0 - WGS84_F)      # 短半轴 (m)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)     # 第一偏心率平方
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)  # 第二偏心率平方

MU = 3.986005e14                         # 地球引力常数 (m^3/s^2)
J2 = 1.08262668e-3                       # 地球二阶带谐系数
OMEGA_E = 7.2921151467e-5                # 地球自转角速度 (rad/s)
C_LIGHT = 299792458.0                    # 光速 (m/s)

# 电离层薄壳模型高度 (m)
IONO_SHELL_HEIGHT = 350000.0


def geodetic_to_ecef(lat_deg, lon_deg, h):
    """大地坐标 -> ECEF。支持标量或数组广播。"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * sin_lat
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(ecef):
    """ECEF -> 大地坐标 (Bowring 迭代法)，返回 (lat_deg, lon_deg, h)。"""
    ecef = np.asarray(ecef, dtype=float)
    x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    # 初值
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def enu_matrix(lat_deg, lon_deg):
    """返回 ECEF->ENU 的旋转矩阵 (3x3)，行向量依次为东、北、天。"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


class LocalFrame:
    """以某参考点为原点的局部 ENU 坐标系。"""

    def __init__(self, lat_deg, lon_deg, h):
        self.lat = float(lat_deg)
        self.lon = float(lon_deg)
        self.h = float(h)
        self.origin_ecef = geodetic_to_ecef(lat_deg, lon_deg, h)
        self.rot = enu_matrix(lat_deg, lon_deg)          # ECEF->ENU

    def enu_to_ecef(self, enu):
        return self.origin_ecef + np.asarray(enu, dtype=float) @ self.rot

    def ecef_to_enu(self, ecef):
        return (np.asarray(ecef, dtype=float) - self.origin_ecef) @ self.rot.T

    def az_el(self, rx_ecef, sat_ecef):
        """计算接收机到卫星的方位角/仰角（度）。sat 可为 (N,3)。"""
        los = np.asarray(sat_ecef, dtype=float) - np.asarray(rx_ecef, dtype=float)
        enu = los @ self.rot.T
        e, n, u = enu[..., 0], enu[..., 1], enu[..., 2]
        rng = np.linalg.norm(los, axis=-1)
        el = np.degrees(np.arctan2(u, np.hypot(e, n)))
        az = np.degrees(np.arctan2(e, n)) % 360.0
        return az, el, rng

    def geodetic_to_enu(self, lat_deg, lon_deg, h):
        return self.ecef_to_enu(geodetic_to_ecef(lat_deg, lon_deg, h))


# ----------------------------------------------------------------------------
# 时间
# ----------------------------------------------------------------------------
def unix_to_jd(t_unix):
    return np.asarray(t_unix, dtype=float) / 86400.0 + 2440587.5


def gmst_rad(t_unix):
    """格林尼治平恒星时 (rad)，IAU1982 表达式。"""
    jd = unix_to_jd(t_unix)
    t = (jd - 2451545.0) / 36525.0
    gmst_deg = (280.46061837
                + 360.98564736629 * (jd - 2451545.0)
                + 0.000387933 * t * t
                - t * t * t / 38710000.0)
    return np.radians(np.mod(gmst_deg, 360.0))


def eci_to_ecef(r_eci, t_unix):
    """ECI -> ECEF（绕 Z 轴旋转 GMST）。r_eci 形状 (...,3)。"""
    theta = gmst_rad(t_unix)
    x, y, z = r_eci[..., 0], r_eci[..., 1], r_eci[..., 2]
    ct, st = np.cos(theta), np.sin(theta)
    return np.stack([ct * x + st * y, -st * x + ct * y, z], axis=-1)


def solve_kepler(m, e, tol=1e-13, max_iter=30):
    """牛顿迭代解偏近点角 E: M = E - e sinE。"""
    m = np.asarray(m, dtype=float)
    e = np.asarray(e, dtype=float)
    ecc_anom = m + e * np.sin(m)          # 良好初值
    for _ in range(max_iter):
        f = ecc_anom - e * np.sin(ecc_anom) - m
        fp = 1.0 - e * np.cos(ecc_anom)
        step = f / fp
        ecc_anom = ecc_anom - step
        if np.max(np.abs(step)) < tol:
            break
    return ecc_anom


def elev_mapping(el_deg):
    """对流层仰角映射函数（Ifadis/Niell 简化式）。"""
    el = np.radians(np.maximum(el_deg, 1.0))
    return 1.0 / (np.sin(el) + 0.00143 / (np.tan(el) + 0.0445))


def oblique_factor(el_deg):
    """电离层倾斜因子 F（Klobuchar 标准式）。"""
    el = np.radians(np.maximum(el_deg, 1.0))
    return 1.0 + 16.0 * (0.53 - el / np.pi) ** 3
