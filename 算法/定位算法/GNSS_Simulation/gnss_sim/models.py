"""真实误差源建模。

包含六类物理效应，全部按"真值 / 模型"分离的方式实现，便于解算器只扣除
其"知道"的那部分：

1. 电离层   : 薄壳模型真值 + Klobuchar 广播模型（含预报误差）
2. 对流层   : Saastamoinen 真值（含干/湿分量时变）+ 标准大气模型
3. 地表多路径: 土壤介电常数(Topp) -> Fresnel 反射 -> 双射线干涉
               + 粗糙面漫反射分量（相干+非相干两类）
4. 天线     : 增益方向图、低仰角滚降、左右旋抑制比、相位中心变化(PCV)
5. 观测噪声 : 由 C/N0 决定的码/相位噪声，多路径引起的 C/N0 起伏
6. 星历钟差 : 广播星历/钟差误差(SISRE)，精密产品误差

参考文献口径：
  * Klobuchar, J. (1987) 单频电离层改正，改正率约 50%~70%
  * Topp et al. (1980) 土壤体积含水量-介电常数经验关系
  * Saastamoinen (1972) 对流层天顶延迟
  * Braasch (1996) 多路径误差包络
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geodesy as geo

TECU_TO_M_L1 = 40.3 / (1575.42e6 ** 2) * 1e16      # 1 TECU 在 L1 上的斜距延迟 ≈ 0.1624 m
IONO_SHELL = 350e3
EARTH_RADIUS_MEAN = 6371000.0


# ============================================================================
# 1. 电离层
# ============================================================================
def pierce_point(lat_deg, lon_deg, az_deg, el_deg, shell_h=IONO_SHELL):
    """计算电离层穿刺点 (IPP) 与倾斜因子 F。"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    az = np.radians(az_deg)
    el = np.radians(np.maximum(el_deg, 0.5))
    re = EARTH_RADIUS_MEAN
    sin_z = re * np.cos(el) / (re + shell_h)
    sin_z = np.clip(sin_z, -1.0, 1.0)
    psi = np.pi / 2.0 - el - np.arcsin(sin_z)
    lat_ip = np.arcsin(np.clip(np.sin(lat) * np.cos(psi)
                               + np.cos(lat) * np.sin(psi) * np.cos(az), -1, 1))
    lon_ip = lon + np.arcsin(np.clip(np.sin(psi) * np.sin(az)
                                     / np.maximum(np.cos(lat_ip), 1e-6), -1, 1))
    f = 1.0 / np.sqrt(np.maximum(1.0 - sin_z ** 2, 1e-9))
    return np.degrees(lat_ip), np.degrees(lon_ip), f


def klobuchar_delay(alpha, beta, lat_deg, lon_deg, az_deg, el_deg, gps_sod):
    """GPS ICD-200 Klobuchar 广播电离层模型，返回 L1 延迟 (m)。

    alpha[4], beta[4] 为广播系数；lat/lon 以度为单位，gps_sod 为当日 GPS 秒。
    """
    phi = np.asarray(lat_deg, dtype=float) / 180.0          # 半周
    lam = np.asarray(lon_deg, dtype=float) / 180.0
    az = np.radians(np.asarray(az_deg, dtype=float)) / np.pi  # 半周
    el = np.radians(np.asarray(el_deg, dtype=float)) / np.pi
    el = np.maximum(el, 1e-3)

    f = 1.0 + 16.0 * (0.53 - el) ** 3
    psi = 0.0137 / (el + 0.11) - 0.022
    phi_i = phi + psi * np.cos(az)
    phi_i = np.clip(phi_i, -0.416, 0.416)
    lam_i = lam + psi * np.sin(az) / np.cos(phi_i * np.pi)
    phi_m = phi_i + 0.064 * np.cos((lam_i - 1.617) * np.pi)

    amp = (alpha[0] + alpha[1] * phi_m + alpha[2] * phi_m ** 2
           + alpha[3] * phi_m ** 3)
    amp = np.maximum(amp, 0.0)
    per = (beta[0] + beta[1] * phi_m + beta[2] * phi_m ** 2
           + beta[3] * phi_m ** 3)
    per = np.maximum(per, 72000.0)

    t = 43200.0 * lam_i + gps_sod
    t = np.mod(t, 86400.0)
    x = 2.0 * np.pi * (t - 50400.0) / per
    ax = np.abs(x)
    delay_s = np.where(
        ax < 1.57,
        f * (5e-9 + amp * (1.0 - x ** 2 / 2.0 + x ** 4 / 24.0)),
        f * 5e-9)
    return delay_s * geo.C_LIGHT


class IonosphereTruth:
    """电离层"真值"：带日变化、纬度结构与随机扰动的 VTEC 场。"""

    def __init__(self, sfu=150.0, rng=None, storm=False):
        rng = rng or np.random.default_rng(0)
        self.sfu = float(sfu)
        self.storm = storm
        self.peak_lt = float(rng.normal(13.2, 0.8))       # 真实日峰值时刻 (LT)
        self.floor_tecu = float(rng.normal(0.045 * sfu, 0.5))     # 夜间底值
        self.amp_tecu = float(rng.normal(0.205 * sfu, 0.12 * sfu))  # 日间振幅
        # 控制段"预报"的模型参数（与真值存在偏差：振幅、峰值时刻、纬度结构）
        self.pred_amp_tecu = self.amp_tecu * float(rng.normal(0.97, 0.28))
        self.pred_floor_tecu = 5e-9 * geo.C_LIGHT / TECU_TO_M_L1
        # 行波扰动（电离层不规则体）
        self.wave_amp = float(rng.uniform(0.5, 1.5)) * (3.0 if storm else 1.0)
        self.wave_phase = float(rng.uniform(0, 2 * np.pi))
        self.storm_factor = 1.0
        if storm:
            self.storm_factor = float(rng.uniform(1.5, 2.2))
        self.rng = rng

    def vtec(self, lat_ip, lon_ip, t_unix):
        """穿刺点垂直 TEC (TECU)。"""
        lt = np.mod(np.asarray(t_unix, dtype=float) / 3600.0 + np.asarray(lon_ip) / 15.0, 24.0)
        daily = np.cos(2.0 * np.pi * (lt - self.peak_lt) / 24.0)
        v = self.floor_tecu + self.amp_tecu * self.storm_factor * np.clip(daily, 0.0, None) ** 1.1
        alat = np.abs(np.asarray(lat_ip, dtype=float))
        # 赤道异常增强 + 中纬槽减弱
        latf = (1.0
                + 0.28 * np.exp(-((alat - 15.0) / 12.0) ** 2)
                - 0.16 * np.exp(-((alat - 48.0) / 15.0) ** 2))
        wave = self.wave_amp * np.sin(2 * np.pi * lt / 9.0 + self.wave_phase)
        return np.maximum(v * latf + wave, 0.5)

    def slant_delay_l1(self, lat_rx, lon_rx, az, el, t_unix):
        """L1 频率上的斜距电离层延迟 (m)。"""
        lat_ip, lon_ip, f = pierce_point(lat_rx, lon_rx, az, el)
        return self.vtec(lat_ip, lon_ip, t_unix) * f * TECU_TO_M_L1

    def broadcast_coefficients(self):
        """由"预报电离层"生成广播 Klobuchar 系数。"""
        peak_m = self.pred_amp_tecu * TECU_TO_M_L1 + 5e-9 * geo.C_LIGHT
        amp_s = max(peak_m / geo.C_LIGHT - 5e-9, 1e-9)
        alpha = [amp_s, 1.2e-9, -5.96e-8, 1.19e-7]
        beta = [1.152e5, 0.0, 0.0, 0.0]
        return np.array(alpha), np.array(beta)


# ============================================================================
# 2. 对流层
# ============================================================================
def saastamoinen_zhd(pressure_hpa, lat_deg, h_m):
    """天顶干延迟 (m)。"""
    return (0.0022768 * pressure_hpa
            / (1.0 - 0.00266 * np.cos(2 * np.radians(lat_deg))
               - 0.00028 * h_m / 1000.0))


def saastamoinen_zwd(temp_c, rh):
    """天顶湿延迟 (m)。temp_c 摄氏度，rh 相对湿度 0~1。"""
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    e = rh * es
    return 0.002277 * (1255.0 / (temp_c + 273.15) + 0.05) * e


class TroposphereTruth:
    """对流层真值：干分量随气压起伏，湿分量按 AR(1) 随机游走。

    使用前需调用 prepare(times) 预生成整个观测时段的湿延迟序列。
    """

    def __init__(self, h_m=80.0, rng=None, wet_sigma=0.020, corr_time=1800.0,
                 wet_bias=0.10):
        rng = rng or np.random.default_rng(0)
        self.rng = rng
        self.h_m = h_m
        self.pressure = float(rng.normal(1010.0, 4.0))
        self.temp_c = float(rng.normal(26.0, 3.0))
        self.rh = float(np.clip(rng.normal(0.60, 0.10), 0.25, 0.95))
        self.zhd = float(saastamoinen_zhd(self.pressure, 34.5, h_m))
        self.zwd_mean = float(saastamoinen_zwd(self.temp_c, self.rh))
        self.zwd_sigma = wet_sigma
        self.corr_time = corr_time
        self.wet_bias = wet_bias          # 真值相对气候均值的系统性偏差
        # 水平梯度（天顶等效，米）
        self.grad_n = float(rng.normal(0, 0.0015))
        self.grad_e = float(rng.normal(0, 0.0015))
        self.zwd_series = None

    def prepare(self, times):
        """预生成湿延迟时间序列（AR(1)，相关时间 corr_time）。"""
        t = np.asarray(times, dtype=float)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 30.0
        rho = np.exp(-dt / self.corr_time)
        innov = np.sqrt(max(1.0 - rho ** 2, 1e-9)) * self.zwd_sigma
        w = np.empty_like(t)
        w[0] = self.rng.normal(0.0, self.zwd_sigma)
        for i in range(1, len(t)):
            w[i] = rho * w[i - 1] + self.rng.normal(0.0, innov)
        self.zwd_series = self.zwd_mean * (1.0 + self.wet_bias) + w
        return self.zwd_series

    def slant_delay(self, az, el, i_epoch=None):
        """斜距对流层延迟 (m)。"""
        m = geo.elev_mapping(el)
        az_r = np.radians(np.asarray(az, dtype=float))
        grad = (self.grad_n * np.cos(az_r) + self.grad_e * np.sin(az_r)) * m
        zwd = (self.zwd_series[i_epoch] if (self.zwd_series is not None
                                           and i_epoch is not None)
               else self.zwd_mean * (1.0 + self.wet_bias))
        return (self.zhd + zwd) * m + grad

    def model_delay(self, az, el):
        """标准大气模型给出的对流层延迟（接收机/广播改正所用）。"""
        p = 1013.25 * (1.0 - 2.2557e-5 * self.h_m) ** 5.2568
        t_c = 20.0 - 0.0065 * self.h_m
        zhd = saastamoinen_zhd(p, 34.5, self.h_m)
        zwd = saastamoinen_zwd(t_c, 0.50)
        return (zhd + zwd) * geo.elev_mapping(el)


# ============================================================================
# 3. 土壤介电特性与地表反射 / 多路径
# ============================================================================
def soil_permittivity(theta_v, sand_frac=0.45):
    """Topp 经验式：体积含水量 -> 土壤相对介电常数实部。

    虚部按导电损耗与束缚水弛豫的简化经验式给出。
    """
    er_real = 3.03 + 9.3 * theta_v + 146.0 * theta_v ** 2 - 76.7 * theta_v ** 3
    er_imag = 0.5 + 8.0 * theta_v
    return er_real, er_imag


def fresnel_rhcp(el_deg, er_real, er_imag):
    """计算地表的 RHCP 同极化/交叉极化反射系数（复数）。

    入参 el_deg 为**卫星仰角**（相对地平线）；入射角（相对地表法线）为
    psi = 90° - el，因此 sin(psi)=cos(el)、cos(psi)=sin(el)：

        Γ_h = (sin el - √(εc - cos²el)) / (sin el + √(εc - cos²el))
        Γ_v = (εc·sin el - √(εc - cos²el)) / (εc·sin el + √(εc - cos²el))

    圆极化分解：Γ_co = (Γ_h + Γ_v)/2（保持右旋），Γ_cross = (Γ_h - Γ_v)/2（变左旋）。

    解析极限（用于自检）：
      * 正入射 el=90°：Γ_co = 0，|Γ_cross| = |(1-√εr)/(1+√εr)|
      * 掠射  el→0 ：Γ_co → -1，Γ_cross → 0
      * 理想导体（εr→∞）：任意仰角下 Γ_co = 0、|Γ_cross| = 1（手性完全反转）
    """
    el = np.radians(np.asarray(el_deg, dtype=float))
    erc = er_real - 1j * er_imag
    cos_e, sin_e = np.cos(el), np.sin(el)
    root = np.sqrt(erc - cos_e ** 2 + 0j)
    g_h = (sin_e - root) / (sin_e + root)              # 水平极化
    g_v = (erc * sin_e - root) / (erc * sin_e + root)  # 垂直极化
    return (g_h + g_v) / 2.0, (g_h - g_v) / 2.0


def roughness_factor(el_deg, sigma_h, wavelength):
    """Rayleigh 粗糙度衰减因子（相干反射的衰减）。"""
    el = np.radians(np.asarray(el_deg, dtype=float))
    k = 2.0 * np.pi / wavelength
    return np.exp(-2.0 * (k * sigma_h * np.sin(el)) ** 2)


@dataclass
class AntennaSpec:
    """天线电性能参数（典型值，来自低成本微带天线与测量型扼流圈天线）。"""
    name: str = "lowcost"
    peak_gain_dbic: float = 3.0
    gain_rolloff_db: float = 11.0      # 低仰角滚降深度
    rolloff_power: float = 1.3
    backlobe_suppression_db: float = 8.0   # 后瓣/地面反射方向抑制
    lhcp_rejection_db: float = 10.0        # 左旋抑制比（轴比）
    pcv_range_mm: float = 7.0              # 相位中心变化幅度
    pcv_azimuth_mm: float = 3.0
    code_noise_k: float = 44.0             # sigma_code = k / sqrt(10^(C/N0/10))
    phase_noise_k: float = 0.25


ANTENNA_LIBRARY = {
    "lowcost": AntennaSpec("低成本微带(探针一体化)", 3.0, 11.0, 1.3, 8.0, 10.0, 7.0, 3.0, 44.0, 0.25),
    "geodetic": AntennaSpec("测量型扼流圈", 5.5, 17.0, 2.2, 22.0, 25.0, 2.0, 0.5, 22.0, 0.18),
    "patch_mast": AntennaSpec("短杆微带天线", 4.0, 9.0, 1.2, 11.0, 12.0, 5.0, 2.0, 40.0, 0.24),
}


class AntennaModel:
    def __init__(self, spec: AntennaSpec, rng=None):
        self.spec = spec
        self.rng = rng or np.random.default_rng(0)
        # 方位向相位中心变化的随机相位（安装朝向差异）
        self.pcv_az_phase = float(self.rng.uniform(0, 2 * np.pi))
        self.pcv_bias = float(self.rng.normal(0, 1.0))     # mm

    def gain_db(self, el_deg):
        s = self.spec
        el = np.asarray(el_deg, dtype=float)
        g = s.peak_gain_dbic - s.gain_rolloff_db * (1.0 - np.sin(np.radians(el))) ** s.rolloff_power
        # 地平线以下急剧下降
        below = el < 0
        if np.any(below):
            g = np.where(below, g - 15.0 * np.minimum(-el, 15.0) / 15.0, g)
        return g

    def pcv_m(self, el_deg, az_deg):
        """相位中心变化 (m)：仰角主项 + 方位二次项。"""
        s = self.spec
        el = np.radians(np.asarray(el_deg, dtype=float))
        az = np.radians(np.asarray(az_deg, dtype=float))
        base = s.pcv_range_mm * 1e-3 * (2.0 * np.sin(el) ** 1.5 - 1.0)
        az_term = s.pcv_azimuth_mm * 1e-3 * np.cos(2.0 * (az - self.pcv_az_phase))
        return base + az_term + self.pcv_bias * 1e-3

    def lhcp_leak(self):
        return 10 ** (-self.spec.lhcp_rejection_db / 20.0)

    def ground_reflection_gain(self):
        return 10 ** (-self.spec.backlobe_suppression_db / 20.0)


class MultipathModel:
    """地表多路径：相干双射线 + 粗糙面漫反射。"""

    def __init__(self, soil, antenna: AntennaModel, rng=None,
                 near_field_factor=1.0, diffuse_amp=0.08, canopy_h=0.0,
                 canopy_atten_db=0.0, canopy_reflect=0.15):
        """near_field_factor: 安装结构/近场附加反射因子，1.0 = 纯双射线几何模型。

        注意：修正 Fresnel 仰角约定后，低仰角处同极化反射系数本身已达 0.6~0.9，
        因此无需再人为放大；该因子仅用于模拟天线近场与地表耦合的附加项。
        """
        self.soil = soil
        self.ant = antenna
        self.rng = rng or np.random.default_rng(0)
        self.near_field = near_field_factor
        self.diffuse_amp = diffuse_amp
        self.canopy_h = canopy_h
        self.canopy_atten_db = canopy_atten_db
        self.canopy_reflect = canopy_reflect
        self.er_real, self.er_imag = soil_permittivity(soil.moisture)

    # ---- 相干分量 -------------------------------------------------------
    def coherent_amplitude(self, el_deg, h_ant, lam):
        """相干多路径复振幅系数 m = a*exp(j*psi)。"""
        g_co, g_cross = fresnel_rhcp(el_deg, self.er_real, self.er_imag)
        rho = roughness_factor(el_deg, self.soil.roughness_m, lam)
        leak = self.ant.lhcp_leak()
        g_ratio = self.ant.ground_reflection_gain() * self.near_field
        # 反射波来自 -el 方向，天线下半空间增益由 backlobe_suppression 描述
        gamma_eff = (g_co + g_cross * leak) * rho
        el = np.asarray(el_deg, dtype=float)
        delay = 2.0 * h_ant * np.sin(np.radians(el))
        phase = -2.0 * np.pi * delay / lam
        m = g_ratio * gamma_eff * np.exp(1j * phase)
        if self.canopy_h > 0:
            # 稀疏作物层的等效反射：体散射去相干后相干分量很小
            canopy_delay = 2.0 * self.canopy_h * np.sin(np.radians(el))
            canopy_phase = -2.0 * np.pi * canopy_delay / lam
            m = m + (self.canopy_reflect * g_ratio * np.exp(1j * canopy_phase)
                     * 10 ** (-self.canopy_atten_db / 20))
        return m, delay, gamma_eff

    def diffuse_amplitude(self, el_deg, dphase, dscale):
        """粗糙面/作物层漫反射的随机复振幅（相位与幅度由外部按 5 min 块给出）。"""
        el = np.asarray(el_deg, dtype=float)
        amp = self.diffuse_amp * dscale * (1.0 + 0.6 * (1.0 - np.sin(np.radians(el))))
        return amp * np.exp(1j * dphase)

    def errors(self, el_deg, h_ant, lam, dphase=None, dscale=None):
        """返回 (码多路径 m, 相位多路径 m, 合成幅度比)。"""
        m_coh, delay, _ = self.coherent_amplitude(el_deg, h_ant, lam)
        m = m_coh
        if dphase is not None:
            m = m + self.diffuse_amplitude(el_deg, dphase, dscale)
        a = np.abs(m)
        psi = np.angle(m)
        # 载波相位多路径: arg(1+m) -> 距离
        phase_err = np.angle(1.0 + m) * lam / (2.0 * np.pi)
        # 码多路径（窄相关器包络，延迟远小于码片时近似）
        chip = geo.C_LIGHT / 1.023e6
        tau_chip = np.abs(delay) / chip
        code_err = (a * np.minimum(tau_chip, 0.1) * chip * np.cos(psi)) / (1.0 + a * np.cos(psi))
        amp_ratio = np.abs(1.0 + m)
        return code_err, phase_err, amp_ratio


# ============================================================================
# 4. 观测噪声与 C/N0
# ============================================================================
CN0_ZENITH = {"G": 45.0, "C": 42.0, "E": 45.0, "R": 43.0}


def cn0_model(system, el_deg, antenna: AntennaModel, amp_ratio=1.0,
              canopy_atten_db=0.0, rng=None, base_offset=0.0):
    """载噪比模型 (dB-Hz)。"""
    el = np.asarray(el_deg, dtype=float)
    ref = CN0_ZENITH.get(system, 44.0) if isinstance(system, str) else \
        np.array([CN0_ZENITH.get(s, 44.0) for s in system])
    g = antenna.gain_db(el)
    g_zen = antenna.gain_db(90.0)
    cn0 = ref + (g - g_zen) - 0.3 - canopy_atten_db
    cn0 = cn0 + 20.0 * np.log10(np.maximum(amp_ratio, 0.2))
    if rng is not None:
        cn0 = cn0 + rng.normal(0.0, 0.4, size=np.shape(cn0))
    return cn0 + base_offset


def code_phase_sigma(cn0_db, antenna: AntennaModel):
    """由 C/N0 给出码/相位测量噪声标准差 (m)。"""
    lin = 10 ** (np.asarray(cn0_db, dtype=float) / 10.0)
    s_code = antenna.spec.code_noise_k / np.sqrt(lin)
    s_phase = antenna.spec.phase_noise_k / np.sqrt(lin)
    return s_code, s_phase


# ============================================================================
# 5. 星历与卫星钟差误差
# ============================================================================
class SatelliteErrorModel:
    """广播星历/钟差误差与精密产品误差。

    广播误差按"轨道误差(SISRE 量级的三维偏差) + 钟差偏差"建模，
    在 1~2 h 的观测时段内视为缓变（符合星历拟合误差的实际特性）。
    """

    def __init__(self, const, rng=None, precise=False):
        rng = rng or np.random.default_rng(0)
        self.rng = rng
        n = len(const)
        self.n = n
        # 真实卫星钟差：偏置 + 慢漂移 + 相对论周期项
        self.clk_bias = rng.normal(0, 3e-5, n)             # s
        self.clk_drift = rng.normal(0, 2e-11, n)           # s/s
        self.clk_period = rng.uniform(3.5e4, 4.5e4, n)
        self.clk_phase = rng.uniform(0, 2 * np.pi, n)
        self.clk_amp = rng.uniform(1e-9, 8e-9, n)
        # 广播星历轨道误差：径向/切向/法向偏差 (m)
        sisre = np.array([{"G": 0.9, "C": 1.2, "E": 0.6, "R": 1.6}[s] for s in const.system])
        if precise:
            sisre = sisre * 0.03
        self.orb_r = rng.normal(0, 0.45 * sisre)
        self.orb_t = rng.normal(0, 0.85 * sisre)
        self.orb_n = rng.normal(0, 0.65 * sisre)
        self.orb_drift = rng.normal(0, 1.0, (n, 3)) * (0.15 * sisre)[:, None]
        # 广播钟差误差 (m)：常量偏置 + 缓慢漂移 (m/h)
        self.clk_err = rng.normal(0, 0.75 * sisre)
        self.clk_err_drift = rng.normal(0, 0.40 * sisre, n)
        # 系统间偏差残留 (m)：多系统联合解算未完全标定的部分
        self.isb_residual = {s: float(rng.normal(0, 0.15)) for s in set(const.system)}

    def clock_true(self, t_unix):
        """真实卫星钟差 (m，等效距离)。"""
        dt = np.asarray(t_unix, dtype=float) - 1.757916e9
        per = 2 * np.pi * dt[..., None] / self.clk_period
        s = (self.clk_bias + self.clk_drift * dt[..., None]
             + self.clk_amp * np.sin(per + self.clk_phase))
        return s * geo.C_LIGHT

    def clock_broadcast(self, t_unix):
        """广播钟差（含误差），漂移项按小时计。"""
        dt = np.asarray(t_unix, dtype=float) - 1.757916e9
        dt_h = dt / 3600.0
        dt_h = dt_h[..., None] if dt_h.ndim > 0 else dt_h
        return self.clock_true(t_unix) + self.clk_err + self.clk_err_drift * dt_h[..., None]

    def orbit_error_ecef(self, sat_ecef, t_unix, idx=None, dt_scale=1.0):
        """广播星历引起的卫星位置误差（ECEF 三维向量, m）。

        idx 为卫星全局索引；给出后按该卫星自身的误差参数取值。
        """
        r = np.asarray(sat_ecef, dtype=float)
        radial = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-9)
        z_axis = np.zeros_like(r)
        z_axis[..., 2] = 1.0
        normal = np.cross(r, z_axis)
        normal = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
        tangential = np.cross(normal, radial)
        sl = (slice(None),) * (r.ndim - 1)
        if idx is None:
            orb_r, orb_t, orb_n = self.orb_r, self.orb_t, self.orb_n
            drift = self.orb_drift
        else:
            idx = np.asarray(idx, dtype=int)
            orb_r, orb_t, orb_n = self.orb_r[idx], self.orb_t[idx], self.orb_n[idx]
            drift = self.orb_drift[idx]
        dt = (np.asarray(t_unix, dtype=float) - 1.757916e9) / 3600.0
        dt = dt[..., None] if dt.ndim > 0 else dt
        drift_term = drift * dt * dt_scale
        return (radial * orb_r[sl][..., None]
                + tangential * orb_t[sl][..., None]
                + normal * orb_n[sl][..., None]
                + drift_term)
