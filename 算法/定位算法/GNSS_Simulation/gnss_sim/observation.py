"""观测生成：把各类真实误差源合成为伪距/载波相位观测。

关键设计是"真值 / 模型"分离：
  * 观测量由真值几何 + 真实误差源合成；
  * 解算器只知道它"应该知道"的模型（广播星历、Klobuchar、标准大气等），
    两者之差自然形成定位误差，无需人为注入误差。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geodesy as geo
from . import models as md
from .config import ScenarioConfig, ProbeSpec, BaseStationConfig

DUAL_FREQ_HZ = 1176.45e6        # L5 / B2a / E5a
TRACKING_CN0_MIN = 28.0         # dB-Hz，低于此值接收机失锁


# ============================================================================
# 接收机状态
# ============================================================================
def make_n_offset(rng, const) -> dict:
    """为每颗卫星/每个频率生成固定的整周模糊度（不随时间变化）。

    N 取 round(2.2e7/λ) + 随机整数，量级对应 ~22000 km 的几何距离，
    与真实的载波相位观测值 L = ρ + N·λ 一致。
    """
    lam1 = geo.C_LIGHT / const.freq
    f2 = np.where(np.isin(const.system, ["G", "C", "E"]), DUAL_FREQ_HZ,
                  const.freq * 1.2)
    lam2 = geo.C_LIGHT / f2
    base = 2.2e7
    return {
        "f1": np.round(base / lam1) + rng.integers(0, 500, len(const)),
        "f2": np.round(base / lam2) + rng.integers(0, 500, len(const)),
    }


class ClockModel:
    """接收机钟差（米）：偏置 + 漂移 + AR(1) 抖动。"""

    def __init__(self, rng, sigma=0.05, corr_time=60.0):
        self.rng = rng
        self.bias = float(rng.normal(0, 3000.0))
        self.drift = float(rng.normal(0, 15.0))
        self.sigma = sigma
        self.corr_time = corr_time
        self.series = None

    def prepare(self, times, t0):
        t = np.asarray(times, dtype=float)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 30.0
        rho = np.exp(-dt / self.corr_time)
        innov = np.sqrt(max(1 - rho ** 2, 1e-9)) * self.sigma
        w = np.empty_like(t)
        w[0] = self.rng.normal(0, self.sigma)
        for i in range(1, len(t)):
            w[i] = rho * w[i - 1] + self.rng.normal(0, innov)
        self.series = self.bias + self.drift * (t - t0) + w
        return self.series


@dataclass
class ReceiverState:
    """一台接收机（探针或基准站）的物理状态。"""
    name: str
    ecef: np.ndarray
    lat: float
    lon: float
    h: float
    antenna: md.AntennaModel
    mp: md.MultipathModel
    el_mask: float = 10.0
    height: float = 0.2                 # 天线相位中心离地高度
    clock: ClockModel = None
    tropo: md.TroposphereTruth = None   # 该站点上空的对流层真值
    dual_freq: bool = False
    n_offset: np.ndarray = None         # 各卫星的整周模糊度随机偏置（不随时间变化）

    @property
    def enu(self):
        return self._enu

    def set_enu(self, enu):
        self._enu = np.asarray(enu, dtype=float)


@dataclass
class FreqObs:
    key: str
    freq: float
    lam: float
    P: np.ndarray
    L: np.ndarray
    iono: np.ndarray            # 真值电离层延迟 (m)
    mp_code: np.ndarray
    mp_phase: np.ndarray
    pcv: np.ndarray
    sigma_code: np.ndarray
    sigma_phase: np.ndarray
    n_true: np.ndarray


@dataclass
class Record:
    """一个接收机在一个历元的完整观测集。"""
    t: float
    i_epoch: int
    idx: np.ndarray                 # 卫星全局索引
    systems: np.ndarray
    sat_true: np.ndarray            # (K,3) 发射时刻的 ECEF 位置
    rho_true: np.ndarray
    trop_true: np.ndarray
    cn0: np.ndarray
    clk_rx_m: float
    rx_true: np.ndarray
    freqs: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.idx)

    @property
    def f1(self) -> FreqObs:
        return self.freqs["f1"]


# ============================================================================
# 世界模型
# ============================================================================
class WorldModel:
    """整个仿真场景的真值世界：大气、卫星误差、健康状态与观测合成。"""

    def __init__(self, scn: ScenarioConfig, const, seed: int = 0, dual_freq=False):
        self.scn = scn
        self.const = const
        self.dual_freq = dual_freq
        self.rng = np.random.default_rng(seed)
        self.frame = geo.LocalFrame(scn.field.lat0, scn.field.lon0, scn.field.h0)

        # --- 大气真值 -----------------------------------------------------
        self.iono = md.IonosphereTruth(scn.iono.sfu, self.rng, scn.iono.storm)
        self.alpha, self.beta = self.iono.broadcast_coefficients()
        field_tropo = md.TroposphereTruth(scn.field.h0, self.rng)
        self.field_tropo = field_tropo
        # CORS 站点（15 km 外）具有不同的气象状态
        self.base_tropo = md.TroposphereTruth(scn.field.h0 + 5.0, self.rng,
                                              wet_bias=float(self.rng.normal(0.15, 0.15)))

        # --- 卫星误差 -----------------------------------------------------
        self.sat_err = md.SatelliteErrorModel(const, self.rng)
        self.healthy = self.rng.random(len(const)) > scn.const.unhealthy_fraction

        # --- 时间 ---------------------------------------------------------
        n_ep = scn.session.epochs
        self.times = scn.session.t_start_unix + np.arange(n_ep) * scn.session.interval_s
        self._sat_tx = None
        self._t_tx = None

    # ------------------------------------------------------------------
    def prepare_session(self):
        """预生成时间序列、卫星位置与漫反射随机状态。"""
        self.field_tropo.prepare(self.times)
        self.base_tropo.prepare(self.times)
        # 发射时刻（一次迭代即可，误差 < 1 mm）
        sat_rx = self.const.propagate_ecef(self.times)          # (E,N,3)
        center = self.frame.enu_to_ecef([0.0, 0.0, 0.0])
        rho0 = np.linalg.norm(sat_rx - center[None, None, :], axis=-1)
        self._t_tx = self.times[:, None] - rho0 / geo.C_LIGHT
        self._sat_tx = self.const.propagate_ecef_matrix(self._t_tx)
        # 漫反射多路径随机状态：每 5 min 一块，逐卫星独立
        n_blocks = int(np.ceil(self.scn.session.duration_s / 300.0)) + 1
        self.diffuse_phase = self.rng.uniform(0, 2 * np.pi, (n_blocks, len(self.const)))
        self.diffuse_scale = self.rng.uniform(0.6, 1.4, (n_blocks, len(self.const)))
        return self

    # ------------------------------------------------------------------
    def make_probe(self, i, enu, spec: ProbeSpec, antenna: md.AntennaModel,
                   mp: md.MultipathModel, height: float) -> ReceiverState:
        ecef = self.frame.enu_to_ecef(enu)
        lat, lon, h = geo.ecef_to_geodetic(ecef)
        rx = ReceiverState(
            name=f"P{i+1:02d}", ecef=ecef, lat=float(lat), lon=float(lon), h=float(h),
            antenna=antenna, mp=mp, el_mask=self.scn.el_mask(spec.mount),
            height=height, clock=ClockModel(self.rng), tropo=self.field_tropo,
            dual_freq=self.dual_freq)
        rx.clock.prepare(self.times, self.times[0])
        rx.set_enu(enu)
        rx.n_offset = make_n_offset(self.rng, self.const)
        return rx

    def make_base(self, cfg: BaseStationConfig) -> ReceiverState:
        """构造基准站接收机（田块中心或 15 km 外 CORS）。"""
        if cfg.kind == "local":
            enu = np.array([0.0, 0.0, 0.0])
            tropo = self.field_tropo
        else:
            enu = np.array([0.0, cfg.distance_km * 1000.0, 0.0])
            tropo = self.base_tropo
        ecef = self.frame.enu_to_ecef(enu)
        lat, lon, h = geo.ecef_to_geodetic(ecef)
        ant_spec = md.ANTENNA_LIBRARY[cfg.antenna]
        ant = md.AntennaModel(ant_spec, self.rng)
        mp = md.MultipathModel(self.scn.soil, ant, self.rng,
                               near_field_factor=1.0, diffuse_amp=0.03)
        rx = ReceiverState(
            name="BASE", ecef=ecef, lat=float(lat), lon=float(lon), h=float(h),
            antenna=ant, mp=mp, el_mask=10.0, height=cfg.height_m,
            clock=ClockModel(self.rng), tropo=tropo, dual_freq=True)
        rx.clock.prepare(self.times, self.times[0])
        rx.set_enu(enu)
        rx.n_offset = make_n_offset(self.rng, self.const)
        return rx

    # ------------------------------------------------------------------
    def build_record(self, rx: ReceiverState, i_epoch: int,
                     dual: bool | None = None) -> Record:
        """合成一个历元的全部观测量。dual 可覆盖世界默认的双频设置。"""
        t = float(self.times[i_epoch])
        dis = self.scn.disable
        sat = self._sat_tx[i_epoch]                     # (N,3)
        los = sat - rx.ecef
        rho_all = np.linalg.norm(los, axis=-1)
        enu = los @ self.frame.rot.T
        el_all = np.degrees(np.arctan2(enu[:, 2], np.hypot(enu[:, 0], enu[:, 1])))
        az_all = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360.0

        vis = (el_all >= rx.el_mask) & self.healthy
        idx = np.nonzero(vis)[0]
        if len(idx) < 4:
            raise RuntimeError(f"{rx.name} 可见卫星不足: {len(idx)}")

        el_v, az_v, rho_v = el_all[idx], az_all[idx], rho_all[idx]
        sys_v = self.const.system[idx]
        t_block = int((t - self.times[0]) // 300.0)
        dphase = self.diffuse_phase[t_block, idx]
        dscale = self.diffuse_scale[t_block, idx]

        # --- 真值大气 -----------------------------------------------------
        iono_l1 = self.iono.slant_delay_l1(rx.lat, rx.lon, az_v, el_v, t)
        if "iono" in dis:
            iono_l1 = np.zeros_like(iono_l1)
        trop = rx.tropo.slant_delay(az_v, el_v, i_epoch)
        if "tropo" in dis:
            trop = np.zeros_like(trop)
        clk_sv = self.sat_err.clock_true(t)[idx]
        clk_rx = float(rx.clock.series[i_epoch])

        # --- 各频率观测量 ------------------------------------------------
        freqs = {}
        f1 = self.const.freq[idx]
        # 电离层真值以 1575.42 MHz 为基准给出，按各系统实际频率缩放
        groups = {"f1": (f1, (1575.42e6 / f1) ** 2)}
        use_dual = self.dual_freq if dual is None else dual
        if use_dual:
            f2 = np.where(np.isin(sys_v, ["G", "C", "E"]), DUAL_FREQ_HZ, f1 * 1.2)
            groups["f2"] = (f2, (1575.42e6 / f2) ** 2)

        cn0_f1 = None
        for key, (freq_arr, iono_scale) in groups.items():
            lam = geo.C_LIGHT / freq_arr
            mp_code, mp_phase, amp_ratio = rx.mp.errors(
                el_v, rx.height, lam, dphase, dscale)
            if "multipath" in dis:
                mp_code = np.zeros_like(mp_code)
                mp_phase = np.zeros_like(mp_phase)
                amp_ratio = np.ones_like(amp_ratio)
            cn0 = md.cn0_model(sys_v, el_v, rx.antenna, amp_ratio,
                               self.scn.soil.canopy_atten_db, self.rng)
            if key == "f1":
                cn0_f1 = cn0
            s_code, s_phase = md.code_phase_sigma(cn0, rx.antenna)
            if "noise" in dis:
                s_code = np.zeros_like(s_code)
                s_phase = np.zeros_like(s_phase)
            pcv = rx.antenna.pcv_m(el_v, az_v)
            if "pcv" in dis:
                pcv = np.zeros_like(pcv)
            iono = iono_l1 * iono_scale
            n_true = rx.n_offset[key][idx]
            P = (rho_v + clk_rx - clk_sv + iono + trop + mp_code
                 + self.rng.normal(0, 1, len(idx)) * s_code)
            L = (rho_v + clk_rx - clk_sv - iono + trop + mp_phase + pcv
                 + n_true * lam
                 + self.rng.normal(0, 1, len(idx)) * s_phase)
            freqs[key] = FreqObs(key=key, freq=freq_arr, lam=lam, P=P, L=L,
                                 iono=iono, mp_code=mp_code, mp_phase=mp_phase,
                                 pcv=pcv, sigma_code=s_code, sigma_phase=s_phase,
                                 n_true=n_true)

        rec = Record(t=t, i_epoch=i_epoch, idx=idx, systems=sys_v,
                     sat_true=sat[idx], rho_true=rho_v, trop_true=trop,
                     cn0=cn0_f1, clk_rx_m=clk_rx, rx_true=rx.ecef, freqs=freqs)
        ok = cn0_f1 >= TRACKING_CN0_MIN
        if not np.all(ok):
            rec = _subset_record(rec, ok)
        return rec


def _subset_record(rec: Record, mask: np.ndarray) -> Record:
    """按掩码裁剪一个历元的观测（用于剔除失锁卫星）。"""
    freqs = {}
    for key, fo in rec.freqs.items():
        freqs[key] = FreqObs(key=fo.key, freq=fo.freq[mask], lam=fo.lam[mask],
                             P=fo.P[mask], L=fo.L[mask], iono=fo.iono[mask],
                             mp_code=fo.mp_code[mask], mp_phase=fo.mp_phase[mask],
                             pcv=fo.pcv[mask], sigma_code=fo.sigma_code[mask],
                             sigma_phase=fo.sigma_phase[mask], n_true=fo.n_true[mask])
    return Record(t=rec.t, i_epoch=rec.i_epoch, idx=rec.idx[mask],
                  systems=rec.systems[mask], sat_true=rec.sat_true[mask],
                  rho_true=rec.rho_true[mask], trop_true=rec.trop_true[mask],
                  cn0=rec.cn0[mask], clk_rx_m=rec.clk_rx_m,
                  rx_true=rec.rx_true, freqs=freqs)
