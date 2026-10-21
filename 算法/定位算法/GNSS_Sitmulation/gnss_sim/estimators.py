"""定位解算器：SPP / SBAS / DGNSS / RTK / PPP。

核心思想：解算器只使用它"应该知道"的模型量（广播星历、Klobuchar、标准大气、
标准相位中心改正等），与观测中的真值之差自然形成定位误差。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geodesy as geo
from . import models as md
from .observation import Record, FreqObs


# ============================================================================
# 改正服务模型
# ============================================================================
@dataclass
class CorrectionService:
    """外部改正源的能力描述。

    各 scale 表示"改正之后仍残留的误差比例"：
      残差 = scale x (真值 - 接收机自算模型值)
    scale=1 表示完全靠接收机自己的广播/标准模型（无外部改正）。
    """
    key: str
    orb_scale: float = 1.0          # 星历误差残留
    clk_scale: float = 1.0          # 卫星钟差误差残留
    iono_scale: float = 1.0         # Klobuchar 模型残差的残留
    tropo_scale: float = 1.0        # 标准大气模型残差的残留
    pcv_known: bool = False         # 是否已知探针天线相位中心改正

    def sat_pos(self, rec: Record, world) -> np.ndarray:
        scale = 0.0 if "ephemeris" in world.scn.disable else self.orb_scale
        err = world.sat_err.orbit_error_ecef(rec.sat_true, rec.t, idx=rec.idx)
        return rec.sat_true + scale * err

    def sat_clk(self, rec: Record, world) -> np.ndarray:
        t = rec.t
        scale = 0.0 if "ephemeris" in world.scn.disable else self.clk_scale
        true = world.sat_err.clock_true(t)[rec.idx]
        bc = world.sat_err.clock_broadcast(t)[rec.idx]
        return true + scale * (bc - true)

    def iono_model(self, rec: Record, world) -> np.ndarray:
        """接收机所用电离层改正模型值 (m)。

        误差预算实验中若关闭了电离层误差源，模型必须与之一致（视为完全改正），
        否则会人为注入一个虚假改正量。
        """
        true = rec.f1.iono
        if "iono" in world.scn.disable:
            return true
        az, el = rec_az_el(rec, world)
        gps_sod = rec.t % 86400.0
        klob = md.klobuchar_delay(world.alpha, world.beta, world.scn.field.lat0,
                                  world.scn.field.lon0, az, el, gps_sod)
        return true - self.iono_scale * (true - klob)

    def tropo_model(self, rec: Record, world) -> np.ndarray:
        if "tropo" in world.scn.disable:
            return rec.trop_true
        az, el = rec_az_el(rec, world)
        std = world.field_tropo.model_delay(az, el)
        return rec.trop_true - self.tropo_scale * (rec.trop_true - std)

    def pcv_model(self, rec: Record, world) -> np.ndarray:
        if self.pcv_known:
            return rec.f1.pcv
        return np.zeros(len(rec))


BROADCAST = CorrectionService("broadcast")
SBAS = CorrectionService("sbas", orb_scale=0.30, clk_scale=0.30,
                         iono_scale=0.25, tropo_scale=0.90)
PRECISE = CorrectionService("precise", orb_scale=0.03, clk_scale=0.04,
                            iono_scale=0.0, tropo_scale=1.0,
                            pcv_known=False)


def rec_az_el(rec: Record, world):
    """由真值几何给出方位角/仰角（接收机自己也能算出）。"""
    los = rec.sat_true - rec.rx_true
    enu = los @ world.frame.rot.T
    el = np.degrees(np.arctan2(enu[:, 2], np.hypot(enu[:, 0], enu[:, 1])))
    az = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360.0
    return az, el


@dataclass
class Solution:
    enu: np.ndarray
    ok: bool = True
    fixed: bool = False
    float_only: bool = False
    n_sat: int = 0
    n_dd: int = 0
    n_fixed: int = 0
    pdop: float = np.nan
    hdop: float = np.nan
    vdop: float = np.nan
    res_rms: float = np.nan
    sigma_amb: float = np.nan
    note: str = ""


def _los_hat(r_sat, x):
    d = r_sat - x
    r = np.linalg.norm(d, axis=-1)
    return d / r[:, None], r


# ---------------------------------------------------------------------------
# 模糊度固定：基于条件方差的序贯取整（简化 LAMBDA/自助法）
# ---------------------------------------------------------------------------
def _ldl(Q):
    """Q = L D L^T，L 为单位下三角，返回 (L, D, 是否正定)。

    D 出现非正值说明 Q 病态（几何退化），调用方应停止继续固定，
    而不是把它当成"精度极高"。
    """
    n = Q.shape[0]
    A = np.array(Q, dtype=float, copy=True)
    L = np.eye(n)
    D = np.zeros(n)
    ok = True
    for j in range(n):
        d = A[j, j]
        if d <= 1e-12:
            ok = False
            D[j] = np.inf          # 标记为不可固定
            d = 1e-12
        D[j] = d if ok else np.inf
        for i in range(j + 1, n):
            L[i, j] = A[i, j] / d
        for i in range(j + 1, n):
            lij = L[i, j] * d
            for k in range(j + 1, n):
                A[i, k] -= lij * L[k, j]
    return L, D, ok


def sequential_fix(n_float, Q, sigma_thresh=0.25):
    """按条件方差从小到大序贯取整，返回 (整数解, 可固定掩码, 条件标准差)。

    先按边缘标准差升序排序，再按 LDL 分解得到的条件方差逐个判定，
    只接受满足阈值的前缀（保证条件分布成立）。Q 非正定时立即停止固定。
    """
    n = len(n_float)
    order = np.argsort(np.diag(Q))
    idx_sorted = order
    Qs = Q[np.ix_(idx_sorted, idx_sorted)]
    L, D, pd_ok = _ldl(Qs)
    sig_cond = np.sqrt(np.maximum(D, 0.0))
    n_cond = np.array(n_float, dtype=float)[idx_sorted]
    n_boot = np.zeros(n)
    mask_sorted = np.zeros(n, dtype=bool)
    for i in range(n):
        n_boot[i] = np.round(n_cond[i])
        if not pd_ok or not np.isfinite(sig_cond[i]) or sig_cond[i] >= sigma_thresh:
            break                     # 只取满足条件的前缀
        mask_sorted[i] = True
        n_cond[i + 1:] += L[i + 1:, i] * (n_boot[i] - n_cond[i])
    n_fixed = np.zeros(n)
    mask = np.zeros(n, dtype=bool)
    n_fixed[idx_sorted] = n_boot
    mask[idx_sorted] = mask_sorted
    sig_out = np.zeros(n)
    sig_out[idx_sorted] = np.where(np.isfinite(sig_cond), sig_cond, np.inf)
    return n_fixed, mask, sig_out


# ============================================================================
# 单点定位 (SPP / SBAS)
# ============================================================================
def solve_spp(rec: Record, service: CorrectionService, world, x0_enu,
              systems=None, use_freq="f1", sig_floor=0.4, max_iter=8):
    """加权最小二乘单点定位，逐系统估计接收机钟差。"""
    fo: FreqObs = rec.freqs[use_freq]
    if systems is not None:
        m = np.isin(rec.systems, list(systems))
        if np.sum(m) < 5:
            return Solution(enu=np.full(3, np.nan), ok=False, note="卫星不足")
        idx = np.nonzero(m)[0]
        fo = _subset_freq(fo, idx)
        systems_v = rec.systems[idx]
        r_sat = service.sat_pos(rec, world)[idx]
        clk_sv = service.sat_clk(rec, world)[idx]
        iono_m = service.iono_model(rec, world)[idx]
        trop_m = service.tropo_model(rec, world)[idx]
    else:
        systems_v = rec.systems
        r_sat = service.sat_pos(rec, world)
        clk_sv = service.sat_clk(rec, world)
        iono_m = service.iono_model(rec, world)
        trop_m = service.tropo_model(rec, world)

    p_corr = fo.P + clk_sv - iono_m - trop_m
    sigma = np.sqrt(fo.sigma_code ** 2 + sig_floor ** 2)
    w = 1.0 / sigma

    sys_list = sorted(set(systems_v))
    e_mat = np.zeros((len(fo.P), len(sys_list)))
    for j, s in enumerate(sys_list):
        e_mat[:, j] = (systems_v == s).astype(float)

    x = world.frame.enu_to_ecef(np.asarray(x0_enu, dtype=float))
    n_param = 3 + len(sys_list)
    a_mat = None
    for _ in range(max_iter):
        los_hat, rho = _los_hat(r_sat, x)
        a_mat = np.hstack([-los_hat, e_mat])
        y = p_corr - rho
        aw = a_mat * w[:, None]
        dx, *_ = np.linalg.lstsq(aw, y * w, rcond=None)
        x = x + dx[:3]
        if np.linalg.norm(dx[:3]) < 1e-4:
            break

    # 协方差与 DOP
    try:
        q = np.linalg.inv(a_mat.T @ (a_mat * (w ** 2)[:, None]))
        rot = world.frame.rot
        q_enu = rot @ q[:3, :3] @ rot.T
        hdop = np.sqrt(q_enu[0, 0] + q_enu[1, 1]) * np.mean(sigma)
        vdop = np.sqrt(q_enu[2, 2]) * np.mean(sigma)
        pdop = np.sqrt(np.trace(q_enu)) * np.mean(sigma)
    except np.linalg.LinAlgError:
        hdop = vdop = pdop = np.nan

    res = p_corr - (np.linalg.norm(r_sat - x, axis=-1))
    return Solution(enu=world.frame.ecef_to_enu(x), n_sat=len(fo.P),
                    hdop=hdop, vdop=vdop, pdop=pdop,
                    res_rms=float(np.sqrt(np.mean(res ** 2))))


def _subset_freq(fo: FreqObs, idx) -> FreqObs:
    return FreqObs(key=fo.key, freq=fo.freq[idx], lam=fo.lam[idx], P=fo.P[idx],
                   L=fo.L[idx], iono=fo.iono[idx], mp_code=fo.mp_code[idx],
                   mp_phase=fo.mp_phase[idx], pcv=fo.pcv[idx],
                   sigma_code=fo.sigma_code[idx], sigma_phase=fo.sigma_phase[idx],
                   n_true=fo.n_true[idx])


# ============================================================================
# 观测量的模型改正（基准站与探针共用）
# ============================================================================
def corrected_observations(rec: Record, service: CorrectionService, world):
    """返回 (P_tilde, L_tilde, r_sat_model, az, el)。

    P̃ = P + clk̂_sv - Î - T̂
    L̃ = L + clk̂_sv + Î - T̂ - p̂cv
    """
    r_sat = service.sat_pos(rec, world)
    clk_sv = service.sat_clk(rec, world)
    iono_m = service.iono_model(rec, world)
    trop_m = service.tropo_model(rec, world)
    pcv_m = service.pcv_model(rec, world)
    p_t = rec.f1.P + clk_sv - iono_m - trop_m
    l_t = rec.f1.L + clk_sv + iono_m - trop_m - pcv_m
    az, el = rec_az_el(rec, world)
    return p_t, l_t, r_sat, az, el


def _match(rec_r: Record, rec_b: Record):
    common, ir, ib = np.intersect1d(rec_r.idx, rec_b.idx, return_indices=True)
    return common, ir, ib


# ============================================================================
# 伪距差分 (DGNSS)
# ============================================================================
def solve_dgnss(rec_r: Record, rec_b: Record, service: CorrectionService, world,
                base_known_ecef, x0_enu, sig_floor=0.3, max_iter=8):
    """基准站伪距改正数 + 探针加权最小二乘（站间单差消卫星钟差）。"""
    common, ir, ib = _match(rec_r, rec_b)
    if len(common) < 5:
        return Solution(enu=np.full(3, np.nan), ok=False, note="共视卫星不足")

    p_r, _, r_sat_r, _, _ = corrected_observations(rec_r, service, world)
    p_b, _, r_sat_b, _, _ = corrected_observations(rec_b, service, world)
    rho_b_model = np.linalg.norm(r_sat_b - base_known_ecef, axis=-1)
    prc = p_b - rho_b_model                       # 基准站伪距改正数

    r_sat = r_sat_r[ir]
    p_diff = p_r[ir] - prc[ib]
    sigma = np.sqrt(rec_r.f1.sigma_code[ir] ** 2
                    + rec_b.f1.sigma_code[ib] ** 2 + sig_floor ** 2)
    w = 1.0 / sigma

    systems_v = rec_r.systems[ir]
    sys_list = sorted(set(systems_v))
    e_mat = np.zeros((len(common), len(sys_list)))
    for j, s in enumerate(sys_list):
        e_mat[:, j] = (systems_v == s).astype(float)

    x = world.frame.enu_to_ecef(np.asarray(x0_enu, dtype=float))
    a_mat = None
    for _ in range(max_iter):
        los_hat, rho = _los_hat(r_sat, x)
        a_mat = np.hstack([-los_hat, e_mat])
        y = p_diff - rho
        aw = a_mat * w[:, None]
        dx, *_ = np.linalg.lstsq(aw, y * w, rcond=None)
        x = x + dx[:3]
        if np.linalg.norm(dx[:3]) < 1e-4:
            break
    res = p_diff - np.linalg.norm(r_sat - x, axis=-1)
    return Solution(enu=world.frame.ecef_to_enu(x), n_sat=len(common),
                    res_rms=float(np.sqrt(np.mean(res ** 2))))


# ============================================================================
# 载波相位双差 (RTK)
# ============================================================================
def solve_rtk(rec_r: Record, rec_b: Record, service: CorrectionService, world,
              base_known_ecef, x0_enu, fix_ambiguity=True,
              sigma_thresh=0.25, frac_thresh=0.25, res_thresh=0.030,
              res_max=0.060):
    """双差 RTK：码+相位联合浮点解 + 整数模糊度固定（部分固定策略）。"""
    common, ir, ib = _match(rec_r, rec_b)
    if len(common) < 5:
        return Solution(enu=np.full(3, np.nan), ok=False, note="共视卫星不足")

    p_r, l_r, r_sat_r, _, el_r_all = corrected_observations(rec_r, service, world)
    p_b, l_b, _, _, _ = corrected_observations(rec_b, service, world)
    r_sat = r_sat_r[ir]
    el_r = el_r_all[ir]

    sd_p = p_r[ir] - p_b[ib]                 # 站间单差（接收机钟差保留，双差中抵消）
    sd_l = l_r[ir] - l_b[ib]

    systems_v = rec_r.systems[ir]
    ref = {}
    for s in set(systems_v):
        m = np.nonzero(systems_v == s)[0]
        ref[s] = int(m[int(np.argmax(el_r[m]))])

    dd_rows, dd_ref, dd_lam = [], [], []
    for s in sorted(ref):
        r0 = ref[s]
        for k in np.nonzero(systems_v == s)[0]:
            if k == r0:
                continue
            dd_rows.append(k); dd_ref.append(r0); dd_lam.append(rec_r.f1.lam[ir][k])
    dd_rows = np.array(dd_rows, dtype=int)
    dd_ref = np.array(dd_ref, dtype=int)
    dd_lam = np.array(dd_lam)
    n_dd = len(dd_rows)
    if n_dd < 4:
        return Solution(enu=np.full(3, np.nan), ok=False, note="双差观测量不足")

    ddp = sd_p[dd_rows] - sd_p[dd_ref]
    ddl = sd_l[dd_rows] - sd_l[dd_ref]

    sig_p = np.sqrt(rec_r.f1.sigma_code[ir][dd_rows] ** 2
                    + rec_b.f1.sigma_code[ib][dd_rows] ** 2
                    + rec_r.f1.sigma_code[ir][dd_ref] ** 2
                    + rec_b.f1.sigma_code[ib][dd_ref] ** 2 + 0.02 ** 2)
    sig_l = np.sqrt(rec_r.f1.sigma_phase[ir][dd_rows] ** 2
                    + rec_b.f1.sigma_phase[ib][dd_rows] ** 2
                    + rec_r.f1.sigma_phase[ir][dd_ref] ** 2
                    + rec_b.f1.sigma_phase[ib][dd_ref] ** 2 + 0.004 ** 2)

    rho_b = np.linalg.norm(r_sat - base_known_ecef, axis=-1)

    def design(x_cur):
        los_hat, rho = _los_hat(r_sat, x_cur)
        d_geom = (rho[dd_rows] - rho[dd_ref]) - (rho_b[dd_rows] - rho_b[dd_ref])
        a_pos = -(los_hat[dd_rows] - los_hat[dd_ref])
        return d_geom, a_pos

    x = world.frame.enu_to_ecef(np.asarray(x0_enu, dtype=float))
    d_geom, a_pos = design(x)
    a_mat = np.vstack([
        np.hstack([a_pos, np.zeros((n_dd, n_dd))]),
        np.hstack([a_pos, np.diag(dd_lam)]),
    ])
    w_vec = np.concatenate([1.0 / sig_p, 1.0 / sig_l])
    y = np.concatenate([ddp - d_geom, ddl - d_geom])
    aw = a_mat * w_vec[:, None]
    try:
        q = np.linalg.inv(aw.T @ aw)
    except np.linalg.LinAlgError:
        return Solution(enu=np.full(3, np.nan), ok=False, note="法方程奇异")
    dx = q @ (aw.T @ (y * w_vec))
    x_float = x + dx[:3]
    n_float = dx[3:]
    q_n = q[3:, 3:]
    sig_n = np.sqrt(np.abs(np.diag(q_n)))

    sol = Solution(enu=world.frame.ecef_to_enu(x_float), n_dd=n_dd,
                   n_sat=len(common), float_only=True,
                   sigma_amb=float(np.min(sig_n)) if n_dd else np.nan)
    if not fix_ambiguity:
        return sol

    n_hat, fixable, sig_cond = sequential_fix(n_float, q_n, sigma_thresh)
    sol.n_fixed = int(np.sum(fixable))
    sol.sigma_amb = float(np.max(sig_cond[fixable])) if sol.n_fixed else float(np.min(sig_cond))
    if sol.n_fixed < 4:
        sol.note = "模糊度未能固定"
        return sol

    # 用已固定的模糊度解位置，并剔除相位残差过大的卫星（防止错误固定）
    res_l = np.zeros(n_dd)
    rms, rmax = np.inf, np.inf
    for _ in range(3):
        rows_f = np.nonzero(fixable)[0]
        if len(rows_f) < 4:
            sol.note = "可用固定卫星不足"
            return sol
        a2 = np.vstack([a_pos[rows_f], a_pos[rows_f]])
        y2 = np.concatenate([(ddp - d_geom)[rows_f],
                             (ddl - n_hat * dd_lam)[rows_f]])
        w2 = np.concatenate([1.0 / sig_p[rows_f], 1.0 / sig_l[rows_f]])
        aw2 = a2 * w2[:, None]
        dx2, *_ = np.linalg.lstsq(aw2, y2 * w2, rcond=None)
        x_fixed = x + dx2[:3]
        d_geom2, _ = design(x_fixed)
        res_l = (ddl - n_hat * dd_lam) - d_geom2
        bad = fixable & (np.abs(res_l) > res_max)
        if not np.any(bad):
            break
        fixable = fixable & ~bad
    rows_f = np.nonzero(fixable)[0]
    rms = float(np.sqrt(np.mean(res_l[rows_f] ** 2)))
    rmax = float(np.max(np.abs(res_l[rows_f])))
    if rms > res_thresh or rmax > res_max:
        sol.note = f"固定被拒绝(残差 RMS {rms*1000:.1f} mm / 最大 {rmax*1000:.1f} mm)"
        return sol
    return Solution(enu=world.frame.ecef_to_enu(x_fixed), n_dd=n_dd,
                    n_sat=len(common), fixed=True, n_fixed=int(np.sum(fixable)),
                    sigma_amb=sol.sigma_amb, res_rms=rms)


# ============================================================================
# 静态多历元 RTK（探针静止数分钟，模糊度可跨历元固定）
# ============================================================================
def solve_rtk_static(recs_r: list[Record], recs_b: list[Record],
                     service: CorrectionService, world, base_known_ecef, x0_enu,
                     fix_ambiguity=True, sigma_thresh=0.25, frac_thresh=0.25,
                     res_thresh=0.030, res_max=0.060, min_epochs=2):
    """静态 RTK：位置固定，双差模糊度在观测时段内不变。

    同时给出逐历元（瞬时）解算的固定成功率。
    """
    if len(recs_r) < min_epochs:
        return Solution(enu=np.full(3, np.nan), ok=False, note="历元不足")

    # 全时段共视卫星
    common_all = None
    for rr, rb in zip(recs_r, recs_b):
        c = np.intersect1d(rr.idx, rb.idx)
        common_all = c if common_all is None else np.intersect1d(common_all, c)
    if common_all is None or len(common_all) < 5:
        return Solution(enu=np.full(3, np.nan), ok=False, note="全时段共视卫星不足")

    # 参考星：按第一个历元的仰角取各系统最高者
    _, el0 = rec_az_el(recs_r[0], world)
    sys0 = recs_r[0].systems
    pos0 = {int(g): k for k, g in enumerate(recs_r[0].idx)}
    sel0 = np.array([pos0[int(g)] for g in common_all])
    ref = {}
    for s in set(sys0[sel0]):
        m = np.nonzero(sys0[sel0] == s)[0]
        ref[s] = int(sel0[m[int(np.argmax(el0[sel0][m]))]])

    # 双差行索引（相对 common_all 的位置）
    rel = {int(g): k for k, g in enumerate(common_all)}
    dd_rows, dd_ref, dd_lam = [], [], []
    for s in sorted(ref):
        r0 = rel[int(recs_r[0].idx[ref[s]])]
        for k, g in enumerate(common_all):
            if k == r0:
                continue
            if sys0[sel0[k]] != s:
                continue
            dd_rows.append(k); dd_ref.append(r0)
            dd_lam.append(float(recs_r[0].f1.lam[sel0[k]]))
    dd_rows = np.array(dd_rows, dtype=int)
    dd_ref = np.array(dd_ref, dtype=int)
    dd_lam = np.array(dd_lam)
    n_dd = len(dd_rows)
    if n_dd < 4:
        return Solution(enu=np.full(3, np.nan), ok=False, note="双差观测量不足")

    x = world.frame.enu_to_ecef(np.asarray(x0_enu, dtype=float))
    n_ep = len(recs_r)

    def build(x_cur):
        """组装多历元法方程（位置 + 跨历元不变的模糊度）。"""
        n_par = 3 + n_dd
        ata = np.zeros((n_par, n_par))
        aty = np.zeros(n_par)
        sig_list = []
        for rr, rb in zip(recs_r, recs_b):
            pos_r = {int(g): k for k, g in enumerate(rr.idx)}
            pos_b = {int(g): k for k, g in enumerate(rb.idx)}
            ir = np.array([pos_r[int(g)] for g in common_all])
            ib = np.array([pos_b[int(g)] for g in common_all])
            p_r, l_r, r_sat, _, _ = corrected_observations(rr, service, world)
            p_b, l_b, _, _, _ = corrected_observations(rb, service, world)
            p_r, l_r, p_b, l_b = p_r[ir], l_r[ir], p_b[ib], l_b[ib]
            r_sat = r_sat[ir]
            sd_p = p_r - p_b
            sd_l = l_r - l_b
            ddp = sd_p[dd_rows] - sd_p[dd_ref]
            ddl = sd_l[dd_rows] - sd_l[dd_ref]
            sig_p = np.sqrt(rr.f1.sigma_code[ir][dd_rows] ** 2
                            + rb.f1.sigma_code[ib][dd_rows] ** 2
                            + rr.f1.sigma_code[ir][dd_ref] ** 2
                            + rb.f1.sigma_code[ib][dd_ref] ** 2 + 0.02 ** 2)
            sig_l = np.sqrt(rr.f1.sigma_phase[ir][dd_rows] ** 2
                            + rb.f1.sigma_phase[ib][dd_rows] ** 2
                            + rr.f1.sigma_phase[ir][dd_ref] ** 2
                            + rb.f1.sigma_phase[ib][dd_ref] ** 2 + 0.004 ** 2)
            los_hat, rho = _los_hat(r_sat, x_cur)
            rho_b = np.linalg.norm(r_sat - base_known_ecef, axis=-1)
            d_geom = (rho[dd_rows] - rho[dd_ref]) - (rho_b[dd_rows] - rho_b[dd_ref])
            a_pos = -(los_hat[dd_rows] - los_hat[dd_ref])
            a_ep = np.vstack([
                np.hstack([a_pos, np.zeros((n_dd, n_dd))]),
                np.hstack([a_pos, np.diag(dd_lam)]),
            ])
            y_ep = np.concatenate([ddp - d_geom, ddl - d_geom])
            w_ep = np.concatenate([1.0 / sig_p, 1.0 / sig_l])
            aw = a_ep * w_ep[:, None]
            ata += aw.T @ aw
            aty += aw.T @ (y_ep * w_ep)
            sig_list.append((sig_p, sig_l))
        return ata, aty, sig_list

    ata, aty, sig_list = build(x)
    try:
        q = np.linalg.inv(ata)
    except np.linalg.LinAlgError:
        return Solution(enu=np.full(3, np.nan), ok=False, note="法方程奇异")
    dx = q @ aty
    x_float = x + dx[:3]
    n_float = dx[3:]
    q_n = q[3:, 3:]
    sig_n = np.sqrt(np.abs(np.diag(q_n)))

    sol = Solution(enu=world.frame.ecef_to_enu(x_float), n_dd=n_dd,
                   n_sat=len(common_all), float_only=True,
                   sigma_amb=float(np.min(sig_n)) if n_dd else np.nan)
    if not fix_ambiguity:
        return sol

    n_hat, fixable, sig_cond = sequential_fix(n_float, q_n, sigma_thresh)
    sol.n_fixed = int(np.sum(fixable))
    sol.sigma_amb = float(np.max(sig_cond[fixable])) if sol.n_fixed else float(np.min(sig_cond))
    if sol.n_fixed < 4:
        sol.note = "模糊度未能固定"
        return sol

    # 固定模糊度后重解位置，并逐卫星用相位残差做外点剔除
    ep_data = []
    for (rr, rb), (sig_p, sig_l) in zip(zip(recs_r, recs_b), sig_list):
        pos_r = {int(g): k for k, g in enumerate(rr.idx)}
        pos_b = {int(g): k for k, g in enumerate(rb.idx)}
        ir = np.array([pos_r[int(g)] for g in common_all])
        ib = np.array([pos_b[int(g)] for g in common_all])
        p_r, l_r, r_sat, _, _ = corrected_observations(rr, service, world)
        p_b, l_b, _, _, _ = corrected_observations(rb, service, world)
        sd_p = p_r[ir] - p_b[ib]
        sd_l = l_r[ir] - l_b[ib]
        ddp = sd_p[dd_rows] - sd_p[dd_ref]
        ddl = sd_l[dd_rows] - sd_l[dd_ref]
        ep_data.append(dict(r_sat=r_sat[ir], ddp=ddp, ddl=ddl,
                            sig_p=sig_p, sig_l=sig_l))

    x_fixed = x_float
    rms, rmax = np.inf, np.inf
    for _ in range(3):
        rows_f = np.nonzero(fixable)[0]
        if len(rows_f) < 4:
            sol.note = "可用固定卫星不足"
            return sol
        ata2 = np.zeros((3, 3)); aty2 = np.zeros(3)
        for d in ep_data:
            los_hat, rho = _los_hat(d["r_sat"], x_fixed)
            rho_b = np.linalg.norm(d["r_sat"] - base_known_ecef, axis=-1)
            d_geom = (rho[dd_rows] - rho[dd_ref]) - (rho_b[dd_rows] - rho_b[dd_ref])
            a_pos = -(los_hat[dd_rows] - los_hat[dd_ref])
            y = np.concatenate([(d["ddp"] - d_geom)[rows_f],
                                (d["ddl"] - d_geom - n_hat * dd_lam)[rows_f]])
            w = np.concatenate([1.0 / d["sig_p"][rows_f],
                                1.0 / d["sig_l"][rows_f]])
            a_full = np.vstack([a_pos[rows_f], a_pos[rows_f]])
            aw = a_full * w[:, None]
            ata2 += aw.T @ aw
            aty2 += aw.T @ (y * w)
        try:
            dx2 = np.linalg.solve(ata2, aty2)
        except np.linalg.LinAlgError:
            sol.note = "固定后法方程奇异"
            return sol
        x_fixed = x_fixed + dx2
        # 逐卫星相位残差（跨历元取最大）
        res_by_row = np.zeros(n_dd)
        for d in ep_data:
            los_hat, rho = _los_hat(d["r_sat"], x_fixed)
            rho_b = np.linalg.norm(d["r_sat"] - base_known_ecef, axis=-1)
            d_geom = (rho[dd_rows] - rho[dd_ref]) - (rho_b[dd_rows] - rho_b[dd_ref])
            res_by_row = np.maximum(res_by_row,
                                    np.abs((d["ddl"] - d_geom - n_hat * dd_lam)))
        bad = fixable & (res_by_row > res_max)
        if not np.any(bad):
            break
        fixable = fixable & ~bad
    rows_f = np.nonzero(fixable)[0]
    rms = float(np.sqrt(np.mean(res_by_row[rows_f] ** 2)))
    rmax = float(np.max(res_by_row[rows_f]))
    if rms > res_thresh or rmax > res_max:
        sol.note = f"固定被拒绝(残差 RMS {rms*1000:.1f} mm / 最大 {rmax*1000:.1f} mm)"
        return sol
    return Solution(enu=world.frame.ecef_to_enu(x_fixed), n_dd=n_dd,
                    n_sat=len(common_all), fixed=True, n_fixed=int(np.sum(fixable)),
                    sigma_amb=sol.sigma_amb, res_rms=rms)


# ============================================================================
# 精密单点定位 (PPP) —— 静态多历元解
# ============================================================================
def iono_free(a1, a2, f1, f2):
    """消电离层组合系数。"""
    c1 = f1 ** 2 / (f1 ** 2 - f2 ** 2)
    c2 = -f2 ** 2 / (f1 ** 2 - f2 ** 2)
    return c1 * a1 + c2 * a2


def solve_ppp(records: list[Record], world, x0_enu, service=PRECISE,
              ztd_bins=2, sig_code_floor=0.3):
    """静态 PPP：多历元消电离层组合最小二乘，估计位置+天顶对流层+浮点模糊度。

    返回 (Solution, 收敛诊断字典)。
    """
    n_ep = len(records)
    if n_ep < 2:
        return Solution(enu=np.full(3, np.nan), ok=False, note="历元不足")

    # 卫星集合（取所有历元出现过的卫星）
    all_sats = sorted(set(int(g) for r in records for g in r.idx))
    sat_pos = {g: k for k, g in enumerate(all_sats)}
    n_sat = len(all_sats)
    # 参数布局: [位置(3)] [逐历元接收机钟差(n_ep)] [天顶对流层(ztd_bins)] [模糊度(n_sat)]
    i_clk = 3
    i_ztd = 3 + n_ep
    i_amb = i_ztd + ztd_bins
    n_param = i_amb + n_sat
    ztd_index = lambda i: min(int(i * ztd_bins / n_ep), ztd_bins - 1)

    rows_a, rows_y, rows_w = [], [], []
    x = world.frame.enu_to_ecef(np.asarray(x0_enu, dtype=float))
    los_cache = []
    # 模糊度初值必须"逐卫星固定"：模糊度参数跨历元共享，
    # 若初值随历元重算，共享参数将无法拟合相位观测。
    n0_by_sat = {}
    for r in records:
        r_sat = service.sat_pos(r, world)
        los = r_sat - x
        rho0 = np.linalg.norm(los, axis=-1)
        los_hat = los / rho0[:, None]
        f1 = r.freqs["f1"]; f2 = r.freqs["f2"]
        c1 = f1.freq ** 2 / (f1.freq ** 2 - f2.freq ** 2)
        c2 = -f2.freq ** 2 / (f1.freq ** 2 - f2.freq ** 2)
        l_if = c1 * f1.L + c2 * f2.L
        clk_sv = service.sat_clk(r, world)
        for k, g in enumerate(r.idx):
            g = int(g)
            if g not in n0_by_sat:
                n0_by_sat[g] = np.round(
                    (l_if[k] + clk_sv[k] - rho0[k]) / f1.lam[k])
        los_cache.append((r, r_sat, rho0, los_hat))

    for _ in range(4):
        rows_a, rows_y, rows_w = [], [], []
        for i_ep, (r, r_sat, rho0, los_hat) in enumerate(los_cache):
            f1 = r.freqs["f1"]; f2 = r.freqs["f2"]
            c1 = f1.freq ** 2 / (f1.freq ** 2 - f2.freq ** 2)
            c2 = -f2.freq ** 2 / (f1.freq ** 2 - f2.freq ** 2)
            p_if = c1 * f1.P + c2 * f2.P
            l_if = c1 * f1.L + c2 * f2.L
            mf = geo.elev_mapping(rec_az_el(r, world)[1])
            n = len(r.idx)
            # 模型改正（精密星历/钟差 + 相位中心）
            clk_sv = service.sat_clk(r, world)
            pcv = service.pcv_model(r, world)
            p_obs = p_if + clk_sv
            l_obs = l_if + clk_sv - pcv
            n0 = np.array([n0_by_sat[int(g)] for g in r.idx])
            sig_p = np.sqrt((c1 * f1.sigma_code) ** 2 + (c2 * f2.sigma_code) ** 2
                            + sig_code_floor ** 2)
            sig_l = np.sqrt((c1 * f1.sigma_phase) ** 2 + (c2 * f2.sigma_phase) ** 2
                            + (0.004) ** 2)

            a_p = np.zeros((n, n_param))
            a_l = np.zeros((n, n_param))
            a_p[:, :3] = -los_hat
            a_l[:, :3] = -los_hat
            a_p[:, i_clk + i_ep] = 1.0          # 接收机钟差（逐历元）
            a_l[:, i_clk + i_ep] = 1.0
            zb = ztd_index(i_ep)
            a_p[:, i_ztd + zb] = mf
            a_l[:, i_ztd + zb] = mf
            for k, g in enumerate(r.idx):
                a_l[k, i_amb + sat_pos[int(g)]] = f1.lam[k]
            rows_a.append(a_p); rows_a.append(a_l)
            rows_y.append(p_obs - rho0)
            rows_y.append(l_obs - rho0 - n0 * f1.lam)
            rows_w.append(1.0 / sig_p); rows_w.append(1.0 / sig_l)

        a_mat = np.vstack(rows_a)
        y = np.concatenate(rows_y)
        w = np.concatenate(rows_w)
        aw = a_mat * w[:, None]
        dx, *_ = np.linalg.lstsq(aw, y * w, rcond=None)
        x = x + dx[:3]
        # 更新几何
        los_cache = []
        for r in records:
            r_sat = service.sat_pos(r, world)
            los = r_sat - x
            rho0 = np.linalg.norm(los, axis=-1)
            los_cache.append((r, r_sat, rho0, los / rho0[:, None]))
        if np.linalg.norm(dx[:3]) < 1e-4:
            break

    try:
        q = np.linalg.inv(a_mat.T @ (a_mat * (w ** 2)[:, None]))
        q_enu = world.frame.rot @ q[:3, :3] @ world.frame.rot.T
        sigma0 = float(np.sqrt(np.sum((aw @ dx - y * w) ** 2)
                               / max(len(y) - n_param, 1)))
    except np.linalg.LinAlgError:
        q_enu = np.full((3, 3), np.nan); sigma0 = np.nan

    sol = Solution(enu=world.frame.ecef_to_enu(x), n_sat=n_sat,
                   n_dd=n_sat, res_rms=sigma0)
    diag = dict(ztd_bins=ztd_bins, n_param=n_param,
                sigma_enu=np.sqrt(np.diag(q_enu)) if np.all(np.isfinite(q_enu)) else None,
                ztd=dx[i_ztd:i_ztd + ztd_bins] if len(dx) > i_ztd else None,
                clk=dx[i_clk:i_ztd] if len(dx) > i_clk else None)
    return sol, diag
