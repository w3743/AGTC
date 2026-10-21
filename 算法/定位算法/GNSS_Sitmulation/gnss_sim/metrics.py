"""精度指标：CEP/RMS/95 分位、定位误差统计与农艺可用性评价。"""
from __future__ import annotations

import numpy as np


def error_stats(err_enu: np.ndarray, prefix="") -> dict:
    """由 ENU 误差样本 (N,3) 计算完整精度指标。

    返回字典包含水平/垂直/三维误差的均值、RMS、50/95/99 分位与最大值。
    """
    e = np.asarray(err_enu, dtype=float)
    e = e[np.all(np.isfinite(e), axis=1)]
    if len(e) == 0:
        return {f"{prefix}N": 0}
    h = np.hypot(e[:, 0], e[:, 1])
    v = np.abs(e[:, 2])
    d3 = np.linalg.norm(e, axis=1)
    out = {
        f"{prefix}n": len(e),
        f"{prefix}h_mean": float(np.mean(h)),
        f"{prefix}h_rms": float(np.sqrt(np.mean(h ** 2))),
        f"{prefix}h_cep50": float(np.percentile(h, 50)),
        f"{prefix}h_cep95": float(np.percentile(h, 95)),
        f"{prefix}h_p99": float(np.percentile(h, 99)),
        f"{prefix}h_max": float(np.max(h)),
        f"{prefix}v_mean": float(np.mean(v)),
        f"{prefix}v_rms": float(np.sqrt(np.mean(v ** 2))),
        f"{prefix}v_p95": float(np.percentile(v, 95)),
        f"{prefix}v_max": float(np.max(v)),
        f"{prefix}d3_rms": float(np.sqrt(np.mean(d3 ** 2))),
        f"{prefix}d3_p95": float(np.percentile(d3, 95)),
        # 分方向偏差（系统性偏差）
        f"{prefix}bias_e": float(np.mean(e[:, 0])),
        f"{prefix}bias_n": float(np.mean(e[:, 1])),
        f"{prefix}bias_u": float(np.mean(e[:, 2])),
    }
    return out


def zone_assignment_rate(true_xy: np.ndarray, est_xy: np.ndarray,
                         zone_size: float) -> float:
    """管理分区归属错误率。

    真实位置与管理网格：估计位置落入错误网格即判为错误。
    true_xy / est_xy 形状 (N,2)。
    """
    t = np.asarray(true_xy, dtype=float)
    e = np.asarray(est_xy, dtype=float)
    ok = np.all(np.isfinite(e), axis=1) & np.all(np.isfinite(t), axis=1)
    if not np.any(ok):
        return float("nan")
    t = t[ok]; e = e[ok]
    zx_t = np.floor(t[:, 0] / zone_size).astype(int)
    zy_t = np.floor(t[:, 1] / zone_size).astype(int)
    zx_e = np.floor(e[:, 0] / zone_size).astype(int)
    zy_e = np.floor(e[:, 1] / zone_size).astype(int)
    wrong = (zx_t != zx_e) | (zy_t != zy_e)
    return float(np.mean(wrong))


def relative_accuracy(err_enu: np.ndarray, positions_xy: np.ndarray) -> dict:
    """探针之间的相对定位精度（两两差分误差）。

    对同一历元的探针位置两两作差，估计差与真值差之差即为相对误差。
    """
    e = np.asarray(err_enu, dtype=float)
    n = len(e)
    if n < 2:
        return {}
    i, j = np.triu_indices(n, k=1)
    d = e[i, :2] - e[j, :2]
    r = np.hypot(d[:, 0], d[:, 1])
    return {"rel_n": len(r), "rel_rms": float(np.sqrt(np.mean(r ** 2))),
            "rel_p95": float(np.percentile(r, 95)),
            "rel_median": float(np.median(r))}


def summarize_by_time(samples: np.ndarray) -> np.ndarray:
    """把 (N_probe, N_epoch, 3) 的误差按探针取时间平均，返回 (N_probe, 3)。"""
    s = np.asarray(samples, dtype=float)
    return np.nanmean(s, axis=1)
