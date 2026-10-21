"""仿真实验驱动：场景生成、多模式解算与统计汇总。"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from . import constellation as con
from . import estimators as est
from . import metrics as mt
from . import models as md
from .config import (MAIN_MODES, ProbeSpec, ScenarioConfig)
from .observation import WorldModel

# 观测历元计划（会话默认 1 h / 30 s，共 121 个历元）
INSTANT_EPOCHS = tuple(range(0, 121, 10))     # 13 个瞬时采样，覆盖 1 h
RTK_EPOCHS = tuple(range(0, 21))              # 0~10 min 静态 RTK
PPP_EPOCHS = tuple(range(0, 61))              # 0~30 min 静态 PPP

INSTANT_MODES = ("spp_gps", "spp_multi", "sbas", "dgnss")


# ============================================================================
# 场景生成
# ============================================================================
def sample_probe_positions(scn: ScenarioConfig, rng) -> np.ndarray:
    """在田块内随机布设探针位置（带最小间距约束）。"""
    half = scn.field.half
    margin = 2.0
    pts: list[np.ndarray] = []
    tries = 0
    while len(pts) < scn.n_probes and tries < 50000:
        tries += 1
        p = rng.uniform(-half + margin, half - margin, 2)
        if all(np.hypot(*(p - q)) >= scn.min_spacing_m for q in pts):
            pts.append(p)
    return np.array(pts)


def make_probe_set(world, scn: ScenarioConfig, rng):
    """按探针规格生成一组接收机（位置、高度、天线朝向均随机）。"""
    positions = sample_probe_positions(scn, rng)
    spec: ProbeSpec = scn.probe
    ant_spec = md.ANTENNA_LIBRARY[spec.antenna]
    probes = []
    for i, xy in enumerate(positions):
        h = spec.resolved_height(rng)
        antenna = md.AntennaModel(ant_spec, rng)
        mp = md.MultipathModel(scn.soil, antenna, rng,
                               canopy_h=scn.soil.canopy_height_m,
                               canopy_atten_db=scn.soil.canopy_atten_db)
        enu = np.array([xy[0], xy[1], h])
        rx = world.make_probe(i, enu, spec, antenna, mp, h)
        probes.append((rx, enu))
    return probes


def make_base_known(world, scn: ScenarioConfig, rng):
    """构造基准站及其"已知坐标"。

    coordinate_source='spp' 时按单点定位精度给出带误差的坐标（现实中最常见），
    'known' 表示已用静态观测精确标定。
    """
    cfg = scn.base
    base = world.make_base(cfg)
    true_ecef = base.ecef.copy()
    if cfg.coordinate_source == "spp":
        known = true_ecef + rng.normal(0, cfg.spp_error_m, 3)
    elif cfg.coordinate_source == "cors":
        known = true_ecef + rng.normal(0, 0.02, 3)
    else:
        known = true_ecef.copy()
    return base, known


# ============================================================================
# 单探针解算
# ============================================================================
def _solve_instant(rec, world, appr, modes, base_rec, base_known):
    out = {}
    if "spp_gps" in modes:
        out["spp_gps"] = est.solve_spp(rec, est.BROADCAST, world, appr,
                                       systems=("G",))
    if "spp_multi" in modes:
        out["spp_multi"] = est.solve_spp(rec, est.BROADCAST, world, appr)
    if "sbas" in modes:
        out["sbas"] = est.solve_spp(rec, est.SBAS, world, appr)
    if "dgnss" in modes and base_rec is not None:
        out["dgnss"] = est.solve_dgnss(rec, base_rec, est.BROADCAST, world,
                                       base_known, appr)
    if "rtk_inst" in modes and base_rec is not None:
        out["rtk_inst"] = est.solve_rtk(rec, base_rec, est.BROADCAST, world,
                                        base_known, appr)
    return out


def run_one_probe(world, scn, rx, enu_true, base_inst, base_rtk, base_known,
                  modes, rng, want_ppp=True, want_rtk_inst=True):
    """对单个探针执行全部模式的解算，返回样本列表。"""
    samples = []
    appr = enu_true + rng.normal(0, 6.0, 3)
    tx, ty = float(enu_true[0]), float(enu_true[1])

    def add(mode, kind, epoch, err, fixed=0, n_sat=0, note=""):
        samples.append(dict(mode=mode, kind=kind, epoch=epoch,
                            ee=float(err[0]), en=float(err[1]), eu=float(err[2]),
                            fixed=int(fixed), n_sat=int(n_sat), note=note,
                            tx=tx, ty=ty))

    # --- 逐历元瞬时解 ---------------------------------------------------
    instant_errors = {m: [] for m in modes if m in INSTANT_MODES}
    for k, i in enumerate(INSTANT_EPOCHS):
        rec = world.build_record(rx, i, dual=False)
        use = set(modes) & set(INSTANT_MODES)
        if want_rtk_inst:
            use.add("rtk_inst")
        sols = _solve_instant(rec, world, appr, use, base_inst[k], base_known)
        for m, s in sols.items():
            if not s.ok or not np.all(np.isfinite(s.enu)):
                continue
            e = s.enu - enu_true
            add(m, "instant", i, e, s.fixed, s.n_sat, s.note)
            if m in instant_errors:
                instant_errors[m].append(e)

    # --- 时段平均（静止观测的降噪效果） ---------------------------------
    for m, errs in instant_errors.items():
        if len(errs) >= 3:
            add(m, "static", -1, np.mean(np.array(errs), axis=0),
                n_sat=len(errs), note="1h平均")

    # --- 静态 RTK（10 min） --------------------------------------------
    if "rtk" in modes and base_rtk is not None:
        recs_r = [world.build_record(rx, i, dual=False) for i in RTK_EPOCHS]
        sol = est.solve_rtk_static(recs_r, base_rtk, est.BROADCAST, world,
                                   base_known, appr)
        if sol.ok and np.all(np.isfinite(sol.enu)):
            add("rtk", "static", -1, sol.enu - enu_true, sol.fixed, sol.n_sat,
                sol.note)

    # --- PPP（30 min 静态） --------------------------------------------
    if "ppp" in modes and want_ppp:
        recs = [world.build_record(rx, i, dual=True) for i in PPP_EPOCHS]
        sol, _ = est.solve_ppp(recs, world, appr, ztd_bins=2)
        if sol.ok and np.all(np.isfinite(sol.enu)):
            add("ppp", "static", -1, sol.enu - enu_true, 0, sol.n_sat, sol.note)
    return samples


# ============================================================================
# 一个场景配置下的蒙特卡洛实验
# ============================================================================
def run_case(scn: ScenarioConfig, label: str, n_layouts=40, n_probes=None,
             seed=0, modes=MAIN_MODES, ppp_every=4, verbose=True):
    """每个 layout 重新抽取探针位置/安装、大气状态、星历钟差误差与多径实现。"""
    if n_probes is not None:
        scn = replace(scn, n_probes=n_probes)
    const = con.build_constellation(scn.const.systems, scn.session.t_start_unix)
    t0 = time.time()
    samples: list[dict] = []

    for li in range(n_layouts):
        seed_i = scn.seed + 1000 * seed + li
        rng = np.random.default_rng(seed_i)
        world = WorldModel(scn, const, seed=seed_i, dual_freq=False)
        world.prepare_session()
        probes = make_probe_set(world, scn, rng)
        base, base_known = make_base_known(world, scn, rng)
        with_base = any(m in modes for m in ("dgnss", "rtk", "rtk_inst"))
        # 基准站观测逐历元只需算一次，供全部探针复用
        base_inst, base_rtk = None, None
        if with_base:
            if any(m in modes for m in ("dgnss", "rtk_inst")):
                base_inst = [world.build_record(base, i, dual=False)
                             for i in INSTANT_EPOCHS]
            if "rtk" in modes:
                base_rtk = [world.build_record(base, i, dual=False)
                            for i in RTK_EPOCHS]
        for pi, (rx, enu) in enumerate(probes):
            rows = run_one_probe(world, scn, rx, enu, base_inst, base_rtk,
                                 base_known, modes, rng,
                                 want_ppp=("ppp" in modes and pi % ppp_every == 0),
                                 want_rtk_inst=("rtk_inst" in modes))
            for r in rows:
                r.update(case=label, layout=li, probe=pi)
                samples.append(r)
        if verbose and (li + 1) % 10 == 0:
            print(f"    [{label}] layout {li+1}/{n_layouts} "
                  f"({time.time()-t0:.1f}s, {len(samples)} 样本)")
    if verbose:
        print(f"    [{label}] 完成: {len(samples)} 样本, 用时 {time.time()-t0:.1f}s")
    return dict(case=label, samples=samples, config=describe(scn),
                n_layouts=n_layouts)


def describe(scn: ScenarioConfig) -> dict:
    return dict(
        n_probes=scn.n_probes, systems="+".join(scn.const.systems),
        soil_moisture=scn.soil.moisture, roughness=scn.soil.roughness_m,
        canopy_h=scn.soil.canopy_height_m, sfu=scn.iono.sfu,
        storm=int(scn.iono.storm), antenna=scn.probe.antenna,
        mount=scn.probe.mount, base_kind=scn.base.kind,
        base_coord=scn.base.coordinate_source,
        base_dist_km=scn.base.distance_km,
        disable=",".join(scn.disable), seed=scn.seed)


# ============================================================================
# 汇总统计
# ============================================================================
def summarize(samples: list[dict], case: str, zone_sizes=(3.0, 6.0, 12.0)):
    """按 (kind, mode) 汇总精度指标。"""
    rows = []
    for kind in ("instant", "static"):
        modes = sorted({s["mode"] for s in samples if s["kind"] == kind})
        for m in modes:
            sub = [s for s in samples if s["kind"] == kind and s["mode"] == m]
            if not sub:
                continue
            e = np.array([[s["ee"], s["en"], s["eu"]] for s in sub])
            st = mt.error_stats(e)
            st.update(case=case, mode=m, kind=kind)
            if m.startswith("rtk"):
                fixed = np.array([s["fixed"] for s in sub], dtype=bool)
                h = np.hypot(e[:, 0], e[:, 1])
                st["fix_rate"] = float(np.mean(fixed))
                st["fix_wrong_rate"] = (float(np.mean(h[fixed] > 0.30))
                                        if np.any(fixed) else np.nan)
                st["fix_h_rms"] = (float(np.sqrt(np.mean(h[fixed] ** 2)))
                                   if np.any(fixed) else np.nan)
            # 管理分区归属错误率
            tru = np.column_stack([[s["tx"] for s in sub], [s["ty"] for s in sub]])
            est_xy = tru + e[:, :2]
            for zs in zone_sizes:
                st[f"zone_err_{zs:g}m"] = mt.zone_assignment_rate(tru, est_xy, zs)
            rows.append(st)
    return rows


def relative_accuracy_by_epoch(samples: list[dict], case: str, mode: str,
                               kind="instant"):
    """同一历元内探针两两相对定位精度。"""
    sub = [s for s in samples
           if s["case"] == case and s["mode"] == mode and s["kind"] == kind]
    groups = {}
    for s in sub:
        groups.setdefault((s["layout"], s["epoch"]), []).append(s)
    rel = []
    for (li, ep), rows in groups.items():
        if len(rows) < 4:
            continue
        e = np.array([[r["ee"], r["en"]] for r in rows])
        i, j = np.triu_indices(len(e), k=1)
        d = e[i] - e[j]
        rel.append(np.hypot(d[:, 0], d[:, 1]))
    if not rel:
        return {}
    allr = np.concatenate(rel)
    return dict(case=case, mode=mode, rel_n=len(allr),
                rel_rms=float(np.sqrt(np.mean(allr ** 2))),
                rel_median=float(np.median(allr)),
                rel_p95=float(np.percentile(allr, 95)))
