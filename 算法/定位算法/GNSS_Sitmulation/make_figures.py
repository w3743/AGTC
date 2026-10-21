"""结果绘图：从 results/*.csv 生成报告用图。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# 若环境中没有 matplotlib，可使用工作区内的本地依赖目录 .pylibs
_LOCAL_LIBS = ROOT / ".pylibs"
if _LOCAL_LIBS.exists():
    sys.path.insert(0, str(_LOCAL_LIBS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib import font_manager

RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 中文字体
CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "Source Han Sans SC", "SimSun", "DengXian"]
_available = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((f for f in CJK_CANDIDATES if f in _available), None)
if CJK:
    matplotlib.rcParams["font.sans-serif"] = [CJK, "DejaVu Sans"]
else:
    print("警告: 未找到中文字体，图中中文可能显示为方框")
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 130
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.3

MODE_LABEL = {
    "spp_gps": "GPS 单点 (SPP)",
    "spp_multi": "多系统单点 (SPP)",
    "sbas": "星基增强 (SBAS)",
    "dgnss": "伪距差分 (DGNSS)",
    "rtk_inst": "RTK 单历元",
    "rtk": "RTK 静态 10 min",
    "ppp": "PPP 静态 30 min",
}
MODE_COLOR = {
    "spp_gps": "#8c564b", "spp_multi": "#d62728", "sbas": "#ff7f0e",
    "dgnss": "#2ca02c", "rtk_inst": "#9467bd", "rtk": "#1f77b4",
    "ppp": "#17becf",
}
ORDER = ["spp_gps", "spp_multi", "sbas", "dgnss", "rtk_inst", "rtk", "ppp"]


def load(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


# ===========================================================================
def fig_field_map():
    df = load("layout_map.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0))
    specs = [("spp_multi", 20.0, "多系统单点定位 (SPP)"),
             ("rtk", 2000.0, "静态 RTK")]
    for ax, (mode, scale, title) in zip(axes, specs):
        sub = df[df["mode"] == mode]
        ax.add_patch(Rectangle((-50, -50), 100, 100, fill=False, lw=1.6,
                               ec="#333333"))
        ax.plot(sub["tx"], sub["ty"], "o", ms=5, mfc="#1f77b4", mec="k",
                mew=0.5, label="探针真实位置", zorder=3)
        ax.quiver(sub["tx"], sub["ty"], sub["ee"] * scale, sub["en"] * scale,
                  angles="xy", scale_units="xy", scale=1, color="#d62728",
                  width=0.004, headwidth=4, zorder=2)
        ax.plot(sub["tx"] + sub["ee"] * scale, sub["ty"] + sub["en"] * scale,
                "x", color="#d62728", ms=4, label="定位结果", zorder=4)
        h = np.hypot(sub["ee"], sub["en"])
        ax.set_title(f"{title}（误差矢量放大 {scale:g} 倍）\n"
                     f"水平误差 RMS = {np.sqrt(np.mean(h**2)):.3f} m,"
                     f" 最大 = {h.max():.3f} m", fontsize=11)
        ax.set_xlabel("东向 (m)"); ax.set_ylabel("北向 (m)")
        ax.set_xlim(-58, 58); ax.set_ylim(-58, 58)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("100 m x 100 m 农田内 20 个土壤探针的定位结果（单历元）",
                 fontsize=13)
    fig.savefig(FIG / "fig_field_map.png"); plt.close(fig)
    print("  fig_field_map.png")


# ===========================================================================
def fig_cdf():
    df = load("samples_main.csv")
    if df is None:
        return
    case = sorted(df["case"].unique())[0]
    sub = df[(df["case"] == case)]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    for ax, kind, title in ((axes[0], "instant", "逐历元解（瞬时）"),
                            (axes[1], "static", "静态观测（时段平均 / 整周固定）")):
        for mode in ORDER:
            s = sub[(sub["kind"] == kind) & (sub["mode"] == mode)]
            if len(s) < 5:
                continue
            h = np.sort(np.hypot(s["ee"], s["en"]).to_numpy())
            y = np.arange(1, len(h) + 1) / len(h)
            ax.semilogx(np.maximum(h, 1e-4), y, lw=1.8,
                        color=MODE_COLOR.get(mode), label=MODE_LABEL.get(mode, mode))
        ax.set_xlabel("水平定位误差 (m)"); ax.set_ylabel("累积概率")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(1e-3, 30); ax.set_ylim(0, 1.02)
        ax.axhline(0.95, color="gray", ls="--", lw=0.8)
        ax.text(1.3e-3, 0.955, "95%", color="gray", fontsize=8)
        ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle(f"水平定位误差累积分布（{case}）", fontsize=13)
    fig.savefig(FIG / "fig_cdf.png"); plt.close(fig)
    print("  fig_cdf.png")


# ===========================================================================
def fig_summary_bar():
    df = load("summary_main.csv")
    if df is None:
        return
    inst = df[df["kind"] == "instant"].set_index(["case", "mode"])
    stat = df[df["kind"] == "static"].set_index(["case", "mode"])
    cases = sorted(df["case"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4),
                             gridspec_kw=dict(width_ratios=[1, 1]))
    w = 0.8 / len(cases)
    for ax, (src, title) in zip(axes, ((inst, "瞬时解（逐历元）"),
                                       (stat, "静态解（1h 平均 / 10min RTK / 30min PPP）"))):
        modes = [m for m in ORDER if (cases[0], m) in src.index]
        x = np.arange(len(modes))
        for k, c in enumerate(cases):
            vals, verr = [], []
            for m in modes:
                key = (c, m)
                vals.append(src.loc[key, "h_cep95"] if key in src.index else np.nan)
                verr.append(src.loc[key, "v_p95"] if key in src.index else np.nan)
            xpos = x + k * w - 0.4 + w / 2
            ax.bar(xpos, vals, w, label=c, color=plt.cm.tab10(k), alpha=0.85)
            # 垂直误差 95%：用同色竖线连到标记，便于对应
            for xp, hv, vv in zip(xpos, vals, verr):
                if np.isfinite(vv):
                    ax.plot([xp, xp], [hv, vv], "-", color=plt.cm.tab10(k),
                            lw=0.8, alpha=0.6, zorder=4)
            ax.plot(xpos, verr, "v", color=plt.cm.tab10(k), mec="k", mew=0.5,
                    ms=5, zorder=5,
                    label="垂直误差 95%" if k == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([MODE_LABEL.get(m, m) for m in modes], rotation=22,
                           ha="right", fontsize=8.5)
        ax.set_ylabel("误差 (m)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("各定位模式的水平误差 95% 分位（柱）与垂直误差 95% 分位（三角）",
                 fontsize=13)
    fig.savefig(FIG / "fig_summary_bar.png"); plt.close(fig)
    print("  fig_summary_bar.png")


# ===========================================================================
def fig_geometry():
    dop = load("geometry_dop.csv")
    npz_path = RES / "geometry_tracks.npz"
    if dop is None or not npz_path.exists():
        return
    z = np.load(npz_path)
    times = z["times"]
    fig = plt.figure(figsize=(13.2, 5.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    key = "G+C+E"
    az, el = z[f"{key}_az"], z[f"{key}_el"]
    colors = plt.cm.tab20(np.linspace(0, 1, az.shape[1]))
    for j in range(az.shape[1]):
        m = el[:, j] >= 5
        ax1.plot(np.radians(az[m, j]), 90 - el[m, j], lw=0.7, alpha=0.75,
                 color=colors[j % 20])
    ax1.set_theta_zero_location("N"); ax1.set_theta_direction(-1)
    ax1.set_rlim(0, 85); ax1.set_rticks([15, 30, 45, 60, 75])
    ax1.set_rlabel_position(135)
    ax1.set_title(f"24 h 卫星轨迹（{key}，仰角>5°）\n径向为天顶距", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2)
    t_h = (times - times[0]) / 3600.0
    for systems in ("G", "G+C", "G+C+E"):
        if f"{systems}_az" not in z:
            continue
        azs, els = z[f"{systems}_az"], z[f"{systems}_el"]
        hd = []
        for k in range(len(times)):
            m = els[k] >= 10
            if np.sum(m) < 4:
                hd.append(np.nan); continue
            e, n = np.radians(azs[k][m]), np.radians(els[k][m])
            hh = np.cos(n); a = np.column_stack([hh * np.sin(e), hh * np.cos(e),
                                                 np.sin(n), np.ones_like(hh)])
            try:
                q = np.linalg.inv(a.T @ a)
                hd.append(np.sqrt(q[0, 0] + q[1, 1]))
            except np.linalg.LinAlgError:
                hd.append(np.nan)
        ax2.plot(t_h, hd, lw=1.4, label=systems)
    ax2.set_xlabel("时间 (h, 起点 2025-09-15 14:00 北京时间)")
    ax2.set_ylabel("HDOP")
    ax2.set_title("HDOP 24 h 变化（仰角掩蔽 10°）", fontsize=11)
    ax2.legend(fontsize=9)
    fig.savefig(FIG / "fig_geometry.png"); plt.close(fig)
    print("  fig_geometry.png")


# ===========================================================================
def fig_error_budget():
    df = load("summary_error_budget.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
    for ax, kind, title in ((axes[0], "instant", "瞬时解"),
                            (axes[1], "static", "静态解")):
        sub = df[df["kind"] == kind]
        cases = list(dict.fromkeys(sub["case"]))
        modes = [m for m in ("spp_multi", "sbas", "dgnss", "rtk", "ppp")
                 if m in set(sub["mode"])]
        y = np.arange(len(cases))
        w = 0.8 / max(len(modes), 1)
        for k, m in enumerate(modes):
            s = sub[sub["mode"] == m].set_index("case")
            vals = [s.loc[c, "h_cep95"] if c in s.index else np.nan for c in cases]
            ax.barh(y + k * w - 0.4 + w / 2, vals, w,
                    color=MODE_COLOR.get(m), alpha=0.88,
                    label=MODE_LABEL.get(m, m))
        ax.set_yticks(y); ax.set_yticklabels(cases, fontsize=8.5)
        ax.set_xscale("log"); ax.set_xlabel("水平误差 95% 分位 (m)")
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("误差预算：逐项关闭误差源后的定位精度", fontsize=13)
    fig.savefig(FIG / "fig_error_budget.png"); plt.close(fig)
    print("  fig_error_budget.png")


# ===========================================================================
def fig_occupation():
    df = load("occupation_time_summary.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for mode, ax_col, label in (("ppp", 0, "PPP（消电离层双频静态解）"),
                                ("spp_multi(时段平均)", 1, "多系统 SPP（时段平均）")):
        s = df[df["mode"] == mode].sort_values("window_min")
        if s.empty:
            continue
        ax = axes[ax_col]
        ax.loglog(s["window_min"], s["h_rms"] * 100, "o-", color="#d62728",
                  label="水平 RMS")
        ax.loglog(s["window_min"], s["h_p95"] * 100, "s--", color="#ff7f0e",
                  label="水平 95%")
        ax.loglog(s["window_min"], s["v_rms"] * 100, "^-", color="#1f77b4",
                  label="垂直 RMS")
        if mode.startswith("spp"):
            ax.set_xscale("log")
            ax.set_yscale("linear")
            ax.set_xlim(0.9, 140)
        ax.set_xlabel("观测时长 (min)"); ax.set_ylabel("误差 (cm)")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(which="both", alpha=0.3)
    fig.suptitle("静态观测时长对定位精度的影响", fontsize=13)
    fig.savefig(FIG / "fig_occupation.png"); plt.close(fig)
    print("  fig_occupation.png")


# ===========================================================================
def fig_sensitivity():
    df = load("summary_sensitivity.csv")
    if df is None:
        return
    # 仅用静态解：RTK/PPP 只有静态结果，且静态解统计更稳健
    df = df[df["kind"] == "static"].copy()
    groups = {
        "土壤含水率": "土壤含水率",
        "电离层": "电离层",
        "天线": "天线安装",
        "安装": "天线安装",
        "星座": "星座组合",
        "低矮作物": "地表覆盖",
        "地表粗糙度": "地表覆盖",
    }

    def gname(case):
        for k, v in groups.items():
            if case.startswith(k):
                return v
        return None

    df["group"] = df["case"].map(gname)
    df = df[df["group"].notna()]
    gnames = list(dict.fromkeys(df["group"]))
    fig, axes = plt.subplots(1, len(gnames), figsize=(4.0 * len(gnames), 4.6),
                             squeeze=False)
    modes = ["spp_multi", "sbas", "dgnss", "rtk", "ppp"]
    for ax, g in zip(axes[0], gnames):
        sub = df[df["group"] == g]
        cases = list(dict.fromkeys(sub["case"]))
        x = np.arange(len(cases))
        w = 0.8 / len(modes)
        for k, m in enumerate(modes):
            s = sub[sub["mode"] == m].set_index("case")
            vals = [s.loc[c, "h_cep95"] if c in s.index else np.nan for c in cases]
            ax.bar(x + k * w - 0.4 + w / 2, vals, w, color=MODE_COLOR[m],
                   alpha=0.9, label=MODE_LABEL[m])
        ax.set_xticks(x)
        ax.set_xticklabels([c.split(":")[-1].strip() for c in cases],
                           rotation=18, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 20)
        ax.set_title(g, fontsize=11)
        ax.set_ylabel("水平误差 95% (m)")
        ax.axhline(0.01, color="gray", ls=":", lw=0.8)
        ax.text(len(cases) - 0.5, 0.011, "1 cm", fontsize=7, color="gray",
                ha="right")
    axes[0][0].legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle("单因素敏感性：静态解水平误差 95% 分位（对数坐标，单位统一为 m）",
                 fontsize=12)
    fig.savefig(FIG / "fig_sensitivity.png"); plt.close(fig)
    print("  fig_sensitivity.png")


# ===========================================================================
def fig_multipath():
    df = load("multipath_vs_soil.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))
    ants = {"lowcost": "低成本微带（探针一体化）", "patch_mast": "短杆微带天线",
            "geodetic": "测量型扼流圈（基准站级）"}
    for ant, ax in zip(ants, axes):
        sub = df[df["antenna"] == ant]
        for theta in sorted(sub["moisture"].unique()):
            s = sub[sub["moisture"] == theta].sort_values("el")
            ax.plot(s["el"], s["phase_mp_mm"], lw=1.6,
                    label=f"含水率 {theta:.2f} (εr={s['permittivity'].iloc[0]:.1f})")
        ax.set_xlabel("卫星仰角 (°)")
        ax.set_ylabel("相干载波多路径误差 (mm)")
        ax.set_title(ants[ant], fontsize=11)
        ax.legend(fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
    fig.suptitle("地表反射引起的载波相位多路径（天线高 0.2 m，相干双射线模型）",
                 fontsize=13)
    fig.savefig(FIG / "fig_multipath.png"); plt.close(fig)
    print("  fig_multipath.png")


# ===========================================================================
def fig_zone():
    df = load("summary_main.csv")
    if df is None:
        return
    cols = [c for c in df.columns if c.startswith("zone_err_")]
    if not cols:
        return
    sizes = [float(c.replace("zone_err_", "").replace("m", "")) for c in cols]
    zcol = "zone_err_6m" if "zone_err_6m" in cols else cols[0]
    zsize = 6.0 if zcol == "zone_err_6m" else sizes[0]
    stat = df[df["kind"] == "static"]
    cases = sorted(stat["case"].unique())
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    modes = [m for m in ORDER if m in set(stat["mode"])]
    x = np.arange(len(modes))
    w = 0.8 / len(cases)
    for k, c in enumerate(cases):
        s = stat[stat["case"] == c].set_index("mode")
        vals = [s.loc[m, zcol] * 100 if m in s.index else np.nan for m in modes]
        ax.bar(x + k * w - 0.4 + w / 2, vals, w, label=c, color=plt.cm.tab10(k),
               alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABEL.get(m, m) for m in modes], rotation=20,
                       ha="right", fontsize=9)
    ax.set_ylabel("分区归属错误率 (%)")
    ax.set_title(f"按 {zsize:g} m x {zsize:g} m 管理分区归属的错误率\n"
                 "（定位误差使探针被划入错误分区）", fontsize=12)
    ax.legend(fontsize=9)
    fig.savefig(FIG / "fig_zone.png"); plt.close(fig)
    print("  fig_zone.png")


def main():
    print("生成图表 ->", FIG)
    for fn in (fig_field_map, fig_cdf, fig_summary_bar, fig_geometry,
               fig_error_budget, fig_occupation, fig_sensitivity,
               fig_multipath, fig_zone):
        try:
            fn()
        except Exception as exc:      # noqa: BLE001
            print(f"  !! {fn.__name__} 失败: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
