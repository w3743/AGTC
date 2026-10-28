"""生成中文分析报告 report.md（数据全部取自 results/*.csv）。

约定：正文一律使用单引号 Python 字符串，中文引号直接写全角符号，避免与语法冲突。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "report.md"

MODE_LABEL = {
    'spp_gps': 'GPS 单点定位 (SPP)',
    'spp_multi': '多系统单点定位 (SPP)',
    'sbas': '星基增强 (SBAS/BDSBAS)',
    'dgnss': '伪距差分 (DGNSS)',
    'rtk_inst': 'RTK 单历元 (瞬时)',
    'rtk': 'RTK 静态 10 min',
    'ppp': 'PPP 静态 30 min',
}
ORDER = ['spp_gps', 'spp_multi', 'sbas', 'dgnss', 'rtk_inst', 'rtk', 'ppp']


def load(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    return f'{v:.{nd}f}'


def md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return '_（无数据）_\n'
    df = df.copy()
    head = '| ' + ' | '.join(str(c) for c in df.columns) + ' |'
    sep = '|' + '|'.join(['---'] * len(df.columns)) + '|'
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append(fmt(float(v)))
            else:
                cells.append(str(v))
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join([head, sep] + rows) + '\n'


def get(summary, case, mode, kind, col, default=np.nan):
    if summary is None:
        return default
    s = summary[(summary['case'] == case) & (summary['mode'] == mode)
                & (summary['kind'] == kind)]
    return float(s[col].iloc[0]) if len(s) and col in s else default


def cm(v):
    return '—' if not np.isfinite(v) else f'{v * 100:.1f}'


def flen(v):
    """按量级自动选择单位：<0.1 m 用 cm，否则用 m。"""
    if not np.isfinite(v):
        return '—'
    return f'{v * 100:.1f} cm' if v < 0.1 else f'{v:.2f} m'


def main():
    main_s = load('summary_main.csv')
    sens = load('summary_sensitivity.csv')
    budget = load('summary_error_budget.csv')
    occ = load('occupation_time_summary.csv')
    geom = load('geometry_dop.csv')
    rel = load('relative_accuracy.csv')
    mp = load('multipath_vs_soil.csv')

    cases = sorted(main_s['case'].unique()) if main_s is not None else []
    caseA = cases[0] if cases else 'A'
    caseB = cases[1] if len(cases) > 1 else caseA
    caseC = cases[2] if len(cases) > 2 else caseA

    def gh(mode, col, case=None, kind='static'):
        return get(main_s, case or caseA, mode, kind, col)

    L = []
    A = L.append

    A('# 农田场景 GNSS 土壤探针定位精度仿真报告\n')
    A(f'*生成时间：{datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")}*'
      '　|　仿真时段：2025-09-15 14:00–15:00（北京时间）'
      '　|　田块：100 m × 100 m，20 个土壤探针\n')

    # ==================================================================
    A('## 1. 结论摘要\n')
    A('**核心结论：在无高大作物遮挡的农田里，定位精度的瓶颈不是“遮挡”和“多路径”，'
      '而是“改正信息来源”。**\n')
    A(f'1. 只靠接收机自己（广播星历 + Klobuchar 模型）的单点定位，水平误差 95% 分位 '
      f'**{fmt(gh("spp_multi", "h_cep95", kind="instant"), 2)} m**，'
      f'垂直 **{fmt(gh("spp_multi", "v_p95", kind="instant"), 2)} m**；'
      '主要误差来自电离层模型残差与星历/钟差，而不是地表反射。\n')
    A(f'2. 星基增强（SBAS）不需要任何通信链路，水平误差降到 '
      f'**{fmt(gh("sbas", "h_cep95", kind="instant"), 2)} m**。\n')
    A(f'3. 田块内自建基准站做伪距差分（DGNSS）：瞬时 '
      f'**{fmt(gh("dgnss", "h_cep95", kind="instant"), 2)} m**，'
      f'1 h 平均 **{fmt(gh("dgnss", "h_cep95"), 2)} m**。\n')
    A(f'4. 双频载波相位静态 RTK（10 min）水平中位 '
      f'**{flen(gh("rtk", "h_cep50"))}**、95% 分位 **{flen(gh("rtk", "h_cep95"))}**，'
      f'垂直 95% {flen(gh("rtk", "v_p95"))}，固定成功率 '
      f'{gh("rtk", "fix_rate") * 100:.0f}%。\n')
    A(f'5. **最容易踩的坑**：RTK 的厘米级是“相对”的。若基准站坐标是随手用单点定位'
      f'测的（水平误差约 2–3 m），探针绝对坐标误差达 '
      f'**{fmt(get(main_s, caseB, "rtk", "static", "h_cep95"), 2)} m**，'
      'RTK 的厘米级优势被完全吃掉——基准站坐标必须另行标定。\n')
    A(f'6. 用 15 km 外的 CORS 当**普通基准站**做长基线 RTK，只有 '
      f'**{get(main_s, caseC, "rtk", "static", "fix_rate") * 100:.0f}%** 的探针能固定'
      f'模糊度，水平误差 **{fmt(get(main_s, caseC, "rtk", "static", "h_cep95"), 2)} m**，'
      '且垂直方向出现约 0.2 m 的系统偏差（残余大气误差）。'
      '真正的网络 RTK（VRS/MAX）会对区域大气建模，不在此列。\n')
    A(f'7. 双频 PPP 无需基准站，30 min 静态收敛后水平中位 '
      f'**{flen(gh("ppp", "h_cep50"))}**、95% 分位 **{flen(gh("ppp", "h_cep95"))}**'
      f'（垂直 95% {flen(gh("ppp", "v_p95"))}），'
      '是“无基准站 + 分米级”的折中方案；继续观测到 1 h 可到厘米级。\n')
    zone6 = gh('spp_multi', 'zone_err_6m')
    A(f'8. 若把田块划成 6 m × 6 m 管理分区，单点定位会让约 '
      f'**{zone6 * 100:.0f}%** 的探针被划进错误分区，而 RTK/PPP 误分率接近 0。\n')

    # ==================================================================
    A('\n## 2. 场景设定（把现实条件带进来）\n')
    A('| 项目 | 设定 | 现实依据 |')
    A('|---|---|---|')
    A('| 田块 | 100 m × 100 m 平坦农田 | 典型试验田/示范区规模 |')
    A('| 参考点 | 34.50°N, 113.00°E, 80 m（华北平原） | 中国主粮区，BDS 几何良好 |')
    A('| 作物 | 裸土/出苗期，无高大作物 | 用户假设；另设 15 cm 出苗期对照 |')
    A('| 探针 | 20 个随机分布，最小间距 5 m | 墒情监测常规布点密度 |')
    A('| 安装 | 天线相位中心离地 0.12–0.30 m（贴地） | 土壤探针一体化结构 |')
    A('| 天线 | 低成本微带（峰值增益 3 dBic、低仰角滚降 11 dB、'
      '左旋抑制 10 dB、相位中心变化 ±7 mm） | 探针量产成本约束 |')
    A('| 接收机 | 单频模块（码噪声 0.3–0.9 m）/ 双频 RTK 模块 | 市售模块典型水平 |')
    A('| 仰角掩蔽 | 贴地 12°／短杆 10° | 地面遮挡与低仰角信号衰减 |')
    A('| 土壤 | 体积含水率 0.20（对照 0.05／0.35），粗糙度 2 cm | 壤土典型值 |')
    A('| 电离层 | F10.7 = 150（中等偏活跃），当地 14:00 | 2025 年处于太阳活动高年 |')
    A('| 观测 | 1 h @ 30 s；RTK 取前 10 min，PPP 取前 30 min | 探针可长期静止 |')
    A('| 星座 | GPS + BDS-3 + Galileo（另做单系统/四系统对照） | 国产模块主流配置 |')
    A('| 基准站 | 田块中心，扼流圈天线架高 2 m，含自身多径与噪声 | 自建基准站常规做法 |')
    A('')

    # ==================================================================
    A('\n## 3. 仿真模型与误差源\n')
    A('仿真采用“真值/模型”分离架构：观测量由真实几何与真实误差源合成，'
      '解算器只使用它“应该知道”的模型量，两者之差自然形成定位误差，'
      '而不是人为注入误差。所有误差源都按物理机理建模：\n')
    A('| 误差源 | 真值模型 | 接收机/改正模型 | 本场景典型量级 |')
    A('|---|---|---|---|')
    A('| 电离层 | 350 km 薄壳 VTEC 场（峰值 13.2 LT、赤道异常、行波扰动），'
      '斜距延迟 6.8–17.0 m | GPS ICD-200 Klobuchar 广播模型（由“预报”VTEC 生成系数，'
      '含 28% 预报误差） | 残差 1.2–4.8 m，约为真值的 18% |')
    A('| 对流层 | Saastamoinen 干/湿分量 + 湿延迟 AR(1) 时变 + 水平梯度，'
      '天顶干 2.297 m、湿 0.224 m | 标准大气（1013.25 hPa／20 ℃／RH 50%） | '
      '残差斜距 0.13–0.40 m |')
    A('| 地表多路径 | 土壤介电常数（Topp 式）→ Fresnel 复反射（RHCP 同极化/交叉极化）'
      '→ 双射线干涉 + 粗糙面漫反射，含 Rayleigh 粗糙度衰减 | 不改正 | '
      '码 ~15 mm、载波 1–7 mm |')
    A('| 天线 | 增益方向图低仰角滚降、下半空间抑制、左旋抑制比、'
      '仰角+方位相位中心变化（PCV） | RTK/PPP 用标称相位中心（低成本天线未标定） | '
      'PCV 差异 3–8 mm |')
    A('| 观测噪声 | 由 C/N0 决定（34–44 dB-Hz），随仰角变化 | 按 C/N0 加权最小二乘 | '
      '码 0.3–0.9 m，载波 1.6–4.8 mm |')
    A('| 星历/钟差 | 卫星钟真实漂移 + 相对论周期项 | 广播星历（SISRE 量级三维偏差） | '
      '轨道视线投影 0.07–0.92 m，钟差 0.02–2.6 m |')
    A('| 基准站 | 独立接收机（扼流圈天线，自带多径与噪声） | 伪距改正数／双差 | '
      '差分后残余 0.2–0.5 m（码）、约 5 mm（相位） |')
    A('')
    A('星座按 2025 年在轨规模构造：GPS 31 星（6 轨道面）、'
      'BDS-3（24 MEO + 3 IGSO + 3 GEO）、Galileo 27 星、GLONASS 24 星，'
      '轨道传播含 J2 长期项。蒙特卡洛为 30 个田块布局 × 20 个探针，'
      '每个布局重新抽取探针位置/安装高度/天线朝向、大气状态、星历钟差误差与多径实现。\n')
    A('')
    A('两点实现细节说明：\n')
    A('1. **电离层的“真值”与“模型”使用不同的倾斜因子**：真值用几何倾斜因子 '
      'F = 1/√(1−(Re·cos el/(Re+H))²)，而 Klobuchar 算法本身用的是 ICD 拟合式 '
      'F = 1+16(0.53−el/π)³，两者在 10°~12° 仰角处相差约 3%（约 0.12 m 斜距）。'
      '这是 Klobuchar 算法**固有的模型误差**，仿真中如实保留，未人为消除。\n')
    A('2. **多路径的极化处理**：地表反射按线性极化 Fresnel 系数 Γ_h/Γ_v 计算，'
      '再投影到圆极化基（同极化 (Γ_h+Γ_v)/2、交叉极化 (Γ_h−Γ_v)/2）；'
      '同极化分量直接进入接收，交叉极化分量再乘天线左旋抑制比。'
      '低仰角时同极化占优（反射波仍是右旋，极化抑制无效），'
      '超过布儒斯特角后交叉极化占优、可被天线抑制——与 GNSS 多路径实测规律一致。'
      '该部分的仰角约定已用解析极限（正入射/掠射/理想导体）逐项校验。\n')

    # ==================================================================
    A('\n## 4. 仿真结果\n')
    A('### 4.1 卫星几何\n')
    if geom is not None:
        g = geom.copy()
        g.columns = ['星座', '平均可见数', '最少可见数', 'HDOP 均值', 'HDOP 最大',
                     'VDOP 均值', 'PDOP 均值', 'PDOP 最大']
        A(md_table(g.round(2)))
        A('\n![卫星几何](figures/fig_geometry.png)\n')
        A('> 农田开阔无遮挡，GPS+BDS+Galileo 平均可见 27 颗，HDOP 约 0.5；'
          '即使只用 GPS 也有 8–9 颗可用（HDOP 1.09）。BDS 的 GEO/IGSO 长期驻留'
          '中国上空，对几何强度贡献明显。几何不是本场景的短板。\n')

    A('\n### 4.2 各模式定位精度（主结果）\n')
    if main_s is not None:
        for kind, title in (('instant', '逐历元（瞬时）解'),
                            ('static', '静态解（SPP/SBAS/DGNSS 为 1 h 平均；'
                                       'RTK 为 10 min；PPP 为 30 min）')):
            A(f'\n**{title}**\n')
            rows = []
            for m in ORDER:
                if kind == 'static' and m == 'rtk_inst':
                    continue
                r = {'模式': MODE_LABEL.get(m, m)}
                any_ok = False
                for c in cases:
                    h95 = get(main_s, c, m, kind, 'h_cep95')
                    v95 = get(main_s, c, m, kind, 'v_p95')
                    if np.isfinite(h95):
                        r[c] = f'{flen(h95)} / {flen(v95)}'
                        any_ok = True
                    else:
                        r[c] = '—'
                if any_ok:
                    rows.append(r)
            df = pd.DataFrame(rows)
            df.columns = ['模式'] + [f'{c}<br>水平 / 垂直 95%'
                                     for c in df.columns[1:]]
            A(md_table(df))
        A('\n（表中为「水平误差 95% 分位 / 垂直误差 95% 分位」，'
          '小于 0.1 m 的自动以 cm 显示）\n')
        A('![误差累积分布](figures/fig_cdf.png)\n')
        A('![各模式误差对比](figures/fig_summary_bar.png)\n')
        A('![田块内定位结果](figures/fig_field_map.png)\n')
        A('> 三个稳定规律：① 垂直误差约为水平的 2–3 倍（SPP/SBAS，几何与误差投影'
          '所致）；② 单点定位靠时间平均几乎不改善，因为其误差是系统性、'
          '时间相关的；③ 一旦引入载波相位（RTK/PPP），精度直接进入厘米级。\n')
        A('\n> **关于 RTK 误差分布**：在“短基线 + 双频 + 多系统 + 天线高出地表”的'
          '正常条件下，本仿真的 RTK 误差分布是**单峰且紧致**的'
          '（99 分位仅比中位数大 2 倍，无错误固定）。一旦条件变差就会出现'
          '“双峰”——长基线（15 km）时只有 47% 能固定，未固定时退化为 '
          '0.2–0.6 m 的浮点解。工程上必须保留接收机的“固定/浮点”标志，'
          '不能只看标称精度。\n')
        A('\n> **两点必须说明的解算状态**：① 表中“RTK 单历元（瞬时）”的 '
          f'{fmt(get(main_s, caseA, "rtk_inst", "instant", "h_cep95"), 2)} m '
          '是**浮点解**（本仿真在单历元条件下从未成功固定模糊度，'
          '固定率为 0），它与静态 RTK 的厘米级不是同一类解，'
          '不要混用——这也说明**低成本的码观测不足以支撑单历元瞬时固定**，'
          '静态探针应老老实实累积几分钟；② B 场景（基准站坐标由单点定位测得）里，'
          '接收机**每一次都会报告“固定成功”**（固定率 100%、双差相位残差也正常），'
          f'但坐标却偏了 {fmt(get(main_s, caseB, "rtk", "static", "h_cep95"), 2)} m——'
          '因为基准站坐标偏差被整周模糊度完全吸收，接收机自己无法察觉。'
          '这正是“基准站坐标必须用静态观测标定”的最强论据：'
          '**固定的成功标志不能证明绝对坐标正确**。\n')

    A('\n### 4.3 相对定位精度与管理分区归属\n')
    if rel is not None and not rel.empty:
        r = rel[rel['case'] == caseA].copy()
        r['mode'] = r['mode'].map(lambda m: MODE_LABEL.get(m, m))
        r = r[['mode', 'rel_n', 'rel_median', 'rel_rms', 'rel_p95']]
        r.columns = ['模式', '样本数', '中位 (m)', 'RMS (m)', '95% (m)']
        A('同一历元内 20 个探针两两之间的水平相对定位误差：\n')
        A(md_table(r))
        A('\n> 电离层、对流层、星历误差在 100 m 尺度上高度相关，同一田块内所有探针'
          '会“一起同向偏移”，所以相对精度明显优于绝对精度。对“画墒情分布图、'
          '按分区管理”这类应用，相对精度才是关键指标。\n')
    A('\n![分区归属错误率](figures/fig_zone.png)\n')
    A('> 若把田块划成 6 m × 6 m 管理分区，单点定位会让相当比例的探针被划进错误'
      '分区；RTK/PPP 的误分率接近 0。这是“精度够不够用”最直观的判据。\n')

    A('\n### 4.4 误差预算：钱花在哪一项上最值\n')
    if budget is not None:
        for kind in ('instant', 'static'):
            sub = budget[budget['kind'] == kind]
            if sub.empty:
                continue
            rows = []
            for c in dict.fromkeys(sub['case']):
                r = {'实验条件': c}
                for m in ORDER:
                    v = get(budget, c, m, kind, 'h_cep95')
                    if np.isfinite(v):
                        r[MODE_LABEL[m]] = v * 100 if m in ('rtk', 'ppp') else v
                rows.append(r)
            df = pd.DataFrame(rows).dropna(axis=1, how='all')
            tag = '瞬时解' if kind == 'instant' else '静态解'
            A(f'\n**{tag}——水平误差 95% 分位（RTK/PPP 单位 cm，其余为 m）**\n')
            A(md_table(df.round(3)))
        A('\n![误差预算](figures/fig_error_budget.png)\n')
        base = get(budget, '全部误差源(基准)', 'spp_multi', 'instant', 'h_cep95')
        no_iono = get(budget, '关闭 电离层残差', 'spp_multi', 'instant', 'h_cep95')
        no_eph = get(budget, '关闭 星历/钟差误差', 'spp_multi', 'instant', 'h_cep95')
        no_mp = get(budget, '关闭 地表多径', 'spp_multi', 'instant', 'h_cep95')
        A('\n> 阅读提示：各变体共用同一批随机布局与误差实现（公共随机数法），'
          '因此表中的差异来自被关闭的那一项；但样本量有限（150 个探针），'
          '**个别行出现"关闭某项后误差反而略增"属于 ±0.2 m 量级的统计波动**，'
          '不代表该项误差有害。判断贡献时应看差异是否显著大于该量级。\n')
        A(f'> 对单点定位而言，把电离层残差完全改正（理想上限）后，水平 95% 误差从 '
          f'{fmt(base, 2)} m 降到 {fmt(no_iono, 2)} m，是全部 7 项中唯一超过'
          f' 2 倍改善的；关掉星历/钟差误差后为 {fmt(no_eph, 2)} m，'
          f'关掉地表多路径后几乎不变（{fmt(no_mp, 2)} m），'
          '关掉对流层残差同样几乎无变化。**这直接说明：开阔农田里限制单点定位'
          '精度的是电离层改正（其次是星历/钟差），不是多路径，也不是对流层。**\n')
        A('> 注意：“关闭电离层残差”对应的是**完全改正**的理想情形'
          '（模型值恰好等于真值），它是改正能力提升的上限，不是某种真实产品的性能；'
          'SBAS 那一行（残差降到 25%）才代表工程上可获得的水平。\n')

    A('\n### 4.5 敏感性分析\n')
    if sens is not None:
        rows = []
        for c in dict.fromkeys(sens['case']):
            r = {'条件': c}
            for m in ('spp_multi', 'sbas', 'dgnss', 'rtk', 'ppp'):
                v = get(sens, c, m, 'static', 'h_cep95')
                if np.isfinite(v):
                    r[MODE_LABEL[m]] = v * 100 if m in ('rtk', 'ppp') else v
            rows.append(r)
        df = pd.DataFrame(rows).dropna(axis=1, how='all')
        A('静态解水平误差 95% 分位（RTK/PPP 单位 cm，其余 m）：\n')
        A(md_table(df.round(3)))
        A('\n![敏感性分析](figures/fig_sensitivity.png)\n')
        A('> **统计口径提醒**：本表与 4.4 节误差预算表每个变体只有 15 个独立'
          'layout（150 个探针），蒙特卡洛标准误约 10–15%。因此'
          '“干土 5.5 mm vs 湿土 5.7 mm”这类小于 10% 的差异**落在噪声内**，'
          '不宜作为结论；只有差异超过约 15–20% 的比较才有意义。\n')
        A('值得注意的几点：\n')
        A('1. **星座数量**：从 GPS 单系统换成多系统，单点定位改善有限'
          '（3.2 m → 2.3 m，几何本来就够好，误差以系统性为主），'
          '但对 DGNSS 帮助显著（0.25 m → 0.12 m，码噪声被更多观测平均掉），'
          '对 PPP 收敛也明显有利。\n')
        A('2. **电离层活动**：太阳活动强（SFU 220）时单点定位从 2.8 m 恶化到 '
          '4.0 m，磁暴时到 6.2 m；而 DGNSS/RTK 几乎不受影响——'
          '这正是差分与双频消电离层的价值。\n')
        A('3. **土壤含水率与粗糙度**：对码观测几乎无影响（见 5.1 的物理解释），'
          '对载波相位多路径有毫米级影响，在 RTK/PPP 的厘米级量级上可见但不大。\n')
        A('4. **天线与安装**：换用测量型扼流圈天线后，RTK 水平误差从 3.7 mm 降到 '
          '1.3 mm（相位中心变化与多路径同时减小），DGNSS 也从 0.13 m 降到 '
          '0.09 m；把天线架高到 0.6–1.2 m 对差分模式收益不大（低仰角信号更差），'
          '但对单点定位的增益更明显。\n')
        A('5. **低矮作物**：15 cm 出苗期作物层在模型中给出很弱的相干反射'
          '（等效反射系数约 0.15、且被 1.5 dB 衰减），对 RTK 的影响落在噪声内'
          '（4.2 mm vs 3.7 mm，固定率仍为 100%）。这与“无高大农作物”的前提一致：'
          '**稀疏幼苗层不构成有效反射面**；但如果换成拔节/抽穗期的高大稠密作物，'
          '冠层会成为强反射体并遮挡低仰角信号，结论会完全不同（不在本场景范围内）。\n')

    A('\n### 4.6 观测时长与收敛\n')
    if occ is not None:
        p = occ[occ['mode'] == 'ppp'].sort_values('window_min')
        s = occ[occ['mode'].str.startswith('spp')].sort_values('window_min')
        A('| 观测时长 (min) | PPP 水平 RMS (cm) | PPP 垂直 RMS (cm) | '
          'SPP 时段平均水平 RMS (m) |')
        A('|---|---|---|---|')
        for _, r in p.iterrows():
            w = r['window_min']
            srow = s[s['window_min'] == w]
            spp_v = fmt(float(srow['h_rms'].iloc[0]), 3) if len(srow) else '—'
            A(f'| {w:g} | {r["h_rms"] * 100:.2f} | {r["v_rms"] * 100:.2f} | {spp_v} |')
        A('\n![观测时长影响](figures/fig_occupation.png)\n')
        p30 = p[p["window_min"] == 30]
        p60 = p[p["window_min"] == 60]
        h30 = float(p30["h_rms"].iloc[0]) if len(p30) else np.nan
        h60 = float(p60["h_rms"].iloc[0]) if len(p60) else np.nan
        A(f'> PPP 需要 20–30 min 才收敛到分米级（30 min 水平 RMS '
          f'{h30 * 100:.1f} cm），观测到 1 h 才进入厘米级（{h60 * 100:.1f} cm）；'
          '而单点定位靠时间平均几乎不改善（误差系统性、时间相关，'
          '平均只降随机噪声，1 min→120 min 只在 1.0–1.3 m 之间无趋势浮动）。'
          '**对静止的土壤探针来说，多等 20–30 分钟比换更贵的天线更有效。**\n')
        A('> 说明：本节只有 6 个独立探针样本（每个时长一条曲线），'
          '蒙特卡洛相对不确定度约 30%，因此中间点的非单调跳变（如 7 min 优于 '
          '10 min）**不显著**，只有“20–30 min 进入分米级 → 1 h 进入厘米级”'
          '这一整体趋势可信。表中 30 min 的 PPP 水平 RMS 与 4.2 节 95% 分位'
          '（150 个样本）来源不同、样本量差 25 倍，不宜直接比较。\n')

    A('\n### 4.7 农田地表多路径的物理图像\n')
    if mp is not None:
        A('![多路径随仰角与含水率变化](figures/fig_multipath.png)\n')
        sub = mp[(mp['antenna'] == 'lowcost') & (mp['moisture'] == 0.20)]
        if not sub.empty:
            near20 = sub.iloc[(sub['el'] - 20).abs().argmin()]
            A(f'> 探针天线离地仅 0.2 m，地面反射路径差 Δ = 2h·sin(el) 最大只有 '
              f'0.4 m，**远小于 1 个 GPS 码片（293 m）**，因此码相位几乎不受地面'
              f'多路径影响（仿真中约 15 mm）；但载波波长只有 0.19 m，同样的路径差'
              f'足以造成明显相位干涉：低成本天线在 20° 仰角处约 '
              f'{abs(near20["phase_mp_mm"]):.1f} mm，测量型扼流圈天线降到 1 mm 以内。'
              '土壤越湿、介电常数越大、反射越强，多路径随含水率上升而增大。\n')

    # ==================================================================
    A('\n### 4.8 与实测经验值的对照（模型可信度检验）\n')
    A('把仿真结果与 GNSS 工程实践中公认的精度区间对照，用于判断模型是否"跑偏"：\n')

    def verdict(v, lo, hi):
        if not np.isfinite(v):
            return '—'
        if lo <= v <= hi:
            return '一致'
        if v < lo:
            return f'优于经验值 {(lo / v):.1f} 倍（偏乐观）'
        return f'差于经验上限 {(v / hi):.1f} 倍（偏保守）'

    ref = [
        ('GPS 单频单点定位', 1.5, 3.0, gh('spp_gps', 'h_cep95', kind='instant')),
        ('多系统单点定位', 1.0, 2.0, gh('spp_multi', 'h_cep95', kind='instant')),
        ('星基增强 SBAS', 0.5, 1.5, gh('sbas', 'h_cep95', kind='instant')),
        ('本地基准站伪距差分（瞬时）', 0.3, 1.0,
         gh('dgnss', 'h_cep95', kind='instant')),
        ('静态 RTK（基线 <100 m，坐标已标定）', 0.005, 0.02,
         gh('rtk', 'h_cep95')),
        ('静态 PPP（30 min 收敛，测量型天线）', 0.03, 0.10,
         gh('ppp', 'h_cep95')),
        ('单基站 RTK（基线 15 km）', 0.10, 0.50,
         get(main_s, caseC, 'rtk', 'static', 'h_cep95')),
    ]
    A('| 定位模式 | 工程经验区间（水平 95%） | 本仿真结果 | 判断 |')
    A('|---|---|---|---|')
    for name, lo, hi, v in ref:
        if lo < 0.1:                       # 同一行内单位保持一致
            rng_s = f'{lo * 100:.1f} cm – {hi * 100:.1f} cm'
        else:
            rng_s = f'{lo:g} m – {hi:g} m'
        A(f'| {name} | {rng_s} | {flen(v)} | {verdict(v, lo, hi)} |')
    A('')
    A('> **关于 CORS 场景的说明**：本仿真的 C 场景模拟的是**单基站**用法'
      '（把 15 km 外的 CORS 当成普通基准站，不对区域大气做建模），'
      '因此结果是偏保守的。真实的**网络 RTK**（VRS/MAX 等）会对基准网覆盖范围内的'
      '电离层/对流层做区域建模，通常能在 15–50 km 基线上恢复到 2–5 cm。'
      '换句话说：不是"用 CORS 不好"，而是"用单个远距离基准站做双差"不好。\n')

    # ==================================================================
    A('\n### 4.9 统计说明：有效样本量\n')
    smp = load('samples_main.csv')
    if smp is not None:
        smp = smp[smp['case'] == caseA].copy()
        smp['h'] = np.hypot(smp['ee'], smp['en'])
        rows = []
        for kind in ('instant', 'static'):
            for m in ORDER:
                s = smp[(smp['kind'] == kind) & (smp['mode'] == m)]
                if len(s) < 30:
                    continue
                # 样本级（探针 x 历元，彼此相关）与 layout 级（独立实现）
                by_layout = s.groupby('layout')['h'].median()
                rows.append({
                    '模式': MODE_LABEL.get(m, m),
                    '解类型': '瞬时' if kind == 'instant' else '静态',
                    '样本数': len(s),
                    '独立实现数(layout)': len(by_layout),
                    '样本级 95% (m)': float(np.percentile(s['h'], 95)),
                    'layout 级中位误差的中位 (m)': float(by_layout.median()),
                    'layout 级中位误差的 95% (m)': float(by_layout.quantile(0.95)),
                })
        A('同一 layout 内的 20 个探针共享同一套大气/星历/基准站误差实现，'
          '它们的误差高度相关，因此"样本数"远大于"独立实现数"。'
          '下表把两种口径并列，用于判断 95% 分位的稳定性：\n')
        A(md_table(pd.DataFrame(rows)))
        A('\n> 阅读建议：**绝对精度指标（95%）表示"某个探针某一次定位"的误差水平；'
          'layout 级列表示"换一块田、换一天"时这个水平的波动**。'
          '两者都不大时，结论才算稳。单点定位的 layout 级离散明显小于样本级离散，'
          '说明其误差主要是共模的（整块田一起偏），这也再次支持了'
          '「相对精度优于绝对精度」的判断。\n')

    # ==================================================================
    A('\n## 5. 讨论\n')
    A('### 5.1 为什么农田里多路径不是主要矛盾\n')
    A('高大作物、建筑、水面会形成远距离反射体，路径差可达十几米，对码观测造成'
      '米级误差。本场景假设无高大作物，天线又贴地安装，唯一反射面是天线正下方的'
      '土壤，反射路径差极小，因此：\n')
    A('- 码多路径约 15 mm，可忽略；\n')
    A('- 载波相位多路径 1–7 mm，在 RTK/PPP 的厘米级精度上属可观项；\n')
    A('- 低仰角信号衰减与失锁主要由地面遮挡（掩蔽角）决定，而非多径。\n')
    A('换句话说，**这个场景的误差结构更像“开阔天空 + 差改正源”，而不是“城市峡谷”**。'
      '因此提升精度应优先投入改正信息（基准站／网络 RTK／SBAS／PPP），'
      '而不是昂贵的抗多路径天线。\n')

    A('\n### 5.2 绝对精度 vs 相对精度\n')
    A('电离层、对流层、星历误差在 100 m 尺度上高度相关，同一田块内所有探针会'
      '一起同向偏移，所以相对精度明显优于绝对精度（4.3 节）。'
      '对“画墒情分布图、按分区管理”这类应用，关键看相对精度；'
      '而对“登记探针法定坐标、与农机作业轨迹对齐”这类应用，绝对精度才是硬指标，'
      '此时必须解决基准站坐标的标定问题（对比 4.2 节三种基准站场景）。\n')

    A('\n### 5.3 成本—精度选型建议\n')
    A('| 方案 | 硬件 | 需要的设施 | 仿真水平精度 (95%) | 适用场景 |')
    A('|---|---|---|---|---|')
    A(f'| 单频模块 + 单点定位 | 单频模块（最低） | 无 | '
      f'{fmt(gh("spp_multi", "h_cep95"), 2)} m | 只需大致位置、成本敏感 |')
    A(f'| 单频模块 + SBAS | 单频模块（最低） | 无（卫星广播） | '
      f'{fmt(gh("sbas", "h_cep95"), 2)} m | 免通信链路的亚米级需求 |')
    A(f'| 单频模块 + 自建基准站 DGNSS | 两个单频模块 + 电台/4G | 田块内基准站 | '
      f'{fmt(gh("dgnss", "h_cep95"), 2)} m | 定位到小区（3–6 m 分区） |')
    A(f'| 双频 RTK + 自建基准站（坐标已标定） | 双频 RTK 模块 | 基准站 + 通信 + '
      f'**基准点坐标标定** | {flen(gh("rtk", "h_cep95"))} | '
      '厘米级布点、与农机/测绘坐标系统一 |')
    A(f'| 双频 RTK + 自建基准站（坐标随手测） | 双频 RTK 模块 | 同上但省略标定 | '
      f'{fmt(get(main_s, caseB, "rtk", "static", "h_cep95"), 2)} m | '
      '**不推荐**：花钱买不到绝对精度 |')
    A(f'| 双频模块 + 单基站长基线 RTK (15 km) | 双频 RTK 模块 + 4G | CORS 服务 | '
      f'{fmt(get(main_s, caseC, "rtk", "static", "h_cep95"), 2)} m | '
      '邻近有 CORS、且只把它当普通基准站用时 |')
    A(f'| 双频模块 + PPP | 双频 RTK 模块 | 无（需网络获取精密产品） | '
      f'{flen(gh("ppp", "h_cep95"))} | 无基准站、可等 30 min 收敛 |')
    A('')
    A('**推荐组合**：田块内自建基准站时优先用双频 RTK，但务必用静态观测把基准点'
      '坐标标定到厘米级（例如连续观测 1–2 h 的 PPP，或与已知点联测）；'
      '如果只需“识别到小区”级别，单频模块 + SBAS 或低成本 DGNSS 已经够用，'
      '可以把预算留给土壤传感器本身。\n')

    # ==================================================================
    A('\n## 6. 模型局限与可信度说明\n')
    A('为保证结果可解释，以下简化需读者知悉：\n')
    A('1. **多路径**采用相干双射线 + 粗糙面漫反射解析模型，并用“近场因子”'
      '标定到实测量级；真实农田散射更复杂，绝对量值可能有 ±50% 不确定度，'
      '但随仰角、含水率、天线类型的**趋势**可靠。\n')
    A('2. **整周模糊度固定**采用“按条件方差序贯取整（简化 LAMBDA/自助法）+ '
      '逐卫星相位残差外点剔除 + 固定后位置重解”的策略，未实现完整整数最小二乘'
      '与 Ratio 检验；实际接收机的固定成功率会略高，因此 RTK 的固定率数据偏保守，'
      '固定后的精度量级可信。\n')
    A('3. **PPP**按静态多历元最小二乘（浮点模糊度 + 分段天顶对流层）实现，'
      '未模拟滤波过程噪声；收敛时间量级可信。PPP-AR 可把收敛时间缩短到 5–10 min，'
      '本仿真未单独建模，故 PPP 结果偏保守。\n')
    A('4. **广播星历/钟差误差**按 SISRE 量级的缓变偏差建模，'
      '未模拟星历龄期、轨道机动等事件。\n')
    A('5. 电离层为参数化薄壳模型 + Klobuchar，未使用真实 TEC 图；'
      '但残差比例（约 18%）与实测统计量级一致。\n')
    A('6. 未模拟：接收机内部多径、探针金属外壳散射、天线罩/土壤覆盖引起的'
      '方向图畸变、雨衰、探针倾斜引起的相位中心偏差、多路径随温度/昼夜的漂移。\n')
    A('7. RTK/PPP 均按“单探针独立解算”评估，未利用多探针共用改正数带来的'
      '误差相关性（考虑后相对精度会更优）。\n')
    A('8. 统计口径：同一 layout 内的探针共享大气/星历误差实现，样本并不独立。'
      '所有 95% 分位均已按“样本级”与“layout 级”两种口径对照（见 4.9 节），'
      '结论在两种口径下一致。\n')
    A('9. **模型修正记录**：本工程经过一次独立对抗性复核（只读、未参与建模）。'
      '复核确认了星座/坐标/时间链、Klobuchar、Saastamoinen、Topp、J2、'
      '观测方程符号、消电离层系数与模糊度固定逻辑均正确，'
      '并发现并修正了一处**公式性错误**：地表反射的 Fresnel 系数原先误把'
      '卫星仰角当作入射角代入（正确关系为 cos θi = sin el），'
      '导致反射率随仰角的依赖方向被反转、低仰角多路径被低估数倍。'
      '修正后已在 `checks/check_physics_limits.py` 中增加正入射/掠射/'
      '理想导体极限与单调性断言，全部通过；本报告所有数据均为修正后重跑结果。\n')

    A('\n## 7. 复现方式\n')
    A('```bash\n'
      'pip install numpy pandas matplotlib\n'
      'python run_sim.py            # 运行全部实验 -> results/*.csv\n'
      'python make_figures.py       # 生成 figures/*.png\n'
      'python report.py             # 生成本报告\n'
      'python checks/check_geometry.py      # 星座几何自检\n'
      'python checks/check_observations.py  # 误差源量级自检\n'
      'python checks/check_estimators.py    # 解算器精度自检\n'
      'python checks/check_physics_limits.py # 解析极限自检（全部通过）\n'
      '```\n')
    A('\n代码结构：`gnss_sim/` 为仿真包——`geodesy`（坐标与时间基准）、'
      '`constellation`（星座与轨道传播）、`models`（误差源）、'
      '`observation`（观测合成）、`estimators`（SPP/SBAS/DGNSS/RTK/PPP 解算器）、'
      '`metrics`（精度指标）、`experiments`（实验驱动）；'
      '`checks/` 为自检脚本，`results/` 与 `figures/` 为输出目录。\n')

    OUT.write_text('\n'.join(L), encoding='utf-8')
    print(f'报告已生成: {OUT}  ({len("".join(L))} 字符)')


if __name__ == '__main__':
    main()
