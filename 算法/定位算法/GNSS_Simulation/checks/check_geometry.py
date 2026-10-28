"""星座几何自检：可见卫星数、DOP、GEO/IGSO 定点经度。

运行: python checks/check_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnss_sim import constellation as con
from gnss_sim import geodesy as geo

T0 = 1757916000.0        # 2025-09-15 06:00:00 UTC
LAT0, LON0, H0 = 34.50, 113.00, 80.0


def main():
    frame = geo.LocalFrame(LAT0, LON0, H0)
    rx = frame.enu_to_ecef([0.0, 0.0, 0.0])
    print(f"参考点 ECEF = {rx}")
    lat, lon, h = geo.ecef_to_geodetic(rx)
    print(f"往返转换: lat={lat:.7f} lon={lon:.7f} h={h:.4f}")

    c = con.build_constellation(("G", "C", "E", "R"), T0)
    print(f"星座规模: {len(c)} 颗")
    for s in ("G", "C", "E", "R"):
        print(f"  {s}: {int(np.sum(c.system == s))} 颗")

    # 轨道半径与周期检查
    for s in ("G", "C", "E", "R"):
        m = c.system == s
        period = 2 * np.pi / c.n[m].mean() / 3600.0
        print(f"  {s}: a={c.a[m].mean()/1e3:.0f} km, 周期={period:.2f} h")

    # GEO/IGSO 定点经度检查（24h 内的 ECEF 经度范围）
    t24 = T0 + np.arange(0, 86400, 600.0)
    sat24 = c.propagate_ecef(t24)
    for i, s in enumerate(c.sats):
        if s.kind in ("GEO", "IGSO"):
            la, lo, hh = geo.ecef_to_geodetic(sat24[:, i, :])
            print(f"  {s.prn} {s.kind}: 经度 {lo.min():8.2f} ~ {lo.max():8.2f} deg, "
                  f"纬度 {la.min():7.2f} ~ {la.max():7.2f} deg, 高度 {hh.mean()/1e3:.0f} km")

    # 可见性与 DOP（24 h，逐 10 min）
    print("\n可见卫星数与 DOP（仰角掩蔽 10 度）:")
    for systems in [("G",), ("C",), ("G", "C", "E"), ("G", "C", "E", "R")]:
        cc = con.build_constellation(systems, T0)
        nvis, hd, vd, pd = [], [], [], []
        for t in t24:
            idx, az, el, rng = cc.visible(frame, rx, float(t), 10.0)
            d = con.dop_from_geometry(az, el)
            nvis.append(len(idx))
            hd.append(d["hdop"]); vd.append(d["vdop"]); pd.append(d["pdop"])
        nvis = np.array(nvis)
        print(f"  {'+'.join(systems):12s} n_vis={nvis.mean():5.1f} "
              f"[{nvis.min():2d},{nvis.max():2d}]  "
              f"HDOP={np.mean(hd):5.2f}  VDOP={np.mean(vd):5.2f}  PDOP={np.mean(pd):5.2f}")

    # 单颗 GPS 卫星 24h 的仰角范围（合理性检查：应有升落）
    idx, az, el, rng = c.visible(frame, rx, T0, 0.0)
    gidx = [i for i in idx if c.system[i] == "G"]
    sample = gidx[0]
    track = []
    for t in t24:
        a, e, r = frame.az_el(rx, c.propagate_ecef(float(t))[0][sample])
        track.append((a, e))
    track = np.array(track)
    print(f"\n样例卫星 {c.prn[sample]} 24h 仰角范围: "
          f"{track[:,1].min():.1f} ~ {track[:,1].max():.1f} deg, "
          f"最大可见仰角持续时长约 {np.sum(track[:,1]>10)*10/60:.1f} h")


if __name__ == "__main__":
    main()
