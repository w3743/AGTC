#!/usr/bin/env python3
"""Plot the Python averaged 1 s 32.4 V regenerative DC-link profile."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "结果" / "dc_link_1s"
CSV = RESULTS / "regen_32.4V_loss0W_dump30mF.csv"
PNG = RESULTS / "regen_32p4V_bus_dump_1s.png"


def main() -> None:
    with CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    t = [float(r["t_s"]) for r in rows]
    bus = [float(r["vbus_v"]) for r in rows]
    i_dump = [float(r["idump_a"]) for r in rows]
    p_each = [(i * 0.5) ** 2 * 2.0 for i in i_dump]
    p_bank = [2 * p for p in p_each]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (ax_v, ax_p) = plt.subplots(2, 1, figsize=(10.4, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [1.0, 1.0]})
    ax_v.plot(t, bus, color="#1261a0", linewidth=1.25, label="VBUS (single board)")
    ax_v.axhline(43.4145, color="#c44736", linestyle="--", linewidth=1.0,
                 label="nominal chopper ON 43.415 V")
    ax_v.axhline(41.4152, color="#a97600", linestyle=":", linewidth=1.0,
                 label="nominal chopper OFF 41.415 V")
    ax_v.axhline(75.0, color="#555555", linestyle="-.", linewidth=0.9,
                 label="DRV8353 recommended upper boundary 75 V")
    ax_v.set_ylabel("Bus voltage (V)")
    ax_v.set_ylim(20, 80)
    ax_v.grid(True, alpha=0.25)
    ax_v.legend(loc="upper right", ncol=2, fontsize=8)

    ax_p.plot(t, p_bank, color="#6b6b6b", linewidth=1.1,
              label="two-resistor bank total")
    ax_p.plot(t, p_each, color="#d16b18", linewidth=1.1,
              label="each 2 Ω resistor (identical branches)")
    ax_p.set_ylabel("Instantaneous resistor power (W)")
    ax_p.set_xlabel("Time (s)")
    ax_p.set_xlim(0, 1.0)
    ax_p.grid(True, alpha=0.25)
    ax_p.legend(loc="upper right", fontsize=8)
    fig.suptitle("Averaged 32.4 V, 1.8 kW bus-regeneration stress case\n"
                 "30 mF nominal local link; 1 s constant power; idealized local chopper")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(PNG, dpi=180)
    print(PNG)


if __name__ == "__main__":
    main()
