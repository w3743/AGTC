"""Corner estimate for the G431/DRV8353 bidirectional phase-current window.

This is a transparent, deterministic corner enumeration, not a circuit simulation
or a hardware qualification. Running it writes 电流窗口误差核算.json beside this
script. Python 3.10+; standard library only.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path


OUT = Path(__file__).with_suffix(".json")
VREF_NOM = 3.3
RSHUNT_NOM = 0.0005
SHUNT_TOL = 0.01
SHUNT_TCR_PPM_PER_C = 50
SHUNT_DELTA_C = 100  # 25 C reference to 125 C datasheet TCR endpoint
CSA_VOS_MAX_V = 0.003
CSA_VOS_DRIFT_TYP_V_PER_C = 10e-6  # typical only; not a guaranteed bound
TLV_VOS_MAX_V = 0.0055
G431_COMP_VOS_MAX_V = 0.009
G431_DAC3_TUE_MAX_LSB = 5
DAC_TRANSFER_DIVISOR = 4096
DAC_MAX_CODE = 4095
VREF_TOL = 0.025
RNEG_TOL = 0.01


def corner_values(nominal: float, tol: float) -> tuple[float, float]:
    return nominal * (1 - tol), nominal * (1 + tol)


def enumerate_current_limits(
    *,
    gain_nom: float,
    gain_corners: tuple[float, float],
    nominal_window_a: float,
    positive_dac: bool,
    rtop_nom: float,
    rbot_nom: float,
) -> dict:
    """Enumerate independent electrical corners for the positive and negative trips."""
    gain_values = gain_corners
    vref_values = corner_values(VREF_NOM, VREF_TOL)
    rshunt_values = [
        RSHUNT_NOM * (1 + tol) * (1 + tcr_ppm * 1e-6 * SHUNT_DELTA_C)
        for tol, tcr_ppm in itertools.product((-SHUNT_TOL, SHUNT_TOL),
                                              (-SHUNT_TCR_PPM_PER_C,
                                               SHUNT_TCR_PPM_PER_C))
    ]
    csa_vos_values = (-CSA_VOS_MAX_V, CSA_VOS_MAX_V)
    rtop_values = corner_values(rtop_nom, RNEG_TOL)
    rbot_values = corner_values(rbot_nom, RNEG_TOL)

    # DAC code is selected from nominal values. DAC VREF and CSA VREF are modeled
    # as the same VREF_CORE rail; board-level source/noise mismatch is not included.
    positive_code = round(
        (0.5 + gain_nom * nominal_window_a * RSHUNT_NOM / VREF_NOM)
        * DAC_TRANSFER_DIVISOR
    )
    positive_code = min(positive_code, DAC_MAX_CODE)
    positive = []
    if positive_dac:
        for vref, gain, rshunt, csa_vos, dac_sign, comp_sign in itertools.product(
            vref_values,
            gain_values,
            rshunt_values,
            csa_vos_values,
            (-1, 1),
            (-1, 1),
        ):
            dac_error = dac_sign * G431_DAC3_TUE_MAX_LSB * vref / DAC_TRANSFER_DIVISOR
            comp_error = comp_sign * G431_COMP_VOS_MAX_V
            v_dac = vref * positive_code / DAC_TRANSFER_DIVISOR + dac_error
            trip = (v_dac + comp_error - vref / 2 - gain * csa_vos) / (gain * rshunt)
            positive.append({
                "trip_A": trip,
                "VREF_V": vref,
                "gain_V_per_V": gain,
                "Rshunt_ohm": rshunt,
                "CSA_VOS_V": csa_vos,
                "DAC_error_V": dac_error,
                "COMP_VOS_V": comp_error,
            })

    negative = []
    for vref, gain, rshunt, csa_vos, tlv_sign, rtop, rbot in itertools.product(
        vref_values,
        gain_values,
        rshunt_values,
        csa_vos_values,
        (-1, 1),
        rtop_values,
        rbot_values,
    ):
        vneg = vref * rbot / (rtop + rbot)
        tlv_error = tlv_sign * TLV_VOS_MAX_V
        trip = (vref / 2 - vneg + gain * csa_vos + tlv_error) / (gain * rshunt)
        negative.append({
            "trip_A": trip,
            "VREF_V": vref,
            "gain_V_per_V": gain,
            "Rshunt_ohm": rshunt,
            "CSA_VOS_V": csa_vos,
            "TLV1704_VOS_V": tlv_error,
            "Rtop_ohm": rtop,
            "Rbottom_ohm": rbot,
        })

    def summarize(corners: list[dict]) -> dict | None:
        if not corners:
            return None
        lo = min(corners, key=lambda row: row["trip_A"])
        hi = max(corners, key=lambda row: row["trip_A"])
        return {
            "min_trip_A": lo["trip_A"],
            "max_trip_A": hi["trip_A"],
            "min_margin_over_120A": lo["trip_A"] - 120,
            "min_corner": lo,
            "max_corner": hi,
            "enumerated_corner_count": len(corners),
        }

    return {
        "nominal_window_A": nominal_window_a,
        "nominal_gain": gain_nom,
        "positive_DAC_code_12bit": positive_code if positive_dac else None,
        "positive_DAC_ideal_voltage_at_3v3_V": (
            VREF_NOM * positive_code / DAC_TRANSFER_DIVISOR if positive_dac else None
        ),
        "negative_divider_nominal": {
            "Rtop_ohm": rtop_nom,
            "Rbottom_ohm": rbot_nom,
            "VNEG_at_3v3_V": VREF_NOM * rbot_nom / (rtop_nom + rbot_nom),
        },
        "positive_trip": summarize(positive),
        "negative_trip_magnitude": summarize(negative),
    }


cases = {
    "comparison_previous_gain10_nominal_plus_minus_140A": enumerate_current_limits(
        gain_nom=10.0,
        gain_corners=(9.7, 10.3),
        nominal_window_a=140.0,
        positive_dac=True,
        rtop_nom=24_900.0,
        rbot_nom=10_000.0,
    ),
    "root_selected_gain10_nominal_plus_minus_150A": enumerate_current_limits(
        gain_nom=10.0,
        gain_corners=(9.7, 10.3),
        nominal_window_a=150.0,
        positive_dac=True,
        rtop_nom=26_700.0,  # nearest practical E96 value to 26.667 kOhm
        rbot_nom=10_000.0,
    ),
    "old_gain20_nominal_plus_minus_130A": enumerate_current_limits(
        gain_nom=20.0,
        gain_corners=(19.4, 20.6),
        nominal_window_a=130.0,
        positive_dac=True,
        rtop_nom=84_500.0,
        rbot_nom=10_000.0,
    ),
}

drift_sensitivity = {
    "CSA_offset_drift_typical_per_C_uV": CSA_VOS_DRIFT_TYP_V_PER_C * 1e6,
    "temperature_span_C": SHUNT_DELTA_C,
    "CSA_typical_drift_over_span_mV_input": CSA_VOS_DRIFT_TYP_V_PER_C * SHUNT_DELTA_C * 1e3,
    "equivalent_current_shift_A_at_0p5mOhm": (
        CSA_VOS_DRIFT_TYP_V_PER_C * SHUNT_DELTA_C / RSHUNT_NOM
    ),
    "shunt_TCR_abs_fraction_over_span": SHUNT_TCR_PPM_PER_C * 1e-6 * SHUNT_DELTA_C,
    "shunt_threshold_shift_at_150A_due_TCR_only_A": (
        150 * SHUNT_TCR_PPM_PER_C * 1e-6 * SHUNT_DELTA_C
    ),
    "qualification_note": (
        "CSA 10uV/C is typical rather than a guaranteed maximum; the illustrative "
        "current shift is not added as a guaranteed corner. KOA TCR is bounded only "
        "within its specified temperature interval; actual pulse self-heating must be measured."
    ),
}

report = {
    "title": "G431 + DRV8353 bidirectional phase-current window corner estimate",
    "status": "engineering_estimate_not_hardware_qualification",
    "assumptions": {
        "VREF_CORE_nom_V": VREF_NOM,
        "VREF_CORE_tolerance": "+/-2.5%",
        "DAC_reference_assumed": "same VREF_CORE rail as CSA VREF+; verify board routing and source impedance",
        "phase_shunt": "0.5mOhm KOA PSF4, initial +/-1%, TCR +/-50ppm/C",
        "temperature_corner": "TCR applied from 25C to 125C only (100C span)",
        "DRV8353_CSA_gain": "GAIN10 range 9.7..10.3; GAIN20 range 19.4..20.6",
        "DRV8353_CSA_input_offset": "+/-3mV bound; input-referred, then multiplied by gain in SO voltage",
        "TLV1704A_input_offset": "+/-5.5mV full-temperature bound",
        "STM32G431_COMP_offset": "+/-9mV conservative symmetric bound from -9..+3mV datasheet range",
        "STM32G431_DAC3_TUE": "+/-5 LSB, DAC 15MSPS accuracy Table 71, internal comparator-only route with output buffer disabled; no separate DAC gain error added because TUE is total unadjusted error",
        "STM32G431_DAC3_comparator_routes": "RM0440 Table 73: DAC3_CH1 can be COMP1/COMP3 INM; DAC3_CH2 can be COMP2 INM. Program both DAC3 channels to same code. Internal-only output mode avoids PA4/PA5 DAC1 pins used by SPI1.",
        "STM32G431_DAC_transfer": "RM0440: VDAC=VREF*DOR/4096; code is bounded to 0..4095",
        "divider_tolerance": "RNEG top and bottom each +/-1%, independent",
        "omitted_effects": [
            "CSA offset drift has only a typical 10uV/C value and no corresponding guaranteed maximum used here",
            "VREF source noise/transient droop and mismatch between coreboard VREF+ and DAC reference",
            "comparator propagation delay, PWM/current ripple, analog routing, grounding, and ADC/DAC settling",
            "temperature-dependent comparator/CSA behavior beyond guaranteed values",
            "calibration and board-to-board component correlation"
        ],
    },
    "formula": {
        "positive": "Itrip=(Vdac + VOS_COMP - VREF/2 - G*VOS_CSA)/(G*Rshunt); DAC code chosen at nominal window using VREF/2 + Gnom*I*Rnom",
        "negative_magnitude": "Itrip=(VREF/2 - VNEG + G*VOS_CSA + VOS_TLV)/(G*Rshunt); VNEG=VREF*Rbottom/(Rtop+Rbottom)",
        "temperature_sensitivity": drift_sensitivity,
    },
    "cases": cases,
    "recommendation": {
        "previous_140A_candidate": "The modeled cold electrical minimums are +122.20A/-121.90A, only about 2.20A/1.90A above 120A; this narrow margin does not cover unbounded CSA drift, reference noise, comparator/Break delay, or PWM ripple, so it was replaced by the selected +/-150A nominal window.",
        "selected_candidate": "Root selected nominal +150A/-150.16A (RNEG top 26.7k/10k and positive DAC code from this script); retain commanded peak candidate <=110A, and validate the computed minimums and transient response on hardware.",
        "hard_limit_status": "No independent +/-120A hard-limit pass is established. Comparator/DAC routing and timed response require bench validation; software current regulation and DRV VDS fault remain distinct protections.",
    },
    "sources": [
        {"title": "TI DRV8353R datasheet", "url": "https://www.ti.com/lit/ds/symlink/drv8353r.pdf"},
        {"title": "TI TLV1704 datasheet", "url": "https://www.ti.com/lit/ds/symlink/tlv1704.pdf"},
        {"title": "ST STM32G431CB datasheet", "url": "https://www.st.com/resource/en/datasheet/stm32g431cb.pdf"},
        {"title": "ST RM0440 STM32G4 reference manual", "url": "https://www.st.com/resource/en/reference_manual/rm0440-stm32g4-series-advanced-armbased-32bit-mcus-stmicroelectronics.pdf"},
        {"title": "KOA PSF4 official datasheet", "url": "https://www.koaspeer.com/pdfs/PSF4.pdf"},
    ],
}

OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    name: {
        "positive": case["positive_trip"]["min_trip_A"],
        "negative": case["negative_trip_magnitude"]["min_trip_A"],
        "positive_max": case["positive_trip"]["max_trip_A"],
        "negative_max": case["negative_trip_magnitude"]["max_trip_A"],
    }
    for name, case in cases.items()
}, indent=2))
print(f"wrote {OUT}")
