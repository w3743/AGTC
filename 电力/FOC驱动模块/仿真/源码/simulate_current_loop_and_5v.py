#!/usr/bin/env python3
"""Small reproducible 20 kHz dq current-loop and LM5008A 5 V startup screen.

This is an engineering screen, not firmware, an LM5008A silicon model, or a
thermal/electrical hardware qualification.  It writes one compact JSON result.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable


HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "结果" / "current_loop_5v_screen.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def quantize_current(i_a: float) -> float:
    """12-bit ADC, 3.3 V range, 1.65 V zero, 5 mV/A CSA gain."""
    v = clip(1.65 + 0.005 * i_a, 0.0, 3.3)
    code = round(v / 3.3 * 4095)
    return ((code / 4095 * 3.3) - 1.65) / 0.005


def regen_current_limit_for_power(
    power_limit_w: float,
    *,
    emf_peak_v: float,
    r_phase_ohm: float,
) -> float:
    """Return the low-current root for Pregen=1.5*(E*|Iq|-R*Iq^2)."""
    if power_limit_w < 0:
        raise ValueError("regen power limit must be non-negative")
    if emf_peak_v <= 0:
        return 0.0 if power_limit_w == 0 else math.inf
    if r_phase_ohm <= 0:
        return power_limit_w / (1.5 * emf_peak_v)
    max_power_w = 1.5 * emf_peak_v * emf_peak_v / (4.0 * r_phase_ohm)
    if power_limit_w >= max_power_w:
        return emf_peak_v / (2.0 * r_phase_ohm)
    discriminant = max(emf_peak_v * emf_peak_v - 4.0 * r_phase_ohm * power_limit_w / 1.5, 0.0)
    return (emf_peak_v - math.sqrt(discriminant)) / (2.0 * r_phase_ohm)


def resolve_regen_capability(
    *,
    per_board_power_w: float | None,
    generation: int | None,
    last_accepted_generation: int | None = None,
    age_s: float | None,
    max_age_s: float,
) -> dict:
    """Validate a newly received capability update; fail closed if invalid.

    The caller must distribute the shared battery/DC-DC sink budget across
    boards before publishing this value.  Compare ``last_accepted_generation``
    ``last_accepted_generation`` is checked only here, in the update-receive
    path. Once accepted, cache the allocation and generation. Do not call this
    function with a last-accepted generation on every PWM cycle; use
    ``resolve_cached_regen_capability`` for cached samples. This model cannot
    prove the sender's cross-board accounting.
    """
    reason = "valid"
    valid = True
    if not isinstance(generation, int) or isinstance(generation, bool) or generation <= 0:
        valid, reason = False, "missing_or_invalid_generation"
    elif last_accepted_generation is not None and (
        not isinstance(last_accepted_generation, int)
        or isinstance(last_accepted_generation, bool)
        or generation <= last_accepted_generation
    ):
        valid, reason = False, "replayed_or_out_of_order_generation"
    elif not isinstance(per_board_power_w, (int, float)) or isinstance(per_board_power_w, bool) or not math.isfinite(per_board_power_w) or per_board_power_w <= 0:
        valid, reason = False, "missing_zero_or_invalid_per_board_allocation"
    elif not isinstance(age_s, (int, float)) or isinstance(age_s, bool) or not math.isfinite(age_s) or age_s < 0:
        valid, reason = False, "missing_or_invalid_age"
    elif not isinstance(max_age_s, (int, float)) or isinstance(max_age_s, bool) or not math.isfinite(max_age_s) or max_age_s <= 0 or age_s > max_age_s:
        valid, reason = False, "expired"
    return {
        "valid": valid,
        "reason": reason,
        "effective_per_board_power_w": float(per_board_power_w) if valid else 0.0,
        "generation": generation,
        "age_s": age_s,
        "max_age_s": max_age_s,
    }


def resolve_cached_regen_capability(
    *,
    per_board_power_w: float | None,
    generation: int | None,
    age_s: float | None,
    max_age_s: float,
) -> dict:
    """Check a previously accepted allocation's TTL without replay checking.

    Call this with the cached sample on PWM updates. Generation replay/order
    checks belong to ``resolve_regen_capability`` when a new message arrives.
    """
    return resolve_regen_capability(
        per_board_power_w=per_board_power_w,
        generation=generation,
        last_accepted_generation=None,
        age_s=age_s,
        max_age_s=max_age_s,
    )


def validate_shared_sink_allocations(
    *,
    per_board_allocations_w: list[float | None],
    qualified_shared_budget_w: float | None,
    budget_qualified: bool,
) -> dict:
    """Reject allocations unless a qualified shared budget covers both boards.

    Any missing/zero/unqualified budget or over-allocation fails closed for
    all boards. The numeric tests below are interface logic examples only;
    the real DC/DC/battery sink budget remains unqualified.
    """
    allocations: list[float] = []
    reason = "valid"
    valid_budget = (
        budget_qualified
        and isinstance(qualified_shared_budget_w, (int, float))
        and not isinstance(qualified_shared_budget_w, bool)
        and math.isfinite(qualified_shared_budget_w)
        and qualified_shared_budget_w > 0.0
    )
    if not valid_budget:
        reason = "shared_budget_unknown_zero_or_unqualified"
    elif any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0.0
        for value in per_board_allocations_w
    ):
        reason = "invalid_board_allocation"
    else:
        allocations = [float(value) for value in per_board_allocations_w]
        if sum(allocations) > float(qualified_shared_budget_w) + 1e-12:
            allocations = []
            reason = "allocations_exceed_qualified_shared_budget"
    valid = reason == "valid"
    if not valid:
        allocations = [0.0 for _ in per_board_allocations_w]
    return {
        "valid": valid,
        "reason": reason,
        "budget_qualified": budget_qualified,
        "qualified_shared_budget_w": qualified_shared_budget_w,
        "per_board_allocations_w": allocations,
        "allocation_sum_w": sum(allocations),
        "allocation_sum_within_budget": valid and sum(allocations) <= float(qualified_shared_budget_w) + 1e-12,
    }


def limit_q_current_request(
    requested_a: float,
    *,
    current_command_limit_a: float | None,
    regen_power_limit_w: float | None,
    emf_peak_v: float,
    r_phase_ohm: float,
    regen_capability_per_board_w: float | None = None,
    regen_capability_generation: int | None = None,
    regen_capability_age_s: float | None = None,
    regen_capability_max_age_s: float = 0.100,
    require_fresh_regen_capability: bool = False,
) -> float:
    """Clamp q current and, when requested, fail closed on invalid regen capability."""
    if current_command_limit_a is not None:
        if current_command_limit_a <= 0:
            raise ValueError("current command limit must be positive")
        requested_a = clip(requested_a, -current_command_limit_a, current_command_limit_a)
    effective_regen_limit_w = regen_power_limit_w
    if requested_a < 0 and require_fresh_regen_capability:
        capability = resolve_cached_regen_capability(
            per_board_power_w=regen_capability_per_board_w,
            generation=regen_capability_generation,
            age_s=regen_capability_age_s,
            max_age_s=regen_capability_max_age_s,
        )
        effective_regen_limit_w = capability["effective_per_board_power_w"]
    if requested_a < 0 and effective_regen_limit_w is not None:
        regen_i_limit_a = regen_current_limit_for_power(
            effective_regen_limit_w,
            emf_peak_v=emf_peak_v,
            r_phase_ohm=r_phase_ohm,
        )
        requested_a = max(requested_a, -regen_i_limit_a)
    return requested_a


def current_loop(
    *,
    duration_s: float,
    r_phase_ohm: float,
    l_phase_h: float,
    vbus_v: float = 32.4,
    emf_peak_v: float = 12.83,
    command_a: float = -110.0,
    slew_a_per_s: float | None = 5000.0,
    current_command_limit_a: float | None = None,
    regen_power_limit_w: float | None = None,
    regen_capability_per_board_w: float | None = None,
    regen_capability_generation: int | None = None,
    regen_capability_age_s: float | None = None,
    regen_capability_max_age_s: float = 0.100,
    require_fresh_regen_capability: bool = False,
    command_schedule: list[tuple[float, float]] | None = None,
    anti_windup_back_calculation: bool = True,
    can_timeout_s: float | None = None,
    nominal_r_ohm: float = 0.017,
    nominal_l_h: float = 20e-6,
    sample_hz: float = 20_000.0,
    plant_substeps: int = 50,
    current_limit_a: float = 120.0,
) -> dict:
    """Amplitude-invariant q-axis plant with 20 kHz sampled PI and PWM ZOH.

    Voltage feedforward uses the legacy 12.83 V phase-peak back-EMF assumption.
    One PWM-period sample/update delay is included.  The plant uses 1 us Euler
    substeps at the default 20 kHz; PWM ripple, dead time and Hall/angle errors
    are excluded.
    """
    period = 1.0 / sample_hz
    subdt = period / plant_substeps
    omega_c = 2.0 * math.pi * 1000.0
    kp = nominal_l_h * omega_c
    ki = nominal_r_ohm * omega_c
    kaw = omega_c
    current_a = 0.0
    integral_v = 0.0
    applied_vq = 0.0
    ref_a = 0.0
    max_abs_a = 0.0
    min_a = 0.0
    max_a = 0.0
    peak_command_a = 0.0
    peak_error_a = 0.0
    sat_samples = 0
    max_abs_integral_v = 0.0
    sample_count = round(duration_s * sample_hz)
    tail_samples = []
    last_actual_vq = 0.0
    max_ref_step_a = 0.0
    previous_ref_a = 0.0
    schedule = sorted(command_schedule or [(0.0, command_a)])
    if not schedule or schedule[0][0] > 0.0:
        schedule.insert(0, (0.0, command_a))
    schedule_segment = 0
    segment_metrics: dict[int, dict] = {}

    for n in range(sample_count):
        t = n * period
        while schedule_segment + 1 < len(schedule) and t >= schedule[schedule_segment + 1][0]:
            schedule_segment += 1
        requested = schedule[schedule_segment][1]
        if can_timeout_s is not None and t >= can_timeout_s:
            requested = 0.0
        requested = limit_q_current_request(
            requested,
            current_command_limit_a=current_command_limit_a,
            regen_power_limit_w=regen_power_limit_w,
            emf_peak_v=emf_peak_v,
            r_phase_ohm=r_phase_ohm,
            regen_capability_per_board_w=regen_capability_per_board_w,
            regen_capability_generation=regen_capability_generation,
            regen_capability_age_s=None if regen_capability_age_s is None else regen_capability_age_s + t,
            regen_capability_max_age_s=regen_capability_max_age_s,
            require_fresh_regen_capability=require_fresh_regen_capability,
        )
        if slew_a_per_s is None:
            ref_a = requested
        else:
            max_step = slew_a_per_s * period
            ref_a += clip(requested - ref_a, -max_step, max_step)
        max_ref_step_a = max(max_ref_step_a, abs(ref_a - previous_ref_a))
        previous_ref_a = ref_a
        measured_a = quantize_current(current_a)
        error_a = ref_a - measured_a
        peak_command_a = max(peak_command_a, abs(ref_a))
        peak_error_a = max(peak_error_a, abs(error_a))
        vlim = max(vbus_v / math.sqrt(3.0), 0.1)
        unsat_vq = emf_peak_v + nominal_r_ohm * ref_a + kp * error_a + integral_v
        new_vq = clip(unsat_vq, -vlim, vlim)
        if abs(new_vq - unsat_vq) > 1e-12:
            sat_samples += 1
        anti_windup_term = kaw * (new_vq - unsat_vq) if anti_windup_back_calculation else 0.0
        integral_v += (ki * error_a + anti_windup_term) * period
        max_abs_integral_v = max(max_abs_integral_v, abs(integral_v))
        seg = segment_metrics.setdefault(schedule_segment, {
            "requested_command_a": schedule[schedule_segment][1],
            "start_s": schedule[schedule_segment][0],
            "sample_count": 0,
            "saturation_samples": 0,
            "max_abs_integrator_v": 0.0,
            "final_reference_a": 0.0,
            "final_current_a": 0.0,
        })
        seg["sample_count"] += 1
        seg["saturation_samples"] += int(abs(new_vq - unsat_vq) > 1e-12)
        seg["max_abs_integrator_v"] = max(seg["max_abs_integrator_v"], abs(integral_v))
        seg["final_reference_a"] = ref_a
        seg["final_current_a"] = current_a

        # Apply prior sample's vector for this PWM period (one-period latency).
        for _ in range(plant_substeps):
            di = (applied_vq - emf_peak_v - r_phase_ohm * current_a) / l_phase_h
            current_a += di * subdt
            max_a = max(max_a, current_a)
            min_a = min(min_a, current_a)
            max_abs_a = max(max_abs_a, abs(current_a))
        applied_vq = new_vq
        last_actual_vq = applied_vq
        if t >= duration_s - 0.010:
            tail_samples.append(current_a)

    return {
        "duration_s": duration_s,
        "r_phase_ohm": r_phase_ohm,
        "l_phase_h": l_phase_h,
        "vbus_v": vbus_v,
        "command_a": command_a,
        "effective_command_after_limits_a": limit_q_current_request(
            command_a,
            current_command_limit_a=current_command_limit_a,
            regen_power_limit_w=regen_power_limit_w,
            emf_peak_v=emf_peak_v,
            r_phase_ohm=r_phase_ohm,
            regen_capability_per_board_w=regen_capability_per_board_w,
            regen_capability_generation=regen_capability_generation,
            regen_capability_age_s=regen_capability_age_s,
            regen_capability_max_age_s=regen_capability_max_age_s,
            require_fresh_regen_capability=require_fresh_regen_capability,
        ),
        "command_schedule": [{"start_s": t, "command_a": q} for t, q in schedule],
        "max_reference_step_a": max_ref_step_a,
        "reference_slew_bound_a": None if slew_a_per_s is None else slew_a_per_s / sample_hz,
        "current_command_limit_a": current_command_limit_a,
        "slew_a_per_s": slew_a_per_s,
        "regen_power_limit_w": regen_power_limit_w,
        "can_timeout_s": can_timeout_s,
        "sample_hz": sample_hz,
        "sample_count": sample_count,
        "plant_substeps_per_sample": plant_substeps,
        "current_limit_a": current_limit_a,
        "measured_current_peak_abs_a": max_abs_a,
        "peak_screen_headroom_a_before_pwm_ripple": current_limit_a - max_abs_a,
        "steady_tail_mean_a": sum(tail_samples) / max(len(tail_samples), 1),
        "steady_tail_peak_abs_a": max((abs(x) for x in tail_samples), default=0.0),
        "iq_min_a": min_a,
        "iq_max_a": max_a,
        "max_command_error_a": peak_error_a,
        "voltage_saturation_sample_fraction": sat_samples / max(sample_count, 1),
        "anti_windup_back_calculation_enabled": anti_windup_back_calculation,
        "anti_windup_max_abs_integrator_v": max_abs_integral_v,
        "anti_windup_segment_metrics": [
            {**metric, "voltage_saturation_fraction": metric["saturation_samples"] / max(metric["sample_count"], 1)}
            for _, metric in sorted(segment_metrics.items())
        ],
        "final_vq_command_v": last_actual_vq,
        "pi_nominal_bandwidth_hz": 1000.0,
        "adc_model": "12-bit, 3.3V, 1.65V zero, 5mV/A, rounded codes",
        "control_delay": "one full 20kHz PWM period",
    }


def current_power_trace(
    *,
    duration_s: float,
    r_phase_ohm: float = 0.010,
    l_phase_h: float = 20e-6,
    command_a: float = -100.0,
    slew_a_per_s: float = 5000.0,
    current_command_limit_a: float | None = None,
    regen_power_limit_w: float | None = None,
    regen_capability_per_board_w: float | None = None,
    regen_capability_generation: int | None = None,
    regen_capability_age_s: float | None = None,
    regen_capability_max_age_s: float = 0.100,
    require_fresh_regen_capability: bool = False,
    can_timeout_s: float | None = None,
    sample_hz: float = 20_000.0,
    plant_substeps: int = 10,
    vbus_for_limit: Callable[[float], float] = lambda _t: 32.4,
) -> list[tuple[float, float, float]]:
    """Return (t, iq, regen_W) on synchronized PWM samples for bus screening."""
    period = 1.0 / sample_hz
    subdt = period / plant_substeps
    omega_c = 2.0 * math.pi * 1000.0
    kp = 20e-6 * omega_c
    ki = 0.017 * omega_c
    kaw = omega_c
    current_a = 0.0
    integral_v = 0.0
    applied_vq = 0.0
    ref_a = 0.0
    rows = []
    for n in range(round(duration_s * sample_hz)):
        t = n * period
        requested = 0.0 if can_timeout_s is not None and t >= can_timeout_s else command_a
        requested = limit_q_current_request(
            requested,
            current_command_limit_a=current_command_limit_a,
            regen_power_limit_w=regen_power_limit_w,
            emf_peak_v=12.83,
            r_phase_ohm=r_phase_ohm,
            regen_capability_per_board_w=regen_capability_per_board_w,
            regen_capability_generation=regen_capability_generation,
            regen_capability_age_s=None if regen_capability_age_s is None else regen_capability_age_s + t,
            regen_capability_max_age_s=regen_capability_max_age_s,
            require_fresh_regen_capability=require_fresh_regen_capability,
        )
        max_step = slew_a_per_s * period
        ref_a += clip(requested - ref_a, -max_step, max_step)
        measured = quantize_current(current_a)
        err = ref_a - measured
        unsat = 12.83 + 0.017 * ref_a + kp * err + integral_v
        vlim = max(vbus_for_limit(t) / math.sqrt(3.0), 0.1)
        new_v = clip(unsat, -vlim, vlim)
        integral_v += (ki * err + kaw * (new_v - unsat)) * period
        # Steady q-axis motor terminal power after copper loss. Positive
        # returned power means the DC bus can receive energy. This excludes
        # transient magnetic energy and mechanical speed change.
        terminal_power_w = 1.5 * (12.83 * current_a + r_phase_ohm * current_a * current_a)
        power_w = max(0.0, -terminal_power_w)
        rows.append((t, current_a, power_w))
        for _ in range(plant_substeps):
            current_a += (applied_vq - 12.83 - r_phase_ohm * current_a) / l_phase_h * subdt
        applied_vq = new_v
    return rows


def regen_bus_case(
    *,
    name: str,
    boards: int,
    external_sink_w: float,
    external_quota_j: float,
    command_a: float = -100.0,
    r_phase_ohm: float = 0.010,
    l_phase_h: float = 20e-6,
    can_timeout_s: float | None = None,
    duration_s: float = 1.0,
    current_command_limit_a: float | None = None,
    regen_power_limit_w: float | None = None,
    regen_capability_per_board_w: float | None = None,
    regen_capability_generation: int | None = None,
    regen_capability_age_s: float | None = None,
    regen_capability_max_age_s: float = 0.100,
    require_fresh_regen_capability: bool = False,
) -> dict:
    """20 kHz bus-energy ledger with hysteretic board-local 1 ohm dump paths."""
    c_bus_f = boards * 0.030
    dt = 1.0 / 20_000.0
    traces = [
        current_power_trace(
            duration_s=duration_s,
            r_phase_ohm=r_phase_ohm,
            l_phase_h=l_phase_h,
            command_a=command_a,
            current_command_limit_a=current_command_limit_a,
            regen_power_limit_w=regen_power_limit_w,
            regen_capability_per_board_w=regen_capability_per_board_w,
            regen_capability_generation=regen_capability_generation,
            regen_capability_age_s=regen_capability_age_s,
            regen_capability_max_age_s=regen_capability_max_age_s,
            require_fresh_regen_capability=require_fresh_regen_capability,
            can_timeout_s=can_timeout_s,
            vbus_for_limit=lambda _t: 32.4,
        )
        for _ in range(boards)
    ]
    vbus = 32.4
    e_bus = 0.5 * c_bus_f * vbus * vbus
    e_sink = 0.0
    e_dump_j = [0.0 for _ in range(boards)]
    active = [False for _ in range(boards)]
    dump_on_s = [0.0 for _ in range(boards)]
    max_v = vbus
    min_v = vbus
    quota_time_s = None
    max_pregen_w = 0.0
    peak_iq = 0.0
    total_generated_j = 0.0
    for k in range(round(duration_s / dt)):
        t = k * dt
        pgen = sum(trace[k][2] for trace in traces)
        peak_iq = max(peak_iq, *(abs(trace[k][1]) for trace in traces))
        max_pregen_w = max(max_pregen_w, pgen)
        total_generated_j += pgen * dt
        available_quota = max(external_quota_j - e_sink, 0.0)
        p_sink = min(external_sink_w, pgen, available_quota / dt)
        if quota_time_s is None and external_quota_j > 0 and e_sink < external_quota_j <= e_sink + p_sink * dt + 1e-12:
            quota_time_s = t + dt
        e_sink += p_sink * dt

        vbus = math.sqrt(max(2.0 * e_bus / c_bus_f, 0.0))
        for j in range(boards):
            if not active[j] and vbus >= 43.415:
                active[j] = True
            elif active[j] and vbus <= 41.415:
                active[j] = False
        p_dump_each = []
        for j in range(boards):
            p_j = (vbus * vbus / (1.0 + 0.01125)) if active[j] else 0.0
            p_dump_each.append(p_j)
            if active[j]:
                e_dump_j[j] += p_j * dt
                dump_on_s[j] += dt
        e_bus += (pgen - p_sink - sum(p_dump_each)) * dt
        e_bus = max(e_bus, 0.0)
        vbus = math.sqrt(2.0 * e_bus / c_bus_f)
        max_v = max(max_v, vbus)
        min_v = min(min_v, vbus)

    return {
        "case": name,
        "boards": boards,
        "duration_s": duration_s,
        "time_step_s": dt,
        "samples_per_second_per_board": 20_000,
        "bus_capacitance_f": c_bus_f,
        "initial_bus_v": 32.4,
        "current_command_a_per_board": command_a,
        "effective_command_after_limits_a_per_board": limit_q_current_request(
            command_a,
            current_command_limit_a=current_command_limit_a,
            regen_power_limit_w=regen_power_limit_w,
            emf_peak_v=12.83,
            r_phase_ohm=r_phase_ohm,
            regen_capability_per_board_w=regen_capability_per_board_w,
            regen_capability_generation=regen_capability_generation,
            regen_capability_age_s=regen_capability_age_s,
            regen_capability_max_age_s=regen_capability_max_age_s,
            require_fresh_regen_capability=require_fresh_regen_capability,
        ),
        "current_command_limit_a": current_command_limit_a,
        "regen_power_limit_w": regen_power_limit_w,
        "regen_capability_per_board_w": regen_capability_per_board_w,
        "regen_capability_generation": regen_capability_generation,
        "regen_capability_age_s_at_start": regen_capability_age_s,
        "regen_capability_max_age_s": regen_capability_max_age_s,
        "require_fresh_regen_capability": require_fresh_regen_capability,
        "current_command_slew_a_per_s": 5000.0,
        "motor_phase_r_ohm": r_phase_ohm,
        "motor_phase_l_h": l_phase_h,
        "power_model": "Pterminal=1.5*(12.83V*iq+R*iq^2); only negative terminal power enters bus ledger, excluding transient L energy and speed change",
        "can_timeout_s": can_timeout_s,
        "external_sink_limit_w": external_sink_w,
        "external_sink_quota_j": external_quota_j,
        "external_sink_used_j": e_sink,
        "external_sink_quota_exhausted_at_s": quota_time_s,
        "regen_energy_into_bus_j": total_generated_j,
        "max_regen_power_w": max_pregen_w,
        "peak_abs_iq_a_per_board": peak_iq,
        "bus_peak_v": max_v,
        "bus_end_v": vbus,
        "dump_energy_per_board_j": e_dump_j,
        "dump_on_time_per_board_s": dump_on_s,
        "dump_resistor_total_r_ohm_per_board": 1.01125,
        "dump_thresholds_v": {"on": 43.415, "off": 41.415},
        "boundary": "fixed speed/back-EMF motor model, one-way source (cannot absorb) except stated finite sink, no mechanical deceleration or dump thermal model",
    }


def buck_ripple(vin_v: float, rt_ohm: float, l_h: float, vout_v: float = 5.0) -> float:
    ton_s = 1.385e-10 * rt_ohm / vin_v
    return max((vin_v - vout_v) * ton_s / l_h, 0.0)


def buck_startup(
    *,
    vin_v: float,
    l_h: float,
    rt_ohm: float,
    c_scale: float = 1.0,
    load_a: float = 0.270,
    overload_step: bool = False,
    load_profile_mode: str = "constant_current",
    startup_load_a: float = 0.05,
    step_at_s: float = 0.010,
    step_duration_s: float = 100e-6,
    duration_s: float = 0.025,
    dt_s: float = 50e-9,
    current_limit_a: float = 0.410,
    include_current_limit_delay: bool = True,
) -> dict:
    """Switching buck startup envelope; ideal COT/diode/limit behavior model.

    The 22uF capacitor is behind 0.82 ohm; direct output includes 0.47uF and
    the WeAct board's nominal 4.7uF+0.1uF VCC input caps. C_SCALE is an explicit
    effective-capacitance sensitivity (not a TDK guarantee).
    """
    rt = rt_ohm
    ton_s = 1.385e-10 * rt / vin_v
    c_direct = (0.47 + 4.7 + 0.1) * 1e-6 * c_scale
    c_branch = 22e-6 * c_scale
    r_ripple = 0.82
    dcr = 0.338
    diode_v = 0.5
    r_l = dcr
    rcl = 1.0e6
    fbref = 2.5
    il = 0.0
    vout = 0.0
    vbranch = 0.0
    t = 0.0
    switch_on = False
    state = "off"
    next_action_s = 0.0
    on_started_s = 0.0
    limit_trip_s: float | None = None
    limit_pending = False
    max_il = 0.0
    min_v = 0.0
    max_v = 0.0
    min_270_window_v = float("inf")
    max_270_window_v = 0.0
    min_overload_v: float | None = None
    max_overload_v: float | None = None
    first_v4p84_s = None
    first_v5p00_s = None
    first_v5p15_s = None
    pulse_count = 0
    current_limit_events = 0
    peak_limit_delay_extra_a = 0.0
    last_i_load = 0.0
    max_load = load_a
    full_load_started = False

    steps = round(duration_s / dt_s)
    for k in range(steps):
        t = k * dt_s
        in_overload = overload_step and step_at_s <= t < step_at_s + step_duration_s
        if load_profile_mode == "startup_50mA_then_full_at_4p84v":
            if vout >= 4.84:
                full_load_started = True
            i_target = load_a if full_load_started else startup_load_a
        else:
            i_target = load_a
        if in_overload:
            i_target = 0.350
        max_load = max(max_load, i_target)
        # The constant-current case bounds early load demand.  A resistor-like
        # ramp is a second envelope for the core-board/LDO startup behavior.
        if load_profile_mode in ("constant_current", "startup_50mA_then_full_at_4p84v"):
            i_load = i_target if vout > 0.02 else 0.0
        elif load_profile_mode == "resistive_5v_equivalent":
            i_load = i_target * clip(vout / 5.0, 0.0, 1.0)
        elif load_profile_mode == "coreboard_uvlo_ramp":
            i_load = i_target * clip((vout - 3.3) / (5.0 - 3.3), 0.0, 1.0)
        else:
            raise ValueError(f"unknown load profile: {load_profile_mode}")
        last_i_load = i_load
        fb = max(vout * 0.5, 0.0)

        if state == "limit_wait" and t >= next_action_s:
            state = "off"
        if state == "min_off" and t >= next_action_s:
            state = "off"
        if state == "on" and limit_pending and limit_trip_s is not None and t >= limit_trip_s:
            switch_on = False
            current_limit_events += 1
            toff = 1.0e-5 / (0.285 + fb / (6.35e-6 * rcl))
            state = "limit_wait"
            next_action_s = t + toff
            limit_pending = False
            limit_trip_s = None
        elif state == "on" and t >= on_started_s + ton_s:
            switch_on = False
            state = "min_off"
            next_action_s = t + 300e-9

        if state == "off" and fb < fbref:
            switch_on = True
            state = "on"
            on_started_s = t
            pulse_count += 1
            limit_pending = False
            limit_trip_s = None

        if switch_on:
            di_dt = (vin_v - vout - r_l * il) / l_h
        else:
            di_dt = -(vout + diode_v + r_l * il) / l_h if il > 0.0 else 0.0
        il_next = max(il + di_dt * dt_s, 0.0)
        max_il = max(max_il, il_next)

        # Peak current detector with a 60ns blanking interval and the TI
        # 350ns response-time upper bound after threshold crossing.
        if switch_on and state == "on" and t - on_started_s >= 70e-9:
            if il_next >= current_limit_a and not limit_pending:
                limit_pending = True
                delay_s = 350e-9 if include_current_limit_delay else 0.0
                limit_trip_s = t + delay_s
                peak_limit_delay_extra_a = max(peak_limit_delay_extra_a, max(il_next - current_limit_a, 0.0))

        i_branch = (vout - vbranch) / r_ripple
        dvout_dt = (il - i_load - i_branch) / c_direct
        dvbranch_dt = i_branch / c_branch
        vout = max(vout + dvout_dt * dt_s, 0.0)
        vbranch = max(vbranch + dvbranch_dt * dt_s, 0.0)
        il = il_next
        min_v = min(min_v, vout)
        max_v = max(max_v, vout)
        if first_v4p84_s is None and vout >= 4.84:
            first_v4p84_s = t
        if first_v5p00_s is None and vout >= 5.00:
            first_v5p00_s = t
        if first_v5p15_s is None and vout >= 5.15:
            first_v5p15_s = t
        if 0.005 <= t <= 0.009:
            min_270_window_v = min(min_270_window_v, vout)
            max_270_window_v = max(max_270_window_v, vout)
        if overload_step and step_at_s <= t < step_at_s + step_duration_s:
            min_overload_v = vout if min_overload_v is None else min(min_overload_v, vout)
            max_overload_v = vout if max_overload_v is None else max(max_overload_v, vout)

    return {
        "vin_v": vin_v,
        "rt_ohm": rt_ohm,
        "inductance_h": l_h,
        "effective_capacitance_scale_assumption": c_scale,
        "c_direct_f": c_direct,
        "c_after_0p82_ohm_f": c_branch,
        "load_a_nominal": load_a,
        "load_profile": load_profile_mode,
        "startup_load_a_until_full_load_transition": startup_load_a if load_profile_mode == "startup_50mA_then_full_at_4p84v" else None,
        "full_load_started": full_load_started,
        "overload_step": {
            "enabled": overload_step,
            "current_a": 0.350 if overload_step else None,
            "start_s": step_at_s if overload_step else None,
            "duration_s": step_duration_s if overload_step else None,
            "vout_min_during_step_v": min_overload_v,
        },
        "dt_s": dt_s,
        "duration_s": duration_s,
        "on_time_s": ton_s,
        "ripple_pp_a_at_5v": buck_ripple(vin_v, rt_ohm, l_h),
        "current_limit_threshold_a": current_limit_a,
        "current_limit_response_included": include_current_limit_delay,
        "max_inductor_current_a": max_il,
        "peak_delay_overshoot_above_threshold_a": peak_limit_delay_extra_a,
        "inductor_saturation_current_datasheet_value_a": 1.45,
        "inductor_saturation_current_typical_a": 1.90,
        "inductor_current_gap_to_listed_saturation_point_a": 1.45 - max_il,
        "current_limit_events": current_limit_events,
        "switching_pulses": pulse_count,
        "vout_min": min_v,
        "vout_max": max_v,
        "time_to_4p84v_s": first_v4p84_s,
        "time_to_5p00v_s": first_v5p00_s,
        "time_to_5p15v_s": first_v5p15_s,
        "vout_5_to_9ms_window_v": [min_270_window_v, max_270_window_v] if min_270_window_v != float("inf") else None,
        "final_vout_v": vout,
        "final_branch_v": vbranch,
        "final_inductor_current_a": il,
    }


def main() -> None:
    current_cases = []
    for slew, name in [(None, "hard_step"), (5000.0, "slew_5000A_per_s")]:
        result = current_loop(
            duration_s=0.120,
            r_phase_ohm=0.017,
            l_phase_h=20e-6,
            slew_a_per_s=slew,
        )
        result["name"] = name
        result["pwm_ripple_bound_pp_a"] = 32.4 / (4.0 * 20_000.0 * 20e-6)
        result["screen_phase_peak_bound_a"] = result["measured_current_peak_abs_a"] + result["pwm_ripple_bound_pp_a"] / 2.0
        result["passes_120a_screen_with_ripple_bound"] = result["screen_phase_peak_bound_a"] <= 120.0
        current_cases.append(result)

    candidate_control_cases = []
    for name, command, current_cap, bus_v, regen_power_limit, capability_w, capability_age, max_age in (
        ("control_q_cap_100A_drive_overlimit_request", 110.0, 100.0, 37.8, None, None, None, 0.100),
        ("control_q_cap_100A_regen_valid_per_board_capability", -110.0, 100.0, 37.8, None, 1800.0, 0.0, 0.250),
        ("control_q_cap_100A_regen_missing_capability_fails_closed", -110.0, 100.0, 37.8, None, None, None, 0.100),
        ("sensitivity_only_q_cap_100A_regen_600W_per_board", -110.0, 100.0, 37.8, None, 600.0, 0.0, 0.250),
    ):
        result = current_loop(
            duration_s=0.120,
            r_phase_ohm=0.017,
            l_phase_h=20e-6,
            vbus_v=bus_v,
            command_a=command,
            slew_a_per_s=5000.0,
            current_command_limit_a=current_cap,
            regen_power_limit_w=regen_power_limit,
            regen_capability_per_board_w=capability_w,
            regen_capability_generation=1 if capability_w is not None else None,
            regen_capability_age_s=capability_age,
            regen_capability_max_age_s=max_age,
            require_fresh_regen_capability=command < 0,
        )
        ripple_pp = bus_v / (4.0 * 20_000.0 * 20e-6)
        result["name"] = name
        result["pwm_ripple_bound_pp_a"] = ripple_pp
        result["screen_phase_peak_bound_a"] = result["measured_current_peak_abs_a"] + ripple_pp / 2.0
        result["passes_120a_screen_with_ripple_bound"] = result["screen_phase_peak_bound_a"] <= 120.0
        result["classification"] = "control-law model case; not hardware qualification"
        result["sensitivity_only"] = name.endswith("600W_per_board")
        candidate_control_cases.append(result)

    bidirectional_transition = current_loop(
        duration_s=0.120,
        r_phase_ohm=0.017,
        l_phase_h=20e-6,
        vbus_v=37.8,
        command_a=110.0,
        command_schedule=[(0.0, 110.0), (0.040, -110.0)],
        slew_a_per_s=5000.0,
        current_command_limit_a=100.0,
        regen_capability_per_board_w=1800.0,
        regen_capability_generation=1,
        regen_capability_age_s=0.0,
        regen_capability_max_age_s=0.250,
        require_fresh_regen_capability=True,
    )
    bidirectional_transition["test"] = "drive_to_regen_command_reversal_with_fresh_per_board_capability"
    bidirectional_transition["passes_reference_slew_bound"] = (
        bidirectional_transition["max_reference_step_a"] <= bidirectional_transition["reference_slew_bound_a"] + 1e-12
    )

    antiwindup_recovery = current_loop(
        duration_s=0.120,
        r_phase_ohm=0.017,
        l_phase_h=20e-6,
        vbus_v=22.5,
        command_a=100.0,
        command_schedule=[(0.0, 100.0), (0.060, 0.0)],
        slew_a_per_s=5000.0,
        current_command_limit_a=100.0,
    )
    antiwindup_recovery["test"] = "low_bus_voltage_saturation_then_zero_command_recovery"
    recovery_segments = antiwindup_recovery["anti_windup_segment_metrics"]
    antiwindup_recovery["saturation_seen_before_recovery"] = bool(recovery_segments and recovery_segments[0]["voltage_saturation_fraction"] > 0.5)
    no_antiwindup_reference = current_loop(
        duration_s=0.120,
        r_phase_ohm=0.017,
        l_phase_h=20e-6,
        vbus_v=22.5,
        command_a=100.0,
        command_schedule=[(0.0, 100.0), (0.060, 0.0)],
        slew_a_per_s=5000.0,
        current_command_limit_a=100.0,
        anti_windup_back_calculation=False,
    )
    antiwindup_recovery["integrator_bounded_vs_no_back_calculation"] = (
        antiwindup_recovery["anti_windup_max_abs_integrator_v"] < no_antiwindup_reference["anti_windup_max_abs_integrator_v"]
    )
    antiwindup_recovery["no_back_calculation_reference_max_abs_integrator_v"] = no_antiwindup_reference["anti_windup_max_abs_integrator_v"]
    antiwindup_recovery["reference_ramped_to_zero"] = abs(antiwindup_recovery["steady_tail_mean_a"]) < 20.0
    antiwindup_recovery["recovery_tail_current_abs_a"] = abs(antiwindup_recovery["steady_tail_mean_a"])

    capability_policy_tests = []
    for name, cap_w, generation, age_s, last_generation, expected_w in (
        ("missing_capability", None, None, None, None, 0.0),
        ("zero_allocation", 0.0, 1, 0.0, None, 0.0),
        ("expired_allocation", 600.0, 1, 0.100001, None, 0.0),
        ("missing_generation", 600.0, None, 0.0, None, 0.0),
        ("replayed_generation_rejected_at_receive", 600.0, 5, 0.0, 5, 0.0),
        ("accepted_generation_reusable_within_ttl", 600.0, 5, 0.050, None, 600.0),
        ("fresh_per_board_allocation", 1800.0, 1, 0.0, None, 1800.0),
    ):
        resolved = resolve_regen_capability(
            per_board_power_w=cap_w,
            generation=generation,
            last_accepted_generation=last_generation,
            age_s=age_s,
            max_age_s=0.100,
        )
        effective_q = limit_q_current_request(
            -110.0,
            current_command_limit_a=100.0,
            regen_power_limit_w=None,
            emf_peak_v=12.83,
            r_phase_ohm=0.017,
            regen_capability_per_board_w=resolved["effective_per_board_power_w"],
            regen_capability_generation=generation,
            regen_capability_age_s=age_s,
            regen_capability_max_age_s=0.100,
            require_fresh_regen_capability=True,
        )
        capability_policy_match = math.isclose(resolved["effective_per_board_power_w"], expected_w, abs_tol=1e-12)
        command_policy_match = effective_q == 0.0 if expected_w == 0.0 else effective_q < 0.0
        capability_policy_tests.append({
            "case": name,
            **resolved,
            "expected_effective_power_w": expected_w,
            "capability_policy_match": capability_policy_match,
            "limited_regen_q_command_a": effective_q,
            "regen_fail_closed_when_unavailable": expected_w == 0.0 and effective_q == 0.0,
            "regen_enabled_when_valid": expected_w > 0.0 and effective_q < 0.0,
            "test_pass": capability_policy_match and command_policy_match,
        })

    cached_generation_pwm_reuse_test = current_loop(
        duration_s=0.120,
        r_phase_ohm=0.017,
        l_phase_h=20e-6,
        vbus_v=37.8,
        command_a=-110.0,
        slew_a_per_s=5000.0,
        current_command_limit_a=100.0,
        regen_capability_per_board_w=600.0,
        regen_capability_generation=5,
        regen_capability_age_s=0.0,
        regen_capability_max_age_s=0.250,
        require_fresh_regen_capability=True,
    )
    cached_generation_pwm_reuse_test["test"] = "accepted_cached_generation_reused_on_each_pwm_until_ttl"
    cached_generation_pwm_reuse_test["same_generation_pwm_interface_has_no_replay_check_argument"] = True
    cached_generation_pwm_reuse_test["test_pass"] = (
        cached_generation_pwm_reuse_test["effective_command_after_limits_a"] < 0.0
        and cached_generation_pwm_reuse_test["steady_tail_mean_a"] < 0.0
        and cached_generation_pwm_reuse_test["max_reference_step_a"] <= cached_generation_pwm_reuse_test["reference_slew_bound_a"] + 1e-12
    )

    ttl_expiry_duration_s = 0.500
    ttl_expiry_ttl_s = 0.100
    ttl_expiry_rows = current_power_trace(
        duration_s=ttl_expiry_duration_s,
        r_phase_ohm=0.010,
        l_phase_h=20e-6,
        command_a=-100.0,
        slew_a_per_s=5000.0,
        current_command_limit_a=100.0,
        regen_capability_per_board_w=1800.0,
        regen_capability_generation=7,
        regen_capability_age_s=0.0,
        regen_capability_max_age_s=ttl_expiry_ttl_s,
        require_fresh_regen_capability=True,
    )
    pwm_period_s = 1.0 / 20_000.0
    ttl_expiry_rows_after_ttl = [row for row in ttl_expiry_rows if row[0] > ttl_expiry_ttl_s]
    ttl_expiry_rows_before_ttl = [row for row in ttl_expiry_rows if row[0] <= ttl_expiry_ttl_s]
    ttl_expiry_case = {
        "case": "per_board_regen_capability_ttl_expiry_under_loaded_minus_100a_request",
        "duration_s": ttl_expiry_duration_s,
        "per_board_power_w_sensitivity_input": 1800.0,
        "allocation_generation": 7,
        "allocation_ttl_s": ttl_expiry_ttl_s,
        "shared_sink_budget_qualified": False,
        "command_a": -100.0,
        "current_command_limit_a": 100.0,
        "slew_a_per_s": 5000.0,
        "phase_r_ohm_sensitivity": 0.010,
        "phase_l_h_sensitivity": 20e-6,
        "model_sample_hz": 20_000,
        "first_pwm_sample_after_expiry_s": ttl_expiry_rows_after_ttl[0][0],
        "current_at_last_sample_within_ttl_a": ttl_expiry_rows_before_ttl[-1][1],
        "peak_abs_current_after_expiry_a": max(abs(row[1]) for row in ttl_expiry_rows_after_ttl),
        "max_regen_power_after_expiry_w": max(row[2] for row in ttl_expiry_rows_after_ttl),
        "regen_energy_before_expiry_j": sum(row[2] for row in ttl_expiry_rows_before_ttl) * pwm_period_s,
        "regen_energy_after_expiry_to_500ms_j": sum(row[2] for row in ttl_expiry_rows_after_ttl) * pwm_period_s,
        "total_regen_energy_to_500ms_j": sum(row[2] for row in ttl_expiry_rows) * pwm_period_s,
        "requested_regen_command_after_expiry_a": 0.0,
        "reference_ramp_down_upper_bound_s_from_minus_100a": 100.0 / 5000.0,
        "current_is_not_instantly_zeroed_by_expiry": True,
        "local_absorber_minimum_energy_for_post_expiry_tail_j": sum(row[2] for row in ttl_expiry_rows_after_ttl) * pwm_period_s,
        "local_absorber_peak_power_for_post_expiry_tail_w": max(row[2] for row in ttl_expiry_rows_after_ttl),
        "closure": "FAIL: request is clamped to zero at capability expiry, but current and regenerated energy persist; real per-board capability and local absorption/isolating path are unqualified",
        "model_boundary": "single-board terminal-power sensitivity only; excludes transient magnetic energy, speed change, actual DC/DC/battery dynamics, and hardware sink qualification",
    }

    shared_sink_allocation_tests = []
    for name, budget_w, qualified, allocations_w, expected_allocations, expected_valid in (
        ("unknown_budget_blocks_both_boards", None, False, [900.0, 900.0], [0.0, 0.0], False),
        ("zero_budget_blocks_both_boards", 0.0, True, [900.0, 900.0], [0.0, 0.0], False),
        ("unqualified_budget_blocks_both_boards", 1800.0, False, [900.0, 900.0], [0.0, 0.0], False),
        ("qualified_budget_normal_allocation", 1800.0, True, [900.0, 900.0], [900.0, 900.0], True),
        ("over_allocation_rejected_for_both_boards", 1800.0, True, [1000.0, 900.0], [0.0, 0.0], False),
    ):
        allocation = validate_shared_sink_allocations(
            per_board_allocations_w=allocations_w,
            qualified_shared_budget_w=budget_w,
            budget_qualified=qualified,
        )
        shared_sink_allocation_tests.append({
            "case": name,
            **allocation,
            "expected_per_board_allocations_w": expected_allocations,
            "expected_valid": expected_valid,
            "test_pass": allocation["valid"] == expected_valid
            and allocation["per_board_allocations_w"] == expected_allocations
            and (not expected_valid or allocation["allocation_sum_within_budget"]),
            "illustrative_logic_test_only": True,
        })

    param_sensitivity = []
    for r_mohm in (10, 25, 50, 100, 200):
        for l_uh in (20, 40, 80, 150, 220, 300):
            for bus in (22.5, 32.4, 37.8):
                for sign in (-1, 1):
                    r = r_mohm * 1e-3
                    l = l_uh * 1e-6
                    full = current_loop(
                        duration_s=0.100,
                        r_phase_ohm=r,
                        l_phase_h=l,
                        vbus_v=bus,
                        command_a=sign * 110.0,
                        slew_a_per_s=5000.0,
                    )
                    ripple_pp = bus / (4.0 * 20_000.0 * l)
                    tail_i = full["steady_tail_mean_a"]
                    pac = 1.5 * (12.83 * tail_i + r * tail_i * tail_i)
                    phase_peak = full["measured_current_peak_abs_a"] + ripple_pp / 2.0
                    param_sensitivity.append({
                        "R_mohm": r_mohm,
                        "L_uH": l_uh,
                        "Vbus_v": bus,
                        "command_a": sign * 110.0,
                        "iq_peak_fundamental_a": full["measured_current_peak_abs_a"],
                        "iq_tail_mean_a": tail_i,
                        "pwm_ripple_bound_pp_a": ripple_pp,
                        "phase_peak_bound_a": phase_peak,
                        "headroom_to_120a_a": 120.0 - phase_peak,
                        "passes_120a_screen": phase_peak <= 120.0,
                        "voltage_saturation_fraction": full["voltage_saturation_sample_fraction"],
                        "terminal_power_tail_w": pac,
                    })

    command_100_grid = []
    for r_mohm in (10, 25, 50, 100, 200):
        for bus in (22.5, 32.4, 37.8):
            for sign in (-1, 1):
                full = current_loop(
                    duration_s=0.100,
                    r_phase_ohm=r_mohm * 1e-3,
                    l_phase_h=20e-6,
                    vbus_v=bus,
                    command_a=sign * 100.0,
                    slew_a_per_s=5000.0,
                )
                ripple_pp = bus / (4.0 * 20_000.0 * 20e-6)
                phase_peak = full["measured_current_peak_abs_a"] + ripple_pp / 2.0
                command_100_grid.append({
                    "R_mohm": r_mohm,
                    "L_uH": 20,
                    "Vbus_v": bus,
                    "command_a": sign * 100.0,
                    "iq_peak_fundamental_a": full["measured_current_peak_abs_a"],
                    "iq_tail_mean_a": full["steady_tail_mean_a"],
                    "pwm_ripple_bound_pp_a": ripple_pp,
                    "phase_peak_bound_a": phase_peak,
                    "headroom_to_120a_a": 120.0 - phase_peak,
                    "passes_120a_screen": phase_peak <= 120.0,
                    "voltage_saturation_fraction": full["voltage_saturation_sample_fraction"],
                })

    voltage_cases = []
    for bus in (22.5, 32.4, 37.8):
        for command in (110.0, -110.0):
            item = current_loop(
                duration_s=0.100,
                r_phase_ohm=0.017,
                l_phase_h=20e-6,
                vbus_v=bus,
                command_a=command,
                slew_a_per_s=5000.0,
            )
            voltage_cases.append(item)

    def evaluate_l_corner_set(l_h: float) -> list[dict]:
        rows = []
        for r_mohm in (10, 25, 50, 100, 200):
            for bus in (22.5, 32.4, 37.8):
                for sign in (-1, 1):
                    full = current_loop(
                        duration_s=0.100,
                        r_phase_ohm=r_mohm * 1e-3,
                        l_phase_h=l_h,
                        vbus_v=bus,
                        command_a=sign * 110.0,
                        slew_a_per_s=5000.0,
                    )
                    ripple_pp = bus / (4.0 * 20_000.0 * l_h)
                    phase_peak = full["measured_current_peak_abs_a"] + ripple_pp / 2.0
                    rows.append({
                        "R_mohm": r_mohm,
                        "L_uH": l_h * 1e6,
                        "Vbus_v": bus,
                        "command_a": sign * 110.0,
                        "iq_peak_fundamental_a": full["measured_current_peak_abs_a"],
                        "pwm_ripple_bound_pp_a": ripple_pp,
                        "phase_peak_bound_a": phase_peak,
                        "headroom_to_120a_a": 120.0 - phase_peak,
                        "passes_120a_screen": phase_peak <= 120.0,
                    })
        return rows

    # Binary-search the lowest effective L in the documented 20..40 uH
    # transition range that passes the full R/Vbus/polarity corner set.
    min_l_low_h = 20e-6
    min_l_high_h = 40e-6
    for _ in range(10):
        l_mid = (min_l_low_h + min_l_high_h) / 2.0
        if max(x["phase_peak_bound_a"] for x in evaluate_l_corner_set(l_mid)) <= 120.0:
            min_l_high_h = l_mid
        else:
            min_l_low_h = l_mid
    min_l_for_110 = min_l_high_h
    min_l_check_cases = evaluate_l_corner_set(min_l_for_110)
    worst_fundamental_peak = max(x["iq_peak_fundamental_a"] for x in min_l_check_cases)

    # The 120 A boundary itself is not useful design margin.  Find the
    # effective phase inductance that leaves at least 3 A under the same
    # full R/Vbus/polarity grid after adding half the analytic PWM ripple.
    practical_l_low_h = 20e-6
    practical_l_high_h = 80e-6
    for _ in range(12):
        l_mid = (practical_l_low_h + practical_l_high_h) / 2.0
        if max(x["phase_peak_bound_a"] for x in evaluate_l_corner_set(l_mid)) <= 117.0:
            practical_l_high_h = l_mid
        else:
            practical_l_low_h = l_mid
    practical_l_for_110 = practical_l_high_h
    practical_l_check_cases = evaluate_l_corner_set(practical_l_for_110)
    candidate_44uh_checks = evaluate_l_corner_set(44e-6)

    command_100 = current_loop(
        duration_s=0.100,
        r_phase_ohm=0.017,
        l_phase_h=20e-6,
        vbus_v=37.8,
        command_a=-100.0,
        slew_a_per_s=5000.0,
    )
    command_100["pwm_ripple_bound_pp_a"] = 37.8 / (4.0 * 20_000.0 * 20e-6)
    command_100["screen_phase_peak_bound_a"] = command_100["measured_current_peak_abs_a"] + command_100["pwm_ripple_bound_pp_a"] / 2.0
    command_100["screen_headroom_to_120a_a"] = 120.0 - command_100["screen_phase_peak_bound_a"]

    command_110_20u_37p8 = current_loop(
        duration_s=0.100,
        r_phase_ohm=0.017,
        l_phase_h=20e-6,
        vbus_v=37.8,
        command_a=-110.0,
        slew_a_per_s=5000.0,
    )
    command_110_20u_37p8["pwm_ripple_bound_pp_a"] = 37.8 / (4.0 * 20_000.0 * 20e-6)
    command_110_20u_37p8["screen_phase_peak_bound_a"] = command_110_20u_37p8["measured_current_peak_abs_a"] + command_110_20u_37p8["pwm_ripple_bound_pp_a"] / 2.0
    command_110_20u_37p8["screen_headroom_to_120a_a"] = 120.0 - command_110_20u_37p8["screen_phase_peak_bound_a"]

    buses = [
        regen_bus_case(name="single_no_absorber_local_dump", boards=1, external_sink_w=0.0, external_quota_j=0.0),
        regen_bus_case(name="single_600W_300J_quota", boards=1, external_sink_w=600.0, external_quota_j=300.0),
        regen_bus_case(name="single_source_disconnected_CAN_timeout", boards=1, external_sink_w=0.0, external_quota_j=0.0, can_timeout_s=0.100),
        regen_bus_case(name="two_boards_shared_60mF_local_dumps", boards=2, external_sink_w=0.0, external_quota_j=0.0),
    ]
    buses[1]["sensitivity_only"] = True
    buses[1]["allocation_note"] = "600W/300J is an external-sink sensitivity case only; it is not a configured or approved per-board capability."

    rt_nom = 280e3
    rt_min = rt_nom * 0.99
    rt_max = rt_nom * 1.01
    l_nom = 220e-6
    l_min = l_nom * 0.80
    ripple_corner = buck_ripple(37.8, rt_max, l_min)
    i_peak_270 = 0.270 + ripple_corner / 2.0
    i_peak_350 = 0.350 + ripple_corner / 2.0
    lm_limit_max = 0.610
    limit_response_s = 350e-9
    # Conservative startup current bound: use VIN as the full inductor voltage
    # during the maximum specified comparator response time (ignore VOUT and
    # winding drops), then compare with Sumida's 1.45 A minimum Isat.
    i_peak_limit_delay_bound = lm_limit_max + 37.8 / l_min * limit_response_s
    fb_min = 2.445
    fb_max = 2.550
    vout_min = fb_min * (1 + (3.01e3 * 0.99) / (3.01e3 * 1.01))
    vout_max = fb_max * (1 + (3.01e3 * 1.01) / (3.01e3 * 0.99))

    buck_cases = [
        buck_startup(vin_v=37.8, l_h=l_min, rt_ohm=rt_max, c_scale=1.0, load_a=0.270),
        buck_startup(vin_v=22.5, l_h=l_min, rt_ohm=rt_max, c_scale=1.0, load_a=0.270),
        buck_startup(vin_v=37.8, l_h=l_min, rt_ohm=rt_max, c_scale=1.0, load_a=0.270, load_profile_mode="resistive_5v_equivalent"),
        buck_startup(vin_v=22.5, l_h=l_min, rt_ohm=rt_max, c_scale=1.0, load_a=0.270, load_profile_mode="resistive_5v_equivalent"),
        buck_startup(vin_v=37.8, l_h=l_min, rt_ohm=rt_max, c_scale=1.0, load_a=0.270, load_profile_mode="coreboard_uvlo_ramp"),
        buck_startup(vin_v=22.5, l_h=l_min, rt_ohm=rt_max, c_scale=1.0, load_a=0.270, load_profile_mode="coreboard_uvlo_ramp"),
        buck_startup(vin_v=37.8, l_h=l_min, rt_ohm=rt_max, c_scale=1.0, load_a=0.270, load_profile_mode="startup_50mA_then_full_at_4p84v"),
        buck_startup(vin_v=22.5, l_h=l_min, rt_ohm=rt_max, c_scale=1.0, load_a=0.270, load_profile_mode="startup_50mA_then_full_at_4p84v"),
        buck_startup(vin_v=37.8, l_h=l_min, rt_ohm=rt_max, c_scale=0.5, load_a=0.270),
        buck_startup(vin_v=37.8, l_h=l_min, rt_ohm=rt_max, c_scale=0.5, load_a=0.270, load_profile_mode="resistive_5v_equivalent"),
        buck_startup(vin_v=37.8, l_h=l_min, rt_ohm=rt_max, c_scale=1.0, load_a=0.270, overload_step=True, load_profile_mode="startup_50mA_then_full_at_4p84v"),
    ]

    # DRV8353 SDO is open-drain.  This RC screen excludes board-specific routing
    # capacitance and the exact SPI sample edge.
    spi_cases = []
    for c_pf in (20.0, 50.0, 100.0):
        tau_s = 4.7e3 * c_pf * 1e-12
        rise70_s = -math.log(0.30) * tau_s
        spi_cases.append({
            "assumed_bus_capacitance_pf": c_pf,
            "pullup_ohm": 4700.0,
            "t_rise_to_0p7VDD_s": rise70_s,
            "sdo_delay_max_s": 30e-9,
            "total_after_clock_edge_s": rise70_s + 30e-9,
            "500ns_half_cycle_margin_s": 500e-9 - rise70_s - 30e-9,
        })

    # Current-sense input filter alternatives. The 1k option is a screen
    # only: use an ADC acquisition window compatible with ST's source-
    # impedance table and include analog-comparator delay in bench tests.
    adc_filter_cases = []
    for r_series_ohm in (47.0, 1000.0):
        tau_s = r_series_ohm * 220e-12
        adc_filter_cases.append({
            "series_resistance_ohm": r_series_ohm,
            "shunt_capacitance_f": 220e-12,
            "rc_time_constant_s": tau_s,
            "t_10_to_90_percent_s": math.log(9.0) * tau_s,
            "settle_to_half_12bit_lsb_after_full_scale_step_s": math.log(2.0 * 4095.0) * tau_s,
            "adc_24p5_cycle_acquisition_at_60mhz_s": 24.5 / 60e6,
            "assumed_csa_linear_output_max_v_at_vref_3p3": 3.05,
            "g431_tt_normal_input_max_v_at_vdd_zero": 0.3,
            "g431_tt_absolute_max_input_v": 4.0,
            "boundary": "STM32G431 TT operating limit is VDD+0.3V (0.3V at VDD=0); 4.0V is absolute maximum only. ST Table 15 states no positive-injection path below the TT maximum input rating. Do not infer powered-off function, rail/reset behavior, or clamp current from these ratings; verify board power sequencing physically. This does not model injection current.",
        })

    result = {
        "generated_by": HERE.name,
        "script_sha256": sha256(HERE),
        "model_status": "executed engineering screen; not MCU firmware, complete PMSM/SVPWM, LM5008A vendor macromodel, thermal model, or bench qualification",
        "current_loop": {
            "legacy_motor_assumptions": {"pole_pairs": 7, "kv_rpm_per_v": 270.0, "speed_rpm": 6000, "phase_back_emf_peak_v": 12.83, "r_phase_ohm": 0.017, "l_phase_h": 20e-6},
            "motor_parameter_sensitivity": "screened R=10/25/50/100/200mΩ, L=20/40/80/150/220/300µH, Vbus=22.5/32.4/37.8V; all are sensitivity points, not measured motor bounds",
            "current_cases": current_cases,
            "parameter_sensitivity": param_sensitivity,
            "command_100a_at_20uh_sensitivity": command_100_grid,
            "command_100a_at_37p8v_17mohm": command_100,
            "command_110a_at_20uh_37p8v": command_110_20u_37p8,
            "bus_voltage_and_polarity_sensitivity": voltage_cases,
            "sample_rate_hz": 20000,
            "limit_screen_a": 120,
            "screened_command_a": 110,
            "reference_control_configuration": {
                "q_current_command_limit_a": 100.0,
                "q_current_slew_limit_a_per_s": 5000.0,
                "pwm_sample_hz": 20000.0,
                "regen_policy": "require fresh per-board allocated sink power; missing, zero, invalid-generation or expired capability clamps negative Iq request to zero",
                "capability_contract": {
                    "power_field": "per_board_power_w, allocated from the shared battery/DC-DC budget before publication",
                    "generation_field": "positive monotonic generation; reject duplicate/replayed/out-of-order generation at update reception, then cache the accepted sample; reusing that accepted generation during its TTL is allowed",
                    "ttl_field": "age_s <= max_age_s; check the cached accepted sample each PWM cycle without re-running the receive-layer replay check; hardware TTL remains to be implemented and bench-verified",
                    "unallocated_shared_capacity_is_not_reused_per_board": True,
                    "shared_sink_allocation_qualified": False,
                },
                "status": "candidate model configuration, not deployed firmware or hardware qualification",
            },
            "current_limit_boundary_is_screen_only": True,
            "pwm_ripple_envelope": "worst-case two-level duty ripple ΔIpp≤Vbus/(4*fPWM*L); add half the bound to fundamental peak, assuming centered synchronous sample; not a switching simulation",
            "110a_fundamental_peak_worst_grid_a": worst_fundamental_peak,
            "110a_minimum_effective_l_model_only_h": min_l_for_110,
            "110a_minimum_l_checks": min_l_check_cases,
            "110a_minimum_effective_l_for_3a_headroom_h": practical_l_for_110,
            "110a_minimum_l_for_3a_headroom_checks": practical_l_check_cases,
            "110a_44uh_candidate_checks": candidate_44uh_checks,
            "110a_l_means_effective_inductance_at_test_operating_point": "measured phase inductance after saturation/temperature/rotor-angle effects; motor not yet received",
            "synchronous_sampling": "one sample per PWM period, abstract centered sample; 12-bit ADC quantization included",
            "local_control_policy": "legacy unbounded 110A/100A screening cases remain unchanged to preserve the prior stress FAILs. The reference configuration applies ±100A q-current request clamping before a 5000A/s slew limiter. That is a control model, not a hardware fast clamp; CAN timeout ramps the reference to zero after 100ms.",
            "regenerative_power_limit_policy": "negative Iq is enabled only by a fresh positive per-board power allocation with generation and TTL. The allocation must already be divided across both FOC boards by the shared-sink owner; missing/zero/stale input fails closed. The 600W input is a sensitivity only, never a default or approved rating.",
            "candidate_control_validation": {
                "overlimit_drive_case": next(x for x in candidate_control_cases if x["name"] == "control_q_cap_100A_drive_overlimit_request"),
                "missing_regen_capability_case": next(x for x in candidate_control_cases if x["name"] == "control_q_cap_100A_regen_missing_capability_fails_closed"),
                "bidirectional_transition": bidirectional_transition,
                "antiwindup_saturation_recovery": antiwindup_recovery,
                "capability_policy_tests": capability_policy_tests,
                "cached_generation_pwm_reuse_test": cached_generation_pwm_reuse_test,
                "ttl_expiry_residual_energy_case": ttl_expiry_case,
                "shared_sink_allocation_tests": shared_sink_allocation_tests,
                "status": "model behavior tests only; no measured motor R/L, firmware execution, hardware sampling or sink allocation evidence",
            },
            "command_tracking_boundary": "peak-envelope PASS only means modeled peak is below 120A; it does not prove the command was tracked. With fixed 12.83V back-EMF and 22.5V bus, positive drive can be voltage saturated; see per-case voltage_saturation_sample_fraction and iq_tail_mean_a",
            "important_boundary": "average dq plant, one update-period delay; no switching/dead-time/Hall angle error/CSA offset. PI gains remain based on nominal 17mΩ/20µH; R/L are unknown until motor arrival. PWM envelope is conservative analytic bound, to verify on scope",
        },
        "candidate_control_cases": candidate_control_cases,
        "bidirectional_transition_test": bidirectional_transition,
        "antiwindup_saturation_recovery_test": antiwindup_recovery,
        "regen_capability_policy_tests": capability_policy_tests,
        "cached_generation_pwm_reuse_test": cached_generation_pwm_reuse_test,
        "ttl_expiry_residual_energy_case": ttl_expiry_case,
        "shared_sink_allocation_tests": shared_sink_allocation_tests,
        "shared_sink_budget_qualification": {
            "qualified": False,
            "per_board_regen_capability_must_be_preallocated": True,
            "cross_board_sum_check_tested": True,
            "actual_battery_dc_dc_absorption_budget_w": None,
        },
        "bus_energy_cases": buses,
        "buck_5v": {
            "architecture": "retain DRV8353R integrated LM5008A buck, pending startup/recovery bench confirmation",
            "rt_nominal_ohm": rt_nom,
            "rt_range_1pct_ohm": [rt_min, rt_max],
            "inductor_nominal_h": l_nom,
            "inductor_minus20pct_h": l_min,
            "lm5008a_min_current_limit_a": 0.410,
            "lm5008a_max_current_limit_a": lm_limit_max,
            "lm5008a_current_limit_response_max_s": limit_response_s,
            "startup_peak_current_upper_bound_a": i_peak_limit_delay_bound,
            "startup_peak_current_upper_bound_method": "Ilimit,max + VIN,max/Lmin * response_delay_max; conservative because it ignores VOUT, diode and winding drops during the delay",
            "inductor_current_gap_to_listed_saturation_point_after_limit_delay_bound_a": 1.45 - i_peak_limit_delay_bound,
            "ripple_worst_corner_pp_a": ripple_corner,
            "steady_peak_at_270ma_a": i_peak_270,
            "steady_min_limit_margin_at_270ma_a": 0.410 - i_peak_270,
            "overload_peak_at_350ma_a": i_peak_350,
            "steady_min_limit_margin_at_350ma_a": 0.410 - i_peak_350,
            "inductor_saturation_current_datasheet_table_label": "Max (Typ.)",
            "inductor_saturation_current_datasheet_value_a": 1.45,
            "inductor_saturation_current_typical_a": 1.90,
            "inductor_saturation_current_definition": "Sumida defines this point where inductance falls 25% from initial value; the 1.45A table value is not used as a thermal-rise rating or relabeled as an Isat minimum",
            "inductor_current_gap_to_listed_saturation_point_over_610ma_limit_a": 1.45 - 0.610,
            "steady_output_corner_v": {"minimum": vout_min, "maximum": vout_max, "method": "FB 2.445..2.550V and equal 3.01kΩ ±1% resistor ratio"},
            "startup_and_step_cases": buck_cases,
            "spi_sdo_rise_cases": spi_cases,
            "current_sense_adc_filter_cases": adc_filter_cases,
            "schematic_topology": {"c_direct_nominal_f": 0.47e-6 + 4.7e-6 + 0.1e-6, "c_ripple_branch_nominal_f": 22e-6, "r_ripple_ohm": 0.82, "coreboard_vcc_caps_source": "WeAct Normal QFN48 V1.0 schematic, nominal 4.7uF+100nF input decoupling"},
            "excluded_startup_load": "WeAct 3.3V LDO output 10uF and its nonlinear start/load response; its 5V-side current was not quantified",
            "capacitor_derating": "0.5 scale is a sensitivity only; exact effective capacitance vs bias/tolerance must be confirmed from TDK data and WeAct board BOM",
            "datasheets": [
                "https://www.ti.com/lit/ds/symlink/lm5008a.pdf",
                "https://www.ti.com/lit/ds/symlink/drv8353.pdf",
                "https://products.sumida.com/products/pdf/CDRH127LD_Ver2.0_E.pdf",
                "https://product.tdk.com/en/search/capacitor/ceramic/mlcc/info?part_no=C3225X7R1C226K250AC",
            ],
            "usb_boundary": "WeAct VBUS→VCC BAT54 path and external 5V reverse path are not electrically proved; prohibit powered USB insertion with coreboard installed until backfeed test",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "script_sha256": result["script_sha256"], "cases": len(current_cases) + len(candidate_control_cases) + len(param_sensitivity) + len(buses) + len(buck_cases), "buck_corner": {k: result["buck_5v"][k] for k in ("ripple_worst_corner_pp_a", "steady_peak_at_270ma_a", "steady_min_limit_margin_at_270ma_a", "overload_peak_at_350ma_a", "steady_output_corner_v")}, "current": [{"name": c["name"], "peak_a": c["measured_current_peak_abs_a"], "tail_a": c["steady_tail_mean_a"], "sat_fraction": c["voltage_saturation_sample_fraction"]} for c in current_cases + candidate_control_cases], "bus": [{"case": b["case"], "vmax": b["bus_peak_v"], "sink_j": b["external_sink_used_j"], "dump_j": b["dump_energy_per_board_j"]} for b in buses], "buck_start": [{"vin": b["vin_v"], "c_scale": b["effective_capacitance_scale_assumption"], "t_4p84_ms": None if b["time_to_4p84v_s"] is None else round(b["time_to_4p84v_s"] * 1e3, 3), "vmax": b["vout_max"], "max_il": b["max_inductor_current_a"], "overload_vmin": b["overload_step"]["vout_min_during_step_v"]} for b in buck_cases]}, indent=2))


if __name__ == "__main__":
    main()
