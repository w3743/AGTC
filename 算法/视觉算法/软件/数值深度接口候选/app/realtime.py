"""Lightweight range overlays and board resource sampling for the 8088 UI."""
from __future__ import annotations

import os
from pathlib import Path
import re
import time

import cv2
import numpy as np


def _box_candidates(depth: np.ndarray, limit: int = 6) -> list[dict]:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= 0.3) & (depth <= 5.0)
    if int(valid.sum()) < 100:
        return []
    values = depth[valid]
    near_limit = min(3.5, max(0.65, float(np.percentile(values, 25)) * 1.2))
    mask = (valid & (depth <= near_limit)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = depth.size
    minimum_area = max(18, int(image_area * 0.0005))
    maximum_area = int(image_area * 0.40)
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < minimum_area or area > maximum_area or width < 5 or height < 5:
            continue
        selected = (labels[y:y + height, x:x + width] == label)
        region = depth[y:y + height, x:x + width][selected]
        region = region[np.isfinite(region) & (region >= 0.3) & (region <= 5.0)]
        if region.size < minimum_area:
            continue
        candidates.append(dict(x=x, y=y, width=width, height=height,
                               area=area, depth_m=float(np.median(region))))
    candidates.sort(key=lambda item: (item["depth_m"], -item["area"]))
    return candidates[:limit]


def make_measure_frame(gray: np.ndarray, depth: np.ndarray, sequence: int) -> tuple[np.ndarray, list[dict]]:
    """Draw depth-derived near-field regions; these boxes do not classify objects."""
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.shape != depth.shape:
        raise ValueError("Measurement view requires a matching rectified GRAY8/depth pair")
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    boxes = _box_candidates(depth)
    for index, box in enumerate(boxes, start=1):
        x, y = box["x"], box["y"]
        x2, y2 = x + box["width"], y + box["height"]
        cv2.rectangle(frame, (x, y), (x2, y2), (64, 235, 172), 1, cv2.LINE_AA)
        cv2.putText(frame, f"OBJ {index} {box['depth_m']:.2f}m", (x, max(13, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (64, 235, 172), 1, cv2.LINE_AA)
    cv2.putText(frame, f"DEPTH RANGE BOXES  seq {int(sequence)}",
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 245, 248), 1, cv2.LINE_AA)
    return frame, boxes


def measure_pixel(depth: np.ndarray, x: int, y: int, projector,
                  depth_factor=None, disparity_offset=None) -> dict:
    depth = np.asarray(depth, dtype=np.float32)
    xyz = projector.pixel_to_xyz(depth, int(x), int(y), depth_factor, disparity_offset)
    if xyz is None:
        return {"ok": False, "message": "所选位置没有可靠深度，请点击有纹理且非黑色区域"}
    xx, yy, zz = xyz
    return dict(ok=True, x=int(x), y=int(y), depth_z_m=round(zz, 4),
                range_m=round(float(np.sqrt(xx * xx + yy * yy + zz * zz)), 4),
                camera_xyz_m=[round(xx, 4), round(yy, 4), round(zz, 4)],
                coordinate_system="camera: X right, Y down, Z forward; metres")


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(errors="replace")
    except (OSError, PermissionError):
        return None


def _read_cpu() -> dict[str, tuple[int, int]]:
    text = _read_text("/proc/stat") or ""
    result = {}
    for line in text.splitlines():
        match = re.match(r"^(cpu\d*)\s+(.+)$", line)
        if not match:
            if result:
                break
            continue
        values = [int(value) for value in match.group(2).split()[:8]]
        if len(values) < 4:
            continue
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        result[match.group(1)] = (total, idle)
    return result


def _memory_kb() -> dict[str, int]:
    result = {}
    for line in (_read_text("/proc/meminfo") or "").splitlines():
        match = re.match(r"^(MemTotal|MemAvailable|SwapTotal|SwapFree):\s+(\d+)", line)
        if match:
            result[match.group(1)] = int(match.group(2))
    return result


def _process_memory(pid: int) -> dict:
    status = _read_text(f"/proc/{pid}/status") or ""
    values = {}
    for line in status.splitlines():
        match = re.match(r"^(VmRSS|VmHWM|VmSize):\s+(\d+)\s+kB", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    smaps = _read_text(f"/proc/{pid}/smaps_rollup") or ""
    match = re.search(r"^Pss:\s+(\d+)\s+kB", smaps, re.MULTILINE)
    if match:
        values["Pss"] = int(match.group(1))
    return values


def _process_dmabuf(pid: int) -> dict:
    text = _read_text("/sys/kernel/debug/dma_buf/bufinfo")
    if not text:
        return {"bytes": None, "objects": None, "exporters": {}}
    total = 0
    count = 0
    exporters = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 7 or not fields[0].isdigit():
            continue
        if not any(re.search(rf"(?:^|\D){pid}(?:-|$)", field) for field in fields[6:]):
            continue
        size, exporter = int(fields[0]), fields[4]
        total += size
        count += 1
        exporters[exporter] = exporters.get(exporter, 0) + size
    return {"bytes": total, "objects": count, "exporters": exporters}


def _devfreq(prefix: str, *, include_load: bool = True) -> list[dict]:
    results = []
    for base in Path("/sys/class/devfreq").glob("*"):
        if prefix.lower() not in base.name.lower():
            continue
        try:
            current = int((base / "cur_freq").read_text().strip())
        except (OSError, ValueError):
            current = None
        load = None
        load_text = _read_text(base / "load") if include_load else None
        if load_text:
            match = re.search(r"(\d+)\s*@", load_text)
            if match:
                load = int(match.group(1))
        results.append(dict(name=base.name, frequency_hz=current, load_percent=load))
    return results


_NPU_LOAD_PATH = "/sys/kernel/debug/rknpu/load"
_NPU_POWER_PATH = "/sys/kernel/debug/rknpu/power"
_NPU_VERSION_PATH = "/sys/kernel/debug/rknpu/version"
_NPU_CORE_COUNT = 3


def _parse_npu_load(raw: str | None) -> dict:
    """Parse the driver's explicitly labelled per-core load counters."""
    core_loads = [None] * _NPU_CORE_COUNT
    errors = []
    if raw is None:
        return dict(status="unavailable", load_percent=core_loads,
                    average_percent=None, max_percent=None,
                    error="unreadable")

    seen = set()
    for field in raw.split(","):
        if not re.search(r"(?<!\w)Core", field, re.IGNORECASE):
            continue
        match = re.search(r"(?<!\w)Core\s*(\d+)\s*:\s*(.*?)\s*$", field,
                          re.IGNORECASE)
        if match is None:
            errors.append("malformed core field")
            continue
        core_index = int(match.group(1))
        if core_index < 0 or core_index >= _NPU_CORE_COUNT:
            errors.append(f"unexpected Core{core_index}")
            continue
        if core_index in seen:
            core_loads[core_index] = None
            errors.append(f"duplicate Core{core_index}")
            continue
        seen.add(core_index)

        value_match = re.fullmatch(r"(\d+)%", match.group(2).strip())
        if value_match is None:
            errors.append(f"invalid Core{core_index} value")
            continue
        value = int(value_match.group(1))
        if not 0 <= value <= 100:
            errors.append(f"out-of-range Core{core_index} value")
            continue
        core_loads[core_index] = value

    for core_index, value in enumerate(core_loads):
        if value is None and core_index not in seen:
            errors.append(f"missing Core{core_index}")

    complete = all(value is not None for value in core_loads) and not errors
    valid_count = sum(value is not None for value in core_loads)
    status = "ok" if complete else ("partial" if valid_count else "unavailable")
    average = round(sum(core_loads) / _NPU_CORE_COUNT, 1) if complete else None
    maximum = max(core_loads) if complete else None
    return dict(status=status, load_percent=core_loads,
                average_percent=average, max_percent=maximum,
                error="; ".join(errors) or None)


class HardwareSampler:
    """Read only proc/sysfs counters; missing counters stay explicitly unavailable."""

    CACHE_INTERVAL_S = 1.0

    def __init__(self):
        self.last_cpu = None
        self._cached_snapshot = None
        self._cached_monotonic = None
        self._cached_pid = None
        self._sequence = 0

    def snapshot(self, pid: int) -> dict:
        now = time.monotonic()
        if (self._cached_snapshot is not None and self._cached_monotonic is not None
                and pid == self._cached_pid
                and now - self._cached_monotonic < self.CACHE_INTERVAL_S):
            result = dict(self._cached_snapshot)
            result["sample_age_ms"] = round(max(0.0, now - self._cached_monotonic) * 1000, 1)
            return result

        current = _read_cpu()
        core_percent = {}
        if self.last_cpu:
            for name, (total, idle) in current.items():
                if name not in self.last_cpu:
                    continue
                old_total, old_idle = self.last_cpu[name]
                elapsed = total - old_total
                if elapsed > 0:
                    core_percent[name] = round(100.0 * (elapsed - (idle - old_idle)) / elapsed, 1)
        self.last_cpu = current
        memory = _memory_kb()
        total_kb = memory.get("MemTotal")
        available_kb = memory.get("MemAvailable")
        used_percent = (None if not total_kb or available_kb is None else
                        round(100.0 * (total_kb - available_kb) / total_kb, 1))

        temperatures = []
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            zone_type = (_read_text(zone / "type") or zone.name).strip()
            raw = _read_text(zone / "temp")
            try:
                value = int(raw.strip())
                celsius = value / 1000.0 if abs(value) > 1000 else float(value)
                if -20 <= celsius <= 150:
                    temperatures.append(dict(name=zone_type, celsius=round(celsius, 1)))
            except (AttributeError, ValueError):
                continue

        npu_text = _read_text(_NPU_LOAD_PATH)
        npu_load = _parse_npu_load(npu_text)
        power_text = _read_text(_NPU_POWER_PATH)
        power = power_text.strip().lower() if power_text else None
        if power not in ("on", "off"):
            power = None
        version_text = _read_text(_NPU_VERSION_PATH)
        version = version_text.strip() or None if version_text is not None else None

        self._sequence += 1
        result = dict(
            sampled_at=time.time(),
            sampled_monotonic=now,
            sample_seq=self._sequence,
            sample_age_ms=0.0,
            cpu_cores=core_percent,
            cpu_average_percent=(round(sum(core_percent.values()) / len(core_percent), 1)
                                 if core_percent else None),
            memory=dict(total_kb=total_kb, available_kb=available_kb,
                        used_percent=used_percent, swap_total_kb=memory.get("SwapTotal"),
                        swap_free_kb=memory.get("SwapFree")),
            service_process=dict(_process_memory(pid), dma_buf=_process_dmabuf(pid)),
            gpu=_devfreq("gpu"),
            npu=dict(npu_load, raw=npu_text, source=_NPU_LOAD_PATH,
                     power=power, driver_version=version,
                     frequency=_devfreq("npu", include_load=False)),
            temperatures=temperatures,
            notes={"gpu_memory": "Mali GPU dedicated memory is not exposed by this BSP; DMA-BUF buffers are shared RAM",
                   "npu_memory": "NPU private memory counter is not exposed; process RSS/PSS is total process RAM, not NPU memory",
                   "npu_zero_load": "A zero reading may mean idle or a driver counter limitation"},
        )
        self._cached_snapshot = result
        self._cached_monotonic = now
        self._cached_pid = pid
        return dict(result)
