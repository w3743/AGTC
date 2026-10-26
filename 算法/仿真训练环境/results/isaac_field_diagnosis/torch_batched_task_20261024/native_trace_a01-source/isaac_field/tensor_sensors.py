"""Narrow batched tensor helpers for LiDAR and depth sensor capture.

The simulator keeps the large sensor buffers on their source device.  This
module performs validity, clipping, and sensor noise application there while
retaining the per-environment NumPy random streams used by ``FieldTask``.
Only one valid-element count per due environment crosses back to the host.

This module intentionally has no Isaac Sim, task, or policy imports.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch


TensorSensorFrame: TypeAlias = dict[str, torch.Tensor | bool]
TensorSensorFrames: TypeAlias = dict[int, TensorSensorFrame]


def _validate_limits(min_range: float, max_range: float) -> tuple[float, float]:
    low, high = float(min_range), float(max_range)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= low:
        raise ValueError("sensor range limits must be finite, positive, and ordered")
    return low, high


def _due_ids(due_env_ids: Sequence[int] | torch.Tensor, batch_size: int) -> list[int]:
    """Validate host-side due IDs without moving sensor data off-device."""
    if isinstance(due_env_ids, torch.Tensor):
        if due_env_ids.device.type != "cpu":
            raise ValueError("due_env_ids must be a CPU tensor")
        if due_env_ids.ndim != 1:
            raise ValueError("due_env_ids must be one-dimensional")
        ids = [int(value) for value in due_env_ids.tolist()]
    else:
        ids = [int(value) for value in due_env_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("due_env_ids must not contain duplicates")
    if any(env_id < 0 or env_id >= batch_size for env_id in ids):
        raise IndexError("due environment ID is outside the sensor batch")
    return ids


def _due_raw_batch(raw_batch: torch.Tensor, due_ids: list[int], expected_ndim: int,
                   name: str) -> torch.Tensor:
    if not isinstance(raw_batch, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if raw_batch.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dimensions including batch")
    if not torch.is_floating_point(raw_batch):
        raise TypeError(f"{name} must have a floating-point dtype")
    if not due_ids:
        return raw_batch.new_empty((0, *raw_batch.shape[1:]), dtype=torch.float32)
    indices = torch.as_tensor(due_ids, dtype=torch.long, device=raw_batch.device)
    # index_select owns fresh storage.  If a dtype conversion is needed, .to()
    # also allocates; for the normal float32 SDK path this avoids a second copy.
    return raw_batch.detach().index_select(0, indices).to(dtype=torch.float32)


def _valid_counts_to_host(valid_counts: torch.Tensor) -> list[int]:
    """Transfer only the small per-environment count vector to CPU."""
    return [int(count) for count in valid_counts.detach().to(device="cpu").tolist()]


def _make_noise_host(due_ids: list[int], valid_counts: list[int],
                     np_rngs: Sequence[np.random.Generator], *,
                     noise_kind: str, noise_a: float, noise_b: float) -> np.ndarray:
    """Draw compact per-environment noise on CPU in legacy RNG order."""
    if due_ids and len(np_rngs) <= max(due_ids):
        raise ValueError("np_rngs must contain one generator per environment ID")
    noise_host = np.empty(sum(valid_counts), dtype=np.float64)
    cursor = 0
    for env_id, count in zip(due_ids, valid_counts, strict=True):
        if count == 0:
            # The reference capture makes an empty NumPy draw, which consumes no
            # state. Skipping it avoids an unnecessary bridge and preserves state.
            continue
        rng = np_rngs[env_id]
        if noise_kind == "uniform":
            noise = rng.uniform(noise_a, noise_b, size=count)
        elif noise_kind == "normal":
            noise = rng.normal(noise_a, noise_b, size=count)
        else:  # Internal programming error rather than a public option.
            raise ValueError(f"unsupported noise kind {noise_kind!r}")
        noise_host[cursor:cursor + count] = noise
        cursor += count
    return noise_host


def _noise_rank_lookup(noise_host: np.ndarray, valid: torch.Tensor) -> torch.Tensor:
    """Upload compact noise once and expand it with fixed-shape device ops.

    The cumulative rank starts at one for the first valid element.  Noise index
    zero is a sentinel, so all-invalid tensors safely gather an all-zero noise
    tensor without a data-dependent output shape or host synchronization.
    """
    if noise_host.size:
        padded_host = np.empty(noise_host.size + 1, dtype=np.float64)
        padded_host[0] = 0.0
        padded_host[1:] = noise_host
        noise_table = torch.as_tensor(padded_host, dtype=torch.float64, device=valid.device)
    else:
        # No valid samples means no NumPy draws and no host-to-device noise copy.
        noise_table = torch.zeros(1, dtype=torch.float64, device=valid.device)
    ranks = valid.reshape(-1).to(dtype=torch.int64).cumsum(dim=0)
    return noise_table.gather(0, ranks)


def capture_lidar_batch(
    ranges: torch.Tensor,
    due_env_ids: Sequence[int] | torch.Tensor,
    np_rngs: Sequence[np.random.Generator],
    *,
    min_range: float,
    max_range: float,
    relative_error: float,
) -> TensorSensorFrames:
    """Capture due LiDAR rows into independent device-resident sensor frames.

    ``ranges`` is the raw ``[N, R]`` GPU sensor buffer.  Valid rays satisfy
    ``finite & (range >= min_range) & (range < max_range)``.  Valid readings
    receive the legacy per-environment uniform multiplicative noise; invalid
    readings are filled with ``max_range``. The returned frames retain the
    device and use the legacy ``ranges``/``valid`` field names.
    """
    low, high = _validate_limits(min_range, max_range)
    error = float(relative_error)
    if not np.isfinite(error) or not 0.0 <= error < 1.0:
        raise ValueError("relative_error must be finite and in [0, 1)")
    due_ids = _due_ids(due_env_ids, ranges.shape[0] if isinstance(ranges, torch.Tensor) and ranges.ndim else 0)
    if not isinstance(ranges, torch.Tensor) or ranges.ndim != 2:
        raise ValueError("ranges must have shape [N, R]")
    raw = _due_raw_batch(ranges, due_ids, 2, "ranges")
    if not due_ids:
        return {}

    valid = torch.isfinite(raw) & (raw >= low) & (raw < high)
    counts = _valid_counts_to_host(valid.sum(dim=1, dtype=torch.int64))
    noise_host = _make_noise_host(
        due_ids, counts, np_rngs, noise_kind="uniform", noise_a=-error, noise_b=error,
    )
    noise = _noise_rank_lookup(noise_host, valid)
    raw_flat = raw.reshape(-1).to(dtype=torch.float64)
    values_flat = (raw_flat * (1.0 + noise)).clamp(min=low, max=high).to(torch.float32)
    # This is a fixed-shape elementwise fill; it does not materialize a dynamic
    # nonzero result or move mask data to the host.
    values_flat.masked_fill_(~valid.reshape(-1), high)
    values = values_flat.reshape_as(raw)

    frames: TensorSensorFrames = {}
    for row, env_id in enumerate(due_ids):
        frames[env_id] = {
            # Row views keep the fresh batched output storage alive and avoid
            # per-environment copies. Neither tensor aliases the SDK input.
            "ranges": values[row],
            "valid": valid[row],
            "_tensor_sensor_frame": True,
        }
    return frames


def capture_depth_batch(
    depth_z: torch.Tensor,
    due_env_ids: Sequence[int] | torch.Tensor,
    np_rngs: Sequence[np.random.Generator],
    *,
    min_range: float,
    max_range: float,
    noise_std: float,
) -> TensorSensorFrames:
    """Capture due camera-Z depth rows into independent device-resident frames.

    ``depth_z`` is the raw ``[N, H, W]`` sensor buffer in metres. Valid pixels
    satisfy ``finite & (z >= min_range) & (z <= max_range)``. Valid pixels
    receive the legacy per-environment additive normal noise; invalid pixels
    are filled with ``max_range``. The returned fields are ``values`` and
    ``valid`` with the batch axis removed.
    """
    low, high = _validate_limits(min_range, max_range)
    std = float(noise_std)
    if not np.isfinite(std) or std < 0.0:
        raise ValueError("noise_std must be finite and non-negative")
    due_ids = _due_ids(
        due_env_ids,
        depth_z.shape[0] if isinstance(depth_z, torch.Tensor) and depth_z.ndim else 0,
    )
    if not isinstance(depth_z, torch.Tensor) or depth_z.ndim != 3:
        raise ValueError("depth_z must have shape [N, H, W]")
    raw = _due_raw_batch(depth_z, due_ids, 3, "depth_z")
    if not due_ids:
        return {}

    valid = torch.isfinite(raw) & (raw >= low) & (raw <= high)
    counts = _valid_counts_to_host(valid.reshape(valid.shape[0], -1).sum(dim=1, dtype=torch.int64))
    noise_host = _make_noise_host(
        due_ids, counts, np_rngs, noise_kind="normal", noise_a=0.0, noise_b=std,
    )
    noise = _noise_rank_lookup(noise_host, valid)
    raw_flat = raw.reshape(-1).to(dtype=torch.float64)
    values_flat = (raw_flat + noise).clamp(min=low, max=high).to(torch.float32)
    values_flat.masked_fill_(~valid.reshape(-1), high)
    values = values_flat.reshape_as(raw)

    frames: TensorSensorFrames = {}
    for row, env_id in enumerate(due_ids):
        frames[env_id] = {
            "values": values[row],
            "valid": valid[row],
            "_tensor_sensor_frame": True,
        }
    return frames
