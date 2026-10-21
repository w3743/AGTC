"""Lossless layouts and color maps for viewing live policy activations."""

from __future__ import annotations

import math

import numpy as np
from matplotlib import colormaps

from ..action_contract import ACTION_LABELS

COLOR_MAP_NAME = "turbo"
COLOR_MAP_LUT = np.rint(
    colormaps[COLOR_MAP_NAME](np.linspace(0.0, 1.0, 256))[:, :3] * 255.0
).astype(np.uint8)
COLOR_MAP_LUT.setflags(write=False)
_SPECTRUM_STOP_INDICES = np.minimum(
    np.floor(np.linspace(0.0, 1.0, 9) * len(COLOR_MAP_LUT)).astype(int),
    len(COLOR_MAP_LUT) - 1,
)
SPECTRUM_STOPS = tuple(tuple(int(channel) for channel in COLOR_MAP_LUT[index])
                       for index in _SPECTRUM_STOP_INDICES)


STATE_LABELS = (
    "任务模式：搜索标记",
    "任务模式：到点标记",
    "目标相对车体前向分量（按场地尺寸缩放；仅到点模式）",
    "目标相对车体侧向分量（按场地尺寸缩放；仅到点模式）",
    "估计车体 X 位置（按场地尺寸映射）",
    "估计车体 Y 位置（按场地尺寸映射）",
    "车体航向正弦",
    "车体航向余弦",
    "测得车体前向速度 / 最大速度",
    "测得车体偏航角速度 / 最大转向率",
    "云台偏航正弦",
    "云台偏航余弦",
    "云台俯仰角 / π",
    *(f"上一动作：{label}" for label in ACTION_LABELS),
    "已用时间 / 时限（上限 1）",
    "任务步长 dt（秒）",
    "目标容差 / 场地尺寸",
    "LiDAR 帧龄 / 2 秒（上限 1）",
    "LiDAR 新帧标记",
    "LiDAR 帧位置差 X / LiDAR 量程",
    "LiDAR 帧位置差 Y / LiDAR 量程",
    "LiDAR 帧时车体航向差 / π",
    "LiDAR 帧时云台偏航差 / π",
    "LiDAR 帧时云台俯仰差 / π",
    "深度图帧龄 / 2 秒（上限 1）",
    "深度图新帧标记",
    "深度图帧位置差 X / LiDAR 量程",
    "深度图帧位置差 Y / LiDAR 量程",
    "深度图帧时车体航向差 / π",
    "深度图帧时云台偏航差 / π",
    "深度图帧时云台俯仰差 / π",
    "相机帧龄 / 2 秒（上限 1）",
    "相机新帧标记",
    "相机帧位置差 X / LiDAR 量程",
    "相机帧位置差 Y / LiDAR 量程",
    "相机帧时车体航向差 / π",
    "相机帧时云台偏航差 / π",
    "相机帧时云台俯仰差 / π",
    "车体横滚角 / π",
    "车体俯仰角 / π",
)


DETECTION_LABELS = (
    "框中心 X / 图像宽度",
    "框中心 Y / 图像高度",
    "框宽 / 图像宽度",
    "框高 / 图像高度",
    "类别 ID",
    "置信度",
    "目标距离 / 深度量程",
    "距离有效标记",
    "候选存在掩码",
    "保留字段（当前为 0）",
)


def _as_real_array(values: np.ndarray) -> np.ndarray:
    """Return a non-empty real-valued array as float64 for NaN-capable layouts."""
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError("values must contain at least one element")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError("values must have a real numeric dtype")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("complex values are not supported")
    return array.astype(np.float64, copy=False)


def _near_square_layout(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lay a C-order vector into a near-square grid, padding with NaN gaps."""
    flat = values.reshape(-1)
    columns = math.ceil(math.sqrt(flat.size))
    rows = math.ceil(flat.size / columns)
    mosaic = np.full((rows, columns), np.nan, dtype=np.float64)
    index_grid = np.full((rows, columns), -1, dtype=np.int64)
    mosaic.flat[: flat.size] = flat
    index_grid.flat[: flat.size] = np.arange(flat.size, dtype=np.int64)
    return mosaic, index_grid


def tensor_mosaic(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make a lossless 2D mosaic and a C-order source-index map.

    Scalars become ``1×1``. Vectors use a near-square grid. Matrices retain
    their shape except for very long one-dimensional matrices, which are laid
    out like vectors. Higher-rank arrays tile their last two dimensions, with
    the preceding dimensions flattened into the channel axis. One NaN pixel
    separates tiles; ``index_grid == -1`` marks all artificial gaps/padding.
    """
    array = _as_real_array(values)
    if array.ndim == 0:
        return array.reshape(1, 1), np.zeros((1, 1), dtype=np.int64)
    if array.ndim == 1:
        return _near_square_layout(array)
    if array.ndim == 2:
        height, width = array.shape
        if min(height, width) == 1 and max(height, width) >= 16:
            return _near_square_layout(array)
        return array.copy(), np.arange(array.size, dtype=np.int64).reshape(array.shape)

    height, width = array.shape[-2:]
    channels = array.reshape(-1, height, width)
    channel_count = channels.shape[0]
    tile_columns = math.ceil(math.sqrt(channel_count))
    tile_rows = math.ceil(channel_count / tile_columns)
    mosaic_height = tile_rows * height + (tile_rows - 1)
    mosaic_width = tile_columns * width + (tile_columns - 1)
    mosaic = np.full((mosaic_height, mosaic_width), np.nan, dtype=np.float64)
    index_grid = np.full((mosaic_height, mosaic_width), -1, dtype=np.int64)

    for channel in range(channel_count):
        tile_row, tile_column = divmod(channel, tile_columns)
        row_start = tile_row * (height + 1)
        column_start = tile_column * (width + 1)
        rows = slice(row_start, row_start + height)
        columns = slice(column_start, column_start + width)
        mosaic[rows, columns] = channels[channel]
        flat_start = channel * height * width
        index_grid[rows, columns] = np.arange(
            flat_start, flat_start + height * width, dtype=np.int64
        ).reshape(height, width)

    return mosaic, index_grid


def colorize(
    values: np.ndarray, *, scale: float | None = None
) -> tuple[np.ndarray, float]:
    """Map signed values across Matplotlib's full-spectrum ``turbo`` colormap.

    If ``scale`` is omitted, the limit is the largest finite absolute value
    (or 1 for all-zero/all-non-finite inputs). A supplied scale is used as-is;
    values outside ``[-scale, scale]`` saturate at the endpoint colors. Zero
    maps to the midpoint of the spectrum. Non-finite input values are magenta.
    """
    array = _as_real_array(values)
    if scale is None:
        finite = np.isfinite(array)
        limit = float(np.max(np.abs(array[finite]))) if np.any(finite) else 0.0
        if limit == 0.0:
            limit = 1.0
    else:
        try:
            limit = float(scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("scale must be finite and positive") from exc
        if not math.isfinite(limit) or limit <= 0.0:
            raise ValueError("scale must be finite and positive")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = np.clip(array / limit, -1.0, 1.0)
    finite = np.isfinite(array)
    safe_normalized = np.where(finite, normalized, 0.0)
    lut_indices = np.rint((safe_normalized + 1.0) * 0.5 * (len(COLOR_MAP_LUT) - 1)).astype(np.intp)
    rgb = COLOR_MAP_LUT[lut_indices].copy()
    rgb[~finite] = (255, 0, 255)
    return rgb, limit


def activation_image(
    values: np.ndarray, *, scale: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return RGB activation mosaic, source-index map, and color limit.

    Artificial NaN separators/padding are rendered black; non-finite source
    elements retain ``colorize``'s magenta color.
    """
    mosaic, index_grid = tensor_mosaic(values)
    rgb, limit = colorize(mosaic, scale=scale)
    rgb[index_grid < 0] = (0, 0, 0)
    return rgb, index_grid, limit


__all__ = [
    "COLOR_MAP_LUT",
    "COLOR_MAP_NAME",
    "DETECTION_LABELS",
    "SPECTRUM_STOPS",
    "STATE_LABELS",
    "activation_image",
    "colorize",
    "tensor_mosaic",
]
