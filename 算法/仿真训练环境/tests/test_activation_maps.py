"""Tests for lossless activation mosaics and diverging pseudocolor."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_field.visualization.activation_maps import (
    COLOR_MAP_LUT,
    COLOR_MAP_NAME,
    DETECTION_LABELS,
    SPECTRUM_STOPS,
    STATE_LABELS,
    activation_image,
    colorize,
    tensor_mosaic,
)


def _assert_source_values_are_recoverable(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mosaic, index_grid = tensor_mosaic(values)
    valid = index_grid >= 0
    source_indices = index_grid[valid]
    assert source_indices.dtype == np.int64
    np.testing.assert_array_equal(np.sort(source_indices), np.arange(values.size))
    np.testing.assert_array_equal(
        mosaic[valid], np.asarray(values, dtype=np.float64).reshape(-1)[source_indices]
    )
    assert np.all(np.isnan(mosaic[~valid]))
    return mosaic, index_grid


def test_scalar_vector_matrix_and_extreme_row_keep_every_source_index():
    scalar, scalar_indices = tensor_mosaic(np.asarray(7.5))
    np.testing.assert_array_equal(scalar, [[7.5]])
    np.testing.assert_array_equal(scalar_indices, [[0]])

    vector = np.arange(13, dtype=np.float32) - 4.0
    vector_grid, vector_indices = _assert_source_values_are_recoverable(vector)
    assert vector_grid.shape[0] > 1 and vector_grid.shape[1] > 1

    matrix = np.arange(15, dtype=np.float64).reshape(3, 5)
    matrix_grid, matrix_indices = tensor_mosaic(matrix)
    assert matrix_grid.shape == matrix.shape
    np.testing.assert_array_equal(matrix_grid, matrix)
    np.testing.assert_array_equal(matrix_indices, np.arange(matrix.size).reshape(matrix.shape))

    long_row = np.arange(257, dtype=np.float32).reshape(1, -1)
    row_grid, row_indices = _assert_source_values_are_recoverable(long_row)
    assert row_grid.shape[0] > 1 and row_grid.shape[1] > 1
    assert row_indices.shape == row_grid.shape


def test_channel_tiles_preserve_flat_mapping_and_mark_only_layout_gaps():
    values = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    mosaic, indices = _assert_source_values_are_recoverable(values)

    # Five 2x3 maps fit in a 2x3 tile grid, with a one-pixel separator.
    assert mosaic.shape == (5, 11)
    assert np.all(indices[:, 3] == -1)
    assert np.all(indices[2, :] == -1)
    assert indices[0, 0] == 0
    assert indices[0, 4] == 6
    assert indices[3, 4] == 24

    higher_rank = np.arange(2 * 3 * 2 * 4, dtype=np.int32).reshape(2, 3, 2, 4)
    _assert_source_values_are_recoverable(higher_rank)


def test_turbo_colormap_samples_standard_full_spectrum_rgb_stops():
    assert COLOR_MAP_NAME == "turbo"
    assert COLOR_MAP_LUT.shape == (256, 3)
    assert COLOR_MAP_LUT.dtype == np.uint8
    np.testing.assert_array_equal(
        COLOR_MAP_LUT[[0, 64, 128, 191, 255]],
        np.asarray(
            (
                (48, 18, 59),
                (40, 188, 235),
                (164, 252, 60),
                (251, 129, 34),
                (122, 4, 3),
            ),
            dtype=np.uint8,
        ),
    )
    assert SPECTRUM_STOPS == (
        (48, 18, 59),
        (70, 107, 227),
        (40, 188, 235),
        (50, 242, 152),
        (164, 252, 60),
        (238, 207, 58),
        (251, 126, 33),
        (208, 47, 5),
        (122, 4, 3),
    )


def test_colorize_maps_signed_zero_to_spectrum_midpoint_and_nonfinite_to_magenta():
    values = np.asarray([-2.0, -0.0, 1.0, np.nan, np.inf, -np.inf])
    rgb, limit = colorize(values)

    assert limit == 2.0
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0], (48, 18, 59))
    np.testing.assert_array_equal(rgb[1], (164, 252, 60))
    np.testing.assert_array_equal(rgb[2], (251, 129, 34))
    np.testing.assert_array_equal(rgb[3:], np.tile((255, 0, 255), (3, 1)))

    zeros_rgb, zeros_limit = colorize(np.zeros((2, 3), dtype=np.float32))
    assert zeros_limit == 1.0
    np.testing.assert_array_equal(zeros_rgb, np.broadcast_to((164, 252, 60), (2, 3, 3)))


def test_fixed_scale_saturates_by_absolute_magnitude_and_rejects_invalid_limits():
    rgb, limit = colorize(np.asarray([-10.0, -1.0, 1.0, 10.0]), scale=2.0)

    assert limit == 2.0
    np.testing.assert_array_equal(
        rgb,
        np.asarray(
            ((48, 18, 59), (40, 188, 235), (251, 129, 34), (122, 4, 3)),
            dtype=np.uint8,
        ),
    )
    for invalid_scale in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            colorize(np.asarray([1.0]), scale=invalid_scale)


def test_activation_image_keeps_source_nan_distinct_from_artificial_gaps():
    values = np.asarray([1.0, np.nan, -1.0, 2.0, 0.0])
    rgb, indices, limit = activation_image(values, scale=2.0)

    assert limit == 2.0
    assert np.all(rgb[indices == -1] == (0, 0, 0))
    np.testing.assert_array_equal(rgb[indices == 1][0], (255, 0, 255))
    assert np.count_nonzero(indices >= 0) == values.size


def test_sensor_label_tables_match_policy_contract_lengths_and_landmarks():
    assert len(STATE_LABELS) == 33
    assert len(DETECTION_LABELS) == 8
    assert STATE_LABELS[0] == "任务模式：0搜索、1到点"
    assert STATE_LABELS[1:3] == (
        "目标相对车体前向分量（按场地尺寸缩放；仅到点模式）",
        "目标相对车体侧向分量（按场地尺寸缩放；仅到点模式）",
    )
    assert STATE_LABELS[7:11] == (
        "上一动作：左履带速度指令",
        "上一动作：右履带速度指令",
        "上一动作：云台偏航角速度指令",
        "上一动作：云台俯仰角速度指令",
    )
    assert STATE_LABELS[11] == "已用时间 / 时限（上限 1）"
    assert STATE_LABELS[12] == "LiDAR 帧龄 / 2 秒（上限 1）"
    assert not any("车体横滚角" in label or "车体俯仰角" in label or "测得车体" in label for label in STATE_LABELS)
    assert not any("噪声配置" in label for label in STATE_LABELS)
    assert DETECTION_LABELS[6] == "目标距离 / 深度量程"
    assert DETECTION_LABELS[7] == "距离有效标记"
    assert not any("候选存在掩码" in label for label in DETECTION_LABELS)
