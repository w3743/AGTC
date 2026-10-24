"""CPU-only contract tests for device-resident batched sensor captures."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import isaac_field.tensor_sensors as tensor_sensors
from isaac_field.tensor_sensors import capture_depth_batch, capture_lidar_batch


def _reference_lidar(ideal, rng, *, min_range, max_range, relative_error):
    """The NumPy operations used by the existing IsaacFieldEnv._capture."""
    ideal = np.asarray(ideal)
    valid = np.isfinite(ideal) & (ideal >= min_range) & (ideal < max_range)
    values = np.full(ideal.shape, max_range, np.float32)
    values[valid] = np.clip(
        ideal[valid]
        * (1 + rng.uniform(-relative_error, relative_error, valid.sum())),
        min_range,
        max_range,
    )
    return values, valid


def _reference_depth(raw, rng, *, min_range, max_range, noise_std):
    """The NumPy operations used by the existing IsaacFieldEnv._capture."""
    raw = np.asarray(raw)
    valid = np.isfinite(raw) & (raw >= min_range) & (raw <= max_range)
    values = np.full_like(raw, max_range)
    values[valid] = np.clip(
        raw[valid] + rng.normal(0, noise_std, valid.sum()), min_range, max_range
    )
    return values, valid


def _rngs(seeds):
    return [np.random.default_rng(seed) for seed in seeds]


def test_lidar_matches_capture_reference_and_preserves_rng_progression():
    raw = torch.tensor(
        [
            [0.12, 0.2, 1.0, 8.0, float("nan"), float("inf"), -float("inf")],
            [0.0, 0.12, 7.999, 3.5, 8.0, 0.4, 0.5],
            [1.2, 2.3, 4.5, 6.7, 0.11, 8.01, 8.0],
        ],
        dtype=torch.float32,
    )
    seeds = [121, 122, 123]
    actual_rngs = _rngs(seeds)
    expected_rngs = _rngs(seeds)
    due_ids = [2, 0]

    frames = capture_lidar_batch(
        raw,
        due_ids,
        actual_rngs,
        min_range=0.12,
        max_range=8.0,
        relative_error=0.05,
    )

    assert list(frames) == due_ids
    for env_id in due_ids:
        expected_values, expected_valid = _reference_lidar(
            raw[env_id].numpy(),
            expected_rngs[env_id],
            min_range=0.12,
            max_range=8.0,
            relative_error=0.05,
        )
        frame = frames[env_id]
        assert frame["_tensor_sensor_frame"] is True
        assert frame["ranges"].shape == raw[env_id].shape
        assert frame["valid"].shape == raw[env_id].shape
        assert frame["ranges"].dtype == torch.float32
        assert frame["valid"].dtype == torch.bool
        assert frame["ranges"].device == raw.device
        assert frame["valid"].device == raw.device
        np.testing.assert_array_equal(frame["valid"].numpy(), expected_valid)
        np.testing.assert_array_equal(frame["ranges"].numpy(), expected_values)
    for env_id in range(len(seeds)):
        np.testing.assert_array_equal(actual_rngs[env_id].random(8), expected_rngs[env_id].random(8))


def test_depth_matches_capture_reference_and_keeps_inclusive_far_boundary():
    raw = torch.tensor(
        [
            [[0.3, 0.4, 5.0], [float("nan"), float("inf"), -float("inf")]],
            [[0.29, 0.31, 4.9], [5.01, 1.25, 2.75]],
        ],
        dtype=torch.float32,
    )
    seeds = [921, 922]
    actual_rngs = _rngs(seeds)
    expected_rngs = _rngs(seeds)
    due_ids = [1, 0]

    frames = capture_depth_batch(
        raw,
        due_ids,
        actual_rngs,
        min_range=0.3,
        max_range=5.0,
        noise_std=0.01,
    )

    assert list(frames) == due_ids
    for env_id in due_ids:
        expected_values, expected_valid = _reference_depth(
            raw[env_id].numpy(),
            expected_rngs[env_id],
            min_range=0.3,
            max_range=5.0,
            noise_std=0.01,
        )
        frame = frames[env_id]
        assert frame["_tensor_sensor_frame"] is True
        assert frame["values"].shape == raw[env_id].shape
        assert frame["valid"].shape == raw[env_id].shape
        assert frame["values"].dtype == torch.float32
        assert frame["valid"].dtype == torch.bool
        assert frame["values"].device == raw.device
        assert frame["valid"].device == raw.device
        np.testing.assert_array_equal(frame["valid"].numpy(), expected_valid)
        np.testing.assert_array_equal(frame["values"].numpy(), expected_values)
    for env_id in range(len(seeds)):
        np.testing.assert_array_equal(actual_rngs[env_id].random(8), expected_rngs[env_id].random(8))


def test_lidar_then_depth_preserves_per_environment_rng_draw_order():
    lidar = torch.tensor(
        [[0.2, 8.0, 0.4], [0.0, 0.5, 1.0]], dtype=torch.float32
    )
    depth = torch.tensor(
        [[[0.3, 0.4], [5.0, 0.1]], [[0.29, 0.5], [4.9, 2.0]]],
        dtype=torch.float32,
    )
    seeds = [51, 52]
    actual_rngs = _rngs(seeds)
    expected_rngs = _rngs(seeds)

    lidar_frames = capture_lidar_batch(
        lidar, [0, 1], actual_rngs,
        min_range=0.12, max_range=8.0, relative_error=0.05,
    )
    depth_frames = capture_depth_batch(
        depth, [0, 1], actual_rngs,
        min_range=0.3, max_range=5.0, noise_std=0.01,
    )
    for env_id in range(2):
        expected_lidar, _ = _reference_lidar(
            lidar[env_id].numpy(), expected_rngs[env_id],
            min_range=0.12, max_range=8.0, relative_error=0.05,
        )
        expected_depth, _ = _reference_depth(
            depth[env_id].numpy(), expected_rngs[env_id],
            min_range=0.3, max_range=5.0, noise_std=0.01,
        )
        np.testing.assert_array_equal(lidar_frames[env_id]["ranges"].numpy(), expected_lidar)
        np.testing.assert_array_equal(depth_frames[env_id]["values"].numpy(), expected_depth)
        np.testing.assert_array_equal(actual_rngs[env_id].random(8), expected_rngs[env_id].random(8))


def test_due_mask_empty_valid_sets_and_zero_noise_preserve_random_streams():
    raw_lidar = torch.tensor([[8.0, float("nan")], [0.2, 0.3]], dtype=torch.float32)
    raw_depth = torch.tensor([[[5.1, float("nan")]], [[0.4, 1.0]]], dtype=torch.float32)
    seeds = [711, 712]
    actual_rngs = _rngs(seeds)
    expected_rngs = _rngs(seeds)

    lidar_frames = capture_lidar_batch(
        raw_lidar, [0], actual_rngs,
        min_range=0.12, max_range=8.0, relative_error=0.0,
    )
    depth_frames = capture_depth_batch(
        raw_depth, [1], actual_rngs,
        min_range=0.3, max_range=5.0, noise_std=0.0,
    )
    assert list(lidar_frames) == [0]
    assert not lidar_frames[0]["valid"].any()
    assert list(depth_frames) == [1]
    assert depth_frames[1]["valid"].all()
    _reference_lidar(
        raw_lidar[0].numpy(), expected_rngs[0],
        min_range=0.12, max_range=8.0, relative_error=0.0,
    )
    _reference_depth(
        raw_depth[1].numpy(), expected_rngs[1],
        min_range=0.3, max_range=5.0, noise_std=0.0,
    )
    for env_id in range(2):
        np.testing.assert_array_equal(actual_rngs[env_id].random(8), expected_rngs[env_id].random(8))

    assert capture_lidar_batch(
        raw_lidar, [], actual_rngs,
        min_range=0.12, max_range=8.0, relative_error=0.05,
    ) == {}
    assert capture_depth_batch(
        raw_depth, [], actual_rngs,
        min_range=0.3, max_range=5.0, noise_std=0.01,
    ) == {}


def test_capture_frames_do_not_alias_reused_sensor_buffers():
    raw_lidar = torch.tensor([[0.25, 0.5, 8.0]], dtype=torch.float32)
    raw_depth = torch.tensor([[[0.5, 1.0], [2.0, 5.0]]], dtype=torch.float32)
    lidar_rngs, depth_rngs = _rngs([31]), _rngs([32])

    lidar_frame = capture_lidar_batch(
        raw_lidar, [0], lidar_rngs,
        min_range=0.12, max_range=8.0, relative_error=0.0,
    )[0]
    depth_frame = capture_depth_batch(
        raw_depth, [0], depth_rngs,
        min_range=0.3, max_range=5.0, noise_std=0.0,
    )[0]
    frozen_lidar = lidar_frame["ranges"].clone()
    frozen_depth = depth_frame["values"].clone()
    raw_lidar.fill_(7.0)
    raw_depth.fill_(4.0)
    torch.testing.assert_close(lidar_frame["ranges"], frozen_lidar, rtol=0, atol=0)
    torch.testing.assert_close(depth_frame["values"], frozen_depth, rtol=0, atol=0)


def test_frames_share_fresh_batched_output_storage_without_input_aliasing():
    raw = torch.tensor([[0.25, 0.5], [0.75, 1.0]], dtype=torch.float32)
    frames = capture_lidar_batch(
        raw, [0, 1], _rngs([61, 62]),
        min_range=0.12, max_range=8.0, relative_error=0.0,
    )
    first_storage = frames[0]["ranges"].untyped_storage().data_ptr()
    second_storage = frames[1]["ranges"].untyped_storage().data_ptr()
    assert first_storage == second_storage
    assert first_storage != raw.untyped_storage().data_ptr()


def test_only_valid_count_vectors_cross_the_host_boundary(monkeypatch):
    seen = []
    original = tensor_sensors._valid_counts_to_host

    def record_count_transfer(counts):
        seen.append((tuple(counts.shape), counts.numel(), counts.dtype))
        return original(counts)

    monkeypatch.setattr(tensor_sensors, "_valid_counts_to_host", record_count_transfer)
    rngs = _rngs([1, 2, 3])
    capture_lidar_batch(
        torch.ones((3, 5), dtype=torch.float32), [2, 0], rngs,
        min_range=0.12, max_range=8.0, relative_error=0.05,
    )
    capture_depth_batch(
        torch.ones((3, 4, 7), dtype=torch.float32), [1], rngs,
        min_range=0.3, max_range=5.0, noise_std=0.01,
    )
    assert seen == [((2,), 2, torch.int64), ((1,), 1, torch.int64)]


def test_tensor_helpers_have_no_raw_or_mask_host_conversion_path():
    source = inspect.getsource(tensor_sensors)
    tree = ast.parse(source)
    cpu_transfer_functions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"cpu", "numpy"}:
            cpu_transfer_functions.append(node.func.attr)
        if node.func.attr == "to":
            keywords = {kw.arg: kw.value for kw in node.keywords}
            device_node = keywords.get("device")
            if isinstance(device_node, ast.Constant) and device_node.value == "cpu":
                owner = next(
                    parent.name
                    for parent in ast.walk(tree)
                    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node in list(ast.walk(parent))
                )
                assert owner == "_valid_counts_to_host"
    assert not cpu_transfer_functions
    assert '.to(device="cpu")' in source
    assert "raw[valid]" not in source
    assert "values[valid]" not in source
    assert "torch.cat(" not in source
    assert "torch.nonzero(" not in source


def test_existing_capture_source_still_defines_the_numpy_reference_contract():
    env_path = Path(__file__).parents[1] / "isaac_field" / "env.py"
    source = env_path.read_text(encoding="utf-8")
    assert "valid=np.isfinite(ideal)&(ideal>=c.lidar_min_range)&(ideal<c.lidar_range)" in source
    assert "t.np_random.uniform(-c.lidar_relative_error,c.lidar_relative_error,valid.sum())" in source
    assert "valid=np.isfinite(raw)&(raw>=c.depth_min_range)&(raw<=c.depth_range)" in source
    assert "t.np_random.normal(0,c.range_noise,valid.sum())" in source
