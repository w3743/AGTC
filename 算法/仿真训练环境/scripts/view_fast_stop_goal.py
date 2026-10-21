"""Paused official-Viser preview with transactional, map-click goal selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import zipfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIAL = ROOT / "results/isaac_field_improvement/fast_stop512_mastery_20261021_attempt2"
DEFAULT_OUTPUT = ROOT / "results/isaac_field_viewer/fast_stop512_click_goal_20261021"
GOAL_RADIUS_M = 0.4
MAP_Z = 0.0
_RAY_EPS = 1.0e-9
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"缺少证据文件：{path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取证据文件 {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"证据文件不是JSON对象：{path}")
    return result


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if (protocol.get("schema") != "isaac-field-mastery-experiment-v1"
            or protocol.get("method") != "mastery"
            or protocol.get("profile") != "fast_stop_512"):
        raise ValueError("trial protocol不是fast_stop_512 mastery profile")
    cfg = protocol.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("protocol缺少config")
    required = {"plants": 0, "terrain_enabled": False, "rock_density": 0.0,
                "goal_tolerance": 0.4, "goal_hold": 0.5, "goal_stop_speed": 0.05,
                "goal_stop_angular_speed": 0.1, "goto_reward_profile": "fast_stop_v1",
                "validate_scene": True}
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"protocol {key}={cfg.get(key)!r}，预览合同要求 {expected!r}")
    near = next((row for row in protocol.get("stages", [])
                 if isinstance(row, dict) and row.get("name") == "near"), None)
    if (near is None or near.get("mode") != "goto" or near.get("plants") != 0
            or list(near.get("goal_distance", [])) != [0.6, 1.2]
            or not math.isclose(float(near.get("goal_angle", math.nan)),
                                math.radians(15), rel_tol=0, abs_tol=1e-12)):
        raise ValueError("protocol缺少near 0.6–1.2m / ±15° goto阶段")
    if not isinstance(protocol.get("profile_physics_contract"), dict):
        raise ValueError("protocol缺少前导向轮与稳定性物理合同")


def _verify_sources(trial: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    snapshot = protocol.get("source_snapshot")
    rows = snapshot.get("files") if isinstance(snapshot, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("protocol缺少source_snapshot.files冻结源清单")
    verified: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source_snapshot含非法条目")
        rel, snapshot_rel, expected = row.get("path"), row.get("snapshot_path"), row.get("sha256")
        if (not isinstance(rel, str) or not rel or Path(rel).is_absolute()
                or not isinstance(snapshot_rel, str) or not isinstance(expected, str)
                or len(expected) != 64):
            raise ValueError(f"source_snapshot条目缺少路径或SHA-256：{row!r}")
        if rel in seen:
            raise ValueError(f"source_snapshot重复路径：{rel}")
        seen.add(rel)
        live, frozen = _resolve(rel), _resolve(snapshot_rel, trial)
        if not _inside(frozen, trial):
            raise ValueError(f"source_snapshot路径越界：{snapshot_rel}")
        for candidate in (live, frozen):
            if not candidate.is_file():
                raise ValueError(f"冻结源文件缺失：{candidate}")
            actual = _sha256(candidate)
            if actual != expected:
                raise ValueError(f"冻结源SHA不匹配：{candidate} expected={expected} actual={actual}")
        verified.append({"path": rel.replace("\\", "/"), "sha256": expected})
    verified.sort(key=lambda item: item["path"])
    return {"file_count": len(verified), "files": verified, "source_sha256": _stable_hash(verified)}


def _checkpoint_record(path: Path, expected: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise ValueError(f"检查点不存在、尚未完成或不是SB3 ZIP：{path}")
    before = path.stat()
    actual = _sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"检查点SHA不匹配：{path} expected={expected} actual={actual}")
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            names = set(archive.namelist())
            data = json.loads(archive.read("data").decode("utf-8"))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"检查点ZIP损坏或仍在写入：{path}: {exc}") from exc
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"检查点ZIP缺少有效SB3 data文件：{path}: {exc}") from exc
    required = {"data", "policy.pth", "policy.optimizer.pth", "pytorch_variables.pth",
                "_stable_baselines3_version", "system_info.txt"}
    if bad is not None or not required.issubset(names):
        raise ValueError(f"检查点ZIP成员不完整：{path}; missing={sorted(required - names)}")
    kwargs = data.get("policy_kwargs", {})
    timesteps = data.get("num_timesteps")
    if (isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps <= 0
            or float(data.get("gamma", math.nan)) != 1.0
            or not isinstance(kwargs, dict) or kwargs.get("lstm_hidden_size") != 512
            or kwargs.get("n_lstm_layers") != 1):
        raise ValueError(f"检查点不是已训练的gamma=1、LSTM 512x1策略：{path}")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or _sha256(path) != actual:
        raise ValueError(f"检查点读取期间仍在变化：{path}")
    return {"path": str(path), "sha256": actual, "bytes": after.st_size,
            "num_timesteps": timesteps, "gamma": 1.0, "lstm_hidden_size": 512,
            "n_lstm_layers": 1}


def _manifest_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    blocks = manifest.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("checkpoint_manifest.blocks缺失")
    rows = {}
    for row in blocks:
        if isinstance(row, dict) and isinstance(row.get("checkpoint"), str):
            digest = row.get("sha256")
            if isinstance(digest, str) and len(digest) == 64:
                rows[row["checkpoint"].replace("\\", "/")] = row
    return rows


def _resolve_checkpoint(trial: Path, protocol: dict[str, Any], selection: str,
                        explicit: Path | None = None) -> dict[str, Any]:
    manifest = _json(trial / "checkpoint_manifest.json")
    rows = _manifest_rows(manifest)
    if explicit is not None:
        raw = explicit.expanduser()
        trial_candidate = _resolve(raw, trial)
        path = trial_candidate if not raw.is_absolute() and trial_candidate.exists() else _resolve(raw)
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = ""
        row = rows.get(rel)
        if row is None:
            raise ValueError("显式检查点没有checkpoint_manifest中的SHA记录")
        record = _checkpoint_record(path, row["sha256"])
        if row.get("num_timesteps") != record["num_timesteps"]:
            raise ValueError("显式检查点num_timesteps与manifest不一致")
        return {**record, "selection": "explicit", "label": "显式检查点",
                "num_timesteps": row["num_timesteps"]}
    if selection == "latest":
        key = manifest.get("latest_checkpoint")
        if not isinstance(key, str) or key.replace("\\", "/") not in rows:
            raise ValueError("manifest.latest_checkpoint没有对应训练block SHA")
        summary_path = trial / "summary.json"
        if summary_path.exists():
            summary = _json(summary_path)
            if summary.get("latest_checkpoint") != key or summary.get("training_finished") is not True:
                raise ValueError("summary与manifest的latest checkpoint不一致或训练未完成")
        row = rows[key.replace("\\", "/")]
        record = _checkpoint_record(_resolve(key), row["sha256"])
        if row.get("num_timesteps") != record["num_timesteps"]:
            raise ValueError("manifest.latest_checkpoint num_timesteps与ZIP内容不一致")
        return {**record, "selection": "latest", "label": "最新已保存检查点",
                "num_timesteps": row["num_timesteps"]}
    if selection != "near-best":
        raise ValueError("selection必须是latest或near-best")

    contract_sha: str | None = None
    reset_signature: str | None = None
    expected_seeds = protocol.get("mastery", {}).get("seed_banks", {}).get("near")
    if not isinstance(expected_seeds, list) or len(expected_seeds) != 16 or len(set(expected_seeds)) != 16:
        raise ValueError("protocol.mastery.seed_banks.near必须有16个唯一seed")
    expected_seeds = sorted(int(seed) for seed in expected_seeds)
    candidates: list[tuple[int, int, int, dict[str, Any], Path]] = []
    probe_dir = trial / "probes"
    if not probe_dir.is_dir():
        raise ValueError("near-best没有已保存的near固定探针")
    for probe_path in sorted(probe_dir.glob("mastery_*_near.json")):
        probe = _json(probe_path)
        tag = probe.get("probe")
        if not isinstance(tag, dict) or tag.get("probe_stage") != "near" or tag.get("purpose") != "mastery":
            continue
        if (probe.get("mode") != "goto" or probe.get("goal_distance") != [0.6, 1.2]
                or not math.isclose(float(probe.get("goal_angle_rad", math.nan)),
                                    math.radians(15), rel_tol=0, abs_tol=1e-12)):
            continue
        if sorted(tag.get("seeds", [])) != expected_seeds or sorted(probe.get("seeds", [])) != expected_seeds:
            continue
        config = tag.get("config")
        eval_config = config.get("evaluation_config", {}) if isinstance(config, dict) else {}
        task_cfg = eval_config.get("task", {}) if isinstance(eval_config, dict) else {}
        if any(task_cfg.get(key) != value for key, value in {
            "plants": 0, "terrain_enabled": False, "rock_density": 0.0, "goal_tolerance": 0.4,
            "goal_hold": 0.5, "goal_stop_speed": 0.05, "goal_stop_angular_speed": 0.1,
        }.items()):
            continue
        digest = _stable_hash(config)
        if contract_sha is None:
            contract_sha = digest
        elif digest != contract_sha:
            continue
        checkpoint = probe.get("checkpoint")
        row = rows.get(str(checkpoint).replace("\\", "/"))
        results = probe.get("rows")
        if row is None or not isinstance(results, list) or not results:
            continue
        if len(results) != len(expected_seeds):
            continue
        rows_by_seed = {row.get("seed"): row for row in results if isinstance(row, dict)}
        if len(rows_by_seed) != len(expected_seeds) or sorted(rows_by_seed) != expected_seeds:
            continue
        resets = []
        for seed in expected_seeds:
            row_item = rows_by_seed[seed]
            initial = row_item.get("reset_initial_condition")
            if (not isinstance(initial, dict)
                    or not isinstance(initial.get("start_pose_xy_yaw"), list)
                    or len(initial["start_pose_xy_yaw"]) != 3
                    or not isinstance(initial.get("goal_xy"), list)
                    or len(initial["goal_xy"]) != 2
                    or row_item.get("goal") != initial["goal_xy"]
                    or row_item.get("trajectory_start_matches_reset") is not True
                    or row_item.get("mode") != "goto"
                    or not isinstance(row_item.get("goal_stop"), dict)):
                resets = []
                break
            resets.append({"seed": seed, "initial": initial, "goal": row_item["goal"]})
        if len(resets) != len(expected_seeds):
            continue
        candidate_reset_signature = _stable_hash(resets)
        if reset_signature is None:
            reset_signature = candidate_reset_signature
        elif candidate_reset_signature != reset_signature:
            continue
        clean, safe = 0, True
        for result in rows_by_seed.values():
            if not isinstance(result, dict) or result.get("damage") != 0 or result.get("collisions") != 0:
                safe = False
                break
            stop = result.get("goal_stop")
            if (result.get("success") is True and result.get("reason") == "goal_reached"
                    and isinstance(stop, dict) and stop.get("within_stop_limits") is True
                    and float(stop.get("hold_s", 0.0)) >= 0.5 - 1e-6):
                clean += 1
        if safe and isinstance(row.get("num_timesteps"), int):
            candidates.append((clean, row["num_timesteps"], len(results), row, _resolve(str(checkpoint))))
    if not candidates:
        raise ValueError("没有满足near固定合同且零损伤/碰撞的有效已训练检查点")
    clean, steps, episode_count, row, path = max(candidates, key=lambda item: (item[0], item[1]))
    return {**_checkpoint_record(path, row.get("sha256")), "selection": "near-best",
            "label": "近距开发候选", "num_timesteps": steps,
            "near_clean_successes": clean, "near_probe_episodes": episode_count,
            "near_probe_contract_sha256": contract_sha,
            "near_reset_group_sha256": reset_signature}


def _active_trial_processes(trial: Path) -> list[dict[str, Any]]:
    needle = str(trial)
    active = []
    if os.name == "posix" and Path("/proc").is_dir():
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                argv_parts = [part.decode("utf-8", "replace") for part in
                              (proc / "cmdline").read_bytes().split(b"\0") if part]
                cwd = (proc / "cwd").resolve()
            except OSError:
                continue
            if _argv_targets_trial(argv_parts, cwd, trial):
                active.append({"pid": int(proc.name), "argv": " ".join(argv_parts)})
        return active
    try:
        import psutil  # type: ignore
    except ImportError as exc:
        raise RuntimeError("无法核实训练进程；--launch需在Isaac Lab运行主机执行") from exc
    for proc in psutil.process_iter(("pid", "cmdline")):
        try:
            argv_parts = list(proc.info.get("cmdline") or [])
            cwd = Path(proc.cwd())
        except (psutil.Error, OSError):
            continue
        if _argv_targets_trial(argv_parts, cwd, trial):
            active.append({"pid": proc.info["pid"], "argv": " ".join(argv_parts)})
    return active


def _argv_targets_trial(argv: list[str], cwd: Path, trial: Path) -> bool:
    if not any(("train_isaac_mastery.py" in part or "run_isaac_mastery.sh" in part)
               for part in argv):
        return False
    output = None
    for index, part in enumerate(argv):
        if part == "--output" and index + 1 < len(argv):
            output = argv[index + 1]
            break
        if part.startswith("--output="):
            output = part.split("=", 1)[1]
            break
    if output is None:
        # The parent shell's positional output precedes any optional flags.
        for index, part in enumerate(argv):
            if "run_isaac_mastery.sh" in part and index + 1 < len(argv):
                output = argv[index + 1]
                break
    if output is None:
        return False
    candidate = Path(output)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        return candidate.resolve() == trial.resolve()
    except OSError:
        return str(candidate) == str(trial)


def validate_completed_trial(trial: str | Path, *, check_process: bool = True) -> dict[str, Any]:
    path = _resolve(trial)
    protocol_path = path / "protocol.json"
    protocol = _json(protocol_path)
    _validate_protocol(protocol)
    sources = _verify_sources(path, protocol)
    status = _json(path / "run_status.json")
    if status.get("status") != "completed" or status.get("exit_code") != 0:
        raise ValueError("trial尚未以run_status completed / exit_code 0结束")
    protocol_sha = _sha256(protocol_path)
    if status.get("protocol_sha256") != protocol_sha:
        raise ValueError("run_status.protocol_sha256与protocol.json不一致")
    summary = _json(path / "summary.json")
    if summary.get("training_finished") is not True or summary.get("profile") != "fast_stop_512":
        raise ValueError("summary没有确认fast_stop_512训练完成")
    outcome = _json(path / "python_outcome.json")
    if outcome.get("status") != "completed" or outcome.get("intended_exit_code") != 0:
        raise ValueError("python_outcome没有确认Python正常完成")
    actual = _json(path / "process_exit.json")
    if actual.get("actual_exit_code") != 0:
        raise ValueError("process_exit没有记录真实父shell exit 0")
    code = path / "python.exit_code"
    if not code.is_file() or code.read_text(encoding="utf-8").strip() != "0":
        raise ValueError("python.exit_code缺失或不为0")
    active = _active_trial_processes(path) if check_process else []
    if active:
        raise ValueError("trial仍有训练进程：" + "; ".join(f"PID {row['pid']} {row['argv']}" for row in active))
    return {"trial": str(path), "protocol_sha256": protocol_sha,
            "source_sha256": sources["source_sha256"], "source_file_count": sources["file_count"],
            "summary": summary}


def intersect_ray_with_map(origin: Any, direction: Any, size: float,
                           plane_z: float = MAP_Z) -> tuple[float, float]:
    o, d = np.asarray(origin, dtype=np.float64).reshape(-1), np.asarray(direction, dtype=np.float64).reshape(-1)
    if o.shape != (3,) or d.shape != (3,) or not np.isfinite(o).all() or not np.isfinite(d).all():
        raise ValueError("点击射线需要有限的三维origin和direction")
    if not math.isfinite(float(size)) or size <= 0 or not math.isfinite(float(plane_z)):
        raise ValueError("地图尺寸或平面高度非法")
    if abs(float(d[2])) <= _RAY_EPS:
        raise ValueError("点击射线与地图平面平行")
    t = (float(plane_z) - float(o[2])) / float(d[2])
    if not math.isfinite(t) or t <= _RAY_EPS:
        raise ValueError("地图平面位于点击射线后方")
    point = o + t * d
    if not np.isfinite(point).all():
        raise ValueError("地图交点不是有限坐标")
    if np.any(point[:2] < 0.0) or np.any(point[:2] > float(size)):
        raise ValueError(f"目标点（{point[0]:.3f}, {point[1]:.3f}）超出0..{size:g}米地图")
    return float(point[0]), float(point[1])


def map_camera_pose(size: float, aspect: float, fov: float = math.radians(60)) -> dict[str, tuple[float, float, float]]:
    size, aspect, fov = float(size), float(aspect), float(fov)
    if not all(map(math.isfinite, (size, aspect, fov))) or size <= 0 or aspect <= 0 or not 0 < fov < math.pi:
        raise ValueError("全图相机需要正的有限地图尺寸、viewport aspect和FOV")
    half_height = size * 0.5 * max(1.0, 1.0 / aspect) * 1.08
    distance = half_height / math.tan(fov * 0.5)
    center = (size * 0.5, size * 0.5)
    return {"eye": (center[0], center[1], distance),
            "target": (center[0], center[1], MAP_Z), "up": (0.0, 1.0, 0.0)}


def vehicle_follow_camera_pose(robot: Any, *, distance: float = 4.0,
                               height: float = 2.5, target_height: float = 0.45
                               ) -> dict[str, tuple[float, float, float]]:
    """Place the camera behind and above the chassis using its world yaw."""
    try:
        x, y, yaw = float(robot["x"]), float(robot["y"]), float(robot["yaw"])
        z = float(robot.get("z", 0.0))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("车辆跟随相机需要有限的x/y/yaw/z状态") from exc
    distance, height, target_height = map(float, (distance, height, target_height))
    if (not all(map(math.isfinite, (x, y, yaw, z, distance, height, target_height)))
            or distance <= 0 or height <= 0 or target_height < 0):
        raise ValueError("车辆跟随相机参数必须为有限的正距离和高度")
    return {
        "eye": (x - distance * math.cos(yaw), y - distance * math.sin(yaw), z + height),
        "target": (x, y, z + target_height),
        "up": (0.0, 0.0, 1.0),
    }


def _safe_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("初始状态含非有限值，拒绝写入预览记录")
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


from isaac_field.viewer import FieldViewerControls


class FastStopGoalControls(FieldViewerControls):
    """Map picker layered onto the frozen official viewer controls."""

    def __init__(self, visualizer, adapter, checkpoint: str | None = None, *,
                 reset_seed: int | None = None,
                 checkpoint_metadata: dict[str, Any] | None = None):
        self._checkpoint_selection = checkpoint_metadata
        super().__init__(visualizer, adapter, checkpoint, reset_seed=reset_seed)
        if checkpoint is not None and not self.policy_loaded:
            raise RuntimeError("基类查看器未能加载检查点；拒绝记录或启动预览")
        if self.policy_loaded:
            # Choose policy control while preserving the base viewer's paused state.
            self._enqueue("control_mode", "policy")
        else:
            self._status = "CPU UI fixture：没有加载策略模型，手动控制保持暂停。"
            self._set_status()
        # Apply on the owner thread before run_viewer's first render so new
        # clients open on the complete map, including for the model-free fixture.
        self._enqueue("map_top_view")

    def _build_sidebar(self, gui, checkpoint: str | None) -> None:
        self._map_size = float(getattr(getattr(self.adapter, "cfg", None), "size", 100.0))
        self._map_pick_enabled = True
        self._vehicle_follow_enabled = False
        self._goal_map_handles: dict[str, Any] = {}
        self._map_status = "选点后从当前位置重置并暂停；按“运行任务”开始。"
        goal = self._current_goal(self._map_size)
        with gui.add_folder("当前模型", expand_by_default=True, order=-20):
            self._model_identity_handle = gui.add_markdown(
                self._model_identity_content(checkpoint, None, self._checkpoint_selection))
        with gui.add_folder("地图点选目标", expand_by_default=True, order=-10):
            gui.add_markdown(
                f"世界XY平面，范围0..{self._map_size:g} m；原点(0, 0)，+Y朝北/屏幕上方。")
            self._map_pick_handle = gui.add_checkbox("启用地图点选", initial_value=True)
            self._map_pick_handle.on_update(
                lambda event: self._enqueue("map_pick_enabled", bool(event.target.value)))
            self._map_view_button = gui.add_button("俯视整张地图")
            self._map_view_button.on_click(lambda _: self._enqueue("map_top_view"))
            self._vehicle_follow_handle = gui.add_checkbox(
                "跟随车辆（车后上方）", initial_value=False)
            self._vehicle_follow_handle.on_update(
                lambda event: self._enqueue("vehicle_follow", bool(event.target.value)))
            self._map_goal_label = gui.add_markdown(
                f"目标（{goal[0]:.2f}, {goal[1]:.2f}）m；到点半径 {GOAL_RADIUS_M:.1f} m。")
        super()._build_sidebar(gui, checkpoint)
        scene = self._server_scene()
        on_click = getattr(scene, "on_click", None)
        if not callable(on_click):
            raise RuntimeError("当前Viser缺少scene.on_click()，无法启用地图点选")
        try:
            subscribe = on_click()
            subscribe(self._on_scene_click)
        except TypeError:
            on_click(self._on_scene_click)
        self._add_map_reference()
        self._update_goal_marker(goal)
        self._update_vehicle_marker()

    @staticmethod
    def _model_identity_content(checkpoint: str | None, runner: Any,
                                selection: dict[str, Any] | None = None) -> str:
        """Describe the active artifact without assigning it an evaluation rank."""
        model = getattr(runner, "model", None) if runner is not None else None
        if model is None:
            return "**模型：** 未加载\n\n**训练步数：** —\n\n**LSTM维度：** —"

        filename = Path(checkpoint).name if checkpoint else "未命名模型"
        steps = getattr(model, "num_timesteps", None)
        if isinstance(steps, (int, np.integer)) and not isinstance(steps, (bool, np.bool_)):
            steps_text = f"{int(steps):,} 步"
        else:
            recorded_steps = selection.get("num_timesteps") if isinstance(selection, dict) else None
            steps_text = f"{int(recorded_steps):,} 步" if isinstance(recorded_steps, int) else "未记录"

        policy = getattr(model, "policy", None)
        dimensions = []
        for role in ("actor", "critic"):
            lstm = getattr(policy, f"lstm_{role}", None)
            hidden = getattr(lstm, "hidden_size", None)
            layers = getattr(lstm, "num_layers", None)
            if isinstance(hidden, int) and isinstance(layers, int):
                dimensions.append(f"{role} {hidden}×{layers}")
        dimension_text = "；".join(dimensions) if dimensions else "未检测到LSTM层"

        selection_text = "显式加载的检查点"
        if isinstance(selection, dict):
            kind = selection.get("selection")
            if kind == "near-best":
                selection_text = "近距开发候选（near-best；不是正式 global best）"
            elif kind == "latest":
                selection_text = "最新已保存检查点"
            elif kind == "explicit":
                selection_text = "显式指定检查点"
        return (f"**模型：** `{filename}`\n\n**训练步数：** {steps_text}\n\n"
                f"**LSTM维度：** {dimension_text}\n\n**评估身份：** {selection_text}")

    def _update_model_identity(self) -> None:
        handle = getattr(self, "_model_identity_handle", None)
        if handle is not None:
            handle.content = self._model_identity_content(
                self._checkpoint, self._runner, self._checkpoint_selection)

    def _load_checkpoint(self, checkpoint: str, *, select_policy: bool) -> None:
        super()._load_checkpoint(checkpoint, select_policy=select_policy)
        metadata = self._checkpoint_selection
        metadata_path = metadata.get("path") if isinstance(metadata, dict) else None
        if isinstance(metadata, dict):
            try:
                same_checkpoint = (metadata_path is not None
                                   and Path(metadata_path).expanduser().resolve()
                                   == Path(self._checkpoint).expanduser().resolve())
            except (OSError, TypeError, ValueError):
                same_checkpoint = metadata_path is not None and str(metadata_path) == str(self._checkpoint)
            if not same_checkpoint:
                self._checkpoint_selection = None
        self._update_model_identity()

    def _server_scene(self):
        viewer = getattr(self.visualizer, "_viewer", None)
        return getattr(getattr(viewer, "_server", None), "scene", None)

    def _on_scene_click(self, event: Any) -> None:
        """Viser callback only snapshots and queues the world ray."""
        try:
            ray = (tuple(event.ray_origin), tuple(event.ray_direction))
        except (AttributeError, TypeError):
            self._enqueue("map_bad_click", "Viser点击没有world ray")
            return
        self._enqueue("map_ray", ray)

    def _apply_command(self, command: str, value: Any) -> None:
        if command == "vehicle_follow":
            requested = bool(value)
            if requested == bool(getattr(self, "_vehicle_follow_enabled", False)):
                return
            self._set_vehicle_follow(requested)
            return
        if command == "map_pick_enabled":
            self._map_pick_enabled = bool(value)
            self._map_status = "地图点选已启用。" if self._map_pick_enabled else "地图点选已关闭。"
            self._update_map_label()
            return
        if command == "map_bad_click":
            self._map_status = str(value)
            self._update_map_label()
            return
        if command == "map_ray":
            if not self._map_pick_enabled:
                return
            try:
                self._apply_map_goal(intersect_ray_with_map(value[0], value[1], self._map_size))
            except (IndexError, TypeError, ValueError) as exc:
                self._map_status = f"目标未改变：{exc}"
                self._update_map_label()
            return
        if command == "map_top_view":
            self._set_top_view()
            return
        super()._apply_command(command, value)

    def _set_vehicle_follow(self, enabled: bool) -> None:
        enabled = bool(enabled)
        handle = getattr(self, "_vehicle_follow_handle", None)
        if handle is not None:
            handle.value = enabled
        if enabled:
            self._map_status = "车辆跟随视角已启用；相机在车后上方随车更新。"
            try:
                self._set_follow_view()
            except Exception:
                self._vehicle_follow_enabled = False
                if handle is not None:
                    handle.value = False
                raise
            self._vehicle_follow_enabled = True
        else:
            self._vehicle_follow_enabled = False
            self._map_status = "车辆跟随已关闭；可拖动相机或选择全图俯视。"
            self._update_map_label()

    def _set_follow_view(self, payload: dict[str, Any] | None = None) -> None:
        """Update current Viser cameras only; this never renders or steps physics."""
        server = getattr(getattr(self.visualizer, "_viewer", None), "_server", None)
        if server is None:
            raise RuntimeError("Viser server不可用")
        if payload is None:
            payload = self.adapter.state_payload()
        robot = payload.get("robot", {})
        pose = vehicle_follow_camera_pose(robot)
        cameras: list[tuple[Any, bool]] = []
        initial = getattr(server, "initial_camera", None)
        if initial is not None:
            cameras.append((initial, True))
        get_clients = getattr(server, "get_clients", None)
        if callable(get_clients):
            clients = get_clients()
            rows = clients.values() if hasattr(clients, "values") else clients
            for client in rows:
                camera = getattr(client, "camera", None)
                if camera is not None and all(camera is not existing for existing, _ in cameras):
                    cameras.append((camera, False))
        applied = 0
        for camera, is_initial in cameras:
            try:
                camera.position, camera.look_at = pose["eye"], pose["target"]
                if is_initial:
                    camera.up = pose["up"]
                else:
                    camera.up_direction = pose["up"]
                applied += 1
            except (AttributeError, TypeError, ValueError):
                continue
        if applied == 0:
            raise RuntimeError("Viser camera缺少可写的跟随视角属性")
        self._update_map_label()

    def _validate_goal_pair(self, start: np.ndarray, goal: np.ndarray):
        """Mirror reset's explicit-pair checks before its physics reset occurs."""
        from isaac_field.task import sample_episode_start_goal

        task = self.adapter.task
        scene = task.scene
        if not bool(getattr(task.cfg, "validate_scene", False)):
            raise ValueError("当前环境未启用完整场景连通性验证，拒绝开始新回合")
        exact_start, exact_goal = sample_episode_start_goal(
            scene, np.random.default_rng(0), start=start, goal=goal,
            random_start=False, mode="goto")
        if not np.allclose(exact_start, start, rtol=0.0, atol=1e-8):
            raise ValueError("完整场景验证未保留当前车辆XY/yaw")
        if not np.allclose(exact_goal, goal, rtol=0.0, atol=1e-8):
            raise ValueError("完整场景验证未保留点击的精确目标")
        validation = scene.validate_connectivity(start[:2], goal)
        if not validation.get("connected", False):
            raise ValueError(f"当前起点到目标不可连通：{validation}")
        return scene

    def _apply_map_goal(self, goal: Any) -> None:
        requested = np.asarray(goal, dtype=np.float64).reshape(-1)
        if (requested.shape != (2,) or not np.isfinite(requested).all()
                or np.any(requested < 0.0) or np.any(requested > self._map_size)):
            raise ValueError(f"目标必须位于0..{self._map_size:g}米地图内")
        payload = self.adapter.state_payload()
        robot = payload.get("robot", {})
        start = np.asarray((robot.get("x"), robot.get("y"), robot.get("yaw")), dtype=np.float64)
        if start.shape != (3,) or not np.isfinite(start).all():
            raise ValueError("当前车辆XY/yaw无效，拒绝改变目标")
        # Do not pause, change the goal/marker, or reset recurrent state until
        # the exact current-pose/target pair passes all scene checks.
        scene = self._validate_goal_pair(start, requested)
        self.adapter.reset(options={"mode": "goto", "scene": scene, "start": start.copy(),
                                    "goal": requested.copy(), "random_start": False})
        if self._runner is not None:
            self._runner.reset()
        self._running = False
        self._terminal = False
        self._last_reward = None
        self._last_info = {}
        self._latest_observation = None
        if self._network_panel is not None:
            self._network_panel.reset()
        self._refresh_observation()
        self._run_button.label = "运行任务"
        self._sync_task_controls("goto")
        self._update_goal_marker(requested)
        self._map_status = (f"从当前位置（{start[0]:.2f}, {start[1]:.2f}）开新回合到 "
                            f"（{requested[0]:.2f}, {requested[1]:.2f}）；已暂停。")
        self._status = "新回合时间、速度和LSTM状态已清零；按“运行任务”开始。"
        self._set_status()
        self._update_map_label()

    def _update_goal_marker(self, goal: Any) -> None:
        scene = self._server_scene()
        if scene is None:
            return
        p = np.asarray(goal, dtype=np.float64).reshape(2)
        angle = np.linspace(0.0, 2 * np.pi, 65, endpoint=False)
        ring = np.column_stack((p[0] + GOAL_RADIUS_M * np.cos(angle),
                                p[1] + GOAL_RADIUS_M * np.sin(angle),
                                np.full(len(angle), MAP_Z + 0.025)))
        segments = np.stack((ring, np.roll(ring, -1, axis=0)), axis=1).astype(np.float32)
        add_lines = getattr(scene, "add_line_segments", None)
        if callable(add_lines):
            handle = self._goal_map_handles.get("ring")
            if handle is None:
                self._goal_map_handles["ring"] = add_lines(
                    "/FieldTask/ViewerGoalTolerance", segments, (1.0, 0.78, 0.12), line_width=2.0)
            else:
                handle.points, handle.visible = segments, True
        add_points = getattr(scene, "add_point_cloud", None)
        if callable(add_points):
            center = np.asarray([[p[0], p[1], MAP_Z + 0.03]], dtype=np.float32)
            handle = self._goal_map_handles.get("center")
            if handle is None:
                self._goal_map_handles["center"] = add_points(
                    "/FieldTask/ViewerGoal", center, (1.0, 0.25, 0.1), point_size=0.08,
                    point_shape="circle", point_shading="flat")
            else:
                handle.points, handle.visible = center, True

    def _add_map_reference(self) -> None:
        scene = self._server_scene()
        add_lines = getattr(scene, "add_line_segments", None)
        if not callable(add_lines):
            return
        z = MAP_Z + 0.015
        boundary = np.asarray([
            [[0, 0, z], [self._map_size, 0, z]],
            [[self._map_size, 0, z], [self._map_size, self._map_size, z]],
            [[self._map_size, self._map_size, z], [0, self._map_size, z]],
            [[0, self._map_size, z], [0, 0, z]],
        ], dtype=np.float32)
        ticks = np.arange(0.0, self._map_size + 1e-8, 10.0)
        grid = []
        for value in ticks:
            if 0.0 < value < self._map_size:
                grid.append([[value, 0.0, z], [value, self._map_size, z]])
                grid.append([[0.0, value, z], [self._map_size, value, z]])
        if grid:
            self._goal_map_handles["grid"] = add_lines(
                "/FieldTask/ViewerMapGrid", np.asarray(grid, dtype=np.float32),
                (0.28, 0.38, 0.52), line_width=0.5)
        self._goal_map_handles["boundary"] = add_lines(
            "/FieldTask/ViewerMapBoundary", boundary, (0.58, 0.68, 0.82), line_width=2.0)
        add_label = getattr(scene, "add_label", None)
        if callable(add_label):
            try:
                self._goal_map_handles["origin_label"] = add_label(
                    "/FieldTask/ViewerMapOrigin", "原点 (0, 0)", (0.15, 0.15, z + 0.02))
            except (TypeError, ValueError):
                pass

    def _update_vehicle_marker(self, payload: dict[str, Any] | None = None) -> None:
        scene = self._server_scene()
        add_points = getattr(scene, "add_point_cloud", None)
        if not callable(add_points):
            return
        try:
            if payload is None:
                payload = self.adapter.state_payload()
            robot = payload["robot"]
            point = np.asarray([[robot["x"], robot["y"], MAP_Z + 0.035]], dtype=np.float32)
            if not np.isfinite(point).all():
                return
        except (KeyError, TypeError, ValueError):
            return
        handle = self._goal_map_handles.get("vehicle")
        if handle is None:
            self._goal_map_handles["vehicle"] = add_points(
                "/FieldTask/ViewerVehiclePosition", point, (0.2, 0.95, 0.55), point_size=0.12,
                point_shape="circle", point_shading="flat")
        else:
            handle.points, handle.visible = point, True

    def _update_map_label(self, payload: dict[str, Any] | None = None) -> None:
        if payload is None:
            goal = self._current_goal(self._map_size)
        else:
            goal = np.asarray(payload.get("goal"), dtype=np.float32).reshape(-1)
            if (goal.shape != (2,) or not np.isfinite(goal).all()
                    or np.any(goal < 0.0) or np.any(goal > self._map_size)):
                goal = self._current_goal(self._map_size)
        details = "停车状态不可用。"
        try:
            task = self.adapter.task
            if payload is None:
                payload = self.adapter.state_payload()
            stop = task._goal_stop_state()
            robot = payload.get("robot", {})
            current = np.asarray((robot.get("x"), robot.get("y")), dtype=np.float64)
            distance = float(np.linalg.norm(goal - current))
            omega = np.asarray(task.body_angular_velocity, dtype=np.float64).reshape(3)
            success = bool((payload.get("metrics") or {}).get("is_success", False))
            details = (f"距离 {distance:.3f} m；真实XY合速度 {float(stop['linear_speed_m_s']):.3f} m/s "
                       f"（低于 {task.cfg.goal_stop_speed:.2f}）；车体角速度 "
                       f"XYZ=({omega[0]:.3f}, {omega[1]:.3f}, {omega[2]:.3f}) rad/s，"
                       f"模长 {float(np.linalg.norm(omega)):.3f}（低于 {task.cfg.goal_stop_angular_speed:.2f}）；"
                       f"hold {float(stop['hold_s']):.2f}/{float(task.cfg.goal_hold):.2f} s；"
                       f"成功={'是' if success else '否'}")
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        self._map_goal_label.content = (
            f"目标（{goal[0]:.2f}, {goal[1]:.2f}）m；到点半径 {GOAL_RADIUS_M:.1f} m。\n\n"
            f"{self._map_status}\n\n{details}"
        )

    def publish(self) -> None:
        super().publish()
        payload = self.adapter.state_payload()
        self._update_vehicle_marker(payload)
        self._update_map_label(payload)
        if getattr(self, "_vehicle_follow_enabled", False):
            self._set_follow_view(payload)

    @staticmethod
    def _aspect(camera: Any) -> float:
        try:
            width, height = float(camera.image_width), float(camera.image_height)
            if width > 0 and height > 0 and math.isfinite(width / height):
                return width / height
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            pass
        try:
            aspect = float(camera.aspect)
            if aspect > 0 and math.isfinite(aspect):
                return aspect
        except (AttributeError, TypeError, ValueError):
            pass
        return 1.0

    def _set_top_view(self) -> None:
        server = getattr(getattr(self.visualizer, "_viewer", None), "_server", None)
        if server is None:
            raise RuntimeError("Viser server不可用")
        cameras: list[tuple[Any, bool]] = []
        initial = getattr(server, "initial_camera", None)
        if initial is not None:
            cameras.append((initial, True))
        get_clients = getattr(server, "get_clients", None)
        if callable(get_clients):
            clients = get_clients()
            cameras.extend((getattr(client, "camera", None), False)
                           for client in (clients.values() if hasattr(clients, "values") else clients))
        applied = 0
        for camera, is_initial in cameras:
            if camera is None:
                continue
            try:
                fov = float(getattr(camera, "fov", math.radians(60)))
                if not 0 < fov < math.pi:
                    fov = math.radians(60)
                pose = map_camera_pose(self._map_size, self._aspect(camera), fov)
                if is_initial:
                    camera.up = pose["up"]
                else:
                    camera.up_direction = pose["up"]
                camera.position, camera.look_at = pose["eye"], pose["target"]
                applied += 1
            except (AttributeError, TypeError, ValueError):
                continue
        if applied == 0:
            raise RuntimeError("Viser camera缺少可写的全图视角属性")
        self._vehicle_follow_enabled = False
        follow_handle = getattr(self, "_vehicle_follow_handle", None)
        if follow_handle is not None:
            follow_handle.value = False
        self._map_status = "全图俯视已启用；地图+Y朝屏幕上方。"
        self._update_map_label()
        self._render_if_connected()


def _record_preview(output: Path, trial: Path, protocol: dict[str, Any],
                    sources: dict[str, Any], checkpoint: dict[str, Any],
                    adapter: Any, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    state = _safe_json(adapter.state_payload())
    record = {"schema": "fast-stop-goal-preview-v1", "trial": str(trial),
              "profile": protocol["profile"], "preview_seed": seed,
              "checkpoint": checkpoint, "protocol_sha256": _sha256(trial / "protocol.json"),
              "source_sha256": sources["source_sha256"], "source_file_count": sources["file_count"],
              "initial_state": state, "initial_time_s": float(adapter.task.time),
              "initial_hold_s": float(adapter.task._hold), "physical_preview": True,
              "viewer_mode": "policy_control_paused"}
    (output / "initial_state.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _launch_viewer(trial: Path, output: Path, port: int, checkpoint: dict[str, Any],
                   protocol: dict[str, Any], sources: dict[str, Any]) -> None:
    from isaac_field.task import Config
    from isaac_field.env import SingleFieldAdapter, make_env
    from isaac_field import viewer as viewer_module
    from scripts import train_isaac_mastery as mastery

    near = next(row for row in protocol["stages"] if row["name"] == "near")
    guide, stability = mastery.foundation._rough_profile_components(
        mastery.foundation.PROFILE_RANDOM_ROUGH_FAST_STOP)
    cfg = Config(**protocol["config"])
    seed = int(protocol.get("evaluation_environment_seed", 90003))
    engine = make_env(cfg=cfg, mode="goto", num_envs=1, seed=seed, auto_reset=False,
                      goal_distance=tuple(near["goal_distance"]), goal_angle=float(near["goal_angle"]),
                      distance_cost=0.0, front_guide_spec=guide,
                      body_stability_config=stability, viewer_port=port)
    adapter = SingleFieldAdapter(engine)
    try:
        mastery.foundation._validate_rough_profile_environment(engine, protocol["profile_physics_contract"])
        adapter.reset(seed=seed, options={"mode": "goto"})
        visualizer = next((item for item in engine.sim.visualizers
                           if getattr(item.cfg, "visualizer_type", None) == "viser"), None)
        if visualizer is None:
            raise RuntimeError("环境没有Isaac Lab官方Viser visualizer")
        factory = viewer_module.attach_field_controls
        recorded = False
        def attach(visualizer_arg, adapter_arg, checkpoint_arg=None, *, reset_seed=None):
            nonlocal recorded
            controls = FastStopGoalControls(
                visualizer_arg, adapter_arg, checkpoint_arg,
                reset_seed=reset_seed, checkpoint_metadata=checkpoint)
            controls.process_pending()
            if not controls.policy_loaded:
                raise RuntimeError("预览控件未加载选定策略；拒绝记录或启动预览")
            if not recorded:
                _record_preview(output, trial, protocol, sources, checkpoint, adapter_arg, seed)
                recorded = True
            return controls
        viewer_module.attach_field_controls = attach
        try:
            print(f"官方Isaac Lab Viser: http://127.0.0.1:{port}/", flush=True)
            viewer_module.run_viewer(adapter, visualizer, checkpoint["path"], reset_seed=seed)
        finally:
            viewer_module.attach_field_controls = factory
    finally:
        adapter.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true", help="CPU-only protocol/source/checkpoint validation")
    actions.add_argument("--check", action="store_true", help="CPU-only validation; default action")
    actions.add_argument("--launch", action="store_true", help="launch only after actual completed exit 0")
    parser.add_argument("--trial", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, help="explicit SB3 policy ZIP")
    parser.add_argument("--selection", choices=("latest", "near-best"), default="near-best")
    parser.add_argument("--port", type=int, default=8783)
    args = parser.parse_args(argv)
    try:
        trial, output = _resolve(args.trial), _resolve(args.output)
        if args.launch:
            gate = validate_completed_trial(trial)
            protocol = _json(trial / "protocol.json")
            sources = _verify_sources(trial, protocol)
            checkpoint = _resolve_checkpoint(trial, protocol, args.selection, args.checkpoint)
            _require_viewer_ports(args.port)
            if output.exists():
                raise FileExistsError(f"拒绝覆盖已有预览输出：{output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"status": "launching", "completion_gate": gate,
                              "checkpoint": checkpoint, "output": str(output)},
                             ensure_ascii=False, indent=2), flush=True)
            _launch_viewer(trial, output, args.port, checkpoint, protocol, sources)
            return 0
        protocol = _json(trial / "protocol.json")
        _validate_protocol(protocol)
        sources = _verify_sources(trial, protocol)
        checkpoint = _resolve_checkpoint(trial, protocol, args.selection, args.checkpoint)
        _require_viewer_ports(args.port)
        print(json.dumps({"status": "check_passed", "launch_allowed": False,
                          "note": "--prepare/--check不启动Isaac Lab且不创建输出目录",
                          "trial": str(trial), "protocol_sha256": _sha256(trial / "protocol.json"),
                          "source_sha256": sources["source_sha256"],
                          "source_file_count": sources["file_count"], "checkpoint": checkpoint,
                          "run_status": _json(trial / "run_status.json")},
                         ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        print(f"PREVIEW REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


def _port_available(port: int, host: str = "127.0.0.1") -> bool:
    if isinstance(port, bool) or not 1 <= int(port) <= 65535:
        raise ValueError("port必须在1..65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def _require_viewer_ports(port: int) -> None:
    if isinstance(port, bool) or not 1 <= int(port) <= 65534:
        raise ValueError("port须在1..65534，以便为既有二维网络面板保留port+1")
    busy = [candidate for candidate in (int(port), int(port) + 1) if not _port_available(candidate)]
    if busy:
        raise ValueError(f"查看器端口已被占用：{busy}")


if __name__ == "__main__":
    raise SystemExit(main())
