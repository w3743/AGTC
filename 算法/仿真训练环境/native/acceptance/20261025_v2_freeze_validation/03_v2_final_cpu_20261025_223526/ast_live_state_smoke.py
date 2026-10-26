from __future__ import annotations
import ast, hashlib, json, sys
from pathlib import Path
import numpy as np

source = Path("/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/train/train_wsl.py")
tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_write_live_state")
module = ast.Module(body=[node], type_ignores=[])
namespace = {"np": np}
exec(compile(module, str(source), "exec"), namespace)
writer = namespace["_write_live_state"]

class Array:
    def __init__(self, value): self.value = value
    def numpy(self): return self.value
class Backend:
    def __init__(self):
        self.state_0 = type("State", (), {})()
        local = np.asarray([
            [0.0, 0.0, 0.10, 0, 0, 0, 1],
            [0.1, 0.2, 0.20, 0, 0, 0, 1],
            [0.3, 0.4, 0.30, 0, 0, 0, 1],
        ], dtype=np.float64)
        self.state_0.body_q = Array(np.concatenate([local + np.asarray([e, -e, 0, 0, 0, 0, 0]) for e in range(4)]))
        self.model = type("Model", (), {"body_label": [f"body/{i}" for i in range(12)]})()
        self.base_body = np.asarray([1, 4, 7, 10], dtype=np.int64)
class Env:
    num_envs = 4
    goal_xy = np.asarray([[1, 2], [-3, 4], [5, -6], [0.7, 0.8]], dtype=np.float64)
    episode_time = np.asarray([0.1, 1.25, 3.5, 9.0], dtype=np.float64)
    def __init__(self): self.backend = Backend()
class Exporter:
    def __init__(self): self.packet = None
    def write(self, **kwargs): self.packet = kwargs

origins = np.asarray([[100, 200, 0], [-30, 50, 0], [7, -8, 0], [0.25, 0.5, 0]], dtype=np.float64)
env = Env()
exporter = Exporter()
ok = writer(exporter, env, origins, 123.0)
assert ok is True
p = exporter.packet
assert p is not None
np.testing.assert_array_equal(p["goals_local_xy"], env.goal_xy)
np.testing.assert_array_equal(p["task_times_s"], env.episode_time)
np.testing.assert_array_equal(p["env_origins_w"], origins)
assert p["sim_time_s"] == 123.0
raw = env.backend.state_0.body_q.numpy()
labels = [str(v) for v in env.backend.model.body_label]
bodies_per_world = len(labels) // env.num_envs
base_local = int(env.backend.base_body[0]) % bodies_per_world
expected = np.zeros((4, bodies_per_world, 7))
for e in range(4):
    first = int(env.backend.base_body[e]) - base_local
    expected[e] = raw[first:first+bodies_per_world].copy()
    expected[e, :, 0] += origins[e, 0]
    expected[e, :, 1] += origins[e, 1]
np.testing.assert_array_equal(p["body_link_pose_w"], expected)
np.testing.assert_array_equal(p["root_link_pose_w"], expected[:, base_local, :])
print(json.dumps({
    "pass": True,
    "source": str(source),
    "function": "_write_live_state",
    "env_count": 4,
    "goals_are_local": True,
    "body_xy_translation_matches_origins": True,
    "per_env_episode_times_preserved": True,
    "root_rows_match_base_body": True,
    "sim_time_s": p["sim_time_s"],
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
}, ensure_ascii=False, sort_keys=True))