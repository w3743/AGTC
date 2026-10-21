"""Mechanical fixture migration; assertions are reviewed separately after execution."""
from pathlib import Path

root = Path(__file__).resolve().parents[3]
paths = [
    "test_isaac_field_policy.py", "test_policy_inspection.py", "test_isaac_field_viewer.py",
    "test_isaac_field_learning.py", "test_fresh_foundation.py", "test_network_2d.py",
    "test_network_panel.py", "test_activation_maps.py", "test_env_integration.py",
]
for name in paths:
    path = root / "tests" / name
    content = path.read_text(encoding="utf-8")
    for old, new in (
        ('sensor-depth-direct-tracks-v4', 'sensor-imu-drive-v5'),
        ('independent_left_right_track_speed_v3', 'independent_left_right_track_speed_explicit_v4'),
        ('depth64-lidar-circular1d-v1', 'depth64-lidar-circular1d-feedback-v2'),
        ('(43,)', '(39,)'), ('(1, 43)', '(1, 39)'), ('np.zeros(43', 'np.zeros(39'),
        ('np.arange(43', 'np.arange(39'), ('Linear(43,', 'Linear(39,'),
        ('"state_dim": 43', '"state_dim": 39'), ('["state_dim"] == 43', '["state_dim"] == 39'),
        ('["shape"] == [43]', '["shape"] == [39]'), ('["fields"]) == 43', '["fields"]) == 39'),
        ('len(STATE_LABELS) == 43', 'len(STATE_LABELS) == 39'),
    ):
        content = content.replace(old, new)
    lines = content.splitlines()
    updated = []
    for index, line in enumerate(lines):
        updated.append(line)
        indent = line[:len(line) - len(line.lstrip())]
        if '"state": gym.spaces.Box(' in line and 'dtype=np.float32)' in line:
            updated.append(indent + '"feedback": gym.spaces.Box(-np.inf, np.inf, shape=(41,), dtype=np.float32),')
        elif '"state": _Space((39,))' in line:
            updated.append(indent + '"feedback": _Space((41,)),')
        elif line.lstrip().startswith('"state":') and ('np.zeros' in line or 'rng.uniform' in line or 'np.full' in line):
            if '"state": np.zeros((39,)' in line or '"state": np.zeros(39,' in line:
                updated.append(indent + '"feedback": np.zeros((41,), dtype=np.float32),')
            elif '(1, 39)' in line:
                updated.append(indent + '"feedback": np.zeros((1, 41), dtype=np.float32),')
            elif '(39,)' in line:
                updated.append(indent + '"feedback": np.zeros((41,), dtype=np.float32),')
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")
print("Updated current-contract fixtures in", len(paths), "test modules")
