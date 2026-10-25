"""Debug the reference state length mismatch (39 vs 33)."""
import json
import math
import sys

import numpy as np

REPO = "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境"
sys.path.insert(0, REPO)
from isaac_field.task import ABSOLUTE_GOAL_INPUT_SCHEMA, Config, FieldTask, state_fields_for_schema

cfg = Config(size=100.0, plants=0, dry_fraction=0.0, terrain_enabled=False, max_time=60.0,
             input_schema=ABSOLUTE_GOAL_INPUT_SCHEMA)
task = FieldTask(cfg, "goto")
print("input_schema", cfg.input_schema, "n fields", len(state_fields_for_schema(cfg.input_schema)))
print("last_action initial", np.asarray(task.last_action).shape)
print("robot", np.asarray(task.robot).shape)
task.robot = np.array([10.0, 20.0, 0.3, 0.0, 0.0])
task.gimbal = np.array([0.1, 0.2])
task.last_action = np.array([0.1, 0.2, 0.3, 0.4])
task.goal = np.array([30.0, 40.0])
task.time = 1.0
task._pose_noise = np.zeros(2)
task._new = {"lidar": 1.0, "depth": 0.0, "camera": 0.0}
task.frames = {}
for kind in ("lidar", "depth", "camera"):
    task.frames[kind] = {"capture": 0.5, "estimated_pose": np.array([10.0, 20.0, 0.3, 0.1, 0.2]),
                         "valid": np.ones(1, bool), "ranges": np.zeros(1, np.float32),
                         "values": np.zeros((1, 1), np.float32), "detections": []}
from isaac_field.feedback import FEEDBACK_DIM, FeedbackSnapshot
task._feedback_snapshot = FeedbackSnapshot(
    feedback=np.zeros(FEEDBACK_DIM, np.float32), sample_time=1.0,
    estimated_heading_rad=0.3, estimated_roll_rad=0.0, estimated_pitch_rad=0.0,
    estimated_quaternion_xyzw=np.array([0, 0, 0, 1], np.float32),
    estimated_gimbal_q_rad=np.array([0.1, 0.2], np.float32), pose_valid=True)
try:
    obs = task._observation()
    print("STATE LEN", obs["state"].shape)
except Exception as e:
    print("ERR", e)
    # replicate the builder to find the extra 6
    estimated_pose = task._estimated_pose()
    estimated_xy = estimated_pose[:2].astype(np.float64)
    delta = task.goal - estimated_xy
    theta = float(estimated_pose[2])
    gimbal_yaw, gimbal_pitch = map(float, estimated_pose[3:5])
    relative = np.array([math.cos(theta) * delta[0] + math.sin(theta) * delta[1],
                         -math.sin(theta) * delta[0] + math.cos(theta) * delta[1]]) / cfg.size
    goal_features = task.goal / cfg.size * 2.0 - 1.0
    state = [float(task.mode == "goto"), *goal_features,
             *(estimated_xy / cfg.size * 2 - 1), math.sin(theta), math.cos(theta),
             *task.last_action, min(1, task.time / cfg.max_time)]
    print("prefix", len(state), "goal_features", np.asarray(goal_features).shape,
          "estimated_xy", estimated_xy.shape, "last_action", np.asarray(task.last_action).shape)
    for k in ("lidar", "depth", "camera"):
        f = task.frames[k]
        age = task.time - f["capture"]
        frame_pose = f["estimated_pose"]
        motion = estimated_xy - frame_pose[:2]
        state.extend([min(age / 2, 1), float(task._new[k]), *(motion / cfg.lidar_range),
                      (theta - frame_pose[2]) / math.pi,
                      (gimbal_yaw - frame_pose[3]) / math.pi,
                      (gimbal_pitch - frame_pose[4]) / math.pi])
        print("after", k, len(state))
