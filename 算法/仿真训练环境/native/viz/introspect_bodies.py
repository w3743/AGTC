import sys, os
ROOT = r"/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native"
sys.path.insert(0, ROOT + "/proto")
from native_env import NativeFieldEnv
env = NativeFieldEnv(num_envs=1, seed=0, physics="newton_xpbd", lidar_backend="numpy", device="cuda:0", feedback_backend="direct", auto_reset=False)
b = env.backend
print("body_count", int(b.model.body_count), "joint_count", int(b.model.joint_count))
for attr in ("body_label", "body_labels", "body_names", "label", "body_labels"):
    if hasattr(b.model, attr):
        print(attr, getattr(b.model, attr))
print("base_body", getattr(b, "base_body", None))
print("body_q shape", b.state_0.body_q.shape)
print("env origins attr", [a for a in dir(env) if "origin" in a.lower()])
if hasattr(env, "env_origins"):
    print("env_origins", env.env_origins)
