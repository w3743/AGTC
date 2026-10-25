"""Quick CPU tuning probe for the observation encoder forward cost."""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from native_env import NativeFieldEnv
from policy_m0 import ActorCritic

print("torch threads", torch.get_num_threads(), "interop", torch.get_num_interop_threads())
print("mkldnn available", torch.backends.mkldnn.is_available(), "enabled", torch.backends.mkldnn.enabled)

env = NativeFieldEnv(num_envs=64, seed=0)
policy = ActorCritic(env.single_observation_space)
obs = {
    "lidar": torch.rand(64, 1, 400) * 0.5,
    "depth": torch.rand(64, 1, 160, 88) * 0.5,
    "detections": torch.zeros(64, 16, 8),
    "state": torch.zeros(64, 33),
    "feedback": torch.zeros(64, 37),
}
with torch.no_grad():
    for _ in range(3):
        policy.encoder(obs)
    for nthreads in (20, 8, 4, 1):
        torch.set_num_threads(nthreads)
        t0 = time.perf_counter()
        for _ in range(10):
            policy.encoder(obs)
        print("ENCODER_FWD_B64 threads=%d ms=%.2f" % (nthreads, (time.perf_counter() - t0) / 10 * 1e3))
    # layer breakdown at 20 threads
    torch.set_num_threads(20)
    feats = {}
    t0 = time.perf_counter()
    for _ in range(10):
        policy.encoder.laser(obs["lidar"])
    print("LIDAR_BRANCH ms=%.2f" % ((time.perf_counter() - t0) / 10 * 1e3))
    t0 = time.perf_counter()
    for _ in range(10):
        policy.encoder.depth(obs["depth"])
    print("DEPTH_BRANCH ms=%.2f" % ((time.perf_counter() - t0) / 10 * 1e3))
    t0 = time.perf_counter()
    for _ in range(10):
        policy.encoder.boxes(obs["detections"])
    print("BOXES_BRANCH ms=%.2f" % ((time.perf_counter() - t0) / 10 * 1e3))
