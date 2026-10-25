"""PPO update cost vs torch thread count (CPU)."""
import sys
import time

import torch

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from native_env import NativeFieldEnv
from policy_m0 import ActorCritic
from train_m0 import ppo_update

env = NativeFieldEnv(num_envs=64, seed=0)
policy = ActorCritic(env.single_observation_space)
opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
B = 1024
obs = {
    "lidar": torch.rand(B, 1, 400) * 0.5,
    "depth": torch.rand(B, 1, 160, 88) * 0.5,
    "detections": torch.zeros(B, 16, 8),
    "state": torch.zeros(B, 33),
    "feedback": torch.zeros(B, 37),
}
actions = torch.zeros(B, 4)
old_logp = torch.zeros(B)
adv = torch.randn(B)
rets = torch.randn(B)
for nthreads in (20, 8, 4, 2, 1):
    torch.set_num_threads(nthreads)
    t0 = time.perf_counter()
    ppo_update(policy, opt, obs, actions, old_logp, adv, rets, epochs=4, minibatch=256)
    dt = time.perf_counter() - t0
    print("PPO_UPDATE_4EPOCH_1024 threads=%d s=%.3f" % (nthreads, dt))
    t0 = time.perf_counter()
    ppo_update(policy, opt, obs, actions, old_logp, adv, rets, epochs=2, minibatch=512)
    print("PPO_UPDATE_2EPOCH_MB512 threads=%d s=%.3f" % (nthreads, time.perf_counter() - t0))
