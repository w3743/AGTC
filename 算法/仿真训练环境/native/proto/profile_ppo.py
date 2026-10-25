"""Measure PPO update cost for the M0 prototype (CPU)."""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/proto")
from native_env import NativeFieldEnv
from policy_m0 import ActorCritic
from train_m0 import ppo_update

torch.set_num_threads(20)
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

# single evaluate + backward
t0 = time.perf_counter()
logp, ent, val = policy.evaluate(obs, actions)
loss = -(logp.mean() + val.mean())
loss.backward()
print("EVAL_BACKWARD_B1024_S %.4f" % (time.perf_counter() - t0))
opt.zero_grad()

# 4 epochs x 4 minibatches
t0 = time.perf_counter()
ppo_update(policy, opt, obs, actions, old_logp, adv, rets, epochs=4, minibatch=256)
print("PPO_UPDATE_1024_S %.4f" % (time.perf_counter() - t0))

# 1 epoch x 1 minibatch (256)
t0 = time.perf_counter()
ppo_update(policy, opt, obs, actions, old_logp, adv, rets, epochs=1, minibatch=256)
print("PPO_UPDATE_1EPOCH_256_S %.4f" % (time.perf_counter() - t0))
