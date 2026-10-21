"""Bounded upstream task probe; no project physics implementation involved."""
import json
import time
from pathlib import Path

import warp as wp
wp.config.enable_backward = False

import torch
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv
from isaaclab_tasks.core.cartpole.cartpole_direct_env_cfg import CartpoleEnvCfg

cfg = CartpoleEnvCfg()
cfg.scene.num_envs = 4
cfg.seed = 7
cfg.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=16, nconmax=16), use_cuda_graph=True)
cfg.sim.visualizer_cfgs = []
started = time.perf_counter()
env = CartpoleEnv(cfg)
try:
    obs, _ = env.reset()
    for _ in range(32):
        obs, reward, terminated, truncated, info = env.step(torch.zeros((4, 1), device=env.device))
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
    result = {'backend': env.sim.physics_manager.__name__, 'device': env.device,
              'torch': torch.__version__, 'steps': 32, 'num_envs': env.num_envs,
              'observation_shape': list(obs['policy'].shape),
              'startup_and_steps_seconds': time.perf_counter() - started}
    output = Path(__file__).resolve().parents[1] / 'results/isaaclab_migration/runtime_probe.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
finally:
    env.close()
