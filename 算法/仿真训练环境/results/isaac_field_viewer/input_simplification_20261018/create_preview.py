from pathlib import Path
import hashlib
import json
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from isaac_field.policy import OBSERVATION_SHAPES, make_model

class PreviewEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            key: gym.spaces.Box(-1.0 if key in ("state", "detections") else 0.0, 1.0, shape, np.float32)
            for key, shape in OBSERVATION_SHAPES.items()
        })
        self.action_space = gym.spaces.Box(-1.0,1.0,(4,),np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {key: np.zeros(shape,np.float32) for key,shape in OBSERVATION_SHAPES.items()}, {}
    def step(self, action):
        raise RuntimeError("Preview initialization does not simulate or train")

output=Path(__file__).resolve().parent
env=DummyVecEnv([PreviewEnv])
model=make_model(env,seed=80003,device="cpu")
model.field_preview_only=True
model.field_observation_config={
    "schema":"sensor-depth-only-v3", "state_dim":43,
    "depth_channels":1, "depth_invalid_value":0.0,
    "measured_velocity_indices":[8,9], "previous_command_indices":[13,14,15,16],
    "previous_command_semantics":"normalized_policy_command",
    "action_components":["forward","differential","gimbal_yaw_rate","gimbal_pitch_rate"],
}
checkpoint=output/"initial_untrained_preview.zip"
if checkpoint.exists():
    raise FileExistsError(checkpoint)
model.save(checkpoint)
old=Path("results/isaac_field_improvement/fresh_foundation_20261017_width470_progressive/resume_8192/stage_04_turn90.zip")
record={"schema":"sensor-depth-only-v3", "trained_timesteps":int(model.num_timesteps),
        "goal_gain":model.policy.features_extractor.goal_gain,
        "observation_shapes":OBSERVATION_SHAPES,
        "checkpoint":str(checkpoint),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "original_checkpoint":str(old),"original_sha256":hashlib.sha256(old.read_bytes()).hexdigest(),
        "purpose":"untrained architecture/activation preview only; no learn call"}
(output/"preview_initialization.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(record,ensure_ascii=False))
env.close()
