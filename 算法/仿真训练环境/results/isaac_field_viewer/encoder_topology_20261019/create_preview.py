from pathlib import Path
import hashlib,json
import gymnasium as gym
import numpy as np
from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv
from isaac_field.policy import OBSERVATION_SHAPES,make_model
class PreviewEnv(gym.Env):
    def __init__(self):
        self.observation_space=gym.spaces.Dict({key:gym.spaces.Box(-1.0 if key in ("state","detections") else 0.,1.,shape,np.float32) for key,shape in OBSERVATION_SHAPES.items()})
        self.action_space=gym.spaces.Box(-1.,1.,(4,),np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        return {key:np.zeros(shape,np.float32) for key,shape in OBSERVATION_SHAPES.items()},{}
    def step(self,action):
        raise RuntimeError("preview creation must not step or train")
output=Path(__file__).resolve().parent
env=DummyVecEnv([PreviewEnv])
model=make_model(env,seed=69001,device="cpu",log_std_init=-2.,n_steps=128,batch_size=128,n_epochs=5,learning_rate=3e-4,gamma=.995,target_kl=.03)
model.field_preview_only=True
model.field_observation_config={"schema":"sensor-depth-only-v3","state_dim":43,"depth_channels":1,"depth_invalid_value":0.,"measured_velocity_indices":[8,9],"previous_command_indices":[13,14,15,16],"previous_command_semantics":"normalized_policy_command","action_components":["forward","differential","gimbal_yaw_rate","gimbal_pitch_rate"]}
checkpoint=output/"untrained_encoder_preview.zip"
if checkpoint.exists(): raise FileExistsError(checkpoint)
model.save(checkpoint)
record={"purpose":"untrained UI preview, separate from training baseline","seed":69001,"training_steps":int(model.num_timesteps),"encoder_config":model.field_encoder_config,"observation_shapes":OBSERVATION_SHAPES,"actor_lstm_input_size":model.policy.lstm_actor.input_size,"actor_lstm_hidden_size":model.policy.lstm_actor.hidden_size,"checkpoint":str(checkpoint),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"encoder_modules":{name:str(module) for name,module in model.policy.features_extractor.named_modules() if not list(module.children())}}
assert record["actor_lstm_input_size"]==160
(output/"preview_initialization.json").write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(record,ensure_ascii=False))
env.close()
