"""Gym registration, following Isaac Lab's external project template."""
import gymnasium as gym

gym.register(id='Isaac-Field-Direct',entry_point='isaac_field.env:IsaacFieldEnv',
             disable_env_checker=True,
             kwargs={'env_cfg_entry_point':'isaac_field.env:FieldEnvCfg'})
