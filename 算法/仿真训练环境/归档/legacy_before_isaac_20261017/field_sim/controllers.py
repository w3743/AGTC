"""Non-learned viewer demo. No Torch dependency and no access to plant truth."""
import numpy as np


class DemoController:
    label = 'rule demo; not a trained AI'

    def action(self, env):
        c = env.cfg
        if env.mode == 'goto':
            target = env.goal
        else:
            row = int(env.time // max(8, (c.size - 4) / .65))
            y = min(c.size - 2, 2 + (row // 2 + 1) * 4)
            target = np.array([c.size - 2 if row % 2 == 0 else 2, y])
        delta = target - (env.robot[:2] + env._pose_noise)
        distance = np.linalg.norm(delta)
        theta = env.robot[2]
        error = (np.arctan2(delta[1], delta[0]) - theta + np.pi) % (2 * np.pi) - np.pi
        lf = env.frames['lidar']
        forward = np.array([np.cos(theta), np.sin(theta)])
        front = lf['directions'][:, :2] @ forward > .85
        clearance = float(np.min(lf['ranges'][front])) if np.any(front) else 1
        speed = min(.75, distance * .6) * max(0, 1 - abs(error) / 1.3)
        turn = np.clip(error * 1.5, -c.max_turn_rate, c.max_turn_rate)
        if clearance < 1.2:
            speed, turn = 0, .9
        scan = np.sin(env.time * .4) * .8 if env.mode == 'search' else 0.0
        yaw_rate = np.clip((scan - env.gimbal[0]) * 2, -c.gimbal_yaw_rate, c.gimbal_yaw_rate)
        if env.mode == 'search':
            phase = env.time % 24
            if phase < 6:
                pitch_target = 0.0
            elif phase < 12:
                pitch_target = .45 * np.sin(np.pi * (phase - 6) / 6)
            elif phase < 18:
                pitch_target = 0.0
            else:
                pitch_target = -.3 * np.sin(np.pi * (phase - 18) / 6)
            pitch_target = np.clip(pitch_target, c.pitch_min, c.pitch_max)
        else:
            pitch_target = 0.0
        pitch_rate = np.clip((pitch_target - env.gimbal[1]) * 2, -c.gimbal_pitch_rate, c.gimbal_pitch_rate)
        return np.array([speed / c.max_speed, turn / c.max_turn_rate,
                         yaw_rate / c.gimbal_yaw_rate, pitch_rate / c.gimbal_pitch_rate], np.float32)
