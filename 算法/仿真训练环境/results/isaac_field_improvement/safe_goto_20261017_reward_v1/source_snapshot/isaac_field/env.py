"""Native batched Isaac Lab task: physical roller drive and live GPU sensors."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

import gymnasium as gym
import numpy as np
import torch
import warp as wp
wp.config.enable_backward = False

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_from_matrix
from isaaclab_newton.physics import NewtonCfg, NewtonShapeCfg, MJWarpSolverCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

from .assets import ROBOT_SPEC, GIMBAL_YAW_JOINT, GIMBAL_PITCH_JOINT, ensure_robot_asset, robot_articulation_cfg, robot_asset_spec_sha256
from .scene_assets import crop_asset, terrain_asset
from .layouts import generate_layout
from .task import Config, FieldTask, Scene, wrap
from .terrain import Terrain

ROOT = Path(__file__).resolve().parents[1]


@configclass
class FieldEnvCfg(DirectRLEnvCfg):
    task: Config = Config()
    mode: str = 'mixed'
    auto_reset: bool = True
    # Curriculum limits affect task sampling, never replace policy control.
    goal_distance: tuple[float, float] | None = None
    goal_angle: float = math.pi
    distance_cost: float = 0.0
    spawn_near_crops: float = 0.8
    # Steering is a differential track-speed demand, not an ideal no-slip yaw rate.
    # None gives it the same available speed range as straight driving.
    steering_track_speed: float | None = None
    # Higher gains produced excessive translation and reverse braking overshoot
    # in native contact tests; .5 is the verified proxy value, not a motor rating.
    drive_velocity_gain: float = 0.5
    drive_effort_limit: float = 2.0
    scene = InteractiveSceneCfg(num_envs=4, env_spacing=110., replicate_physics=True)
    sim = sim_utils.SimulationCfg(
        dt=.005, render_interval=10,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=256, nconmax=128, iterations=40, ls_iterations=30,
                                      integrator='implicitfast', use_mujoco_contacts=False),
            default_shape_cfg=NewtonShapeCfg(margin=.002, gap=.005, mu=.8),
            load_visual_shapes=True, use_cuda_graph=True,
        ),
    )
    decimation = 10
    episode_length_s = 900.
    action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
    observation_space = gym.spaces.Dict({
        'lidar': gym.spaces.Box(0., 1., (2, 400), np.float32),
        'depth': gym.spaces.Box(0., 1., (2, 160, 88), np.float32),
        'detections': gym.spaces.Box(-1., 1., (16, 10), np.float32),
        'state': gym.spaces.Box(-1., 1., (44,), np.float32),
    })
    state_space = 0
    compute_final_obs = True
    seed = 7


class IsaacFieldEnv(DirectRLEnv):
    cfg: FieldEnvCfg

    def __init__(self, cfg: FieldEnvCfg | None = None, render_mode=None, **kwargs):
        cfg = cfg or FieldEnvCfg()
        self.field_cfg = c = cfg.task
        for name, value in (('max_speed', c.max_speed), ('acceleration', c.acceleration),
                            ('drive_velocity_gain', cfg.drive_velocity_gain),
                            ('drive_effort_limit', cfg.drive_effort_limit)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if cfg.steering_track_speed is not None and (
                not math.isfinite(cfg.steering_track_speed) or cfg.steering_track_speed < 0):
            raise ValueError('steering_track_speed must be finite and non-negative')
        if cfg.mode not in ('mixed', 'goto', 'search'):
            raise ValueError('mode must be mixed/search/goto')
        if not math.isclose(cfg.sim.dt * cfg.decimation, c.dt):
            raise ValueError('task dt must match physics dt * decimation')
        cfg.episode_length_s = c.max_time
        cfg.scene.env_spacing = c.size + 10.
        cfg.observation_space = gym.spaces.Dict({
            'lidar': gym.spaces.Box(0., 1., (2, c.lidar_rays), np.float32),
            'depth': gym.spaces.Box(0., 1., (2, c.depth_height, c.depth_width), np.float32),
            'detections': gym.spaces.Box(-1., 1., (c.max_detections, 10), np.float32),
            'state': gym.spaces.Box(-1., 1., (44,), np.float32),
        })
        self._rng = np.random.default_rng(cfg.seed)
        self._geometry_seed = int(cfg.seed or 0) + 1_000_003
        self._reset_options = None
        self.layout_sampler = None
        self.episode_results = []
        self._sensor_cache = {}
        # One physically imported terrain realization per run, shared across worlds;
        # episode maps randomize plant positions, spawn, yaw and goals on this surface.
        self.terrain = Terrain(c.size, seed=int(cfg.seed or 0),
            ground_amplitude=c.terrain_amplitude if c.terrain_enabled else 0.,
            wavelength_range=(c.terrain_wavelength_min, c.terrain_wavelength_max),
            rock_density=c.rock_density if c.terrain_enabled else 0., large_rock_fraction=c.large_rock_fraction)
        prototype = FieldTask(c, 'goto')
        prototype_scene=Scene.generate(c,np.random.default_rng(cfg.seed),None,
                                       np.array([c.size-2,c.size-2]),terrain=self.terrain)
        self._targets = np.zeros(c.plants,dtype=bool)
        dry_count=min(c.plants,max(1,int(round(c.plants*c.dry_fraction)))) if c.dry_fraction>0 else 0
        if dry_count:
            self._targets[np.random.default_rng(self._geometry_seed).choice(c.plants,dry_count,replace=False)]=True
        template = Scene(c, prototype_scene.xy, self._targets, np.random.default_rng(self._geometry_seed), terrain=self.terrain)
        fingerprint = hashlib.sha256(json.dumps({'config': asdict(c), 'seed': cfg.seed,
                                               'asset_revision': 8}, sort_keys=True).encode()).hexdigest()[:16]
        asset_dir = ROOT / 'results/isaac_field/assets' / fingerprint
        asset_dir.mkdir(parents=True, exist_ok=True)
        terrain_path = asset_dir / 'terrain.usda'
        if not terrain_path.exists():
            terrain_asset(self.terrain, terrain_path)
        crop_path=asset_dir/'crop.usda'
        if c.plants and not crop_path.exists():
            crop_asset(template,asset_dir)
        cfg.scene.field = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Field',
                                      spawn=sim_utils.UsdFileCfg(usd_path=str(terrain_path)))
        robot = robot_articulation_cfg(ensure_robot_asset(ROOT / 'results/isaac_field/assets'))
        robot.prim_path = '{ENV_REGEX_NS}/Robot'
        robot.init_state.pos = (2., 2., .3)
        robot.actuators = {
            'drive': ImplicitActuatorCfg(joint_names_expr=['.*_wheel_joint'], stiffness=0., damping=cfg.drive_velocity_gain,
                                        joint_effort_limit=cfg.drive_effort_limit, joint_velocity_limit=110.),
            'gimbal': ImplicitActuatorCfg(joint_names_expr=['gimbal_.*_joint'], stiffness=1., damping=.1,
                                         joint_effort_limit=1., joint_velocity_limit=3.),
        }
        cfg.scene.robot = robot
        if c.plants:
            cfg.scene.corn = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Plants',
                                        spawn=sim_utils.UsdFileCfg(usd_path=str(crop_path)))
        mount = '{ENV_REGEX_NS}/Robot/gimbal_yaw_link/gimbal_pitch_link'
        cfg.scene.lidar = RayCasterCfg(prim_path=mount, update_period=0., mesh_prim_paths=[],
            pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0., 0.),
                horizontal_fov_range=(-180., 180.), horizontal_res=360./c.lidar_rays),
            ray_alignment='base', max_distance=c.lidar_range)
        cfg.scene.rangefinder = RayCasterCfg(prim_path=mount, update_period=0., mesh_prim_paths=[],
            pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0., 0.),
                horizontal_fov_range=(0., 0.), horizontal_res=1.),
            ray_alignment='base', max_distance=c.camera_range)
        cfg.scene.depth = CameraCfg(prim_path=mount+'/Stereo', width=c.depth_width, height=c.depth_height,
            offset=CameraCfg.OffsetCfg(convention='world'), update_period=0., update_latest_camera_pose=True,
            data_types=['distance_to_image_plane'], renderer_cfg=NewtonWarpRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(prototype.depth_intrinsics.ravel().tolist(),
                c.depth_width, c.depth_height, clipping_range=(c.depth_min_range,c.depth_range)))
        detector_width = 64
        detector_focal = detector_width / (2 * math.tan(math.pi / 6))
        detector_k = [detector_focal,0,31.5,0,detector_focal,31.5,0,0,1]
        cfg.scene.detector = CameraCfg(prim_path=mount+'/Detector', width=detector_width, height=detector_width,
            offset=CameraCfg.OffsetCfg(convention='world'), update_period=0.,
            data_types=['instance_segmentation','distance_to_image_plane'],
            renderer_cfg=NewtonWarpRendererCfg(colorize_instance_segmentation=False),
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(detector_k,64,64,
                clipping_range=(.12,c.camera_range)))
        self.tasks = [FieldTask(c, 'goto') for _ in range(cfg.scene.num_envs)]
        super().__init__(cfg, render_mode, **kwargs)
        self.rover = self.scene['robot']
        self._wheel_names = [f'{w.name}_wheel_joint' for w in ROBOT_SPEC.wheel_specs]
        self._wheel_ids = torch.tensor([self.rover.joint_names.index(n) for n in self._wheel_names], device=self.device)
        self._wheel_sides = torch.tensor([-1. if w.side=='left' else 1. for w in ROBOT_SPEC.wheel_specs],device=self.device)
        self._wheel_track_ids = (self._wheel_sides > 0).long()
        self._gimbal_ids = torch.tensor([self.rover.joint_names.index(n) for n in
                                        (GIMBAL_YAW_JOINT,GIMBAL_PITCH_JOINT)],device=self.device)
        self._mount_id = self.rover.body_names.index('gimbal_pitch_link')
        self.actions = torch.zeros((self.num_envs,4),device=self.device)
        self.command = torch.zeros((self.num_envs,2),device=self.device)
        self.track_command = torch.zeros_like(self.command)
        self.wheel_target_velocity = torch.zeros((self.num_envs,14),device=self.device)
        self.gimbal_target = torch.zeros_like(self.command)
        self._reward = torch.zeros(self.num_envs,device=self.device)
        self._last_results = [None] * self.num_envs
        # Static kinematic roots intentionally have no articulation/DOFs. Use
        # Newton's public state buffers for their episode-only pose updates.
        model=self.sim.physics_manager.get_model()
        indices=np.full((self.num_envs,c.plants),-1,np.int64)
        for j,label in enumerate(model.body_label):
            match=re.search(r'/env_(\d+)/Plants/Corn_(\d+)$',label)
            if match:
                env_id,plant_id=map(int,match.groups())
                indices[env_id,plant_id]=j
        if np.any(indices<0):
            raise RuntimeError('not all authored plant bodies were imported by Newton')
        self._plant_body_ids=torch.as_tensor(indices,device=self.device)
        for i,t in enumerate(self.tasks):
            t.attach_sensors(lambda kind, i=i: self._capture(i,kind))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_options = options
        return super().reset(seed=seed, options=options)

    def _reset_idx(self, env_ids):
        ids = [int(i) for i in env_ids.tolist()]
        super()._reset_idx(env_ids)
        poses = self.rover.data.default_root_pose.torch[env_ids].clone()
        plant_poses = []
        for row,i in enumerate(ids):
            t = self.tasks[i]
            options = dict(self._reset_options or {})
            options.setdefault('mode', 'search' if self.cfg.mode=='mixed' and self._rng.random()<.5
                               else 'goto' if self.cfg.mode=='mixed' else self.cfg.mode)
            options.setdefault('terrain', self.terrain)
            # Validate the exact shapes that were authored, rather than a
            # second, newly randomized set of leaf/root radii.
            supplied=options.get('scene')
            if supplied is None and self.layout_sampler is not None:
                supplied=self.layout_sampler(self._rng)
            episode_seed=int(self._rng.integers(0,2**31))
            layout_rng=np.random.default_rng(episode_seed)
            for attempt in range(64):
                if supplied is not None:
                    xy=supplied.xy
                else:
                    xy=generate_layout(self.field_cfg,layout_rng,terrain=self.terrain,
                        start=options.get('start'),goal=options.get('goal'))
                scene=Scene(self.field_cfg,xy,self._targets,np.random.default_rng(self._geometry_seed),terrain=self.terrain)
                candidate_options={**options,'scene':scene}
                if 'goal' not in options:
                    candidate_options['goal']=scene.sample_start(layout_rng)[:2]
                try:
                    t.prepare(seed=episode_seed,options=candidate_options)
                    break
                except ValueError:
                    if supplied is not None or attempt==63:
                        raise
            if options.get('random_start', 'start' not in options):
                # Sample against the actual authored plant radii, after the map is final.
                t.robot[:3]=t.scene.sample_start(t.np_random)
                if self.field_cfg.plants and t.np_random.random()<self.cfg.spawn_near_crops:
                    lo=np.maximum(t.scene.xy.min(0)-1.5, .6)
                    hi=np.minimum(t.scene.xy.max(0)+1.5, self.field_cfg.size-.6)
                    for _ in range(2000):
                        xy=t.np_random.uniform(lo,hi)
                        yaw=t.np_random.uniform(-math.pi,math.pi)
                        if t.scene.pose_clear(xy,yaw,include_small_rocks=True):
                            t.robot[:3]=[*xy,yaw]
                            break
                t.trail.clear();t.trail.append(t.robot[:2].tolist())
            elif not t.scene.navigable(t.robot[:2]):
                raise ValueError('explicit spawn overlaps the actual vehicle-inflated geometry')
            if self.cfg.goal_distance is not None and 'goal' not in options and t.mode=='goto':
                for _ in range(100):
                    angle=t.robot[2]+t.np_random.uniform(-self.cfg.goal_angle,self.cfg.goal_angle)
                    distance=t.np_random.uniform(*self.cfg.goal_distance)
                    goal=t.robot[:2]+distance*np.array([math.cos(angle),math.sin(angle)])
                    if t.scene.navigable(goal):
                        t.goal=goal
                        break
                else:
                    raise RuntimeError('no valid curriculum goal found')
            # Spawn tangent to the local surface, with every roller above its
            # actual contact height. A level chassis on a slope can start with
            # deeply embedded front/rear rollers and receive a false launch impulse.
            slope=np.asarray(self.terrain.slope(t.robot[:2]),dtype=float)
            normal=np.array([-slope[0],-slope[1],1.]);normal/=np.linalg.norm(normal)
            forward=np.array([math.cos(t.robot[2]),math.sin(t.robot[2]),0.])
            forward[2]=np.dot(slope,forward[:2]);forward/=np.linalg.norm(forward)
            left=np.cross(normal,forward);left/=np.linalg.norm(left)
            rotation=np.stack((forward,left,np.cross(forward,left)),-1)
            bottoms=np.array([[w.x_m,w.y_m,w.z_m-ROBOT_SPEC.wheel_radius_m] for w in ROBOT_SPEC.wheel_specs])@rotation.T
            ground=np.asarray(self.terrain.height(t.robot[:2]+bottoms[:,:2]))
            spawn_z=float(np.max(ground-bottoms[:,2]))+.01
            xyz=np.array([*t.robot[:2],spawn_z])
            poses[row,:3] = torch.as_tensor(xyz,device=self.device)+self.scene.env_origins[i]
            poses[row,3:] = quat_from_matrix(torch.as_tensor(rotation,dtype=torch.float32,device=self.device))
            pp=np.zeros((self.field_cfg.plants,7),np.float32)
            if len(pp):
                pp[:,:2]=t.scene.xy
                pp[:,2]=t.scene.root_z
                pp[:,6]=1.
            plant_poses.append(pp)
            self._last_results[i]=None
        self.rover.write_root_pose_to_sim_index(root_pose=poses,env_ids=env_ids)
        self.rover.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(ids),6),device=self.device),env_ids=env_ids)
        joint_pos=self.rover.data.default_joint_pos.torch[env_ids].clone()*0
        self.rover.write_joint_position_to_sim_index(position=joint_pos,env_ids=env_ids)
        self.rover.write_joint_velocity_to_sim_index(velocity=joint_pos,env_ids=env_ids)
        self.rover.actuators.target_command.set_position_index(value=joint_pos,env_ids=env_ids)
        self.rover.actuators.target_command.set_velocity_index(value=joint_pos,env_ids=env_ids)
        if self.field_cfg.plants:
            pp=torch.as_tensor(np.stack(plant_poses),device=self.device)
            pp[:,:,:3]+=self.scene.env_origins[env_ids,None,:]
            body_ids=self._plant_body_ids[env_ids]
            manager=self.sim.physics_manager
            for state in (manager.get_state_0(),manager.get_state_1()):
                wp.to_torch(state.body_q)[body_ids]=pp
                wp.to_torch(state.body_qd)[body_ids]=0
            manager.invalidate_body_state(env_ids=wp.from_torch(env_ids.to(torch.int32)))
        self.command[env_ids]=0
        self.track_command[env_ids]=0
        self.wheel_target_velocity[env_ids]=0
        self.gimbal_target[env_ids]=0
        self.actions[env_ids]=0
        self._sensor_cache.clear()
        self._refresh_state()
        self._reset_options=None

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError('actions must be finite')
        self.actions = actions.clamp(-1.,1.)
        self._before = [t.robot.copy() for t in self.tasks]
        self._old_distance = [float(np.linalg.norm(t.goal-t.robot[:2])) for t in self.tasks]
        speed = self.field_cfg.max_speed
        steering = speed if self.cfg.steering_track_speed is None else self.cfg.steering_track_speed
        forward = self.actions[:, 0] * speed
        differential = self.actions[:, 1] * steering
        desired = torch.stack((forward - differential, forward + differential), dim=1)
        # Common scaling preserves the requested left/right ratio on saturation.
        desired /= (desired.abs().amax(dim=1, keepdim=True) / speed).clamp_min(1.)
        change = self.field_cfg.acceleration * self.step_dt
        self.track_command += (desired - self.track_command).clamp(-change, change)
        self.wheel_target_velocity = self.track_command[:, self._wheel_track_ids] / ROBOT_SPEC.wheel_radius_m
        # Nominal no-slip twist is retained for diagnostics only. State/reward
        # continue using the actual chassis velocities measured from physics.
        self.command[:, 0] = self.track_command.mean(dim=1)
        self.command[:, 1] = (self.track_command[:, 1] - self.track_command[:, 0]) / ROBOT_SPEC.track_separation_m
        self.gimbal_target+=self.actions[:,2:]*torch.tensor([self.field_cfg.gimbal_yaw_rate,self.field_cfg.gimbal_pitch_rate],device=self.device)*self.step_dt
        self.gimbal_target[:,0]=(self.gimbal_target[:,0]+math.pi)%(2*math.pi)-math.pi
        self.gimbal_target[:,1].clamp_(self.field_cfg.pitch_min,self.field_cfg.pitch_max)
        self._sensor_cache.clear()

    def _apply_action(self):
        self.rover.actuators.target_command.set_velocity_index(value=self.wheel_target_velocity,joint_ids=self._wheel_ids)
        self.rover.actuators.target_command.set_position_index(value=self.gimbal_target,joint_ids=self._gimbal_ids)

    def _refresh_state(self):
        poses=self.rover.data.root_link_pose_w.torch
        velocities=self.rover.data.root_com_vel_w.torch
        angles=torch.stack(euler_xyz_from_quat(poses[:,3:]),-1).detach().cpu().numpy()
        rotation=matrix_from_quat(poses[:,3:])
        forward_speed=(velocities[:,:3]*rotation[:,:,0]).sum(-1).detach().cpu().numpy()
        root=poses.detach().cpu().numpy()
        origin_offsets=self.scene.env_origins.detach().cpu().numpy()
        mount=self.rover.data.body_link_pose_w.torch[:,self._mount_id]
        mount_rotation=matrix_from_quat(mount[:,3:]).detach().cpu().numpy()
        mount_pose=mount.detach().cpu().numpy()
        joints=self.rover.data.joint_pos.torch[:,self._gimbal_ids].detach().cpu().numpy()
        angular=velocities[:,5].detach().cpu().numpy()
        for i,t in enumerate(self.tasks):
            t.robot[:]=[*list(root[i,:2]-origin_offsets[i,:2]),wrap(float(angles[i,2])),float(forward_speed[i]),float(angular[i])]
            t.gimbal[:]=joints[i]
            t.body_pose=(float(root[i,2]-origin_offsets[i,2]),float(angles[i,1]),float(angles[i,0]))
            t.view_origin=(mount_pose[i,:3]-origin_offsets[i]).astype(np.float32)
            t.view_basis=(mount_rotation[i,:,0],-mount_rotation[i,:,1],mount_rotation[i,:,2])

    def _capture(self,i,kind):
        t,c=self.tasks[i],self.field_cfg
        if kind not in self._sensor_cache:
            if kind=='lidar':
                sensor=self.scene['lidar'].data
                self._sensor_cache[kind]=sensor.ray_distances.torch.detach().cpu().numpy().copy()
            elif kind=='depth':
                data=self.scene['depth'].data
                self._sensor_cache[kind]=(data.output['distance_to_image_plane'].torch.detach().cpu().numpy()[...,0].copy(),
                                         data.intrinsic_matrices.torch.detach().cpu().numpy().copy())
            else:
                sensor=self.scene['detector'].data
                self._sensor_cache[kind]=(
                    sensor.output['instance_segmentation'].torch.detach().cpu().numpy()[...,0].copy(),
                    sensor.info['instance_segmentation'],
                    self.scene['rangefinder'].data.ray_distances.torch.detach().cpu().numpy().copy())
        if kind=='lidar':
            ideal=self._sensor_cache[kind][i]
            valid=np.isfinite(ideal)&(ideal>=c.lidar_min_range)&(ideal<c.lidar_range)
            values=np.full(c.lidar_rays,c.lidar_range,np.float32)
            values[valid]=np.clip(ideal[valid]*(1+t.np_random.uniform(-c.lidar_relative_error,c.lidar_relative_error,valid.sum())),c.lidar_min_range,c.lidar_range)
            return {'ranges':values,'valid':valid}
        if kind=='depth':
            frames,intrinsics=self._sensor_cache[kind]
            raw=frames[i]
            valid=np.isfinite(raw)&(raw>=c.depth_min_range)&(raw<=c.depth_range)
            values=np.full_like(raw,c.depth_range)
            values[valid]=np.clip(raw[valid]+t.np_random.normal(0,c.range_noise,valid.sum()),c.depth_min_range,c.depth_range)
            return {'values':values,'valid':valid,'intrinsics':intrinsics[i],
                    'basis':np.stack(t.view_basis).copy(),
                    'calibration_id':'isaac-centered-from-'+str(t.depth_intrinsics[0,0]),
                    'calibration_source':'Isaac Lab Camera.data.intrinsic_matrices; scaled native focal length, centered principal point'}
        masks,metadata,laser=self._sensor_cache[kind]
        image=masks[i]
        labels=metadata.get('idToLabels',{})
        detections=[]
        yy,xx=np.indices(image.shape)
        cone=(xx-31.5)**2+(yy-31.5)**2<=32**2
        central_id=int(image[32,32])
        for instance in np.unique(image[cone]):
            path=labels.get(str(int(instance)),labels.get(int(instance),''))
            if not isinstance(path,str):
                path=str(path)
            match=re.search(r'/Corn_(\d+)/Leaf_(\d+)',path)
            if match is None:
                continue
            plant,leaf=map(int,match.groups())
            pixels=(image==instance)&cone
            ys,xs=np.nonzero(pixels)
            if len(xs)<2 or t.np_random.random()<c.detection_drop:
                continue
            confidence=float(np.clip(.70+.28*min(1.,len(xs)/20)+t.np_random.normal(0,.015),0,.99))
            predicted=int(bool(t.scene.leaf_dry[leaf])^(t.np_random.random()<c.class_error))
            distance=float(laser[i,0])
            measured=(float(distance+t.np_random.normal(0,c.range_noise))
                      if instance==central_id and .12<=distance<=c.camera_range else None)
            detections.append({'box':[(xs.min()+xs.max()+1)/128,(ys.min()+ys.max()+1)/128,
                                       (xs.max()-xs.min()+1)/64,(ys.max()-ys.min()+1)/64],
                               'class_id':predicted,'confidence':confidence,'range':measured,
                               'plant':plant,'leaf':leaf})
        detections.sort(key=lambda d:d['confidence'],reverse=True)
        return {'detections':detections[:c.max_detections]}

    def _get_dones(self):
        self._refresh_state()
        actions=self.actions.detach().cpu().numpy()
        rewards=[]; terminated=[]; truncated=[]
        for i,t in enumerate(self.tasks):
            if t.done and not self.cfg.auto_reset:
                rewards.append(0.);terminated.append(False);truncated.append(False)
                continue
            boundary=bool(np.any(t.robot[:2]<0.) or np.any(t.robot[:2]>self.field_cfg.size))
            rock=False
            terrain=t.scene.terrain
            if terrain is not None and np.any(terrain.rock_blocking):
                delta=terrain.rock_xy[terrain.rock_blocking]-t.robot[:2]
                co,si=math.cos(t.robot[2]),math.sin(t.robot[2])
                x=delta[:,0]*co+delta[:,1]*si
                y=-delta[:,0]*si+delta[:,1]*co
                rock=bool(np.any(np.maximum(abs(x)-self.field_cfg.robot_length/2,0)**2+
                     np.maximum(abs(y)-self.field_cfg.robot_width/2,0)**2<=terrain.rock_radius[terrain.rock_blocking]**2))
            result=t.finish(actions[i],self._before[i],self._old_distance[i],rock,boundary)
            obs,reward,term,trunc,info=result
            if t.mode=='goto':
                shaping=self.cfg.distance_cost*info['distance']*self.step_dt
                reward-=shaping
                t.reward-=shaping;t.total_reward-=shaping
            if boundary or abs(t.body_pose[1])>1.2 or abs(t.body_pose[2])>1.2:
                t.done=True;t.reason='outside_or_overturned';term=True;trunc=False;reward-=5.
                t.reward=reward;t.total_reward-=5.
                info=t._info()
            self._last_results[i]=(obs,reward,term,trunc,info)
            rewards.append(reward);terminated.append(term and self.cfg.auto_reset);truncated.append(trunc and self.cfg.auto_reset)
            if term or trunc:
                self.episode_results.append({'mode':t.mode,'return':t.total_reward,**info})
                self.episode_results=self.episode_results[-2000:]
        self._reward=torch.tensor(rewards,device=self.device)
        return torch.tensor(terminated,device=self.device),torch.tensor(truncated,device=self.device)

    def _get_rewards(self):
        return self._reward

    def _get_observations(self):
        if any(not t.frames for t in self.tasks):
            # Both explicit and same-step autoresets must render the new poses.
            self.scene.write_data_to_sim()
            self.sim.forward()
            for name in ('lidar','rangefinder','depth','detector'):
                self.scene[name].update(0.,force_recompute=True)
        self._refresh_state()
        observations=[]
        for t in self.tasks:
            if not t.frames:
                obs,_=t.initialize_sensors()
            else:
                obs=t._observation()
            observations.append(obs)
        return {'policy':{key:torch.as_tensor(np.stack([o[key] for o in observations]),device=self.device)
                          for key in observations[0]}}

    def plant_positions(self):
        """Actual Newton plant poses, in stable plant-index order (debug/tests)."""
        return wp.to_torch(self.sim.physics_manager.get_state_0().body_q)[self._plant_body_ids,:3]

    @property
    def drive_config(self):
        spec = ROBOT_SPEC
        per_wheel_effort_nm = float(self.cfg.drive_effort_limit)
        max_track_speed_mps = float(self.field_cfg.max_speed)
        equivalent_motor_torque_nm = (
            per_wheel_effort_nm
            * spec.wheel_count_per_side
            * spec.drive_sprocket_radius_m
            / (spec.drive_reduction_ratio * spec.wheel_radius_m)
        )
        max_motor_rpm = (
            max_track_speed_mps
            / spec.drive_sprocket_radius_m
            * spec.drive_reduction_ratio
            * 60.0
            / (2.0 * math.pi)
        )
        return dict(
            interface='forward_and_differential_track_speed_v2',
            max_track_speed_mps=max_track_speed_mps,
            steering_track_speed_mps=(
                max_track_speed_mps
                if self.cfg.steering_track_speed is None
                else self.cfg.steering_track_speed
            ),
            track_acceleration_mps2=self.field_cfg.acceleration,
            velocity_gain=self.cfg.drive_velocity_gain,
            wheel_effort_limit_nm=per_wheel_effort_nm,
            wheel_effort_limit_basis='per_proxy_wheel_joint_effort; not measured motor torque',
            drive_sprocket_radius_m=spec.drive_sprocket_radius_m,
            drive_sprocket_diameter_m=spec.drive_sprocket_diameter_m,
            drive_reduction_ratio=spec.drive_reduction_ratio,
            proxy_wheel_radius_m=spec.wheel_radius_m,
            proxy_wheel_count_per_side=spec.wheel_count_per_side,
            track_ground_contact_length_m=spec.track_ground_contact_length_m,
            robot_asset_spec_sha256=robot_asset_spec_sha256(spec),
            drive_efficiency_assumption=1.0,
            equivalent_motor_torque_per_side_nm=equivalent_motor_torque_nm,
            max_motor_rpm_at_max_track_speed=max_motor_rpm,
            drivetrain_note=(
                'Torque and RPM are idealized proxy-equivalent values: one motor per side, '
                'unit efficiency, 60 mm active sprocket, and 45 mm proxy support rollers. '
                f'The {per_wheel_effort_nm:g} Nm per-wheel effort cap is not a measured motor rating.'
            ),
        )

    @property
    def observation_config(self):
        return dict(
            schema='state-last-command-v2',
            state_dim=44,
            measured_velocity_indices=[8, 9],
            previous_command_indices=[13, 14, 15, 16],
            previous_command_semantics='normalized_policy_command',
            action_components=['forward', 'differential', 'gimbal_yaw_rate', 'gimbal_pitch_rate'],
        )


def make_env(cfg=None, mode='mixed', num_envs=1, seed=7, auto_reset=True, *,
             goal_distance=None, goal_angle=math.pi, distance_cost=0., viewer_port=None):
    env_cfg=FieldEnvCfg(task=cfg or Config(),mode=mode,auto_reset=auto_reset,seed=seed,
                        goal_distance=goal_distance,goal_angle=goal_angle,distance_cost=distance_cost)
    env_cfg.scene.num_envs=num_envs
    if viewer_port is not None:
        from isaaclab_visualizers.viser import ViserVisualizerCfg
        env_cfg.sim.visualizer_cfgs=[ViserVisualizerCfg(port=viewer_port,bind_address='127.0.0.1',
            display_address='localhost',eye=(4.,-4.,3.),lookat=(2.,2.,0.),open_browser=False)]
    return IsaacFieldEnv(env_cfg)


class SingleFieldAdapter(gym.Env):
    """Single-row interface for policy evaluation and official Viser controls."""
    metadata={'render_modes':[]}
    def __init__(self, env:IsaacFieldEnv):
        if env.num_envs!=1:
            raise ValueError('SingleFieldAdapter requires num_envs=1')
        self.engine=env
        self.cfg=env.field_cfg
        self.observation_space=env.single_observation_space['policy']
        self.action_space=env.single_action_space

    @property
    def task(self):return self.engine.tasks[0]
    def __getattr__(self,name):return getattr(self.task,name)
    def current_observation(self):return self.task._observation()
    def reset(self,*,seed=None,options=None):
        self.engine.reset(seed=seed,options=options)
        return self.current_observation(),self.task._info()
    def step(self,action):
        if self.task.done:raise RuntimeError('episode ended; reset first')
        self.engine.step(torch.as_tensor(np.asarray(action,np.float32)[None],device=self.engine.device))
        return self.engine._last_results[0]
    def set_goal(self,goal):
        self.task.set_goal(goal)
        return self.current_observation(),self.task._info()
    def scene_payload(self):return self.task.scene_payload()
    def state_payload(self):
        state=self.task.state_payload()
        state.update(backend='Isaac Lab / Newton MJWarp',drive='14 driven rollers',
                     collision_metric='geometric footprint overlap with blocking rocks')
        return state
    def close(self):self.engine.close()
