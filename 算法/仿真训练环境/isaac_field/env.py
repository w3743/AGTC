"""Native batched Isaac Lab task: physical roller drive and live GPU sensors."""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re

import gymnasium as gym
import numpy as np
import torch
import warp as wp
wp.config.enable_backward = False

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCfg, ImuCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_from_matrix
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from .assets import (
    ROBOT_SPEC,
    FrontGuideSpec,
    GIMBAL_YAW_JOINT,
    GIMBAL_PITCH_JOINT,
    robot_articulation_cfg,
    robot_asset_spec_sha256,
)
from .body_stability import BodyStabilityAccumulator, BodyStabilityConfig, body_stability_sample
from .action_contract import ACTION_COMPONENTS, action_config as policy_action_config
from .drive_control import SideIntegralDriveController
from .physx_drive import (
    ensure_physx_robot_asset,
    audit_physx_coupling,
    build_physics_contact_metadata,
    same_side_wheel_coupling_metadata,
)
from .runtime import ensure_runtime, simulator_identity
from .live_state import LiveStateExporter, create_sensor_sample_exporter_from_environment
from .feedback import (
    ABSOLUTE_GOAL_INPUT_SCHEMA, FEEDBACK_DIM, STATE_DIM, FeedbackSample,
    feedback_config,
)
from .tensor_sensors import capture_depth_batch, capture_lidar_batch
from .tensor_task import (
    REASON_GOAL_REACHED,
    REASON_OUTSIDE_OR_OVERTURNED,
    REASON_TIME_LIMIT,
    fast_stop_v2_step,
    pack_emptyfield_goto_observation,
)
from .scene_assets import crop_asset, terrain_asset
from .layouts import generate_layout
from .task import (
    Config,
    EpisodePoseSamplingError,
    FAST_STOP_REWARD_PROFILES,
    FieldTask,
    Scene,
    sample_episode_start_goal,
    state_fields_for_schema,
    wrap,
)
from .terrain import Terrain

ROOT = Path(__file__).resolve().parents[1]


def _bridge_fabric_frame_view_setting(use_fabric, settings_manager=None):
    """Mirror SimCfg.use_fabric to the legacy setting read by FabricFrameView."""
    if settings_manager is None:
        from isaaclab.app.settings_manager import get_settings_manager
        settings_manager = get_settings_manager()
    enabled = bool(use_fabric)
    settings_manager.set("/physics/fabricEnabled", enabled)
    return enabled


def _disable_default_viewport_updates_if_headless(sim):
    """Stop only Kit's default viewport tick for headless training."""
    if sim.has_gui:
        return {"headless": False, "changed": False, "status": "gui_enabled"}
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    if viewport is None:
        return {"headless": True, "changed": False, "status": "no_active_viewport"}
    before = bool(viewport.updates_enabled)
    viewport.updates_enabled = False
    return {
        "headless": True,
        "changed": before,
        "status": "disabled",
        "api": "omni.kit.viewport.utility.get_active_viewport",
        "updates_enabled_before": before,
        "updates_enabled_after": bool(viewport.updates_enabled),
    }


class _CameraRigidPosePublisher:
    """Reuse upstream GPU conversion for actual PhysX rigid-body poses."""
    def __init__(self, sim):
        from isaaclab.scene_data.scene_data_provider import SceneDataProvider
        from isaaclab.scene_data.scene_data_backend import SceneDataFormat
        from isaaclab_physx.renderers.fabric import FabricBackend, FabricBackendCfg
        backend = sim.get_scene_data_provider().backend

        class TransformPublication:
            native_transform_formats = (SceneDataFormat.Transform,)

            def __getattr__(self, name):
                return getattr(backend, name)

            def get_transforms(self, output_format):
                return backend.transforms

        self.sim = sim
        self.backend = backend
        self.provider = SceneDataProvider(TransformPublication())
        self.publisher = FabricBackend(FabricBackendCfg(stage=sim.stage, device=str(sim.device)))
        self._last_timestamp = None
        self.publication_count = 0

    def publish(self):
        stamp = (self.backend.transforms_timestamp, self.backend.geometry_timestamp)
        if stamp == self._last_timestamp:
            return False
        self.sim.render_context.update_scene_state(self.sim.get_physics_step_count())
        self.publisher.update_transforms(self.provider)
        self._last_timestamp = (self.backend.transforms_timestamp, self.backend.geometry_timestamp)
        self.publication_count += 1
        return True

    def close(self):
        self.publisher.close()


def _forward_publish_scene_and_render(sim):
    """Forward once and publish real GPU poses before camera capture."""
    sim.forward()
    publisher = getattr(sim, '_agtc_camera_rigid_pose_publisher', None)
    if publisher is None:
        sim.render_context.update_scene_state(sim.get_physics_step_count())
    else:
        publisher.publish()
    sim.render()


def ray_ranges(data):
    """Metric ranges from public PhysX hit points; these patterns start at the sensor origin."""
    return torch.linalg.vector_norm(data.ray_hits_w.torch - data.pos_w.torch[:, None, :], dim=-1)


@configclass
class FieldSceneCfg(InteractiveSceneCfg):
    # The official launcher scans config before IsaacFieldEnv builds geometry.
    # Declare camera intent here so its normal scan selects the RTX experience;
    # the constructor replaces this with the task's calibrated camera config.
    depth = CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/gimbal_yaw_link/gimbal_pitch_link/Stereo',
        width=88, height=160, data_types=['distance_to_image_plane'],
        renderer_cfg=IsaacRtxRendererCfg(), spawn=None)


@configclass
class FieldEnvCfg(DirectRLEnvCfg):
    task: Config = Config()
    mode: str = 'mixed'
    task_backend: str = 'reference'
    auto_reset: bool = True
    # Curriculum limits affect task sampling, never replace policy control.
    goal_distance: tuple[float, float] | None = None
    goal_angle: float = math.pi
    distance_cost: float = 0.0
    spawn_near_crops: float = 0.8
    # Explicit torque integration needs a gain below I_spin / physics_dt.
    # This proxy value is a numerical model parameter, not a motor rating.
    drive_velocity_gain: float = 0.04
    # Simulation candidate only: side-mean wheel speed-error integral feedforward.
    drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s: float = 0.1
    drive_effort_limit: float = 2.0
    couple_same_side_wheels: bool = True
    # Optional pair of passive front-guide rollers. These are asset/contact
    # geometry only and never enter the 14-joint drive actuator contract.
    # configclass recursively walks dataclasses, so keep the frozen geometry
    # object serialized until IsaacFieldEnv restores it after config wrapping.
    front_guide_spec: dict | None = None
    # Optional body-attitude reward shaping; never added to policy observations.
    # configclass recursively copies/wraps nested values, so keep the frozen
    # reward config here as a plain serialized mapping and restore its type in
    # IsaacFieldEnv after the config wrapper has run.
    body_stability_config: dict | None = None
    # Model parameters, not calibrated N5065 electrical/gear measurements.
    motor_torque_constant_nm_per_a: float = 0.035
    drive_efficiency_assumption: float = 1.0
    imu_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    imu_offset_rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scene = FieldSceneCfg(num_envs=4, env_spacing=110., replicate_physics=True)
    sim = sim_utils.SimulationCfg(
        dt=.001, render_interval=50,
        physics=PhysxCfg(solver_type=1, min_position_iteration_count=40,
                         max_position_iteration_count=40, min_velocity_iteration_count=1,
                         max_velocity_iteration_count=1),
        physics_material=PhysxRigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.8,
            restitution=0., friction_combine_mode='average', restitution_combine_mode='average'),
    )
    decimation = 50
    episode_length_s = 900.
    action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
    observation_space = gym.spaces.Dict({
        'lidar': gym.spaces.Box(0., 1., (1, 400), np.float32),
        # Isaac Lab's SB3 adapter recognizes float images with [-1, 1] bounds
        # as already normalized. Actual metric-depth values remain in [0, 1].
        'depth': gym.spaces.Box(-1., 1., (1, 160, 88), np.float32),
        'detections': gym.spaces.Box(-1., 1., (16, 8), np.float32),
        'state': gym.spaces.Box(-1., 1., (STATE_DIM,), np.float32),
        'feedback': gym.spaces.Box(-np.inf, np.inf, (FEEDBACK_DIM,), np.float32),
    })
    state_space = 0
    compute_final_obs = True
    seed = 7


class IsaacFieldEnv(DirectRLEnv):
    cfg: FieldEnvCfg

    def __init__(self, cfg: FieldEnvCfg | None = None, render_mode=None, **kwargs):
        cfg = cfg or FieldEnvCfg()
        self.task_backend = getattr(cfg, 'task_backend', 'reference')
        self._validate_task_backend(cfg, self.task_backend)
        if not isinstance(cfg.sim.physics, PhysxCfg):
            raise ValueError('IsaacFieldEnv requires the Isaac Sim PhysX backend')
        ensure_runtime(cfg, device=cfg.sim.device)
        self.front_guide_spec = (
            None if cfg.front_guide_spec is None else FrontGuideSpec(**cfg.front_guide_spec)
        )
        self.body_stability_config = (
            None if cfg.body_stability_config is None
            else BodyStabilityConfig(**cfg.body_stability_config)
        )
        self.field_cfg = c = cfg.task
        for name, value in (('max_speed', c.max_speed), ('acceleration', c.acceleration),
                            ('drive_velocity_gain', cfg.drive_velocity_gain),
                            ('drive_effort_limit', cfg.drive_effort_limit),
                            ('motor_torque_constant_nm_per_a', cfg.motor_torque_constant_nm_per_a),
                            ('drive_efficiency_assumption', cfg.drive_efficiency_assumption)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if (not math.isfinite(cfg.drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s)
                or cfg.drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s < 0):
            raise ValueError('drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s must be finite and non-negative')
        if cfg.mode not in ('mixed', 'goto', 'search'):
            raise ValueError('mode must be mixed/search/goto')
        spin_inertia = .5 * ROBOT_SPEC.wheel_mass_kg * ROBOT_SPEC.wheel_radius_m ** 2
        if cfg.drive_velocity_gain * cfg.sim.dt > spin_inertia:
            raise ValueError('explicit drive gain must satisfy gain * physics_dt <= proxy wheel spin inertia')
        if not math.isclose(cfg.sim.dt * cfg.decimation, c.dt):
            raise ValueError('task dt must match physics dt * decimation')
        cfg.episode_length_s = c.max_time
        cfg.scene.env_spacing = c.size + 10.
        cfg.observation_space = gym.spaces.Dict({
            'lidar': gym.spaces.Box(0., 1., (1, c.lidar_rays), np.float32),
            'depth': gym.spaces.Box(-1., 1., (1, c.depth_height, c.depth_width), np.float32),
            'detections': gym.spaces.Box(-1., 1., (c.max_detections, 8), np.float32),
            'state': gym.spaces.Box(-1., 1., (STATE_DIM,), np.float32),
            'feedback': gym.spaces.Box(-np.inf, np.inf, (FEEDBACK_DIM,), np.float32),
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
        asset_fingerprint = {'config': asdict(c), 'seed': cfg.seed, 'asset_revision': 9,
                             'physics_backend': 'physx'}
        if self.front_guide_spec is not None:
            asset_fingerprint['robot_asset_spec_sha256'] = robot_asset_spec_sha256(
                ROBOT_SPEC, self.front_guide_spec,
            )
        fingerprint = hashlib.sha256(json.dumps(asset_fingerprint, sort_keys=True).encode()).hexdigest()[:16]
        asset_dir = ROOT / 'results/isaac_field/assets' / fingerprint
        asset_dir.mkdir(parents=True, exist_ok=True)
        terrain_path = asset_dir / 'terrain.usda'
        if not terrain_path.exists():
            terrain_asset(self.terrain, terrain_path)
        crop_path=asset_dir/'crop.usda'
        if c.plants and not crop_path.exists():
            crop_asset(template,asset_dir, physics_backend='physx')
        cfg.scene.field = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Field',
            spawn=sim_utils.UsdFileCfg(usd_path=str(terrain_path),
                physics_material=cfg.sim.physics_material,
                collision_props=[PhysxCollisionCfg(contact_offset=.005, rest_offset=0.)]))
        robot_asset = ensure_physx_robot_asset(
            ROOT / 'results/isaac_field/assets', front_guide_spec=self.front_guide_spec,
            couple_support_wheels=cfg.couple_same_side_wheels,
        )
        robot = robot_articulation_cfg(robot_asset, front_guide_spec=self.front_guide_spec)
        robot.prim_path = '{ENV_REGEX_NS}/Robot'
        robot.init_state.pos = (2., 2., .3)
        robot.actuators = {
            # The shared explicit-actuator adapter clamps the combined PD and
            # feedforward effort before writing it to the PhysX joint view.
            'drive': IdealPDActuatorCfg(joint_names_expr=['.*_wheel_joint'], stiffness=0., damping=cfg.drive_velocity_gain,
                                        actuator_effort_limit=cfg.drive_effort_limit,
                                        joint_effort_limit=cfg.drive_effort_limit, joint_velocity_limit=110.),
            'gimbal': ImplicitActuatorCfg(joint_names_expr=['gimbal_.*_joint'], stiffness=1., damping=.1,
                                         joint_effort_limit=1., joint_velocity_limit=3.),
        }
        cfg.scene.robot = robot
        cfg.scene.light = AssetBaseCfg(prim_path='/World/FieldLight',
            spawn=sim_utils.DomeLightCfg(intensity=2000., color=(.75, .75, .75)))
        if c.plants:
            cfg.scene.corn = RigidObjectCollectionCfg(rigid_objects={
                f'Corn_{i}': RigidObjectCfg(prim_path=f'{{ENV_REGEX_NS}}/Plants/Corn_{i}',
                    spawn=sim_utils.UsdFileCfg(usd_path=str(asset_dir / 'corn' / f'corn_{i:04}.usda')),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(*map(float, template.xy[i]), float(template.root_z[i]))))
                for i in range(c.plants)})
        mount = '{ENV_REGEX_NS}/Robot/gimbal_yaw_link/gimbal_pitch_link'
        cfg.scene.imu = ImuCfg(prim_path=mount, update_period=0.,
            offset=ImuCfg.OffsetCfg(pos=cfg.imu_offset_pos, rot=cfg.imu_offset_rot_xyzw))
        ray_targets = [MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr='{ENV_REGEX_NS}/Field', is_shared=True, track_mesh_transforms=False)]
        if c.plants:
            # Include visual leaf meshes and stem geometry; collision masks do
            # not make plants transparent to the physical range sensors.
            # One body expression per target also keeps PhysX's body-major
            # view order aligned with the caster's environment-major table.
            ray_targets.extend(MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr=f'{{ENV_REGEX_NS}}/Plants/Corn_{i}', is_shared=True,
                track_mesh_transforms=True) for i in range(c.plants))
        cfg.scene.lidar = MultiMeshRayCasterCfg(prim_path=mount, update_period=0., mesh_prim_paths=ray_targets,
            pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0., 0.),
                horizontal_fov_range=(-180., 180.), horizontal_res=360./c.lidar_rays),
            ray_alignment='base', max_distance=c.lidar_range)
        cfg.scene.rangefinder = MultiMeshRayCasterCfg(prim_path=mount, update_period=0., mesh_prim_paths=ray_targets,
            pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0., 0.),
                horizontal_fov_range=(0., 0.), horizontal_res=1.),
            ray_alignment='base', max_distance=c.camera_range)
        cfg.scene.depth = CameraCfg(prim_path=mount+'/Stereo', width=c.depth_width, height=c.depth_height,
            offset=CameraCfg.OffsetCfg(convention='world'), update_period=0., update_latest_camera_pose=True,
            data_types=['distance_to_image_plane'], renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(prototype.depth_intrinsics.ravel().tolist(),
                c.depth_width, c.depth_height, clipping_range=(c.depth_min_range,c.depth_range)))
        detector_width = 64
        detector_focal = detector_width / (2 * math.tan(math.pi / 6))
        detector_k = [detector_focal,0,31.5,0,detector_focal,31.5,0,0,1]
        cfg.scene.detector = CameraCfg(prim_path=mount+'/Detector', width=detector_width, height=detector_width,
            offset=CameraCfg.OffsetCfg(convention='world'), update_period=0., update_latest_camera_pose=True,
            data_types=['instance_segmentation','distance_to_image_plane'],
            renderer_cfg=IsaacRtxRendererCfg(colorize_instance_segmentation=False),
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(detector_k,64,64,
                clipping_range=(.12,c.camera_range)))
        self.tasks = [FieldTask(c, 'goto') for _ in range(cfg.scene.num_envs)]
        for task in self.tasks:
            task.feedback_model_config = {'sensor_mount_rot_xyzw': list(cfg.imu_offset_rot_xyzw)}
            task.body_angular_velocity = np.zeros(3, dtype=np.float64)
            task.root_com_velocity_xy = np.zeros(2, dtype=np.float64)
            task.body_stability_sample = None
            task.body_stability_stats = BodyStabilityAccumulator()
        guide_coupling_enabled = (
            self.front_guide_spec is not None
            and self.front_guide_spec.rotation_coupling == 'same_side_track_speed'
        )
        # Camera FrameView reads /physics/fabricEnabled, while current PhysX
        # setup publishes SimCfg.use_fabric as /isaaclab/fabric_enabled.
        _bridge_fabric_frame_view_setting(cfg.sim.use_fabric)
        super().__init__(cfg, render_mode, **kwargs)
        self._default_viewport_updates = _disable_default_viewport_updates_if_headless(self.sim)
        self._same_side_mimic_model_audit = audit_physx_coupling(self)
        self._same_side_wheel_coupling_metadata = same_side_wheel_coupling_metadata(
            cfg.couple_same_side_wheels,
            guide_relations_per_world=(
                2 if guide_coupling_enabled else 0
            ) if self.front_guide_spec is not None else None,
        )
        self._physics_contact_metadata = build_physics_contact_metadata(self)
        self.rover = self.scene['robot']
        self._camera_rigid_pose_publisher = _CameraRigidPosePublisher(self.sim)
        self.sim._agtc_camera_rigid_pose_publisher = self._camera_rigid_pose_publisher
        live_state_path = os.environ.get('ISAAC_FIELD_LIVE_STATE_FILE')
        self._live_state_exporter = None
        if live_state_path:
            self._live_state_exporter = LiveStateExporter(
                live_state_path,
                robot_usd_path=cfg.scene.robot.spawn.usd_path,
                terrain_usd_path=cfg.scene.field.spawn.usd_path,
                body_names=self.rover.body_names,
            )
        self._sensor_sample_exporter = create_sensor_sample_exporter_from_environment(
            run_id=(self._live_state_exporter.run_id if self._live_state_exporter is not None else None),
            num_envs=self.num_envs,
            lidar_rays=self.field_cfg.lidar_rays,
            depth_height=self.field_cfg.depth_height,
            depth_width=self.field_cfg.depth_width,
        )
        self._sensor_sample_lidar_directions_local = None
        self._wheel_names = [f'{w.name}_wheel_joint' for w in ROBOT_SPEC.wheel_specs]
        self._wheel_ids = torch.tensor([self.rover.joint_names.index(n) for n in self._wheel_names], device=self.device)
        self._wheel_sides = torch.tensor([-1. if w.side=='left' else 1. for w in ROBOT_SPEC.wheel_specs],device=self.device)
        self._wheel_track_ids = (self._wheel_sides > 0).long()
        wheel_track_ids = self._wheel_track_ids.detach().cpu().numpy()
        self._wheel_side_masks = (wheel_track_ids == 0, wheel_track_ids == 1)
        self._gimbal_ids = torch.tensor([self.rover.joint_names.index(n) for n in
                                        (GIMBAL_YAW_JOINT,GIMBAL_PITCH_JOINT)],device=self.device)
        self._mount_id = self.rover.body_names.index('gimbal_pitch_link')
        self._imu_offset_rotation = matrix_from_quat(torch.as_tensor(
            self.cfg.imu_offset_rot_xyzw, device=self.device, dtype=torch.float32
        )[None]).detach().cpu().numpy()[0].copy()
        spec = ROBOT_SPEC
        self._torque_to_iq = (spec.drive_sprocket_radius_m / (spec.wheel_radius_m
            * spec.drive_reduction_ratio * self.cfg.drive_efficiency_assumption
            * self.cfg.motor_torque_constant_nm_per_a))
        self._omega_to_motor_rpm = (spec.wheel_radius_m / spec.drive_sprocket_radius_m
            * spec.drive_reduction_ratio * 60. / (2. * math.pi))
        self._state_observation_fresh = False
        self._feedback_host_cache = None
        self._finished_observation_cache_mask = None
        self._sim_tensor_cache = None
        self._tensor_finished_observation_cache = None
        self._tensor_observation_cache = None
        self._tensor_sensor_frame_cache = {}
        self._tensor_capture_env_ids = None
        self._tensor_capture_initial_env_ids = frozenset()
        self.actions = torch.zeros((self.num_envs,4),device=self.device)
        self.command = torch.zeros((self.num_envs,2),device=self.device)
        self.track_command = torch.zeros_like(self.command)
        self.wheel_target_velocity = torch.zeros((self.num_envs,14),device=self.device)
        self._drive_pi = SideIntegralDriveController(
            self.num_envs,
            self._wheel_track_ids,
            device=self.device,
            kp=cfg.drive_velocity_gain,
            ki=cfg.drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s,
            dt=self.step_dt,
            limit=cfg.drive_effort_limit,
        )
        self.gimbal_target = torch.zeros_like(self.command)
        self._reward = torch.zeros(self.num_envs,device=self.device)
        self._last_results = [None] * self.num_envs
        if self.task_backend == 'torch_batched':
            self._tensor_goal_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float64)
            self._tensor_episode_time = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float64)
            self._tensor_hold = torch.zeros_like(self._tensor_episode_time)
            self._tensor_initial_goal_distance = torch.zeros_like(self._tensor_episode_time)
            self._tensor_previous_physics_failure = torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.bool,
            )
            self._tensor_last_actions = torch.zeros((self.num_envs, 4), device=self.device)
            self._tensor_old_distance = torch.zeros_like(self._tensor_episode_time)
            self._tensor_lidar_ranges = torch.full(
                (self.num_envs, c.lidar_rays), c.lidar_range, device=self.device, dtype=torch.float32,
            )
            self._tensor_lidar_valid = torch.zeros(
                (self.num_envs, c.lidar_rays), device=self.device, dtype=torch.bool,
            )
            self._tensor_depth_values = torch.full(
                (self.num_envs, c.depth_height, c.depth_width), c.depth_range,
                device=self.device, dtype=torch.float32,
            )
            self._tensor_depth_valid = torch.zeros(
                (self.num_envs, c.depth_height, c.depth_width), device=self.device, dtype=torch.bool,
            )
            self._tensor_lidar_frame_seq = [None] * self.num_envs
            self._tensor_depth_frame_seq = [None] * self.num_envs
            self._tensor_lidar_policy_cache = None
            self._tensor_depth_policy_cache = None
        # Kinematic plant roots use the backend-neutral public collection API.
        self.plants = self.scene['corn'] if c.plants else None
        for i,t in enumerate(self.tasks):
            t.attach_sensors(lambda kind, i=i: self._capture(i,kind))
            if self.task_backend == 'torch_batched':
                t._tensor_observation_callback = lambda i=i: self._tensor_observation_row(i)
                t._tensor_state_sync_callback = lambda i=i: self._sync_tensor_task_state((i,))

    def _setup_scene(self):
        super()._setup_scene()
        from pxr import UsdPhysics
        actuator = self.cfg.scene.robot.actuators['gimbal']
        self._gimbal_drive_schema = []
        for prim in self.sim.stage.Traverse():
            if prim.GetName() not in (GIMBAL_YAW_JOINT, GIMBAL_PITCH_JOINT):
                continue
            was_enabled = bool(prim.HasAPI(UsdPhysics.DriveAPI, 'angular'))
            drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(float(actuator.stiffness))
            drive.CreateDampingAttr(float(actuator.damping))
            drive.CreateMaxForceAttr(float(actuator.joint_effort_limit))
            self._gimbal_drive_schema.append({
                'path': str(prim.GetPath()), 'angular_drive_previously_authored': was_enabled,
            })

    @staticmethod
    def _validate_task_backend(cfg, task_backend: str) -> None:
        if task_backend not in ('reference', 'torch_batched'):
            raise ValueError("task_backend must be 'reference' or 'torch_batched'")
        if task_backend == 'reference':
            return
        task = cfg.task
        perf_physics = os.environ.get('ISAAC_FIELD_PERF_PHYSICS', '').strip() == '1'
        if perf_physics:
            policy_dt = float(cfg.sim.dt) * int(cfg.decimation)
            if not math.isclose(policy_dt, float(task.dt), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    'ISAAC_FIELD_PERF_PHYSICS requires sim.dt * decimation == task.dt; '
                    f'got {cfg.sim.dt} * {cfg.decimation} != {task.dt}'
                )
            expected_dt = float(cfg.sim.dt)
            expected_decimation = int(cfg.decimation)
        else:
            expected_dt, expected_decimation = 0.001, 50
        checks = (
            ('mode', cfg.mode, 'goto'),
            ('task.size', float(task.size), 12.0),
            ('task.plants', int(task.plants), 0),
            ('task.terrain_enabled', bool(task.terrain_enabled), False),
            ('task.rock_density', float(task.rock_density), 0.0),
            ('task.goto_reward_profile', task.goto_reward_profile, 'fast_stop_v2'),
            ('task.input_schema', task.input_schema, ABSOLUTE_GOAL_INPUT_SCHEMA),
            ('task.fast_stop_failure_remaining_time_cost',
             task.fast_stop_failure_remaining_time_cost, True),
            ('task.goal_tolerance', float(task.goal_tolerance), 0.4),
            ('task.goal_stop_speed', float(task.goal_stop_speed), 0.05),
            ('task.goal_stop_angular_speed', float(task.goal_stop_angular_speed), 0.1),
            ('task.goal_hold', float(task.goal_hold), 0.5),
            ('task.goal_progress_budget', task.goal_progress_budget, 5.0),
            ('task.max_time', float(task.max_time), 60.0),
            ('task.dt', float(task.dt), 0.05),
            ('sim.dt', float(cfg.sim.dt), expected_dt),
            ('decimation', int(cfg.decimation), expected_decimation),
            ('sim.render_interval', int(cfg.sim.render_interval), 50),
            ('sim.physics.min_position_iteration_count',
             int(cfg.sim.physics.min_position_iteration_count), 40),
            ('sim.physics.max_position_iteration_count',
             int(cfg.sim.physics.max_position_iteration_count), 40),
            ('sim.physics.min_velocity_iteration_count',
             int(cfg.sim.physics.min_velocity_iteration_count), 1),
            ('sim.physics.max_velocity_iteration_count',
             int(cfg.sim.physics.max_velocity_iteration_count), 1),
            ('sim.physics.solver_type', int(cfg.sim.physics.solver_type), 1),
        )
        mismatches = []
        for name, actual, expected in checks:
            if (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                    and isinstance(expected, (int, float)) and not isinstance(expected, bool)):
                matches = math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
            else:
                matches = actual == expected
            if not matches:
                mismatches.append(f'{name}={actual!r} (requires {expected!r})')
        if int(task.lidar_rays) != 400:
            mismatches.append(f'task.lidar_rays={task.lidar_rays!r} (requires 400)')
        if (int(task.depth_height), int(task.depth_width)) != (160, 88):
            mismatches.append(
                f'task depth shape={(task.depth_height, task.depth_width)!r} (requires (160, 88))'
            )
        if int(task.max_detections) != 16:
            mismatches.append(f'task.max_detections={task.max_detections!r} (requires 16)')
        if mismatches:
            raise ValueError('torch_batched task backend does not support: ' + '; '.join(mismatches))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_options = options
        return super().reset(seed=seed, options=options)

    def _sync_tensor_task_state(self, env_ids=None):
        """Copy only small task scalars into persistent device-side buffers."""
        if self.task_backend != 'torch_batched' or not hasattr(self, '_tensor_goal_xy'):
            return
        ids = list(range(self.num_envs)) if env_ids is None else [int(i) for i in env_ids]
        if not ids:
            return
        index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        selected = [self.tasks[i] for i in ids]
        self._tensor_goal_xy[index] = torch.as_tensor(
            np.stack([task.goal for task in selected]), dtype=torch.float64, device=self.device,
        )
        self._tensor_episode_time[index] = torch.as_tensor(
            [task.time for task in selected], dtype=torch.float64, device=self.device,
        )
        self._tensor_hold[index] = torch.as_tensor(
            [task._hold for task in selected], dtype=torch.float64, device=self.device,
        )
        self._tensor_initial_goal_distance[index] = torch.as_tensor(
            [task._initial_goal_distance for task in selected], dtype=torch.float64, device=self.device,
        )
        self._tensor_previous_physics_failure[index] = torch.as_tensor(
            [task.physics_failure for task in selected], dtype=torch.bool, device=self.device,
        )
        self._tensor_last_actions[index] = torch.as_tensor(
            np.stack([task.last_action for task in selected]), dtype=torch.float32, device=self.device,
        )
        self._tensor_old_distance[index] = 0.0
        for env_id in ids:
            self._tensor_lidar_frame_seq[env_id] = None
            self._tensor_depth_frame_seq[env_id] = None
        self._tensor_lidar_policy_cache = None
        self._tensor_depth_policy_cache = None
        self._tensor_observation_cache = None
        self._tensor_finished_observation_cache = None

    def _reset_idx(self, env_ids):
        self._invalidate_cached_samples()
        ids = [int(i) for i in env_ids.tolist()]
        super()._reset_idx(env_ids)
        poses = self.rover.data.default_root_pose.torch[env_ids].clone()
        plant_poses = []
        for row,i in enumerate(ids):
            t = self.tasks[i]
            t.body_angular_velocity = np.zeros(3, dtype=np.float64)
            t.root_com_velocity_xy = np.zeros(2, dtype=np.float64)
            t.body_stability_sample = None
            t.body_stability_stats = BodyStabilityAccumulator()
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
            random_start=options.get('random_start', 'start' not in options)
            if not isinstance(random_start, (bool, np.bool_)):
                raise ValueError('random_start must be a boolean')
            if random_start and 'start' in options:
                raise ValueError('start and random_start=True cannot be supplied together')
            # Reject malformed explicit coordinates before retrying random maps;
            # map-specific clearance failures below may legitimately be retried.
            radius=math.hypot(self.field_cfg.robot_length,self.field_cfg.robot_width)/2
            if 'start' in options:
                fixed_start=np.asarray(options['start'],dtype=np.float64).reshape(-1)
                if (fixed_start.shape!=(3,) or not np.isfinite(fixed_start).all()
                        or np.any(fixed_start[:2]<radius)
                        or np.any(fixed_start[:2]>self.field_cfg.size-radius)):
                    raise ValueError('start must contain finite x, y, yaw with the vehicle inside the field')
            if 'goal' in options:
                fixed_goal=np.asarray(options['goal'],dtype=np.float64).reshape(-1)
                if fixed_goal.shape!=(2,) or not np.isfinite(fixed_goal).all():
                    raise ValueError('goal must contain two finite coordinates')
            map_attempts=1 if supplied is not None else 64
            last_start=options.get('start')
            last_goal=options.get('goal')
            last_failure=None
            for attempt in range(map_attempts):
                if supplied is not None:
                    xy=supplied.xy
                else:
                    xy=generate_layout(self.field_cfg,layout_rng,terrain=self.terrain,
                        start=options.get('start'),goal=options.get('goal'))
                scene=Scene(self.field_cfg,xy,self._targets,np.random.default_rng(self._geometry_seed),terrain=self.terrain)
                try:
                    sampled_start,sampled_goal=sample_episode_start_goal(
                        scene,layout_rng,
                        start=options.get('start'),goal=options.get('goal'),
                        random_start=random_start,
                        spawn_near_crops=self.cfg.spawn_near_crops if random_start else 0.0,
                        mode=options['mode'],
                        goal_distance=(self.cfg.goal_distance
                                       if 'goal' not in options and options['mode']=='goto' else None),
                        goal_angle=self.cfg.goal_angle,
                    )
                    last_start,last_goal=sampled_start,sampled_goal
                    # prepare() validates this exact pair. Nothing below reset
                    # may replace either endpoint after the full-scene check.
                    candidate_options={**options,'scene':scene,'start':sampled_start,
                                       'goal':sampled_goal,'random_start':False}
                    t.prepare(seed=episode_seed,options=candidate_options)
                    break
                except ValueError as exc:
                    last_failure=exc
                    if isinstance(exc,EpisodePoseSamplingError):
                        if exc.start is not None:
                            last_start=exc.start
                        if exc.goal is not None:
                            last_goal=exc.goal
                    if supplied is not None or attempt==map_attempts-1:
                        raise ValueError(
                            f"reset failed: env_id={i}, episode_seed={episode_seed}, "
                            f"attempts={attempt+1}/{map_attempts}, "
                            f"start={None if last_start is None else np.asarray(last_start).tolist()}, "
                            f"goal={None if last_goal is None else np.asarray(last_goal).tolist()}: {exc}"
                        ) from exc
            # prepare() also serves simulator-independent callers and clears
            # native kinematic samples; native resets then seed an explicit
            # zero until the next root_com_vel_w refresh.
            t.root_com_velocity_xy = np.zeros(2, dtype=np.float64)
            t.body_angular_velocity = np.zeros(3, dtype=np.float64)
            # Spawn tangent to the local surface, with every roller above its
            # actual contact height. A level chassis on a slope can start with
            # deeply embedded front/rear rollers and receive a false launch impulse.
            slope=np.asarray(self.terrain.slope(t.robot[:2]),dtype=float)
            normal=np.array([-slope[0],-slope[1],1.]);normal/=np.linalg.norm(normal)
            forward=np.array([math.cos(t.robot[2]),math.sin(t.robot[2]),0.])
            forward[2]=np.dot(slope,forward[:2]);forward/=np.linalg.norm(forward)
            left=np.cross(normal,forward);left/=np.linalg.norm(left)
            rotation=np.stack((forward,left,np.cross(forward,left)),-1)
            contact_geometry = [
                (w.x_m, w.y_m, w.z_m, ROBOT_SPEC.wheel_radius_m)
                for w in ROBOT_SPEC.wheel_specs
            ]
            if self.front_guide_spec is not None:
                contact_geometry.extend(
                    (w.x_m, w.y_m, w.z_m, w.radius_m)
                    for w in self.front_guide_spec.wheel_specs(ROBOT_SPEC)
                )
            bottoms = np.asarray(
                [[x, y, z - radius] for x, y, z, radius in contact_geometry],
                dtype=np.float64,
            ) @ rotation.T
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
            self.plants.write_body_pose_to_sim_index(body_poses=pp, env_ids=env_ids)
            self.plants.write_body_velocity_to_sim_index(
                body_velocities=torch.zeros((*pp.shape[:2], 6), device=self.device), env_ids=env_ids)
        self.command[env_ids]=0
        self.track_command[env_ids]=0
        self.wheel_target_velocity[env_ids]=0
        self._drive_pi.reset(env_ids)
        self.rover.actuators.target_command.set_effort_index(
            value=torch.zeros((len(ids), len(self._wheel_ids)), device=self.device, dtype=self._drive_pi.integral.dtype),
            joint_ids=self._wheel_ids,
            env_ids=env_ids,
        )
        self.gimbal_target[env_ids]=0
        self.actions[env_ids]=0
        self._sensor_cache.clear()
        self._refresh_state()
        self._sync_tensor_task_state(ids)
        self._reset_options=None

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError('actions must be finite')
        self._finished_observation_cache_mask = None
        if getattr(self, 'task_backend', 'reference') == 'torch_batched':
            self._tensor_finished_observation_cache = None
            self._tensor_observation_cache = None
            self._tensor_sensor_frame_cache.clear()
        self.actions = actions.clamp(-1.,1.)
        if getattr(self, 'task_backend', 'reference') == 'torch_batched':
            root_xy = (
                self.rover.data.root_link_pose_w.torch[:, :2]
                - self.scene.env_origins[:, :2]
            ).to(dtype=torch.float64)
            self._tensor_old_distance = torch.linalg.vector_norm(
                self._tensor_goal_xy - root_xy, dim=1,
            )
            self._before = None
            self._old_distance = self._tensor_old_distance
        else:
            self._before = [t.robot.copy() for t in self.tasks]
            self._old_distance = [float(np.linalg.norm(t.goal-t.robot[:2])) for t in self.tasks]
        desired = self.actions[:, :2] * self.field_cfg.max_speed
        change = self.field_cfg.acceleration * self.step_dt
        self.track_command += (desired - self.track_command).clamp(-change, change)
        self.wheel_target_velocity = self.track_command[:, self._wheel_track_ids] / ROBOT_SPEC.wheel_radius_m
        wheel_qd = self.rover.data.joint_vel.torch[:, self._wheel_ids]
        self._drive_pi.update(self.wheel_target_velocity, wheel_qd)
        # Nominal no-slip twist is retained for diagnostics only. State/reward
        # continue using the actual chassis velocities measured from physics.
        self.command[:, 0] = self.track_command.mean(dim=1)
        self.command[:, 1] = (self.track_command[:, 1] - self.track_command[:, 0]) / ROBOT_SPEC.track_separation_m
        self.gimbal_target+=self.actions[:,2:]*torch.tensor([self.field_cfg.gimbal_yaw_rate,self.field_cfg.gimbal_pitch_rate],device=self.device)*self.step_dt
        self.gimbal_target[:,0]=(self.gimbal_target[:,0]+math.pi)%(2*math.pi)-math.pi
        self.gimbal_target[:,1].clamp_(self.field_cfg.pitch_min,self.field_cfg.pitch_max)
        self._sensor_cache.clear()

        # The target buffers persist across physics steps. This action is held
        # for the full decimation window, so write them once per policy step;
        # actuator computation and PhysX submission still run on every substep.
        target_command = self.rover.actuators.target_command
        target_command.set_velocity_index(value=self.wheel_target_velocity, joint_ids=self._wheel_ids)
        target_command.set_position_index(value=self.gimbal_target, joint_ids=self._gimbal_ids)
        target_command.set_effort_index(value=self._drive_pi.feedforward, joint_ids=self._wheel_ids)

    def _apply_action(self):
        # Isaac Lab invokes this once per physics substep. The persistent target
        # buffers are written in _pre_physics_step; per-substep actuator compute
        # and scene.write_data_to_sim remain responsible for applying them.
        pass

    def _invalidate_cached_samples(self):
        """Force the next observation to sample state after reset or rendering."""
        self._state_observation_fresh = False
        self._feedback_host_cache = None
        self._finished_observation_cache_mask = None
        if getattr(self, 'task_backend', 'reference') == 'torch_batched':
            self._tensor_finished_observation_cache = None
            self._tensor_observation_cache = None
            self._tensor_lidar_policy_cache = None
            self._tensor_depth_policy_cache = None
            sensor_frame_cache = getattr(self, '_tensor_sensor_frame_cache', None)
            if sensor_frame_cache is not None:
                sensor_frame_cache.clear()

    def _collect_sim_cpu_samples(self, *, include_feedback=False):
        """Move the current small state/feedback sample to host memory once."""
        poses=self.rover.data.root_link_pose_w.torch
        velocities=self.rover.data.root_com_vel_w.torch
        angles=torch.stack(euler_xyz_from_quat(poses[:,3:]),-1)
        rotation=matrix_from_quat(poses[:,3:])
        forward_speed=(velocities[:,:3]*rotation[:,:,0]).sum(-1, keepdim=True)
        # Simulator truth for chassis stability: PhysX supplies angular
        # velocity in world axes; rotate it into body axes. The moving-gimbal
        # IMU and commanded rates are not substitutes for this signal.
        body_angular_velocity = torch.bmm(
            rotation.transpose(1, 2), velocities[:, 3:6].unsqueeze(-1),
        ).squeeze(-1)
        origin_offsets=self.scene.env_origins
        mount=self.rover.data.body_link_pose_w.torch[:,self._mount_id]
        mount_rotation=matrix_from_quat(mount[:,3:])
        joints=self.rover.data.joint_pos.torch[:,self._gimbal_ids]
        state_tensors = {
            'angles': angles,
            'forward_speed': forward_speed,
            'body_angular_velocity': body_angular_velocity,
            'root': poses,
            'origin_offsets': origin_offsets,
            'mount_rotation': mount_rotation.reshape(mount_rotation.shape[0], 9),
            'mount_pose': mount,
            'joints': joints,
            'angular': velocities[:,5:6],
            'root_com_velocity_xy': velocities[:,:2],
        }
        live_state_exporter = getattr(self, '_live_state_exporter', None)
        if live_state_exporter is not None:
            state_tensors['body_link_pose_w'] = self.rover.data.body_link_pose_w.torch
        if getattr(self, 'task_backend', 'reference') == 'torch_batched':
            # Reuse the simulator-owned device tensors for the batched task
            # step; the packed host sample below remains the VQF/diagnostic bridge.
            self._sim_tensor_cache = state_tensors
        feedback_tensors = None
        if include_feedback:
            imu = self.scene['imu'].data
            feedback_tensors = {
                'acceleration': imu.lin_acc_b.torch,
                'angular_velocity': imu.ang_vel_b.torch,
                'gimbal_qd': self.rover.data.joint_vel.torch[:,self._gimbal_ids],
                'wheel_qd': self.rover.data.joint_vel.torch[:,self._wheel_ids],
                'wheel_torque': self.rover.actuators.applied_effort.torch[:,self._wheel_ids],
            }
        tensors = [*state_tensors.values()]
        if feedback_tensors is not None:
            tensors.extend(feedback_tensors.values())
        flat_tensors = [value.reshape(value.shape[0], -1) for value in tensors]
        packed = torch.cat(flat_tensors, dim=1).detach().cpu().numpy()
        cursor = 0

        def unpack(values):
            nonlocal cursor
            result = {}
            for name, value in values.items():
                width = value.numel() // value.shape[0]
                shape = tuple(value.shape[1:])
                result[name] = packed[:, cursor:cursor + width].reshape(
                    (packed.shape[0], *shape)
                )
                cursor += width
            return result

        state = unpack(state_tensors)
        state['mount_rotation'] = state['mount_rotation'].reshape(-1, 3, 3)
        if live_state_exporter is not None:
            live_state_exporter.write(
                root_link_pose_w=state['root'],
                body_link_pose_w=state['body_link_pose_w'],
                env_origins_w=state['origin_offsets'],
                goals_local_xy=np.asarray([task.goal for task in self.tasks]),
                task_times_s=np.asarray([task.time for task in self.tasks]),
                sim_time_s=self._sim_step_counter * self.cfg.sim.dt,
            )
        feedback = None if feedback_tensors is None else unpack(feedback_tensors)
        if feedback is not None:
            feedback['gimbal_q'] = state['joints']
            feedback['mount_rotation'] = state['mount_rotation']
        return state, feedback

    def close(self):
        publisher = getattr(self, '_camera_rigid_pose_publisher', None)
        if publisher is not None:
            self._camera_rigid_pose_publisher = None
            if getattr(self.sim, '_agtc_camera_rigid_pose_publisher', None) is publisher:
                del self.sim._agtc_camera_rigid_pose_publisher
            publisher.close()
        super().close()

    def _refresh_state(self, *, include_feedback=False):
        publisher = getattr(self, '_camera_rigid_pose_publisher', None)
        if publisher is not None:
            publisher.publish()
        self._finished_observation_cache_mask = None
        if getattr(self, 'task_backend', 'reference') == 'torch_batched':
            self._tensor_finished_observation_cache = None
            self._tensor_observation_cache = None
        state, feedback = self._collect_sim_cpu_samples(include_feedback=include_feedback)
        self._feedback_host_cache = feedback
        for i,t in enumerate(self.tasks):
            t.robot[:]=[*list(state['root'][i,:2]-state['origin_offsets'][i,:2]),
                        wrap(float(state['angles'][i,2])),float(state['forward_speed'][i,0]),
                        float(state['angular'][i,0])]
            t.gimbal[:]=state['joints'][i]
            t.body_pose=(float(state['root'][i,2]-state['origin_offsets'][i,2]),
                         float(state['angles'][i,1]),float(state['angles'][i,0]))
            t.root_com_velocity_xy = state['root_com_velocity_xy'][i].copy()
            t.body_angular_velocity = state['body_angular_velocity'][i].copy()
            t.view_origin=(state['mount_pose'][i,:3]-state['origin_offsets'][i]).astype(np.float32)
            mount_rotation = state['mount_rotation'][i]
            t.view_basis=(mount_rotation[:,0].copy(),-mount_rotation[:,1].copy(),
                          mount_rotation[:,2].copy())
        self._state_observation_fresh = True

    def _refresh_feedback(self, *, after_step: bool) -> None:
        """Sample physical signals once; observation reads never advance the filter.

        Explicit PD drive effort is the post-limit torque actually written into
        PhysX. Its conversion to an equivalent motor Iq is an uncalibrated
        drivetrain model, not a measurement of hardware phase current.
        """
        self._finished_observation_cache_mask = None
        if getattr(self, 'task_backend', 'reference') == 'torch_batched':
            self._tensor_finished_observation_cache = None
            self._tensor_observation_cache = None
        if self._feedback_host_cache is None:
            _, self._feedback_host_cache = self._collect_sim_cpu_samples(include_feedback=True)
        sample = self._feedback_host_cache
        acceleration = sample['acceleration']
        angular_velocity = sample['angular_velocity']
        q = sample['gimbal_q']
        qd = sample['gimbal_qd']
        wheel_qd = sample['wheel_qd']
        wheel_torque = sample['wheel_torque']
        # The mount rotation is included in the packed host sample and reused
        # for the virtual magnetometer, alongside the immutable configured offset.
        mount_rotation = sample['mount_rotation']
        # Virtual magnetometer signal only. This world field is an explicit
        # assumed ENU reference; orientation is estimated from noisy sensors.
        offset_rotation = self._imu_offset_rotation
        sensor_rotation = mount_rotation @ offset_rotation
        mag = np.einsum('nji,j->ni', sensor_rotation, np.array([0., 3000., -6000.]))
        iq = np.stack([wheel_torque[:, mask].sum(axis=1) * self._torque_to_iq
                       for mask in self._wheel_side_masks], axis=1)
        rpm = np.stack([wheel_qd[:, mask].mean(axis=1) * self._omega_to_motor_rpm
                        for mask in self._wheel_side_masks], axis=1)
        for i, task in enumerate(self.tasks):
            if task.done and after_step and not self.cfg.auto_reset:
                continue
            now = task.time + (self.step_dt if after_step else 0.)
            fresh_mag = (not after_step) or int(round(now / self.step_dt)) % 2 == 0
            task.update_feedback(FeedbackSample(
                sample_time=now, accel_mps2=acceleration[i], gyro_rps=angular_velocity[i],
                # ATK manual ID4 (magnetometer + temperature) is fixed at 10 Hz.
                mag_counts=mag[i] if fresh_mag else None,
                temperature_c=25. if fresh_mag else None,
                gimbal_q_rad=q[i], gimbal_qd_rps=qd[i],
                motor_iq_a=iq[i] if after_step else np.zeros(2),
                motor_rpm=rpm[i] if after_step else np.zeros(2),
            ))
        self._feedback_host_cache = None

    def _capture_tensor_sensor_batch(self, kind):
        """Capture only due rows while keeping LiDAR/depth payloads on-device."""
        scope = self._tensor_capture_env_ids
        if scope is None:
            scope = tuple(range(self.num_envs))
            initial_ids = frozenset()
        else:
            initial_ids = self._tensor_capture_initial_env_ids
        due_ids = [int(i) for i in scope
                   if self.tasks[int(i)].sensor_due(kind, initial=int(i) in initial_ids)]
        if not due_ids:
            raise RuntimeError(f"tensor sensor callback for {kind!r} has no due environments")

        if kind == 'lidar':
            if kind not in self._sensor_cache:
                self._sensor_cache[kind] = ray_ranges(self.scene['lidar'].data)
            sensor = self.scene['lidar']
            sensor_pose_host = None
            if self._sensor_sample_exporter is not None:
                data = sensor.data
                sensor_pose_host = torch.cat(
                    (data.pos_w.torch.to(dtype=torch.float32), data.quat_w.torch.to(dtype=torch.float32)),
                    dim=1,
                ).detach().cpu().numpy().copy()
                if self._sensor_sample_lidar_directions_local is None:
                    self._sensor_sample_lidar_directions_local = (
                        sensor.ray_directions.torch[0].detach().cpu().numpy().copy()
                    )
            frames = capture_lidar_batch(
                self._sensor_cache[kind], due_ids,
                [task.np_random for task in self.tasks],
                min_range=self.field_cfg.lidar_min_range,
                max_range=self.field_cfg.lidar_range,
                relative_error=self.field_cfg.lidar_relative_error,
            )
            if sensor_pose_host is not None:
                for env_id, frame in frames.items():
                    frame.update(
                        sensor_origin_w=sensor_pose_host[env_id, :3].copy(),
                        sensor_quat_w_xyzw=sensor_pose_host[env_id, 3:7].copy(),
                        ray_directions_local=self._sensor_sample_lidar_directions_local,
                    )
            return frames
        if kind == 'depth':
            if kind not in self._sensor_cache:
                data = self.scene['depth'].data
                self._sensor_cache[kind] = data.output['distance_to_image_plane'].torch[..., 0]
            data = self.scene['depth'].data
            pose_intrinsics_host = None
            if self._sensor_sample_exporter is not None:
                pose_intrinsics_host = torch.cat(
                    (
                        data.pos_w.torch.to(dtype=torch.float32),
                        data.quat_w_ros.torch.to(dtype=torch.float32),
                        data.intrinsic_matrices.torch.reshape(self.num_envs, 9).to(dtype=torch.float32),
                    ),
                    dim=1,
                ).detach().cpu().numpy().copy()
            elif 'depth_intrinsics_host' not in self._sensor_cache:
                self._sensor_cache['depth_intrinsics_host'] = (
                    data.intrinsic_matrices.torch.detach().cpu().numpy().copy()
                )
            frames = capture_depth_batch(
                self._sensor_cache[kind], due_ids,
                [task.np_random for task in self.tasks],
                min_range=self.field_cfg.depth_min_range,
                max_range=self.field_cfg.depth_range,
                noise_std=self.field_cfg.range_noise,
            )
            for env_id, frame in frames.items():
                task = self.tasks[env_id]
                metadata = None if pose_intrinsics_host is None else pose_intrinsics_host[env_id]
                intrinsics = (
                    self._sensor_cache['depth_intrinsics_host'][env_id].copy()
                    if metadata is None else metadata[7:16].reshape(3, 3).copy()
                )
                frame.update(
                    intrinsics=intrinsics,
                    basis=np.stack(task.view_basis).astype(np.float32, copy=True),
                    calibration_id='isaac-centered-from-' + str(task.depth_intrinsics[0, 0]),
                    calibration_source=(
                        'Isaac Lab Camera.data.intrinsic_matrices; scaled native focal length, centered principal point'
                    ),
                )
                if metadata is not None:
                    frame.update(
                        sensor_origin_w=metadata[0:3].copy(),
                        sensor_quat_w_ros_xyzw=metadata[3:7].copy(),
                    )
            return frames
        if kind == 'camera':
            # Empty-field fastpath retains the task camera clock/latency but
            # needs no segmentation readback or stochastic Corn detection.
            return {env_id: {'detections': []} for env_id in due_ids}
        raise ValueError(f"unknown sensor kind: {kind}")

    def _capture_tensor_sensor(self, env_id, kind):
        cache = self._tensor_sensor_frame_cache
        if kind not in cache:
            previous_scope = self._tensor_capture_env_ids
            previous_initial = self._tensor_capture_initial_env_ids
            if previous_scope is None:
                self._tensor_capture_env_ids = (int(env_id),)
                self._tensor_capture_initial_env_ids = frozenset()
            try:
                cache[kind] = self._capture_tensor_sensor_batch(kind)
            finally:
                self._tensor_capture_env_ids = previous_scope
                self._tensor_capture_initial_env_ids = previous_initial
        try:
            return cache[kind][int(env_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"environment {env_id} requested {kind!r} outside the due capture batch"
            ) from exc

    def _capture(self,i,kind):
        if getattr(self, 'task_backend', 'reference') == 'torch_batched':
            return self._capture_tensor_sensor(i, kind)
        t,c=self.tasks[i],self.field_cfg
        if kind not in self._sensor_cache:
            if kind=='lidar':
                sensor=self.scene['lidar'].data
                self._sensor_cache[kind]=ray_ranges(sensor).detach().cpu().numpy().copy()
            elif kind=='depth':
                data=self.scene['depth'].data
                self._sensor_cache[kind]=(data.output['distance_to_image_plane'].torch.detach().cpu().numpy()[...,0].copy(),
                                         data.intrinsic_matrices.torch.detach().cpu().numpy().copy())
            else:
                sensor=self.scene['detector'].data
                self._sensor_cache[kind]=(
                    sensor.output['instance_segmentation'].torch.detach().cpu().numpy()[...,0].copy(),
                    sensor.info['instance_segmentation'],
                    ray_ranges(self.scene['rangefinder'].data).detach().cpu().numpy().copy())
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
        labels=(metadata or {}).get('idToLabels',{})
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

    def _tensor_body_stability(self, angles, body_angular_velocity):
        """Vectorized equivalent of body_stability_sample for the active batch."""
        roll = angles[:, 0].to(dtype=torch.float64)
        pitch = angles[:, 1].to(dtype=torch.float64)
        angular = body_angular_velocity.to(dtype=torch.float64)
        tilt = torch.acos(torch.clamp(torch.cos(roll) * torch.cos(pitch), min=-1.0, max=1.0))
        config = self.body_stability_config or BodyStabilityConfig()
        excess = torch.clamp(tilt - float(config.excess_tilt_threshold_rad), min=0.0)
        rate_sq = angular[:, 0].square() + angular[:, 1].square()
        if self.body_stability_config is None:
            penalty = torch.zeros_like(tilt)
        else:
            penalty = self.step_dt * (
                float(config.tilt_weight) * excess.square()
                + float(config.angular_rate_weight) * rate_sq
            )
        return {
            'roll': roll, 'pitch': pitch, 'tilt': tilt, 'excess': excess,
            'angular': angular, 'rate_sq': rate_sq, 'penalty': penalty,
        }

    def _build_tensor_observations(self):
        """Build the policy batch from existing task clocks and device frames."""
        tasks = self.tasks
        device = torch.device(self.device)
        lidar_frames = [task.frames['lidar'] for task in tasks]
        depth_frames = [task.frames['depth'] for task in tasks]
        if any(frame.get('_tensor_sensor_frame') is not True
               for frame in (*lidar_frames, *depth_frames)):
            raise RuntimeError('torch_batched observations require device-resident LiDAR and depth frames')
        lidar_frame_changed = False
        depth_frame_changed = False
        for env_id, (lidar_frame, depth_frame) in enumerate(zip(lidar_frames, depth_frames, strict=True)):
            if lidar_frame['ranges'].device != device or depth_frame['values'].device != device:
                raise RuntimeError('torch_batched sensor frames must remain on the environment device')
            if self._tensor_lidar_frame_seq[env_id] != lidar_frame['seq']:
                self._tensor_lidar_ranges[env_id].copy_(lidar_frame['ranges'])
                self._tensor_lidar_valid[env_id].copy_(lidar_frame['valid'])
                self._tensor_lidar_frame_seq[env_id] = lidar_frame['seq']
                lidar_frame_changed = True
            if self._tensor_depth_frame_seq[env_id] != depth_frame['seq']:
                self._tensor_depth_values[env_id].copy_(depth_frame['values'])
                self._tensor_depth_valid[env_id].copy_(depth_frame['valid'])
                self._tensor_depth_frame_seq[env_id] = depth_frame['seq']
                depth_frame_changed = True
        if lidar_frame_changed or self._tensor_lidar_policy_cache is None:
            self._tensor_lidar_policy_cache = torch.where(
                self._tensor_lidar_valid,
                self._tensor_lidar_ranges / self.field_cfg.lidar_range,
                torch.zeros_like(self._tensor_lidar_ranges),
            ).unsqueeze(1)
        if depth_frame_changed or self._tensor_depth_policy_cache is None:
            self._tensor_depth_policy_cache = torch.where(
                self._tensor_depth_valid,
                self._tensor_depth_values / self.field_cfg.depth_range,
                torch.zeros_like(self._tensor_depth_values),
            ).unsqueeze(1)

        # VQF/UWB estimates and frame metadata are CPU-owned task state. Pack
        # them into one small transfer; the large sensor arrays stay in their
        # persistent device buffers and are only copied when their seq changes.
        metadata_host = np.empty((self.num_envs, 63), dtype=np.float64)
        for i, task in enumerate(tasks):
            metadata_host[i, 0:3] = [task.frames[k]['capture'] for k in ('lidar', 'depth', 'camera')]
            metadata_host[i, 3:18] = np.asarray(
                [task.frames[k]['estimated_pose'] for k in ('lidar', 'depth', 'camera')],
                dtype=np.float64,
            ).reshape(15)
            metadata_host[i, 18:21] = [task._new[k] for k in ('lidar', 'depth', 'camera')]
            metadata_host[i, 21:26] = task._estimated_pose()
            metadata_host[i, 26:63] = (
                np.zeros(FEEDBACK_DIM, dtype=np.float64)
                if task._feedback_snapshot is None
                else np.asarray(task._feedback_snapshot.feedback, dtype=np.float64)
            )
        metadata = torch.as_tensor(metadata_host, dtype=torch.float64, device=device)
        frame_capture_times = metadata[:, 0:3]
        frame_estimated_poses = metadata[:, 3:18].reshape(self.num_envs, 3, 5)
        frame_new = metadata[:, 18:21].to(dtype=torch.bool)
        estimated_pose = metadata[:, 21:26]
        feedback = metadata[:, 26:63]
        observations = pack_emptyfield_goto_observation(
            lidar_ranges=self._tensor_lidar_ranges,
            lidar_valid=self._tensor_lidar_valid,
            depth_values=self._tensor_depth_values,
            depth_valid=self._tensor_depth_valid,
            normalized_lidar=self._tensor_lidar_policy_cache,
            normalized_depth=self._tensor_depth_policy_cache,
            empty_detections=torch.empty((self.num_envs, 0, 8), dtype=torch.float32, device=device),
            frame_capture_times=frame_capture_times,
            frame_estimated_poses=frame_estimated_poses,
            frame_new=frame_new,
            estimated_pose=estimated_pose,
            goal_xy=self._tensor_goal_xy,
            last_actions=self._tensor_last_actions,
            episode_time=self._tensor_episode_time,
            feedback=feedback,
            size=self.field_cfg.size,
            lidar_range=self.field_cfg.lidar_range,
            depth_range=self.field_cfg.depth_range,
            max_time=self.field_cfg.max_time,
            max_detections=self.field_cfg.max_detections,
        )
        sensor_exporter = getattr(self, '_sensor_sample_exporter', None)
        if sensor_exporter is not None:
            sensor_exporter.publish_if_changed(
                lidar_frames=lidar_frames,
                depth_frames=depth_frames,
                lidar_ranges=self._tensor_lidar_ranges,
                lidar_valid=self._tensor_lidar_valid,
                depth_values=self._tensor_depth_values,
                depth_valid=self._tensor_depth_valid,
            )
        return observations

    def _tensor_observation_row(self, env_id):
        if self._tensor_observation_cache is None:
            self._tensor_observation_cache = self._build_tensor_observations()
        return {key: value[int(env_id)] for key, value in self._tensor_observation_cache.items()}

    def _get_tensor_batched_dones(self):
        self._finished_observation_cache_mask = None
        self._tensor_finished_observation_cache = None
        self._tensor_sensor_frame_cache.clear()
        self._sensor_cache.clear()
        self._refresh_state(include_feedback=True)
        self._refresh_feedback(after_step=True)
        state = self._sim_tensor_cache
        if not isinstance(state, dict):
            raise RuntimeError('torch_batched state tensors were not retained for this PhysX step')

        root = state['root']
        origins = state['origin_offsets']
        position_xy = root[:, :2] - origins[:, :2]
        angles = state['angles']
        body_angular_velocity = state['body_angular_velocity']
        boundary = torch.any(
            (position_xy < 0.0) | (position_xy > self.field_cfg.size), dim=1,
        )
        pitch_failure = torch.abs(angles[:, 1]) > 1.2
        roll_failure = torch.abs(angles[:, 0]) > 1.2
        active_host = [not (task.done and not self.cfg.auto_reset) for task in self.tasks]
        active = torch.as_tensor(active_host, dtype=torch.bool, device=self.device)
        zeros = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        transition = fast_stop_v2_step(
            position_xy=position_xy,
            old_distance=self._tensor_old_distance,
            goal_xy=self._tensor_goal_xy,
            root_com_velocity_xy=state['root_com_velocity_xy'],
            body_angular_velocity=body_angular_velocity,
            body_tilt_pitch_roll=torch.stack((angles[:, 1], angles[:, 0]), dim=1),
            boundary=boundary,
            physics_failure_event=zeros,
            previous_physics_failure=self._tensor_previous_physics_failure,
            episode_time=self._tensor_episode_time,
            hold_elapsed=self._tensor_hold,
            initial_goal_distance=self._tensor_initial_goal_distance,
            active_mask=active,
            dt=self.step_dt,
            max_time=self.field_cfg.max_time,
            goal_tolerance=self.field_cfg.goal_tolerance,
            goal_hold=self.field_cfg.goal_hold,
            goal_stop_speed=self.field_cfg.goal_stop_speed,
            goal_stop_angular_speed=self.field_cfg.goal_stop_angular_speed,
            time_penalty=self.field_cfg.time_penalty,
            goal_progress_budget=self.field_cfg.goal_progress_budget,
        )
        stability = self._tensor_body_stability(angles, body_angular_velocity)
        penalty = torch.where(active, stability['penalty'], torch.zeros_like(stability['penalty']))
        reward = transition['rewards'].to(dtype=torch.float64) - penalty
        reward = reward.to(dtype=torch.float32)
        self._reward = reward
        self._tensor_episode_time.copy_(transition['next_time'])
        self._tensor_hold.copy_(transition['next_hold'])
        self._tensor_previous_physics_failure.copy_(transition['next_physics_failure'])
        self._tensor_last_actions.copy_(torch.where(active[:, None], self.actions, self._tensor_last_actions))

        component_names = tuple(transition['components'])
        component_batch = torch.stack(
            [transition['components'][name].to(dtype=torch.float64) for name in component_names], dim=1,
        )
        stability_batch = torch.stack((
            stability['roll'], stability['pitch'], stability['tilt'], stability['excess'],
            stability['angular'][:, 0], stability['angular'][:, 1], stability['angular'][:, 2],
            stability['rate_sq'], stability['penalty'],
        ), dim=1)
        host_columns = torch.cat((
            component_batch,
            reward.to(dtype=torch.float64).unsqueeze(1),
            transition['done'].to(dtype=torch.float64).unsqueeze(1),
            transition['truncated'].to(dtype=torch.float64).unsqueeze(1),
            transition['reason_codes'].to(dtype=torch.float64).unsqueeze(1),
            transition['next_time'].to(dtype=torch.float64).unsqueeze(1),
            transition['next_hold'].to(dtype=torch.float64).unsqueeze(1),
            transition['next_physics_failure'].to(dtype=torch.float64).unsqueeze(1),
            boundary.to(dtype=torch.float64).unsqueeze(1),
            pitch_failure.to(dtype=torch.float64).unsqueeze(1),
            roll_failure.to(dtype=torch.float64).unsqueeze(1),
            stability_batch,
            self.actions.to(dtype=torch.float64),
        ), dim=1).detach().cpu().numpy()

        reward_column = len(component_names)
        done_column = reward_column + 1
        truncated_column = done_column + 1
        reason_column = truncated_column + 1
        time_column = reason_column + 1
        hold_column = time_column + 1
        physics_column = hold_column + 1
        boundary_column = physics_column + 1
        pitch_column = boundary_column + 1
        roll_column = pitch_column + 1
        stability_start = roll_column + 1
        action_start = stability_start + 9
        diagnostic_infos = []
        finished_observation_cache_mask = [False] * self.num_envs
        action_host = host_columns[:, action_start:action_start + 4]
        for i, task in enumerate(self.tasks):
            if not active_host[i]:
                diagnostic_infos.append({
                    'reward_components': {},
                    'inactive': True,
                    'body_stability': copy.deepcopy(task.body_stability_sample),
                    'body_stability_summary': task.body_stability_stats.summary(),
                    'termination_detail': {
                        'boundary': False,
                        'overtilt_roll': False,
                        'overtilt_pitch': False,
                    },
                })
                continue

            row = host_columns[i]
            reason_code = int(row[reason_column])
            task.time = float(row[time_column])
            task._hold = float(row[hold_column])
            task.physics_failure = bool(row[physics_column])
            task.done = bool(row[done_column])
            task.reason = {
                0: '',
                REASON_GOAL_REACHED: 'goal_reached',
                REASON_OUTSIDE_OR_OVERTURNED: 'outside_or_overturned',
                REASON_TIME_LIMIT: 'time_limit',
            }[reason_code]
            task.last_action = np.asarray(action_host[i], dtype=np.float32).copy()
            task.reward = float(row[reward_column])
            task.total_reward += task.reward
            task.reward_components = {
                name: float(row[index]) for index, name in enumerate(component_names)
            }
            if reason_code not in (REASON_OUTSIDE_OR_OVERTURNED, REASON_TIME_LIMIT):
                task.reward_components.pop('terminal_failure', None)
                task.reward_components.pop('failure_remaining_time_cost', None)
            task.trail.append(task.robot[:2].tolist())

            sample = {
                'enabled': self.body_stability_config is not None,
                'roll_rad': float(row[stability_start]),
                'pitch_rad': float(row[stability_start + 1]),
                'tilt_rad': float(row[stability_start + 2]),
                'excess_tilt_rad': float(row[stability_start + 3]),
                'body_angular_velocity_rad_s': [
                    float(row[stability_start + index]) for index in (4, 5, 6)
                ],
                'roll_pitch_rate_sq': float(row[stability_start + 7]),
                'penalty': float(row[stability_start + 8]),
            }
            task.body_stability_sample = sample
            task.body_stability_stats.update(sample, self.step_dt)
            task.reward_components['body_stability'] = -sample['penalty']
            info = task._info()
            info['body_stability'] = copy.deepcopy(sample)
            info['body_stability_summary'] = task.body_stability_stats.summary()
            boundary_value = bool(row[boundary_column])
            pitch_value = bool(row[pitch_column])
            roll_value = bool(row[roll_column])
            info['termination_detail'] = {
                'boundary': boundary_value,
                'overtilt_roll': roll_value,
                'overtilt_pitch': pitch_value,
            }
            diagnostic_infos.append(copy.deepcopy(info))
            finished_observation_cache_mask[i] = True
            if task.done:
                self.episode_results.append({'mode': task.mode, 'return': task.total_reward, **copy.deepcopy(info)})
                self.episode_results = self.episode_results[-2000:]

        sense_ids = tuple(i for i, is_active in enumerate(active_host) if is_active)
        self._tensor_capture_env_ids = sense_ids
        self._tensor_capture_initial_env_ids = frozenset()
        self._tensor_sensor_frame_cache.clear()
        for i in sense_ids:
            self.tasks[i].update_uwb_pose_noise()
        try:
            for i in sense_ids:
                self.tasks[i]._sense()
        finally:
            self._tensor_capture_env_ids = None
            self._tensor_capture_initial_env_ids = frozenset()

        observations = self._build_tensor_observations()
        self._tensor_observation_cache = observations
        self._tensor_finished_observation_cache = observations
        for i, task in enumerate(self.tasks):
            if not active_host[i]:
                continue
            obs_row = {key: value[i] for key, value in observations.items()}
            info = diagnostic_infos[i]
            self._last_results[i] = (
                obs_row, task.reward, bool(row[done_column]), bool(row[truncated_column]), copy.deepcopy(info),
            )
        self.diagnostic_step_infos = tuple(diagnostic_infos)
        self._finished_observation_cache_mask = tuple(finished_observation_cache_mask)
        terminated = transition['terminated'] & bool(self.cfg.auto_reset)
        truncated = transition['truncated'] & bool(self.cfg.auto_reset)
        return terminated, truncated

    def _get_tensor_batched_observations(self):
        initialize_sensors = any(not task.frames for task in self.tasks)
        pending_cache_mask = getattr(self, '_finished_observation_cache_mask', None)
        self._finished_observation_cache_mask = None
        reuse_finished_observations = (
            pending_cache_mask is not None
            and len(pending_cache_mask) == len(self.tasks)
            and not initialize_sensors
            and self._state_observation_fresh
            and self._tensor_finished_observation_cache is not None
        )
        if initialize_sensors:
            self._invalidate_cached_samples()
            self._sensor_cache.clear()
            self._tensor_sensor_frame_cache.clear()
            self.scene.write_data_to_sim()
            _forward_publish_scene_and_render(self.sim)
            for name in ('lidar', 'rangefinder', 'depth', 'detector'):
                self.scene[name].update(0., force_recompute=True)
        if not self._state_observation_fresh:
            self._refresh_state(include_feedback=initialize_sensors)
        if initialize_sensors:
            self._refresh_feedback(after_step=False)
            initial_ids = tuple(i for i, task in enumerate(self.tasks) if not task.frames)
            self._tensor_capture_env_ids = initial_ids
            self._tensor_capture_initial_env_ids = frozenset(initial_ids)
            for i in initial_ids:
                self.tasks[i].update_uwb_pose_noise()
            try:
                for i in initial_ids:
                    self.tasks[i].initialize_sensors(build_observation=False)
            finally:
                self._tensor_capture_env_ids = None
                self._tensor_capture_initial_env_ids = frozenset()

        reuse_rows = (
            [bool(value and self._last_results[i] is not None)
             for i, value in enumerate(pending_cache_mask)]
            if reuse_finished_observations else []
        )
        if reuse_finished_observations and all(reuse_rows):
            observations = self._tensor_finished_observation_cache
        elif reuse_finished_observations:
            fresh_observations = self._build_tensor_observations()
            reuse_mask = torch.as_tensor(
                reuse_rows,
                dtype=torch.bool, device=self.device,
            )
            observations = {
                key: torch.where(
                    reuse_mask.reshape((-1,) + (1,) * (fresh.ndim - 1)),
                    self._tensor_finished_observation_cache[key], fresh,
                )
                for key, fresh in fresh_observations.items()
            }
        else:
            observations = self._build_tensor_observations()
        self._tensor_observation_cache = observations
        self._state_observation_fresh = False
        return {'policy': observations}

    def _get_dones(self):
        if getattr(self, 'task_backend', 'reference') == 'torch_batched':
            return self._get_tensor_batched_dones()
        self._finished_observation_cache_mask = None
        self._refresh_state(include_feedback=True)
        self._refresh_feedback(after_step=True)
        actions=self.actions.detach().cpu().numpy()
        rewards=[]; terminated=[]; truncated=[]; diagnostic_infos=[]
        finished_observation_cache_mask = [False] * len(self.tasks)
        for i,t in enumerate(self.tasks):
            if t.done and not self.cfg.auto_reset:
                rewards.append(0.);terminated.append(False);truncated.append(False)
                diagnostic_infos.append({
                    "reward_components": {},
                    "inactive": True,
                    "body_stability": copy.deepcopy(t.body_stability_sample),
                    "body_stability_summary": t.body_stability_stats.summary(),
                    "termination_detail": {
                        "boundary": False,
                        "overtilt_roll": False,
                        "overtilt_pitch": False,
                    },
                })
                continue
            boundary=bool(np.any(t.robot[:2]<0.) or np.any(t.robot[:2]>self.field_cfg.size))
            overtilt_roll = abs(t.body_pose[2]) > 1.2
            overtilt_pitch = abs(t.body_pose[1]) > 1.2
            fast_stop = (
                self.field_cfg.goto_reward_profile in FAST_STOP_REWARD_PROFILES
                and t.mode == 'goto'
            )
            rock=False
            terrain=t.scene.terrain
            if terrain is not None and np.any(terrain.rock_blocking):
                delta=terrain.rock_xy[terrain.rock_blocking]-t.robot[:2]
                co,si=math.cos(t.robot[2]),math.sin(t.robot[2])
                x=delta[:,0]*co+delta[:,1]*si
                y=-delta[:,0]*si+delta[:,1]*co
                rock=bool(np.any(np.maximum(abs(x)-self.field_cfg.robot_length/2,0)**2+
                     np.maximum(abs(y)-self.field_cfg.robot_width/2,0)**2<=terrain.rock_radius[terrain.rock_blocking]**2))
            if fast_stop:
                result=t.finish(
                    actions[i], self._before[i], self._old_distance[i], rock, boundary,
                    physics_failure=boundary or overtilt_roll or overtilt_pitch,
                )
            else:
                # Keep the legacy call shape and scoring order untouched.
                result=t.finish(actions[i],self._before[i],self._old_distance[i],rock,boundary)
            obs,reward,term,trunc,info=result
            finished_observation_cache_mask[i] = True
            if t.mode=='goto' and not fast_stop:
                shaping=self.cfg.distance_cost*info['distance']*self.step_dt
                reward-=shaping
                t.reward-=shaping;t.total_reward-=shaping
                t.reward_components["distance_cost_shaping"] = float(-shaping)
            stability_sample = body_stability_sample(
                roll=t.body_pose[2],
                pitch=t.body_pose[1],
                body_angular_velocity=t.body_angular_velocity,
                dt=self.step_dt,
                config=self.body_stability_config,
            )
            t.body_stability_sample = stability_sample
            t.body_stability_stats.update(stability_sample, self.step_dt)
            stability_penalty = float(stability_sample["penalty"])
            t.reward_components["body_stability"] = -stability_penalty
            if self.body_stability_config is not None and stability_penalty:
                reward -= stability_penalty
                t.reward -= stability_penalty
                t.total_reward -= stability_penalty
            if boundary or overtilt_roll or overtilt_pitch:
                t.done=True;t.reason='outside_or_overturned';term=True;trunc=False
                if not fast_stop:
                    reward-=5.
                    t.reward=reward;t.total_reward-=5.
                    t.reward_components["outside_or_overturned"] = -5.0
            info=t._info()
            info["body_stability"] = copy.deepcopy(stability_sample)
            info["body_stability_summary"] = t.body_stability_stats.summary()
            info["termination_detail"] = {
                "boundary": bool(boundary),
                "overtilt_roll": bool(overtilt_roll),
                "overtilt_pitch": bool(overtilt_pitch),
            }
            diagnostic_infos.append(copy.deepcopy(info))
            self._last_results[i]=(obs,reward,term,trunc,copy.deepcopy(info))
            rewards.append(reward);terminated.append(term and self.cfg.auto_reset);truncated.append(trunc and self.cfg.auto_reset)
            if term or trunc:
                self.episode_results.append({'mode':t.mode,'return':t.total_reward,**copy.deepcopy(info)})
                self.episode_results=self.episode_results[-2000:]
        self._reward=torch.tensor(rewards,device=self.device)
        # The official fast SB3 wrapper intentionally drops custom infos. Keep
        # immutable transition snapshots here for an opt-in diagnostics callback,
        # including terminal steps before DirectRLEnv performs same-step reset.
        self.diagnostic_step_infos=tuple(diagnostic_infos)
        self._finished_observation_cache_mask = tuple(finished_observation_cache_mask)
        return torch.tensor(terminated,device=self.device),torch.tensor(truncated,device=self.device)

    def _get_rewards(self):
        return self._reward

    def _get_observations(self):
        if getattr(self, 'task_backend', 'reference') == 'torch_batched':
            return self._get_tensor_batched_observations()
        initialize_sensors = any(not t.frames for t in self.tasks)
        pending_cache_mask = getattr(self, '_finished_observation_cache_mask', None)
        self._finished_observation_cache_mask = None
        reuse_finished_observations = (
            pending_cache_mask is not None
            and len(pending_cache_mask) == len(self.tasks)
            and not initialize_sensors
            and self._state_observation_fresh
        )
        if initialize_sensors:
            # Reset poses are written before the explicit render/forward. Drop
            # any pre-reset sample so the state below matches those sensors.
            self._invalidate_cached_samples()
            # Both explicit and same-step autoresets must render the new poses.
            self.scene.write_data_to_sim()
            _forward_publish_scene_and_render(self.sim)
            for name in ('lidar','rangefinder','depth','detector'):
                self.scene[name].update(0.,force_recompute=True)
        if not self._state_observation_fresh:
            self._refresh_state(include_feedback=initialize_sensors)
        if initialize_sensors:
            self._refresh_feedback(after_step=False)
        observations=[]
        for i,t in enumerate(self.tasks):
            if not t.frames:
                obs,_=t.initialize_sensors()
            elif (reuse_finished_observations and pending_cache_mask[i]
                  and self._last_results[i] is not None):
                obs=self._last_results[i][0]
            else:
                obs=t._observation()
            observations.append(obs)
        self._state_observation_fresh = False
        return {'policy':{key:torch.as_tensor(np.stack([o[key] for o in observations]),device=self.device)
                          for key in observations[0]}}

    def plant_positions(self):
        """Actual PhysX plant poses, in stable plant-index order (debug/tests)."""
        if self.plants is None:
            return torch.empty((self.num_envs, 0, 3), device=self.device)
        return self.plants.data.body_link_pose_w.torch[..., :3]

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
        drive_metadata = dict(
            interface='independent_left_right_track_speed_coupled_pi_v7',
            actuator_model='explicit_IdealPD_velocity_servo',
            controller='native_per_wheel_P_plus_two_state_side_integral_feedforward',
            motor_torque_constant_nm_per_a=self.cfg.motor_torque_constant_nm_per_a,
            current_feedback_source='equivalent_from_physx_post_clamp_applied_effort; model_not_measured',
            max_track_speed_mps=max_track_speed_mps,
            track_acceleration_mps2=self.field_cfg.acceleration,
            velocity_gain=self.cfg.drive_velocity_gain,
            integral_gain_nm_per_proxy_wheel_per_rad_s_s=(
                self.cfg.drive_integral_gain_nm_per_proxy_wheel_per_rad_s_s
            ),
            integral_update_period_s=float(self.step_dt),
            integral_state_count_per_env=2,
            integral_state_unit='Nm_per_proxy_wheel',
            integral_error_source='per_side_mean_target_minus_actual_wheel_qd_rad_s',
            integral_broadcast='same_side_integral_ff_to_each_of_7_proxy_wheels',
            integral_limit_nm_per_proxy_wheel=per_wheel_effort_nm,
            integral_antiwindup=(
                'conditional_on_side_mean_P_estimate; final_safety_is_explicit_actuator_per_wheel_clamp'
            ),
            integral_gain_status='uncalibrated simulation candidate; not real motor/controller calibration',
            wheel_effort_limit_nm=per_wheel_effort_nm,
            actuator_effort_limit_nm=per_wheel_effort_nm,
            joint_effort_limit_nm=per_wheel_effort_nm,
            effort_clamp_source='explicit_actuator_adapter_after_combined_PD_and_feedforward',
            wheel_effort_limit_basis='actual_PhysX_adapter_actuator_output_clamp_per_proxy_wheel_joint; not measured motor torque',
            gimbal_solver_joint_effort_limit_nm=1.0,
            gimbal_effort_limit_basis='implicit_PhysX_joint_effort_limit',
            drive_sprocket_radius_m=spec.drive_sprocket_radius_m,
            drive_sprocket_diameter_m=spec.drive_sprocket_diameter_m,
            drive_reduction_ratio=spec.drive_reduction_ratio,
            proxy_wheel_radius_m=spec.wheel_radius_m,
            proxy_wheel_count_per_side=spec.wheel_count_per_side,
            track_ground_contact_length_m=spec.track_ground_contact_length_m,
            robot_asset_spec_sha256=robot_asset_spec_sha256(
                spec, front_guide_spec=self.front_guide_spec,
            ),
            drive_efficiency_assumption=self.cfg.drive_efficiency_assumption,
            equivalent_motor_torque_per_side_nm=equivalent_motor_torque_nm,
            max_motor_rpm_at_max_track_speed=max_motor_rpm,
            drivetrain_note=(
                'Torque and RPM are idealized proxy-equivalent values: one motor per side, '
                'unit efficiency, 60 mm active sprocket, and 45 mm proxy support rollers. '
                f'The {per_wheel_effort_nm:g} Nm per-wheel effort cap is not a measured motor rating.'
            ),
            same_side_wheel_coupling=copy.deepcopy(self._same_side_wheel_coupling_metadata),
            physics_contact=copy.deepcopy(self._physics_contact_metadata),
        )
        if self.front_guide_spec is not None:
            guide_wheels = self.front_guide_spec.wheel_specs(spec)
            guide_coupled = self.front_guide_spec.rotation_coupling == 'same_side_track_speed'
            guide_ratio = spec.wheel_radius_m / self.front_guide_spec.radius_m
            drive_metadata["front_guide"] = {
                "enabled": True,
                "model": (
                    "same_side_track_speed_coupled_front_guides_separate_proxy_not_continuous_track"
                    if guide_coupled else
                    "passive_free_rotating_guide_wheels_separate_proxy_not_continuous_track"
                ),
                "wheels": [asdict(wheel) for wheel in guide_wheels],
                "drive_motor_count": 0,
                "added_drive_torque_nm": 0.0,
                "rotation_coupling": self.front_guide_spec.rotation_coupling,
                "coupling": {
                    "enabled": guide_coupled,
                    "relation": (
                        "q_guide = (R_main/R_guide)*q_same_side_wheel0"
                        if guide_coupled else None
                    ),
                    "relations_per_world": 2 if guide_coupled else 0,
                    "leader_joint_by_side": (
                        {
                            "left": "left_wheel_0_wheel_joint",
                            "right": "right_wheel_0_wheel_joint",
                        } if guide_coupled else None
                    ),
                    "ratio_by_side": (
                        {"left": guide_ratio, "right": guide_ratio}
                        if guide_coupled else None
                    ),
                    "guide_actuator_count": 0,
                    "added_pid_count": 0,
                },
            }
        if self.body_stability_config is not None:
            drive_metadata["body_stability"] = {
                "enabled": True,
                "config": self.body_stability_config.as_dict(),
                "rate_source": "root_link_com_angular_velocity_world_rotated_to_body",
                "policy_observation_included": False,
                "hard_tilt_termination_rad": 1.2,
            }
        return drive_metadata

    @property
    def observation_config(self):
        c = self.field_cfg
        schema = c.input_schema
        contract = dict(
            schema=schema,
            state_dim=STATE_DIM,
            state_fields=list(state_fields_for_schema(schema)),
            observation_shapes={
                'lidar': [1, c.lidar_rays],
                'depth': [1, c.depth_height, c.depth_width],
                'detections': [c.max_detections, 8],
                'state': [STATE_DIM],
                'feedback': [FEEDBACK_DIM],
            },
            lidar_fields=['normalized_range'],
            lidar_invalid_value=0.0,
            detection_fields=['cx', 'cy', 'width', 'height', 'class_id', 'confidence', 'range', 'range_valid'],
            detection_presence='positive width and height',
            feedback=feedback_config(input_schema=schema),
            feedback_dim=FEEDBACK_DIM,
            estimated_pose=self.tasks[0].estimated_pose_config,
            imu_mount_body='gimbal_pitch_link',
            imu_offset_pos_m=list(self.cfg.imu_offset_pos),
            imu_offset_rot_xyzw=list(self.cfg.imu_offset_rot_xyzw),
            depth_channels=1,
            depth_invalid_value=0.0,
            depth_value_range=[0.0, 1.0],
            depth_space_bounds=[-1.0, 1.0],
            measured_velocity_indices=[],
            heading_source='sensor_fusion_head_attitude_and_measured_gimbal_inverse',
            previous_command_indices=[7, 8, 9, 10],
            previous_command_semantics='normalized_direct_track_speed_policy_command',
            action_components=list(ACTION_COMPONENTS),
        )
        if schema == ABSOLUTE_GOAL_INPUT_SCHEMA:
            contract["absolute_goal_coordinates"] = {
                "source": "task.goal",
                "coordinate_frame": "field_world_xy",
                "origin_xy_m": [0.0, 0.0],
                "unit": "m",
                "field_size_m": float(c.size),
                "normalization": "world_xy / field_size_m * 2 - 1",
                "state_clip": [-1.0, 1.0],
            }
        return contract

    @property
    def backend_identity(self):
        return simulator_identity()

    @property
    def action_config(self):
        return policy_action_config()

    @property
    def damage_config(self):
        """JSON-safe task damage metric and geometry contract."""
        return dict(self.tasks[0].damage_config) if self.tasks else {}


def make_env(cfg=None, mode='mixed', num_envs=1, seed=7, auto_reset=True, *,
             goal_distance=None, goal_angle=math.pi, distance_cost=0., viewer_port=None,
             couple_same_side_wheels=True, front_guide_spec=None,
             body_stability_config=None, task_backend='reference', rsl=False):
    if front_guide_spec is not None:
        if not isinstance(front_guide_spec, FrontGuideSpec):
            raise TypeError("front_guide_spec must be FrontGuideSpec or None")
        front_guide_spec = asdict(front_guide_spec)
    if body_stability_config is not None:
        if not isinstance(body_stability_config, BodyStabilityConfig):
            raise TypeError("body_stability_config must be BodyStabilityConfig or None")
        body_stability_config = body_stability_config.as_dict()
    env_cfg=FieldEnvCfg(task=cfg or Config(),mode=mode,auto_reset=auto_reset,seed=seed,
                        task_backend=task_backend,
                        goal_distance=goal_distance,goal_angle=goal_angle,distance_cost=distance_cost,
                        couple_same_side_wheels=couple_same_side_wheels,
                        front_guide_spec=front_guide_spec,
                        body_stability_config=body_stability_config)
    env_cfg.scene.num_envs=num_envs
    if viewer_port is not None:
        from isaaclab_visualizers.viser import ViserVisualizerCfg
        env_cfg.sim.visualizer_cfgs=[ViserVisualizerCfg(port=viewer_port,bind_address='127.0.0.1',
            display_address='localhost',eye=(4.,-4.,3.),lookat=(2.,2.,0.),open_browser=False)]
    if rsl:
        from .rsl_rl_env import IsaacFieldRslEnv
        return IsaacFieldRslEnv(env_cfg)
    return IsaacFieldEnv(env_cfg)


class SingleFieldAdapter(gym.Env):
    """Single-row interface for policy evaluation and official Viser controls."""
    metadata={'render_modes':[]}
    def __init__(self, env:IsaacFieldEnv):
        if env.num_envs!=1:
            raise ValueError('SingleFieldAdapter requires num_envs=1')
        self.engine=env
        self.cfg=env.field_cfg
        space = env.single_observation_space
        self.observation_space = space['policy'] if 'policy' in space.spaces else space
        self.action_space=env.single_action_space

    @property
    def task(self):return self.engine.tasks[0]
    @property
    def observation_config(self):return self.engine.observation_config
    def __getattr__(self,name):return getattr(self.task,name)
    def current_observation(self):return self.task._observation()
    def reset(self,*,seed=None,options=None):
        self.engine.reset(seed=seed,options=options)
        info = self.task._info()
        info["terrain_seed"] = getattr(self.task.scene, "terrain_seed", None)
        info["mode"] = self.task.mode
        return self.current_observation(), info
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
        state.update(backend='Isaac Sim / Isaac Lab / PhysX',drive='14 driven rollers',
                     collision_metric='geometric footprint overlap with blocking rocks')
        return state
    def close(self):self.engine.close()
