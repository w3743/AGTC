"""Small USD assets used by the Isaac FieldSim adapter.

This module authors ordinary OpenUSD physics and geometry schemas.  It does not
start Isaac Sim, install packages, or import the legacy ``field_sim`` package.
The rover is a free-root articulation with seven rotating wheel proxies per
side; those rollers approximate track contact but do not model a continuous
belt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class WheelSpec:
    name: str
    side: str
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class FrontGuideWheelSpec:
    """Placement and contact geometry for one front guide wheel without a separate drive."""

    name: str
    side: str
    x_m: float
    y_m: float
    z_m: float
    radius_m: float
    width_m: float
    mass_kg: float

    @property
    def link_name(self) -> str:
        return f"{self.name}_link"

    @property
    def joint_name(self) -> str:
        # Deliberately excludes ``wheel_joint`` so the existing actuator regex
        # cannot match this joint. A Newton runtime hook may couple it to the
        # same-side track speed without authoring an independent DriveAPI.
        return f"{self.name}_joint"


@dataclass(frozen=True)
class RobotAssetSpec:
    """Explicit proxy dimensions and masses for the local rover asset."""

    body_length_m: float = 0.8
    body_width_m: float = 0.3
    # A simple box height is required for collision and inertia.  The source
    # request fixes length/width, so this is an asset-generation proxy.
    body_height_m: float = 0.15
    total_mass_kg: float = 30.0
    chassis_mass_kg: float = 26.4
    wheel_mass_kg: float = 0.25
    gimbal_link_mass_kg: float = 0.05
    wheel_radius_m: float = 0.045
    # User-confirmed track width; the wheel cylinders are proxies for this
    # contact band, while their 45 mm radius remains a separate proxy value.
    wheel_width_m: float = 0.06
    # Hardware-confirmed track geometry.  These describe the drivetrain and
    # contact span; they are not the geometry of the rotating roller proxies.
    track_ground_contact_length_m: float = 0.595
    drive_sprocket_radius_m: float = 0.03
    drive_reduction_ratio: float = 4.0
    # User-confirmed 470 mm overall width includes 60 mm tracks on both sides;
    # their center separation is therefore 410 mm (the body itself is 300 mm).
    track_separation_m: float = 0.41
    wheel_count_per_side: int = 7
    wheel_z_m: float = -0.105
    reference_height_m: float = 0.15
    gimbal_offset_z_m: float = 0.25
    pitch_limits_deg: tuple[float, float] = (-25.0, 60.0)

    @property
    def wheel_spacing_m(self) -> float:
        """Center spacing that spans the confirmed track-ground length."""
        if self.wheel_count_per_side < 2:
            return 0.0
        return self.track_ground_contact_length_m / (self.wheel_count_per_side - 1)

    @property
    def drive_sprocket_diameter_m(self) -> float:
        return 2.0 * self.drive_sprocket_radius_m

    @property
    def vehicle_width_m(self) -> float:
        """Overall lateral safety envelope of chassis and both track proxies."""
        return max(self.body_width_m, self.track_separation_m + self.wheel_width_m)

    @property
    def assembled_mass_kg(self) -> float:
        return (
            self.chassis_mass_kg
            + 2.0 * self.wheel_count_per_side * self.wheel_mass_kg
            + 2.0 * self.gimbal_link_mass_kg
        )

    @property
    def chassis_com_offset_z_m(self) -> float:
        """Small chassis-link offset that keeps the full assembly COM at the proxy height."""
        other_moment = 2.0 * self.wheel_count_per_side * self.wheel_mass_kg * self.wheel_z_m
        other_moment += 2.0 * self.gimbal_link_mass_kg * self.gimbal_offset_z_m
        return -other_moment / self.chassis_mass_kg

    @property
    def wheel_specs(self) -> tuple[WheelSpec, ...]:
        half_track = self.track_separation_m / 2.0
        x_positions = tuple(
            (index - (self.wheel_count_per_side - 1) / 2.0) * self.wheel_spacing_m
            for index in range(self.wheel_count_per_side)
        )
        return tuple(
            WheelSpec(f"{side}_wheel_{index}", side, x, y, self.wheel_z_m)
            for side, y in (("left", half_track), ("right", -half_track))
            for index, x in enumerate(x_positions)
        )


@dataclass(frozen=True)
class FrontGuideSpec:
    """Trial proxy for one front guide wheel on each track.

    These dimensions are explicit modeling assumptions, not measurements of
    the physical vehicle. ``z_offset_from_support_m`` raises each guide-wheel
    axis above the existing support-wheel axes. Guide rotation has no
    independent actuator; ``rotation_coupling`` declares whether it is free
    or follows the same-side track speed through the runtime physics hook.
    """

    x_m: float = 0.42
    z_offset_from_support_m: float = 0.05
    radius_m: float = 0.045
    width_m: float = 0.06
    mass_per_wheel_kg: float = 0.25
    rotation_coupling: str = "same_side_track_speed"

    def __post_init__(self) -> None:
        for name in ("x_m", "z_offset_from_support_m", "radius_m", "width_m", "mass_per_wheel_kg"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.z_offset_from_support_m <= 0.0:
            raise ValueError("z_offset_from_support_m must be positive")
        if self.radius_m <= 0.0 or self.width_m <= 0.0 or self.mass_per_wheel_kg <= 0.0:
            raise ValueError("front guide radius, width, and mass must be positive")
        if (
            not isinstance(self.rotation_coupling, str)
            or self.rotation_coupling not in {"free", "same_side_track_speed"}
        ):
            raise ValueError(
                "rotation_coupling must be 'free' or 'same_side_track_speed'"
            )

    def wheel_specs(self, robot_spec: RobotAssetSpec | None = None) -> tuple[FrontGuideWheelSpec, ...]:
        """Return the left/right guide geometry using the robot track spacing."""
        robot = ROBOT_SPEC if robot_spec is None else robot_spec
        half_track = robot.track_separation_m / 2.0
        z_m = robot.wheel_z_m + self.z_offset_from_support_m
        return tuple(
            FrontGuideWheelSpec(
                name=f"{side}_front_guide",
                side=side,
                x_m=self.x_m,
                y_m=y,
                z_m=z_m,
                radius_m=self.radius_m,
                width_m=self.width_m,
                mass_kg=self.mass_per_wheel_kg,
            )
            for side, y in (("left", half_track), ("right", -half_track))
        )


ROBOT_SPEC = RobotAssetSpec()
ROBOT_ASSET_FILENAME = "field_robot.usda"
ROBOT_ROOT_PATH = "/FieldRobot"
BASE_LINK_PATH = f"{ROBOT_ROOT_PATH}/base_link"
GIMBAL_YAW_LINK_PATH = f"{ROBOT_ROOT_PATH}/gimbal_yaw_link"
GIMBAL_PITCH_LINK_PATH = f"{GIMBAL_YAW_LINK_PATH}/gimbal_pitch_link"
GIMBAL_YAW_JOINT = "gimbal_yaw_joint"
GIMBAL_PITCH_JOINT = "gimbal_pitch_joint"


def robot_asset_spec_sha256(
    spec: RobotAssetSpec,
    front_guide_spec: FrontGuideSpec | None = None,
) -> str:
    """Return a stable digest of every declared robot-asset input.

    The historical no-guide payload is intentionally unchanged so existing
    14-wheel asset contracts keep their original digest.
    """
    payload_value: Any = asdict(spec)
    if front_guide_spec is not None:
        payload_value = {
            "frontGuideSpec": asdict(front_guide_spec),
            "robotAssetSpec": asdict(spec),
        }
    payload = json.dumps(payload_value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _box_diagonal_inertia(mass: float, size: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = size
    return (
        mass * (y * y + z * z) / 12.0,
        mass * (x * x + z * z) / 12.0,
        mass * (x * x + y * y) / 12.0,
    )


def _cylinder_y_diagonal_inertia(mass: float, radius: float, length: float) -> tuple[float, float, float]:
    axial = 0.5 * mass * radius * radius
    radial = mass * (3.0 * radius * radius + length * length) / 12.0
    return radial, axial, radial


@lru_cache(maxsize=1)
def _usd_modules():
    """Load pxr only when an authoring function is called."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics, Vt  # noqa: PLC0415

    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics, Vt


def _set_translate(xformable: Any, xyz: tuple[float, float, float]) -> None:
    Gf, *_ = _usd_modules()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*xyz))


def _author_mass_link(
    stage: Any,
    path: str,
    mass_kg: float,
    inertia: tuple[float, float, float],
    *,
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    Gf, _, _, UsdGeom, UsdPhysics, _, _ = _usd_modules()
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass_kg))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return prim


def _author_cube(stage: Any, path: str, size: tuple[float, float, float], *, collision: bool) -> Any:
    Gf, _, _, UsdGeom, UsdPhysics, _, _ = _usd_modules()
    cube = UsdGeom.Cube.Define(stage, path)
    # A unit-side cube scaled to the requested dimensions keeps the SDF/USD
    # geom extents easy to inspect without relying on the Cube default size 2.
    cube.CreateSizeAttr().Set(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def _author_cylinder(stage: Any, path: str, radius: float, length: float, *, collision: bool) -> Any:
    _, _, _, UsdGeom, UsdPhysics, _, _ = _usd_modules()
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr().Set(float(radius))
    cylinder.CreateHeightAttr().Set(float(length))
    cylinder.CreateAxisAttr().Set("Y")
    if collision:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    return cylinder.GetPrim()


def _author_revolute(
    stage: Any,
    path: str,
    body0_path: str,
    body1_path: str,
    axis: str,
    local_pos0: tuple[float, float, float],
    *,
    local_pos1: tuple[float, float, float] = (0.0, 0.0, 0.0),
    local_rot0: Any = None,
    local_rot1: Any = None,
    limits_deg: tuple[float, float] | None = None,
) -> Any:
    Gf, Sdf, _, _, UsdPhysics, _, _ = _usd_modules()
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos1))
    identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(local_rot0 if local_rot0 is not None else identity)
    joint.CreateLocalRot1Attr().Set(local_rot1 if local_rot1 is not None else identity)
    if limits_deg is not None:
        joint.CreateLowerLimitAttr().Set(float(limits_deg[0]))
        joint.CreateUpperLimitAttr().Set(float(limits_deg[1]))
    return joint


def _author_robot(
    stage: Any,
    spec: RobotAssetSpec,
    front_guide_spec: FrontGuideSpec | None = None,
) -> None:
    Gf, Sdf, _, UsdGeom, UsdPhysics, _, _ = _usd_modules()
    guide_wheels = () if front_guide_spec is None else front_guide_spec.wheel_specs(spec)
    guide_mass_kg = sum(wheel.mass_kg for wheel in guide_wheels)
    chassis_mass_kg = spec.chassis_mass_kg - guide_mass_kg
    if chassis_mass_kg <= 0.0:
        raise ValueError("front guide wheel masses leave no positive chassis mass")
    other_x_moment = sum(wheel.mass_kg * wheel.x_m for wheel in guide_wheels)
    other_z_moment = 2.0 * spec.wheel_count_per_side * spec.wheel_mass_kg * spec.wheel_z_m
    other_z_moment += 2.0 * spec.gimbal_link_mass_kg * spec.gimbal_offset_z_m
    other_z_moment += sum(wheel.mass_kg * wheel.z_m for wheel in guide_wheels)
    chassis_com_offset_x_m = -other_x_moment / chassis_mass_kg
    chassis_com_offset_z_m = -other_z_moment / chassis_mass_kg
    assembled_mass_kg = (
        chassis_mass_kg
        + 2.0 * spec.wheel_count_per_side * spec.wheel_mass_kg
        + 2.0 * spec.gimbal_link_mass_kg
        + guide_mass_kg
    )

    root = UsdGeom.Xform.Define(stage, ROBOT_ROOT_PATH).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    root.CreateAttribute("field:totalMassKg", Sdf.ValueTypeNames.Float).Set(spec.total_mass_kg)
    root.CreateAttribute("field:assembledMassKg", Sdf.ValueTypeNames.Float).Set(assembled_mass_kg)
    root.CreateAttribute("field:referenceHeightM", Sdf.ValueTypeNames.Float).Set(spec.reference_height_m)
    root.CreateAttribute("field:cameraHeightAboveGroundProxyM", Sdf.ValueTypeNames.Float).Set(
        spec.reference_height_m + spec.gimbal_offset_z_m
    )
    root.CreateAttribute("field:gimbalPitchAxis", Sdf.ValueTypeNames.String).Set("local -Y; positive raises forward +X")
    root.CreateAttribute("field:trackModel", Sdf.ValueTypeNames.String).Set(
        "seven rotating wheel proxies per side; no continuous belt"
    )
    root.CreateAttribute("field:trackGroundContactLengthM", Sdf.ValueTypeNames.Float).Set(
        spec.track_ground_contact_length_m
    )
    root.CreateAttribute("field:trackContactWidthM", Sdf.ValueTypeNames.Float).Set(spec.wheel_width_m)
    root.CreateAttribute("field:trackCenterSeparationM", Sdf.ValueTypeNames.Float).Set(
        spec.track_separation_m
    )
    root.CreateAttribute("field:bodyWidthM", Sdf.ValueTypeNames.Float).Set(spec.body_width_m)
    root.CreateAttribute("field:vehicleOverallWidthM", Sdf.ValueTypeNames.Float).Set(
        spec.vehicle_width_m
    )
    root.CreateAttribute("field:trackSeparationStatus", Sdf.ValueTypeNames.String).Set(
        "User-confirmed 0.47 m overall width includes 0.06 m tracks; 0.41 m center separation derived; 0.30 m body width excludes tracks."
    )
    root.CreateAttribute("field:driveSprocketRadiusM", Sdf.ValueTypeNames.Float).Set(
        spec.drive_sprocket_radius_m
    )
    root.CreateAttribute("field:driveSprocketDiameterM", Sdf.ValueTypeNames.Float).Set(
        spec.drive_sprocket_diameter_m
    )
    root.CreateAttribute("field:driveReductionRatio", Sdf.ValueTypeNames.Float).Set(
        spec.drive_reduction_ratio
    )
    root.CreateAttribute("field:proxyRollerRadiusM", Sdf.ValueTypeNames.Float).Set(spec.wheel_radius_m)
    root.CreateAttribute("field:driveGeometryNote", Sdf.ValueTypeNames.String).Set(
        "User-confirmed 60 mm active drive sprocket is separate from 45 mm rotating support-roller proxies."
    )
    root.CreateAttribute("field:comHeightAssumptionM", Sdf.ValueTypeNames.Float).Set(spec.reference_height_m)
    root.CreateAttribute("field:baseComOffsetZM", Sdf.ValueTypeNames.Float).Set(chassis_com_offset_z_m)
    if front_guide_spec is not None:
        left_guide, right_guide = guide_wheels
        root.CreateAttribute("field:frontGuideWheelCount", Sdf.ValueTypeNames.Int).Set(len(guide_wheels))
        root.CreateAttribute("field:frontGuideRotationCoupling", Sdf.ValueTypeNames.String).Set(
            front_guide_spec.rotation_coupling
        )
        root.CreateAttribute("field:frontGuidePassive", Sdf.ValueTypeNames.Bool).Set(True)
        root.CreateAttribute("field:frontGuidePassiveSemantics", Sdf.ValueTypeNames.String).Set(
            "No independent DriveAPI; rotation coupling is specified separately and realized by the runtime physics hook."
        )
        root.CreateAttribute("field:frontGuideRadiusM", Sdf.ValueTypeNames.Float).Set(
            front_guide_spec.radius_m
        )
        root.CreateAttribute(
            "field:frontGuideAngularVelocityRatioIfCoupled",
            Sdf.ValueTypeNames.Float,
        ).Set(spec.wheel_radius_m / front_guide_spec.radius_m)
        root.CreateAttribute("field:frontGuideWidthM", Sdf.ValueTypeNames.Float).Set(
            front_guide_spec.width_m
        )
        root.CreateAttribute("field:frontGuideMassPerWheelKg", Sdf.ValueTypeNames.Float).Set(
            front_guide_spec.mass_per_wheel_kg
        )
        root.CreateAttribute("field:frontGuideAxisCenterXM", Sdf.ValueTypeNames.Float).Set(
            front_guide_spec.x_m
        )
        root.CreateAttribute("field:frontGuideZOffsetFromSupportM", Sdf.ValueTypeNames.Float).Set(
            front_guide_spec.z_offset_from_support_m
        )
        root.CreateAttribute("field:frontGuideAxisCenterZM", Sdf.ValueTypeNames.Float).Set(
            left_guide.z_m
        )
        root.CreateAttribute("field:frontGuideLeftCenterYM", Sdf.ValueTypeNames.Float).Set(
            left_guide.y_m
        )
        root.CreateAttribute("field:frontGuideRightCenterYM", Sdf.ValueTypeNames.Float).Set(
            right_guide.y_m
        )
        root.CreateAttribute("field:baseComOffsetXM", Sdf.ValueTypeNames.Float).Set(
            chassis_com_offset_x_m
        )
        if front_guide_spec.rotation_coupling == "same_side_track_speed":
            rotation_note = (
                "Guide angular velocity follows same-side track linear speed through the runtime radius ratio "
                "R_main/R_guide; no independent actuator is authored."
            )
        else:
            rotation_note = "Guide revolute joints rotate freely and have no independent actuator."
        root.CreateAttribute("field:frontGuideModelNote", Sdf.ValueTypeNames.String).Set(
            "Trial-only wheel-proxy approximation; " + rotation_note + " Dimensions and masses are assumptions, "
            "not measured or calibrated vehicle properties, and this model does not represent a continuous track."
        )

    base_inertia = _box_diagonal_inertia(
        chassis_mass_kg, (spec.body_length_m, spec.body_width_m, spec.body_height_m)
    )
    _author_mass_link(
        stage,
        BASE_LINK_PATH,
        chassis_mass_kg,
        base_inertia,
        center_of_mass=(chassis_com_offset_x_m, 0.0, chassis_com_offset_z_m),
    )
    _author_cube(
        stage,
        f"{BASE_LINK_PATH}/body_collision_visual",
        (spec.body_length_m, spec.body_width_m, spec.body_height_m),
        collision=True,
    )

    for wheel in spec.wheel_specs:
        link_path = f"{BASE_LINK_PATH}/{wheel.name}_wheel_link"
        link = _author_mass_link(
            stage,
            link_path,
            spec.wheel_mass_kg,
            _cylinder_y_diagonal_inertia(spec.wheel_mass_kg, spec.wheel_radius_m, spec.wheel_width_m),
        )
        _set_translate(UsdGeom.Xformable(link), (wheel.x_m, wheel.y_m, wheel.z_m))
        link.CreateAttribute("field:trackSide", Sdf.ValueTypeNames.String).Set(wheel.side)
        _author_cylinder(
            stage,
            f"{link_path}/wheel_collision_visual",
            spec.wheel_radius_m,
            spec.wheel_width_m,
            collision=True,
        )
        _author_revolute(
            stage,
            f"{ROBOT_ROOT_PATH}/{wheel.name}_wheel_joint",
            BASE_LINK_PATH,
            link_path,
            "Y",
            (wheel.x_m, wheel.y_m, wheel.z_m),
        )

    for wheel in guide_wheels:
        link_path = f"{BASE_LINK_PATH}/{wheel.link_name}"
        link = _author_mass_link(
            stage,
            link_path,
            wheel.mass_kg,
            _cylinder_y_diagonal_inertia(wheel.mass_kg, wheel.radius_m, wheel.width_m),
        )
        _set_translate(UsdGeom.Xformable(link), (wheel.x_m, wheel.y_m, wheel.z_m))
        link.CreateAttribute("field:trackSide", Sdf.ValueTypeNames.String).Set(wheel.side)
        link.CreateAttribute("field:passiveFrontGuide", Sdf.ValueTypeNames.Bool).Set(True)
        _author_cylinder(
            stage,
            f"{link_path}/front_guide_collision_visual",
            wheel.radius_m,
            wheel.width_m,
            collision=True,
        )
        _author_revolute(
            stage,
            f"{ROBOT_ROOT_PATH}/{wheel.joint_name}",
            BASE_LINK_PATH,
            link_path,
            "Y",
            (wheel.x_m, wheel.y_m, wheel.z_m),
        )

    yaw_inertia = _box_diagonal_inertia(0.05, (0.04, 0.04, 0.03))
    yaw = _author_mass_link(stage, GIMBAL_YAW_LINK_PATH, spec.gimbal_link_mass_kg, yaw_inertia)
    _set_translate(UsdGeom.Xformable(yaw), (0.0, 0.0, spec.gimbal_offset_z_m))
    yaw.CreateAttribute("field:sensorMount", Sdf.ValueTypeNames.Bool).Set(True)
    _author_revolute(
        stage,
        f"{ROBOT_ROOT_PATH}/{GIMBAL_YAW_JOINT}",
        BASE_LINK_PATH,
        GIMBAL_YAW_LINK_PATH,
        "Z",
        (0.0, 0.0, spec.gimbal_offset_z_m),
    )

    pitch_inertia = _box_diagonal_inertia(0.05, (0.08, 0.08, 0.04))
    pitch = _author_mass_link(stage, GIMBAL_PITCH_LINK_PATH, spec.gimbal_link_mass_kg, pitch_inertia)
    pitch.CreateAttribute("field:sensorMount", Sdf.ValueTypeNames.Bool).Set(True)
    # USD's axis token is unsigned.  Rotate both joint frames 180 degrees about
    # Z so local +Y is local -Y in the gimbal frame; positive pitch then raises
    # a forward (+X) sensor.  The frame is parented under yaw, so yaw rotates it.
    negative_y = Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0))
    _author_revolute(
        stage,
        f"{ROBOT_ROOT_PATH}/{GIMBAL_PITCH_JOINT}",
        GIMBAL_YAW_LINK_PATH,
        GIMBAL_PITCH_LINK_PATH,
        "Y",
        (0.0, 0.0, 0.0),
        local_rot0=negative_y,
        local_rot1=negative_y,
        limits_deg=spec.pitch_limits_deg,
    )


def _existing_robot_asset_spec_sha256(path: Path) -> str | None:
    try:
        _, _, Usd, *_ = _usd_modules()
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            return None
        root = stage.GetPrimAtPath(ROBOT_ROOT_PATH)
        if not root or not root.IsValid():
            return None
        attr = root.GetAttribute("field:assetSpecSha256")
        return str(attr.Get()) if attr and attr.IsValid() and attr.Get() else None
    except Exception:
        return None


def ensure_robot_asset(
    output_dir: str | os.PathLike[str],
    *,
    force: bool = False,
    front_guide_spec: FrontGuideSpec | None = None,
) -> Path:
    """Create the deterministic local rover USDA and return its path.

    Args:
        output_dir: Base directory for the deterministic robot asset.
        force: Reauthor the file if it already exists.
        front_guide_spec: Optional front-guide wheel proxy with no independent
            actuator. Its rotation mode is stored in a separate spec-hash
            directory so it cannot replace the historical 14-wheel asset.

    The root prim is a floating articulation.  Spawn it at ``(x, y, 0.15)``
    to put the camera mount at 0.40 m above flat ground.  The pitch sensor mount
    itself has no collision geometry, so attaching a camera at its origin does
    not add a sensor-housing collider.
    """
    directory = Path(output_dir)
    spec = ROBOT_SPEC
    spec_sha256 = robot_asset_spec_sha256(spec, front_guide_spec)
    if front_guide_spec is not None:
        directory = directory / f"front_guides_{spec_sha256}"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / ROBOT_ASSET_FILENAME
    if output.is_file() and not force and _existing_robot_asset_spec_sha256(output) == spec_sha256:
        return output

    Gf, _, Usd, UsdGeom, _, _, _ = _usd_modules()
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    stage = Usd.Stage.CreateNew(str(temporary))
    try:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, ROBOT_ROOT_PATH).GetPrim()
        stage.SetDefaultPrim(root)
        root.CreateAttribute("field:assetSpecSha256", _usd_modules()[1].ValueTypeNames.String).Set(spec_sha256)
        if front_guide_spec is None:
            _author_robot(stage, spec)
        else:
            _author_robot(stage, spec, front_guide_spec)
        stage.GetRootLayer().Save()
    except Exception:
        del stage
        temporary.unlink(missing_ok=True)
        raise
    del stage
    os.replace(temporary, output)
    return output


def _validate_leaf_mesh(
    points: Sequence[Sequence[float]],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    normals: Sequence[Sequence[float]] | None,
    color: Sequence[float],
) -> tuple[list[tuple[float, float, float]], list[int], list[int], list[tuple[float, float, float]] | None, tuple[float, float, float]]:
    if not points:
        raise ValueError("corn leaf mesh needs at least one point")
    parsed_points = [tuple(float(component) for component in point) for point in points]
    if any(len(point) != 3 or not all(math.isfinite(value) for value in point) for point in parsed_points):
        raise ValueError("points must be finite 3D coordinates")
    counts = [int(count) for count in face_vertex_counts]
    indices = [int(index) for index in face_vertex_indices]
    if not counts or any(count < 3 for count in counts) or sum(counts) != len(indices):
        raise ValueError("face_vertex_counts must describe polygon faces matching face_vertex_indices")
    if any(index < 0 or index >= len(parsed_points) for index in indices):
        raise ValueError("face_vertex_indices contain an out-of-range point index")
    parsed_normals = None
    if normals is not None:
        parsed_normals = [tuple(float(component) for component in normal) for normal in normals]
        if len(parsed_normals) != len(parsed_points) or any(
            len(normal) != 3 or not all(math.isfinite(value) for value in normal) for normal in parsed_normals
        ):
            raise ValueError("vertex normals must be finite 3D vectors, one per point")
    parsed_color = tuple(float(component) for component in color)
    if len(parsed_color) != 3 or not all(math.isfinite(value) for value in parsed_color):
        raise ValueError("color must be a finite RGB triplet")
    if any(value < 0.0 or value > 1.0 for value in parsed_color):
        raise ValueError("color components must be in [0, 1]")
    return parsed_points, counts, indices, parsed_normals, parsed_color


def author_corn_leaf(
    stage: Any,
    prim_path: str,
    *,
    plant_id: int,
    leaf_id: int,
    points: Sequence[Sequence[float]],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
    normals: Sequence[Sequence[float]] | None,
    color: Sequence[float],
    damage_state: str = "healthy",
) -> str:
    """Author one non-colliding corn-leaf USD mesh from caller-supplied geometry.

    Geometry and the current scene's color are passed in explicitly; this
    function intentionally does not import or inspect ``field_sim``.  Plant,
    leaf, and damage identifiers are stored both as USD attributes and semantic
    labels for task adapters.
    """
    if damage_state not in {"healthy", "dry"}:
        raise ValueError("damage_state must be 'healthy' or 'dry'")
    if int(plant_id) < 0 or int(leaf_id) < 0:
        raise ValueError("plant_id and leaf_id must be non-negative")
    parsed_points, counts, indices, parsed_normals, parsed_color = _validate_leaf_mesh(
        points, face_vertex_counts, face_vertex_indices, normals, color
    )
    Gf, Sdf, _, UsdGeom, UsdPhysics, UsdSemantics, Vt = _usd_modules()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"USD prim already exists: {prim_path}")
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*point) for point in parsed_points]))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
    mesh.CreateDoubleSidedAttr().Set(True)
    mesh.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*parsed_color)]))
    if parsed_normals is not None:
        mesh.CreateNormalsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*normal) for normal in parsed_normals]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    prim = mesh.GetPrim()
    prim.CreateAttribute("field:plantId", Sdf.ValueTypeNames.Int).Set(int(plant_id))
    prim.CreateAttribute("field:leafId", Sdf.ValueTypeNames.Int).Set(int(leaf_id))
    prim.CreateAttribute("field:damageState", Sdf.ValueTypeNames.String).Set(damage_state)
    labels = UsdSemantics.LabelsAPI.Apply(prim, "class")
    labels.CreateLabelsAttr().Set([f"corn_leaf_{damage_state}"])
    for instance_name, label in (("plant_id", f"plant_{int(plant_id)}"), ("leaf_id", f"leaf_{int(leaf_id)}")):
        UsdSemantics.LabelsAPI.Apply(prim, instance_name).CreateLabelsAttr().Set([label])

    # Explicitly guard against accidental collision authoring if this helper is
    # later edited to add visual/material code.
    if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise AssertionError("corn leaves must remain visual-only and non-colliding")
    return prim_path


def robot_articulation_cfg(
    asset_path: str | os.PathLike[str],
    *,
    root_height_m: float | None = None,
    front_guide_spec: FrontGuideSpec | None = None,
):
    """Return an IsaacLab ``ArticulationCfg`` for the authored free-root asset.

    The USD carries the physical links, joints, and mass properties.  Wheel and
    gimbal drive gains should be configured by the task's actuator policy so
    they can be tuned without regenerating geometry.
    """
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg  # noqa: PLC0415
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    from isaaclab.assets import ArticulationCfg  # noqa: PLC0415

    height = ROBOT_SPEC.reference_height_m if root_height_m is None else float(root_height_m)
    joint_pos = {GIMBAL_YAW_JOINT: 0.0, GIMBAL_PITCH_JOINT: 0.0}
    if front_guide_spec is not None:
        joint_pos.update({wheel.joint_name: 0.0 for wheel in front_guide_spec.wheel_specs()})
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(asset_path).resolve()),
            articulation_props=[NewtonArticulationCfg(self_collision_enabled=False)],
            fix_root_link=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, height),
            joint_pos=joint_pos,
        ),
        actuators={},
    )

