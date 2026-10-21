"""Author task geometry once as standard USD, consumed by Isaac Lab/Newton."""
from pathlib import Path
import numpy as np

from .assets import author_corn_leaf


def _mesh(stage, path, points, triangles, color, collision=False):
    from pxr import UsdGeom, UsdPhysics, Vt
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = np.asarray(points, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.ravel()))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([color])
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    return mesh


def terrain_asset(terrain, output: Path, *, spacing=.25):
    """Same sampled relief for contact, sensing and visualization; rocks use cap meshes."""
    from pxr import Usd, UsdGeom
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    root = UsdGeom.Xform.Define(stage, '/Terrain').GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    size = terrain.size
    n = int(np.ceil(size / spacing)) + 1
    axis = np.linspace(0., size, n)
    x, y = np.meshgrid(axis, axis, indexing='xy')
    xy = np.stack((x.ravel(), y.ravel()), -1)
    points = np.c_[xy, terrain.height(xy)]
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing='ij')
    a = (i * n + j).ravel()
    triangles = np.concatenate((np.stack((a, a + 1, a + n), -1),
                                np.stack((a + 1, a + n + 1, a + n), -1)))
    _mesh(stage, '/Terrain/Ground', points, triangles, (.32, .20, .10), True)
    # Hemispherical rock caps, closed against their local ground plane.
    for k, (xy0, radius, height, base) in enumerate(zip(
            terrain.rock_xy, terrain.rock_radius, terrain.rock_height, terrain.rock_base_z)):
        angles = np.arange(16) * (2 * np.pi / 16)
        points = [[*xy0, base + height]]
        for theta in np.linspace(np.pi / 8, np.pi / 2, 4):
            for phi in angles:
                points.append([xy0[0] + radius * np.sin(theta) * np.cos(phi),
                               xy0[1] + radius * np.sin(theta) * np.sin(phi),
                               base + height * np.cos(theta)])
        faces = [[0, 1 + j, 1 + (j + 1) % 16] for j in range(16)]
        for ring in range(3):
            for j in range(16):
                a, b = 1 + ring * 16 + j, 1 + ring * 16 + (j + 1) % 16
                faces.extend(((a, a + 16, b), (b, a + 16, b + 16)))
        _mesh(stage, f'/Terrain/Rock_{k}', points, faces, (.40, .40, .38), True)
    stage.GetRootLayer().Save()
    return output


def plant_assets(scene, output_dir: Path):
    """Fixed per-plant shapes; episode randomization moves their kinematic roots."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for plant in range(len(scene.xy)):
        output = output_dir / f'corn_{plant:04}.usda'
        stage = Usd.Stage.CreateNew(str(output))
        root = UsdGeom.Xform.Define(stage, '/Corn').GetPrim()
        stage.SetDefaultPrim(root)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        rigid = UsdPhysics.RigidBodyAPI.Apply(root)
        rigid.CreateKinematicEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr(.01)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001, .001, .001))
        # A world-fixed kinematic root becomes a MuJoCo mocap body. A free
        # kinematic root instead adds six locked DOFs per plant to the solver.
        anchor = UsdPhysics.FixedJoint.Define(stage, '/Corn/Anchor')
        anchor.CreateBody1Rel().SetTargets(['/Corn'])
        stem = UsdGeom.Cylinder.Define(stage, '/Corn/Stem')
        stem.CreateRadiusAttr(float(scene.stem_radius[plant]))
        stem.CreateHeightAttr(float(scene.height[plant]))
        stem.CreateAxisAttr('Z')
        stem.AddTranslateOp().Set((0., 0., float(scene.height[plant] / 2)))
        stem.CreateDisplayColorAttr([(.15, .35, .06)])
        origin = np.array([*scene.xy[plant], scene.root_z[plant]])
        for leaf in np.flatnonzero(scene.leaf_plant == plant):
            # Tapered blade with enough segments for vertical occlusion, no physics collider.
            longitudinal = np.linspace(-1., 1., 9)
            ring = []
            for sign, samples in ((1, longitudinal), (-1, longitudinal[::-1])):
                for t in samples:
                    width = np.sqrt(max(0., 1 - t*t)) * (0.55 + .45 * (1 - t) / 2)
                    ring.append(scene.centers[leaf] - origin
                                + scene.u[leaf] * scene.axes[leaf, 0] * t
                                + scene.v[leaf] * scene.axes[leaf, 1] * width * sign)
            points = np.vstack((scene.centers[leaf] - origin, ring))
            # The ring runs clockwise relative to cross(u,v). Author both
            # windings: Newton's instanced Viser path drops USD doubleSided.
            # Coincident opposite faces retain the same physical leaf surface.
            front = np.array([[0, 1+(j+1) % len(ring), 1+j] for j in range(len(ring))])
            faces = np.concatenate((front, front[:, ::-1]))
            dry = bool(scene.leaf_dry[leaf])
            author_corn_leaf(stage, f'/Corn/Leaf_{leaf}', plant_id=plant, leaf_id=int(leaf),
                             points=points.tolist(), face_vertex_counts=[3]*len(faces),
                             face_vertex_indices=faces.ravel().tolist(), normals=None,
                             color=(.55,.36,.10) if dry else (.16,.43,.05),
                             damage_state='dry' if dry else 'healthy')
        stage.GetRootLayer().Save()
        outputs.append(output)
    return outputs


def crop_asset(scene, output_dir: Path):
    """One USD source for all fixed kinematic plants, without dynamic joints."""
    from pxr import Usd, UsdGeom
    paths=plant_assets(scene,output_dir/'corn')
    output=output_dir/'crop.usda'
    stage=Usd.Stage.CreateNew(str(output))
    root=UsdGeom.Xform.Define(stage,'/Plants').GetPrim()
    stage.SetDefaultPrim(root)
    for i,path in enumerate(paths):
        prim=stage.DefinePrim(f'/Plants/Corn_{i}','Xform')
        prim.GetReferences().AddReference(str(path))
        UsdGeom.Xformable(prim).AddTranslateOp().Set((*map(float,scene.xy[i]),float(scene.root_z[i])))
    UsdGeom.SetStageUpAxis(stage,'Z')
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.GetRootLayer().Save()
    return output
