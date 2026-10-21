// Debug-only renderer of the exact simulation geometry. Never a policy input.
import * as THREE from './vendor/three.module.min.js';

function terrainHeight(terrain, x, y) {
  let height = Number.isFinite(Number(terrain.base_height)) ? Number(terrain.base_height) : 0;
  const vectors = terrain.wave_vectors4x2 || [];
  const amplitudes = terrain.wave_amplitudes4 || [];
  const phases = terrain.wave_phases4 || [];
  for (let i = 0; i < Math.min(vectors.length, amplitudes.length, phases.length); i++) {
    height += amplitudes[i] * Math.sin(x * vectors[i][0] + y * vectors[i][1] + phases[i]);
  }
  return height;
}

class FieldCameraView {
  constructor(canvas, statusElement) {
    this.canvas = canvas;
    this.status = statusElement;
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false,
      powerPreference: 'low-power'});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x101a1d);
    this.world = new THREE.Scene();
    this.world.background = new THREE.Color(0xb6cbd0);
    this.camera = new THREE.PerspectiveCamera(60, 1, .025, 100);
    this.camera.up.set(0, 0, 1);
    const skyLight = new THREE.HemisphereLight(0xe8f3f5, 0x594c2b, 2.1);
    skyLight.position.set(0, 0, 1);
    this.world.add(skyLight);
    const sun = new THREE.DirectionalLight(0xfff5dc, 2.2);
    sun.position.set(-15, -10, 30);
    this.world.add(sun);
    this.geometry = new THREE.Group();
    this.world.add(this.geometry);
    this.lastFrame = -Infinity;
    this.width = this.height = 0;
    this.loaded = false;
    this.lost = false;
    this.lastPose = '';
    this.frameCount = 0;
    canvas.dataset.renderer = 'webgl2';
    canvas.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      this.lost = true;
      this.message('三维图形上下文已丢失；刷新页面恢复。俯视与控制仍可用。');
    });
    this.resize();
  }

  message(text) {
    if (this.status) this.status.textContent = text;
  }

  clearGeometry() {
    const geometries = new Set(), materials = new Set();
    this.geometry.traverse(object => {
      if (object.geometry) geometries.add(object.geometry);
      const list = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of list) if (material) materials.add(material);
      if (object.isInstancedMesh) object.dispose();
    });
    for (const geometry of geometries) geometry.dispose();
    for (const material of materials) material.dispose();
    this.geometry.clear();
  }

  setScene(payload) {
    if (!payload?.truth_debug_only || !payload.leaf_geometry)
      throw new Error('缺少仿真叶片几何，请重启更新后的仿真服务。');
    this.clearGeometry();
    const size = Number(payload.size), plants = payload.plants || [];
    const leaves = payload.leaf_geometry;
    this.range = Number(payload.config?.lidar_range) || 8;
    this.heightM = Number(payload.config?.camera_height) || .4;
    this.camera.far = Math.max(this.range, 30);
    this.world.fog = new THREE.Fog(0xb6cbd0, this.camera.far * .7, this.camera.far);

    const terrain = payload.terrain || {};
    const terrainSegments = Math.min(256, Math.max(1, Math.ceil(size * 2)));
    const groundGeometry = new THREE.PlaneGeometry(size, size, terrainSegments, terrainSegments);
    const groundPositions = groundGeometry.attributes.position;
    for (let i = 0; i < groundPositions.count; i++) {
      const x = groundPositions.getX(i) + size / 2;
      const y = groundPositions.getY(i) + size / 2;
      groundPositions.setZ(i, terrainHeight(terrain, x, y));
    }
    groundGeometry.computeVertexNormals();
    const ground = new THREE.Mesh(groundGeometry,
      new THREE.MeshLambertMaterial({color: 0x766547}));
    ground.position.set(size / 2, size / 2, 0);
    this.geometry.add(ground);
    const edgePoints = [];
    const addEdgePoint = (x, y) => edgePoints.push(new THREE.Vector3(x, y, terrainHeight(terrain, x, y) + .015));
    for (let i = 0; i <= terrainSegments; i++) addEdgePoint(size * i / terrainSegments, 0);
    for (let i = 1; i <= terrainSegments; i++) addEdgePoint(size, size * i / terrainSegments);
    for (let i = 1; i <= terrainSegments; i++) addEdgePoint(size - size * i / terrainSegments, size);
    for (let i = 1; i <= terrainSegments; i++) addEdgePoint(0, size - size * i / terrainSegments);
    this.geometry.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(edgePoints),
      new THREE.LineBasicMaterial({color: 0xe5c16a})));

    const dummy = new THREE.Object3D();
    const rocks = terrain.rocks || [];
    if (rocks.length) {
      const cap = new THREE.SphereGeometry(1, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2);
      cap.rotateX(Math.PI / 2);
      const stones = new THREE.InstancedMesh(cap,
        new THREE.MeshLambertMaterial({color: 0x899195}), rocks.length);
      for (let i = 0; i < rocks.length; i++) {
        const rock = rocks[i];
        dummy.position.set(rock.x, rock.y, rock.z);
        dummy.rotation.set(0, 0, 0);
        dummy.scale.set(rock.radius, rock.radius, rock.height);
        dummy.updateMatrix();
        stones.setMatrixAt(i, dummy.matrix);
      }
      stones.instanceMatrix.needsUpdate = true;
      stones.computeBoundingSphere();
      this.geometry.add(stones);
    }
    if (plants.length) {
      const stems = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 10),
        new THREE.MeshLambertMaterial({color: 0x668b36}), plants.length);
      for (let i = 0; i < plants.length; i++) {
        const p = plants[i];
        dummy.position.set(p.x, p.y, (Number(p.base_z) || 0) + p.height / 2);
        dummy.rotation.set(Math.PI / 2, 0, 0);
        const stemRadius = Number(p.stem_radius ?? p.radius);
        dummy.scale.set(stemRadius, p.height, stemRadius);
        dummy.updateMatrix();
        stems.setMatrixAt(i, dummy.matrix);
      }
      stems.instanceMatrix.needsUpdate = true;
      stems.computeBoundingSphere();
      this.geometry.add(stems);
    }

    const leafDisc = new THREE.CircleGeometry(1, 24);
    if (leaves.shape === 'tapered_blade_v1') {
      // Match Scene.cast(): y² <= (1-x²)(1-|x|), for normalized x in [-1,1].
      const positions = leafDisc.attributes.position;
      for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i), y = positions.getY(i);
        positions.setY(i, y * Math.sqrt(Math.max(0, 1 - Math.abs(x))));
      }
      positions.needsUpdate = true;
      leafDisc.computeBoundingSphere();
    }
    const x = new THREE.Vector3(), y = new THREE.Vector3(), normal = new THREE.Vector3();
    const scale = new THREE.Vector3(), center = new THREE.Vector3(), matrix = new THREE.Matrix4();
    for (const dry of [false, true]) {
      const indices = leaves.dry.flatMap((value, index) => !!value === dry ? [index] : []);
      if (!indices.length) continue;
      const material = new THREE.MeshLambertMaterial({color: dry ? 0xb17b42 : 0x4a8639,
        side: THREE.DoubleSide});
      const mesh = new THREE.InstancedMesh(leafDisc, material, indices.length);
      for (let j = 0; j < indices.length; j++) {
        const i = indices[j];
        x.fromArray(leaves.u[i]); y.fromArray(leaves.v[i]); normal.fromArray(leaves.normals[i]);
        scale.set(leaves.axes[i][0], leaves.axes[i][1], 1);
        matrix.makeBasis(x, y, normal).scale(scale);
        center.fromArray(leaves.centers[i]);
        matrix.setPosition(center);
        mesh.setMatrixAt(j, matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
      this.geometry.add(mesh);
    }
    this.loaded = true;
    this.lastFrame = -Infinity;
    this.lastPose = '';
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.round(rect.width), height = Math.round(rect.height);
    if (width < 2 || height < 2) return;
    if (this.width !== width || this.height !== height) {
      this.width = width; this.height = height;
      this.renderer.setSize(width, height, false);
      this.lastFrame = -Infinity;
      this.lastPose = '';
    }
  }

  render(state) {
    if (!state?.view_pose || !this.loaded || this.lost) return;
    this.resize();
    if (this.width < 2 || this.height < 2) return;
    const now = performance.now();
    if (now - this.lastFrame < 50) return;
    const pose = state.view_pose;
    const poseKey = [...pose.origin, ...pose.forward, ...pose.up, pose.half_angle].join(',');
    if (poseKey === this.lastPose) return;
    this.lastFrame = now;
    this.lastPose = poseKey;
    const origin = new THREE.Vector3(...pose.origin);
    const forward = new THREE.Vector3(...pose.forward);
    this.camera.position.copy(origin);
    this.camera.up.fromArray(pose.up);
    this.camera.lookAt(origin.clone().add(forward));
    // A square projection gives the same two tangent axes as the sensor cone.
    // Letterboxing keeps physical angles undistorted in a wide UI panel.
    const side = Math.min(this.width, this.height);
    this.camera.aspect = 1;
    this.camera.fov = THREE.MathUtils.radToDeg(2 * pose.half_angle);
    this.camera.updateProjectionMatrix();
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, this.width, this.height);
    this.renderer.setClearColor(0x101a1d);
    this.renderer.clear();
    this.renderer.setViewport((this.width - side) / 2, (this.height - side) / 2, side, side);
    this.renderer.setScissor((this.width - side) / 2, (this.height - side) / 2, side, side);
    this.renderer.setScissorTest(true);
    this.renderer.render(this.world, this.camera);
    this.renderer.setScissorTest(false);
    this.canvas.dataset.frames = String(++this.frameCount);
    const yaw = THREE.MathUtils.radToDeg(Math.atan2(pose.forward[1], pose.forward[0]));
    const pitch = THREE.MathUtils.radToDeg(Math.asin(Math.max(-1, Math.min(1, pose.forward[2]))));
    this.message(`实时云台视角 · 离地 ${this.heightM.toFixed(2)} m · 方位 ${yaw.toFixed(0)}° · 俯仰 ${pitch.toFixed(0)}° · 几何预览`);
  }

  dispose() {
    this.clearGeometry();
    this.renderer.dispose();
  }
}

window.FieldCameraView = FieldCameraView;
window.dispatchEvent(new Event('field-camera-ready'));
