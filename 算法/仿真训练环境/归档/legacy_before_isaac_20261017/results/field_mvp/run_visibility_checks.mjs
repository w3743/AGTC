import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const html = fs.readFileSync(path.join(root, 'field_sim/web/index.html'), 'utf8');
const refs = JSON.parse(fs.readFileSync(path.join(here, 'visibility_reference.json'), 'utf8'));

function extractFunction(name) {
  const match = new RegExp(`\\bfunction\\s+${name}\\s*\\(`).exec(html);
  assert(match, `could not find function ${name} in field_sim/web/index.html`);
  const start = match.index;
  const bodyStart = html.indexOf('{', match.index + match[0].length);
  assert(bodyStart >= 0, `could not find body for ${name}`);
  let depth = 0, quote = '', lineComment = false, blockComment = false, escaped = false;
  for (let i = bodyStart; i < html.length; i++) {
    const ch = html[i], next = html[i + 1];
    if (lineComment) { if (ch === '\n') lineComment = false; continue; }
    if (blockComment) { if (ch === '*' && next === '/') { blockComment = false; i++; } continue; }
    if (quote) {
      if (escaped) { escaped = false; continue; }
      if (ch === '\\') { escaped = true; continue; }
      if (ch === quote) quote = '';
      continue;
    }
    if (ch === '/' && next === '/') { lineComment = true; i++; continue; }
    if (ch === '/' && next === '*') { blockComment = true; i++; continue; }
    if (ch === '"' || ch === "'" || ch === '`') { quote = ch; continue; }
    if (ch === '{') depth++;
    else if (ch === '}' && --depth === 0) return html.slice(start, i + 1);
  }
  throw new Error(`unclosed function body for ${name}`);
}

const functionNames = [
  'arr', 'angle', 'rotateCapturedBody', 'depthCaptureFrame', 'depthPointWorld',
  'depthFrustumCorners', 'buildDepthWorldPoints', 'buildLidarWorldEndpoints',
  'lidarMaxRange',
  'ensureSensorData', 'worldToCanvas', 'drawSensorOverlay', 'drawSensors', 'drawLidarMapEndpoints', 'drawLidarRangeContour',
];
const declarations = functionNames.map(extractFunction).join('\n');

class MockContext {
  constructor(details = false) {
    this.details = details;
    this.path = [];
    this.fillPaths = [];
    this.strokePaths = [];
    this.fillEntries = [];
    this.strokeEntries = [];
    this.clearCount = 0;
    this.clipCount = 0;
    this.drawImageCount = 0;
    this.fillCount = 0;
    this.strokeCount = 0;
    this.lineDash = [];
  }
  beginPath() { this.path = []; }
  moveTo(x, y) { if (this.details) this.path.push({ op: 'moveTo', x, y }); }
  lineTo(x, y) { if (this.details) this.path.push({ op: 'lineTo', x, y }); }
  rect(x, y, w, h) { if (this.details) this.path.push({ op: 'rect', x, y, w, h }); }
  arc(x, y, r, a0, a1) { if (this.details) this.path.push({ op: 'arc', x, y, r, a0, a1 }); }
  closePath() { if (this.details) this.path.push({ op: 'closePath' }); }
  fill() { this.fillCount++; if (this.details) { this.fillPaths.push(this.path.slice()); this.fillEntries.push({ path: this.path.slice(), style: this.fillStyle }); } }
  stroke() { this.strokeCount++; if (this.details) { this.strokePaths.push(this.path.slice()); this.strokeEntries.push({ path: this.path.slice(), style: this.strokeStyle, lineWidth: this.lineWidth, dash: [...this.lineDash] }); } }
  clearRect() { this.clearCount++; }
  drawImage() { this.drawImageCount++; }
  clip() { this.clipCount++; }
  setTransform() {}
  setLineDash(dash) { this.lineDash = [...dash]; }
  save() {}
  restore() {}
}

function makeHarness(app, field, fctx, document) {
  return new Function('app', 'field', 'fctx', 'document', `${declarations}
    let depthBuildCalls=0, lidarBuildCalls=0;
    const rawDepthBuilder=buildDepthWorldPoints, rawLidarBuilder=buildLidarWorldEndpoints;
    buildDepthWorldPoints=function(...args){depthBuildCalls++;return rawDepthBuilder(...args);};
    buildLidarWorldEndpoints=function(...args){lidarBuildCalls++;return rawLidarBuilder(...args);};
    return {arr, angle, depthCaptureFrame, depthPointWorld, depthFrustumCorners,
      buildDepthWorldPoints, buildLidarWorldEndpoints, ensureSensorData, drawSensorOverlay, drawSensors, drawLidarMapEndpoints, drawLidarRangeContour,
      getBuildCounts:()=>({depth:depthBuildCalls,lidar:lidarBuildCalls})};
  `)(app, field, fctx, document);
}

function close(actual, expected, tolerance, label) {
  assert.equal(actual.length, expected.length, `${label}: length mismatch`);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs(Number(actual[i]) - Number(expected[i])) <= tolerance,
      `${label}[${i}] ${actual[i]} != ${expected[i]} ± ${tolerance}`);
  }
}

function frameFromCase(c, seq = 1) {
  return {
    seq,
    pose: c.pose,
    body_attitude: c.body_attitude,
    origin: c.origin,
    intrinsics: c.intrinsics,
    values: c.values,
    valid: c.valid,
    max_range: 5,
  };
}

const report = { geometry: [], cache_and_canvas: {}, timing_node_mock: {} };

const drawSource = extractFunction('draw');
const drawOrder = ['drawSensors(s)', 'drawLidarRangeContour()', 'drawPlantStates(s)', 'drawLidarMapEndpoints()', 'drawRobot(']
  .map(call => drawSource.indexOf(call));
assert(drawOrder.every(i => i >= 0) && drawOrder.every((i, n) => n === 0 || drawOrder[n - 1] < i),
  'draw order must place valid LiDAR markers after plant status rings and before the robot heading');
report.layer_order = ['depth_sensor_overlay', 'lidar_range_contour', 'plant_status_rings', 'lidar_endpoints', 'robot_heading'];

// Cross-check browser-JS functions against the separate Python reference for
// known world headings, positive/negative gimbal pitch, and body roll/pitch.
for (const c of refs) {
  const app = { sceneId: 'reference', sensorRevision: 0, sensorOverlayKey: '', depthMapData: null, lidarMapData: null };
  const field = { width: 400, height: 400 };
  const api = makeHarness(app, field, new MockContext(), { createElement() { throw new Error('unused'); } });
  const depth = frameFromCase(c);
  const frame = api.depthCaptureFrame(depth);
  assert(frame, `${c.name}: capture frame rejected`);
  close(frame.origin, c.origin, 1e-9, `${c.name} captured origin`);
  close(frame.f, c.basis.forward, 1e-10, `${c.name} forward basis`);
  close(frame.right, c.basis.right, 1e-10, `${c.name} right basis`);
  close(frame.up, c.basis.up, 1e-10, `${c.name} up basis`);

  const pointErrors = [];
  for (const p of c.points) {
    const got = api.depthPointWorld(frame, p.u, p.v, p.z);
    close(got, p.world, 1e-9, `${c.name} pixel(${p.u},${p.v}) captured world XYZ`);
    pointErrors.push(Math.max(...got.map((value, i) => Math.abs(value - p.world[i]))));
    assert.ok(Math.abs(p.camera_z_dot - p.z) < 1e-10, `${c.name}: Python reference cameraZ check failed`);
    if (p.u !== c.intrinsics[0][2] || p.v !== c.intrinsics[1][2]) {
      assert.ok(p.slant > p.z, `${c.name}: off-axis ray should have longer slant than cameraZ`);
    }
  }

  const corners = api.depthFrustumCorners(frame, 5);
  for (let i = 0; i < 4; i++) close(corners[i], c.frustum[i], 1e-9, `${c.name} frustum corner ${i}`);
  const cloud = api.buildDepthWorldPoints(depth, 14080);
  assert.equal(cloud.validCount, c.points.length, `${c.name}: valid holes must stay omitted`);
  assert.equal(cloud.displayCount, c.points.length, `${c.name}: all valid sample pixels retained`);
  assert(cloud.points instanceof Float32Array);
  assert.equal(cloud.points.length, 2 * c.points.length);
  for (let i = 0; i < c.points.length; i++) {
    close([cloud.points[2 * i], cloud.points[2 * i + 1]], c.points[i].world.slice(0, 2), 4e-6,
      `${c.name} XY map point ${i}`);
  }

  report.geometry.push({ name: c.name, validPixels: c.points.length,
    maxWorldErrorM: Math.max(0, ...pointErrors), frustumCornersChecked: 4,
    offAxisCameraZAndSlantChecked: true });
}

// Cardinal chassis headings establish the map's XY convention explicitly.
{
  const app = { sceneId: 'cardinal' };
  const api = makeHarness(app, { width: 20, height: 20 }, new MockContext(), { createElement() {} });
  const cases = [[0, [1, 0, 0]], [Math.PI / 2, [0, 1, 0]],
    [Math.PI, [-1, 0, 0]], [-Math.PI / 2, [0, -1, 0]]];
  for (const [yaw, expected] of cases) {
    const frame = api.depthCaptureFrame({ pose: [0, 0, yaw, 0, 0], body_attitude: [0, 0],
      origin: [0, 0, .4], intrinsics: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      values: [[1]], valid: [[true]] });
    assert(frame);
    close(frame.f, expected, 1e-12, `cardinal yaw ${yaw}`);
  }
  report.cardinal_yaw = { headingsChecked: ['east +X', 'north +Y', 'west -X', 'south -Y'] };
}

// LiDAR uses the captured world ray as supplied, includes only valid returns,
// and computes origin + direction * measured range without renormalizing.
{
  const app = { sceneId: 'lidar' };
  const api = makeHarness(app, { width: 100, height: 100 }, new MockContext(), { createElement() {} });
  const lidar = { origin: [4, 8, 0.6],
    directions: [[0, 1, 0], [1, 0, 0], [0.6, 0.8, 0], [2, 0, 1]],
    ranges: [2, 5, 3, 1.25], valid: [true, false, true, true] };
  const endpoints = api.buildLidarWorldEndpoints(lidar);
  assert.equal(endpoints.count, 3);
  assert(endpoints.points instanceof Float32Array);
  close(Array.from(endpoints.points), [4, 10, 5.8, 10.4, 6.5, 8], 2e-6,
    'LiDAR valid origin+worlddirection*range XY endpoints');
  report.lidar = { validEndpoints: endpoints.count, invalidReturnsSkipped: 1,
    deliberatelyNonUnitDirectionWasNotNormalized: true };
}

// Exercise the actual data/cache/render functions with an instrumented 2D
// canvas. No browser, server, or product state is started or changed.
{
  const overlayCtx = new MockContext(true), fctx = new MockContext(true);
  const app = { sceneId: 'mock-scene', sensorRevision: 0, sensorOverlayKey: '',
    sensorOverlay: null, depthMapData: null, lidarMapData: null,
    dpr: 1, w: 400, h: 400,
    map: { x: 0, y: 0, side: 400, size: 100, k: 4 } };
  const field = { width: 400, height: 400 };
  const document = { createElement(tag) {
    assert.equal(tag, 'canvas');
    return { width: 0, height: 0, getContext(type) { assert.equal(type, '2d'); return overlayCtx; } };
  } };
  const api = makeHarness(app, field, fctx, document);
  const c = refs[0], depth = frameFromCase(c, 41);
  const lidarDirections = Array.from({ length: 400 }, (_, i) => [Math.cos(i * Math.PI / 200), Math.sin(i * Math.PI / 200), 0]);
  const lidarRanges = Array.from({ length: 400 }, () => 4), lidarValid = Array(400).fill(false);
  for (const i of [50, 200, 399]) lidarValid[i] = true;
  const lidar = { seq: 9, origin: [10, 20, .4], max_range: 8,
    directions: lidarDirections, ranges: lidarRanges, valid: lidarValid };
  const state = { scene_id: app.sceneId, depth, lidar,
    view_pose: { origin: [91, -33, 10], forward: [0, -1, 0], up: [0, 0, 1] } };

  api.drawSensors(state);
  const data = api.getBuildCounts();
  assert.deepEqual(data, { depth: 1, lidar: 1 }, 'one projection per sequence');
  assert.equal(app.depthMapData.validCount, c.points.length);
  const capturedXY = Array.from(app.depthMapData.points);
  assert.equal(app.lidarMapData.count, 3, 'only three valid beams out of 400 are projected');
  assert.equal(app.lidarMapData.maxRange, 8);
  assert.equal(app.lidarMapData.rangeContour.length, 800, 'range contour uses all 400 captured directions');
  assert.equal(overlayCtx.clipCount, 0, 'sensor overlay is clipped by canvas edge only');

  const expectedCorners = api.depthFrustumCorners(app.depthMapData.frame, app.depthMapData.maxRange)
    .map(p => [app.map.x + p[0] * app.map.k, app.map.y + (app.map.size - p[1]) * app.map.k]);
  const frustumStroke = overlayCtx.strokePaths.find(path => path.some(op => op.op === 'lineTo'));
  assert(frustumStroke, 'frustum path was stroked');
  const lineEnds = frustumStroke.filter(op => op.op === 'lineTo').map(op => [op.x, op.y]);
  for (const corner of expectedCorners) {
    assert(lineEnds.some(p => Math.hypot(p[0] - corner[0], p[1] - corner[1]) < 1e-7),
      `frustum outline missing projected corner ${corner}`);
  }
  assert(overlayCtx.fillPaths.every(path => path.every(op => op.op !== 'lineTo')),
    'frustum lines are never filled and cannot mask terrain');
  const pointFill = overlayCtx.fillPaths.find(path => path.length && path.every(op => op.op === 'rect'));
  assert(pointFill, 'valid depth pixels drawn as point markers');
  assert.equal(pointFill.length, c.points.length, 'invalid depth holes produce no map markers');
  assert(overlayCtx.fillPaths.every(path => path.every(op => op.op !== 'arc')),
    'LiDAR is removed from the cached sensor overlay and drawn as a later independent layer');
  const flatContour = Array.from(app.lidarMapData.rangeContour);
  const flatXs = Array.from({ length: 400 }, (_, i) => flatContour[i * 2]);
  const flatYs = Array.from({ length: 400 }, (_, i) => flatContour[i * 2 + 1]);
  close([Math.min(...flatXs), Math.max(...flatXs)], [lidar.origin[0] - 8, lidar.origin[0] + 8], 2e-6,
    'horizontal scan circle X radius');
  close([Math.min(...flatYs), Math.max(...flatYs)], [lidar.origin[1] - 8, lidar.origin[1] + 8], 2e-6,
    'horizontal scan circle Y radius');
  api.drawLidarRangeContour();
  const getRangePath = () => fctx.strokePaths.findLast(path => path.length > 300 && path.some(op => op.op === 'closePath'));
  const flatRangePath = getRangePath();
  assert(flatRangePath, 'closed range contour is stroked');
  assert.equal(flatRangePath.length, 401, '400 captured beam directions form the closed perimeter');
  const rangeStroke = fctx.strokeEntries.findLast(e => e.path === flatRangePath);
  const rangeStrokeEntry = fctx.strokeEntries.findLast(e => e.path.length === 401 && e.path.some(op => op.op === 'closePath'));
  assert(rangeStrokeEntry);
  assert.equal(rangeStrokeEntry.style, 'rgba(108,217,209,.72)');
  assert.equal(rangeStrokeEntry.lineWidth, 1.1);
  assert.deepEqual(rangeStrokeEntry.dash, [4, 4]);
  assert(!fctx.fillPaths.some(path => path.length > 300), 'range contour is stroke only, never filled');
  api.drawLidarMapEndpoints();
  const getRayPath = () => fctx.strokePaths.findLast(path => path.length === 6 && path.every(op => op.op === 'moveTo' || op.op === 'lineTo'));
  const rayPath = getRayPath();
  assert(rayPath, 'one fine-line path draws exactly the three valid LiDAR beams');
  const expectedMapPoint = (x, y) => [app.map.x + x * app.map.k, app.map.y + (app.map.size - y) * app.map.k];
  const expectedOrigin = expectedMapPoint(lidar.origin[0], lidar.origin[1]);
  const expectedEndpoints = [50, 200, 399].map(i => expectedMapPoint(
    lidar.origin[0] + lidarDirections[i][0] * lidarRanges[i],
    lidar.origin[1] + lidarDirections[i][1] * lidarRanges[i]));
  const lineOps = rayPath;
  for (let i = 0; i < 3; i++) {
    close([lineOps[i * 2].x, lineOps[i * 2].y], expectedOrigin, 1e-8, `valid LiDAR beam ${i} origin`);
    close([lineOps[i * 2 + 1].x, lineOps[i * 2 + 1].y], expectedEndpoints[i], 5e-6, `valid LiDAR beam ${i} endpoint`);
  }
  const rayStroke = fctx.strokeEntries.find(e => e.path.length === 6 && e.path.every(op => op.op === 'moveTo' || op.op === 'lineTo'));
  assert.equal(rayStroke.style, 'rgba(108,217,209,.24)', 'beam line uses low-alpha cyan');
  assert.equal(rayStroke.lineWidth, .7, 'beam line is thin');
  const darkMarkers = fctx.fillEntries.filter(e => e.style === '#07100f' && e.path.some(op => op.op === 'closePath'));
  const brightDiamonds = fctx.strokeEntries.filter(e => e.style === '#8df7ef' && e.path.some(op => op.op === 'closePath'));
  const brightCores = fctx.fillEntries.filter(e => e.style === '#e4fffc' && e.path.some(op => op.op === 'arc'));
  assert.equal(darkMarkers.length, 3, 'three dark marker outlines are drawn');
  assert.equal(brightDiamonds.length, 3, 'three bright cyan diamonds are drawn');
  assert.equal(brightCores.length, 3, 'three bright centers are drawn');
  const firstClearCount = overlayCtx.clearCount, firstStrokeCount = overlayCtx.strokeCount;
  const firstRayPath = rayPath.map(op => ({ ...op }));
  const firstRangePath = flatRangePath.map(op => ({ ...op }));

  const changedCurrentView = { ...state, view_pose: { origin: [-1, -2, 3], forward: [1, 0, 0], up: [0, 0, 1] } };
  api.drawSensors(changedCurrentView);
  api.drawLidarRangeContour();
  api.drawLidarMapEndpoints();
  assert.deepEqual(api.getBuildCounts(), { depth: 1, lidar: 1 }, 'same sensor sequences do not reproject');
  assert.deepEqual(Array.from(app.depthMapData.points), capturedXY, 'current view pose cannot move captured depth points');
  assert.equal(overlayCtx.clearCount, firstClearCount, 'same view reuses cached overlay canvas');
  assert.equal(overlayCtx.strokeCount, firstStrokeCount);

  app.map = { ...app.map, x: 17, y: -21, k: 5, side: 500 };
  api.drawSensors(changedCurrentView);
  api.drawLidarRangeContour();
  api.drawLidarMapEndpoints();
  assert.equal(overlayCtx.clearCount, firstClearCount + 1, 'pan/zoom transform redraws overlay');
  assert.equal(overlayCtx.strokeCount, firstStrokeCount + 1, 'pan/zoom redraws the frustum');
  assert.deepEqual(api.getBuildCounts(), { depth: 1, lidar: 1 }, 'pan/zoom reuses world geometry');
  const zoomedRayPath = getRayPath();
  assert.notDeepEqual(zoomedRayPath, firstRayPath, 'pan/zoom reprojects cached world endpoints to the new screen transform');
  const zoomedRangePath = getRangePath();
  assert.notDeepEqual(zoomedRangePath, firstRangePath, 'pan/zoom redraws the cached world range contour');
  assert.equal(fctx.clipCount, 0, 'LiDAR returns outside the plot polygon are not hidden by map-edge clipping');
  const overlayBlitsAfterTransform = fctx.drawImageCount;

  const movedLidar = { ...lidar, seq: lidar.seq + 1, origin: [11, 19, .4] };
  const lidarOnlyState = { ...changedCurrentView, lidar: movedLidar };
  api.drawSensors(lidarOnlyState);
  api.drawLidarRangeContour();
  api.drawLidarMapEndpoints();
  assert.deepEqual(api.getBuildCounts(), { depth: 1, lidar: 2 }, 'new LiDAR seq reprojects LiDAR only');
  assert.equal(overlayCtx.clearCount, firstClearCount + 1, 'new LiDAR seq does not invalidate the depth overlay');
  const movedRayPath = getRayPath();
  const movedOrigin = expectedMapPoint(movedLidar.origin[0], movedLidar.origin[1]);
  close([movedRayPath[0].x, movedRayPath[0].y], movedOrigin, 1e-8, 'LiDAR origin stays paired with its captured frame');
  const movedContour = Array.from(app.lidarMapData.rangeContour);
  close([movedContour[0], movedContour[1]], [movedLidar.origin[0] + lidarDirections[0][0] * lidar.max_range,
    movedLidar.origin[1] + lidarDirections[0][1] * lidar.max_range], 2e-6,
    'range contour uses new sequence captured origin and directions');

  const depthOnlyState = { ...lidarOnlyState, depth: { ...depth, seq: depth.seq + 1 } };
  api.drawSensors(depthOnlyState);
  api.drawLidarMapEndpoints();
  assert.deepEqual(api.getBuildCounts(), { depth: 2, lidar: 2 }, 'new depth seq reprojects depth only');
  assert.equal(overlayCtx.clearCount, firstClearCount + 2, 'new sequence redraws overlay canvas');
  assert.equal(fctx.drawImageCount, overlayBlitsAfterTransform + 2, 'overlay is composited on both new depth frames');
  report.cache_and_canvas = { initialDepthBuilds: 1, initialLidarBuilds: 1,
    duplicateSequenceRebuilds: 0, changedViewRebuilds: 0, panZoomRebuilds: 0,
    panZoomOverlayRedraws: 1, lidarSequenceRebuilds: 1, depthSequenceRebuilds: 1,
    frustumStrokeSegments: lineEnds.length, frustumFillPaths: 0,
    depthMarkers: pointFill.length, validLidarEndpoints: 3, invalidLiDARBeamsSkipped: 397,
    lidarBeamLines: rayPath.filter(op => op.op === 'lineTo').length,
    lidarDarkOutlines: darkMarkers.length, lidarBrightDiamonds: brightDiamonds.length, lidarBrightCores: brightCores.length,
    rangeContourVertices: app.lidarMapData.rangeContour.length / 2, rangeContourStrokeOnly: true,
    flatContourSemiaxesM: [8, 8],
    depthOverlayClipCalls: overlayCtx.clipCount, lidarLayerClipCalls: fctx.clipCount };
}

// An empty return frame still carries the nominal scan-plane perimeter.
// At 60 degrees with no chassis tilt, all captured rays project to a 4-by-8 m
// ellipse in XY while producing zero return markers.
{
  const overlayCtx = new MockContext(true), fctx = new MockContext(true);
  const app = { sceneId: 'tilted-empty', sensorRevision: 0, sensorOverlayKey: '', sensorOverlay: null,
    depthMapData: null, lidarMapData: null, dpr: 1, w: 400, h: 400,
    map: { x: 0, y: 0, side: 400, size: 100, k: 4 } };
  const field = { width: 400, height: 400 };
  const api = makeHarness(app, field, fctx, { createElement() {
    return { width: 0, height: 0, getContext() { return overlayCtx; } };
  } });
  const scanPitch = Math.PI / 3, maxRange = 8, origin = [30, 30, .6];
  const directions = Array.from({ length: 400 }, (_, i) => {
    const az = i * Math.PI / 200;
    return [Math.cos(scanPitch) * Math.cos(az), Math.sin(az), Math.sin(scanPitch) * Math.cos(az)];
  });
  const lidar = { seq: 22, origin, max_range: maxRange, directions,
    ranges: Array(400).fill(maxRange), valid: Array(400).fill(false) };
  const state = { scene_id: app.sceneId, lidar, depth: frameFromCase(refs[0], 5),
    robot: { x: 95, y: 4, yaw: -1.8 },
    view_pose: { origin: [95, 4, 20], forward: [0, 1, 0], up: [0, 0, 1] } };
  api.drawSensors(state);
  assert.equal(app.lidarMapData.count, 0, 'all-invalid frame has zero return points');
  assert.equal(app.lidarMapData.rangeContour.length, 800, 'all 400 captured directions still produce the contour');
  const contourWorld = Array.from(app.lidarMapData.rangeContour);
  const xs = Array.from({ length: 400 }, (_, i) => contourWorld[2 * i]);
  const ys = Array.from({ length: 400 }, (_, i) => contourWorld[2 * i + 1]);
  close([Math.min(...xs), Math.max(...xs)], [origin[0] - 4, origin[0] + 4], 2e-6,
    '60 degree scan XY short axis');
  close([Math.min(...ys), Math.max(...ys)], [origin[1] - 8, origin[1] + 8], 2e-6,
    '60 degree scan XY long axis');
  close([contourWorld[0], contourWorld[1]], [origin[0] + directions[0][0] * maxRange,
    origin[1] + directions[0][1] * maxRange], 2e-6, 'contour uses captured origin and world direction');
  api.drawLidarRangeContour();
  api.drawLidarMapEndpoints();
  assert.equal(fctx.strokeEntries.length, 1, 'all-invalid frame draws only its contour outline');
  assert.equal(fctx.fillEntries.length, 0, 'all-invalid frame draws no endpoint markers or fill');
  const ellipsePath = fctx.strokeEntries[0].path;
  assert.equal(ellipsePath.length, 401);
  assert.equal(fctx.strokeEntries[0].style, 'rgba(108,217,209,.72)');
  assert.deepEqual(fctx.strokeEntries[0].dash, [4, 4]);
  const ellipseFirst = ellipsePath[0];
  close([ellipseFirst.x, ellipseFirst.y],
    [(origin[0] + directions[0][0] * maxRange) * app.map.k,
      (app.map.size - origin[1] - directions[0][1] * maxRange) * app.map.k],
    5e-6, 'rendered contour begins at captured first beam endpoint');

  const depthBuildsBefore = api.getBuildCounts();
  const contourRef = app.lidarMapData.rangeContour;
  api.drawSensors({ ...state, view_pose: { origin: [-500, -500, 0], forward: [-1, 0, 0], up: [0, 0, 1] } });
  assert.deepEqual(api.getBuildCounts(), depthBuildsBefore, 'same empty LiDAR seq does not recompute contour or endpoints');
  assert.equal(app.lidarMapData.rangeContour, contourRef, 'same seq reuses cached range contour');
  const firstPath = fctx.strokeEntries[0].path.map(op => ({ ...op }));
  app.map = { ...app.map, x: 13, y: -7, k: 5, side: 500 };
  api.drawLidarRangeContour();
  assert.deepEqual(api.getBuildCounts(), depthBuildsBefore, 'pan/zoom redraw uses cached world contour');
  assert.notDeepEqual(fctx.strokeEntries.at(-1).path, firstPath, 'pan/zoom redraws contour in screen space');
  assert.equal(fctx.clipCount, 0, 'contour is not clipped to the field polygon');
  report.lidar_range_contour = { flatCircleSemiaxesM: [8, 8], pitched60EllipseSemiaxesM: [4, 8],
    allInvalidBeamCount: 400, allInvalidReturnMarkers: 0, contourDirections: 400,
    contourOnlyStroke: true, sameSeqReprojectionCount: 0, panZoomRedrawWithoutProjection: true,
    capturedOriginAndWorldDirections: true, clipMapCalls: fctx.clipCount };
}

// Representative full 88x160 frame timing. This measures JavaScript geometry
// and the mocked Canvas command loop only, not a browser frame rate or raster.
if (!process.argv.includes('--quick')) {
{
  const app = { sceneId: 'timing', sensorRevision: 0, sensorOverlayKey: '', sensorOverlay: null,
    depthMapData: null, lidarMapData: null, dpr: 1, w: 800, h: 600,
    map: { x: -100, y: -100, side: 1000, size: 100, k: 10 } };
  const field = { width: 800, height: 600 };
  const overlayCtx = new MockContext(false), fctx = new MockContext(false);
  const api = makeHarness(app, field, fctx, { createElement() {
    return { width: 0, height: 0, getContext() { return overlayCtx; } };
  } });
  const width = 88, height = 160;
  const values = Array.from({ length: height }, (_, v) => Array.from({ length: width }, (_, u) =>
    0.31 + ((u * 7 + v * 3) % 469) / 100));
  const valid = Array.from({ length: height }, (_, v) => Array.from({ length: width }, (_, u) => (u + 3 * v) % 10 !== 0));
  const depth = { seq: 1, pose: [50, 50, 1.1, -0.2, 0.1], body_attitude: [0.06, -0.05],
    origin: [50, 50, 0.4], intrinsics: [[98.3248, 0, 44.01125], [0, 98.3248, 80.55786], [0, 0, 1]],
    values, valid, max_range: 5 };
  const lidar = { seq: 1, origin: [50, 50, .4], max_range: 8,
    directions: Array.from({ length: 400 }, (_, i) => [Math.cos(i * Math.PI / 200), Math.sin(i * Math.PI / 200), 0]),
    ranges: Array.from({ length: 400 }, (_, i) => 1 + (i % 70) / 10),
    valid: Array.from({ length: 400 }, (_, i) => i % 4 !== 0) };

  const projectionMs = [];
  for (let i = 0; i < 11; i++) {
    const t0 = performance.now();
    const cloud = api.buildDepthWorldPoints(depth, 14080);
    const dt = performance.now() - t0;
    assert.equal(cloud.displayCount, cloud.validCount);
    if (i > 0) projectionMs.push(dt);
  }
  api.ensureSensorData({ scene_id: 'timing', depth, lidar });
  const drawMs = [];
  for (let i = 0; i < 11; i++) {
    app.sensorOverlayKey = '';
    const t0 = performance.now();
    api.drawSensorOverlay({});
    const dt = performance.now() - t0;
    if (i > 0) drawMs.push(dt);
  }
  const median = a => a.sort((x, y) => x - y)[Math.floor(a.length / 2)];
  report.timing_node_mock = { canvasResolution: `${width}x${height}`, validDepthPoints: app.depthMapData.displayCount,
    validLidarEndpoints: app.lidarMapData.count, projectionMedianMs: +median(projectionMs).toFixed(3),
    overlayDrawMedianMs: +median(drawMs).toFixed(3), combinedMedianMs: +(median(projectionMs) + median(drawMs)).toFixed(3),
    projectionRuns: projectionMs.length, overlayRuns: drawMs.length,
    note: 'Node.js geometry + instrumented mock Canvas API; not a browser render, GPU raster time, or full-app FPS.' };
}
}

const out = path.join(here, process.argv.includes('--quick') ? 'lidar_marker_check_results.json' : 'visibility_check_results.json');
fs.writeFileSync(out, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
console.log(`saved ${out}`);
