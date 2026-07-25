import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const EXCLUSIVE_FRAME_GROUPS = Object.freeze([
  "omega_shared",
  "r3_native",
  "lingbot_native",
  "hloc_native",
]);

const FRAME_LABELS = {
  omega_shared: "VGGT Omega",
  r3_native: "R3 depth",
  lingbot_native: "LingBot depth",
  hloc_native: "HLoc sparse SfM",
};

const FRAME_HINTS = {
  omega_shared: "shared VGGT reconstruction frame · estimated display ground available",
  r3_native: "sampled native depth cloud · not aligned to Omega",
  lingbot_native: "sampled native depth cloud · not aligned to Omega",
  hloc_native: "sparse feature-track reconstruction · density is not a quality comparison",
};

const PATH_COLORS = {
  raw: 0x4f9dff,
  bspline: 0xffb54a,
  kalman: 0x61dfc2,
  rts: 0xf16fa8,
};

const root = document.getElementById("method-scene3d");
const fallback = document.getElementById("scene-canvas");
const layerStates = new Map();
const renderGroups = new Map();
let renderer;
let scene;
let camera;
let controls;
let manifest;
let manifestUrl;
let activeFrameGroup = "omega_shared";
let activeSample = 0;
let pointSize = 1.4;
let toolbar;
let layerTray;
let layerDetails;
let viewNote;
let groundButton;
let animationFrame = 0;
let resizeObserver;
let omegaAlignmentMode = null;
let visibleLayerNames = new Set(["points", "raw", "rts"]);
let globalPointBudget = null;
let frameSelectionToken = 0;
let referenceGrid;
const referenceVisibility = { cameras: true, grid: true };

function markFallback(reason) {
  cancelAnimationFrame(animationFrame);
  resizeObserver?.disconnect();
  controls?.dispose();
  scene?.traverse((object) => {
    object.geometry?.dispose?.();
    if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose?.());
    else object.material?.dispose?.();
  });
  renderGroups.clear();
  renderer?.dispose();
  renderer?.forceContextLoss?.();
  document.body.classList.remove("webgl-ready");
  document.body.classList.add("webgl-unavailable");
  if (root) root.hidden = true;
  if (fallback) fallback.removeAttribute("aria-hidden");
  const message = document.getElementById("canvas-fallback");
  if (message && reason) message.textContent = `WebGL unavailable: ${reason}. Canvas view retained.`;
}

function layerState(id) {
  if (!layerStates.has(id)) {
    layerStates.set(id, { status: "idle", object: null, error: null, budget: 0 });
  }
  return layerStates.get(id);
}

function validateRelativeLayer(layer) {
  if (!layer || layer.scale_status !== "relative_only") {
    throw new Error(`${layer?.id || "geometry layer"} is not relative_only`);
  }
  if (!EXCLUSIVE_FRAME_GROUPS.includes(layer.coordinate_frame)) {
    throw new Error(`${layer.id} has an unsupported coordinate frame`);
  }
}

function validateBuffer(buffer, declared, label) {
  if (!declared || !Number.isInteger(declared.bytes) || declared.bytes < 0) {
    throw new Error(`${label} lacks a valid declared byte count`);
  }
  if (buffer.byteLength !== declared.bytes) {
    throw new Error(`${label} byte mismatch: expected ${declared.bytes}, received ${buffer.byteLength}`);
  }
  if (!Array.isArray(declared.shape) || declared.shape.length !== 2 || declared.shape[1] !== 3) {
    throw new Error(`${label} must declare an N x 3 shape`);
  }
  const componentBytes = declared.dtype === "float32_le" ? 4 : declared.dtype === "uint8" ? 1 : 0;
  if (!componentBytes || declared.shape[0] * declared.shape[1] * componentBytes !== declared.bytes) {
    throw new Error(`${label} shape, dtype, and bytes disagree`);
  }
  return declared.shape[0];
}

async function fetchBuffer(asset, label) {
  const url = new URL(asset.path, window.location.href);
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${label} request failed (${response.status})`);
  const buffer = await response.arrayBuffer();
  validateBuffer(buffer, asset, label);
  return buffer;
}

function waitForOmegaBuffers() {
  const current = window.FPVViewer?.getSceneData?.();
  if (current?.points && current?.colors) return Promise.resolve(current);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Omega buffers did not become ready")), 12000);
    document.addEventListener("fpv-scene-ready", () => {
      clearTimeout(timeout);
      const data = window.FPVViewer?.getSceneData?.();
      data?.points && data?.colors ? resolve(data) : reject(new Error("Omega buffers are missing"));
    }, { once: true });
  });
}

async function reuseOmegaBuffers(layer) {
  const data = await waitForOmegaBuffers();
  updateOmegaDisplayTransform();
  if (data.points.byteLength !== layer.positions.bytes || data.colors.byteLength !== layer.colors.bytes) {
    throw new Error("Omega typed arrays do not match the validated manifest");
  }
  return { positions: data.points, colors: data.colors };
}

function uniformPointColor(layer) {
  const style = layer.render_style;
  if (
    style?.color_mode !== "uniform"
    || style?.source !== "display_only_not_method_output"
    || typeof style.color !== "string"
    || !/^#[0-9a-f]{6}$/i.test(style.color)
  ) {
    throw new Error(`${layer.id} colorless geometry requires an explicit display-only uniform color`);
  }
  return style.color;
}

function geometryFromBuffers(layer, positions, colors) {
  const total = layer.point_count;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const materialOptions = {
    size: pointSize,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0.94,
  };
  if (colors) {
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
    materialOptions.vertexColors = true;
  } else {
    materialOptions.color = uniformPointColor(layer);
  }
  const material = new THREE.PointsMaterial(materialOptions);
  geometry.setDrawRange(0, total);
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  return points;
}

async function fetchCameraCenters(layer) {
  if (!layer.cameras?.path) return [];
  const url = new URL(layer.cameras.path, window.location.href);
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`camera path request failed (${response.status})`);
  const payload = await response.json();
  if (payload.scale_status !== "relative_only") throw new Error("camera path is not relative_only");
  const records = Array.isArray(payload.samples) ? payload.samples : payload.registered_images || [];
  return records.map((sample) => sample.center).filter((center) => Array.isArray(center) && center.length === 3 && center.every(Number.isFinite));
}

function lineFromPoints(points, color, segmentStarts = []) {
  if (!Array.isArray(points) || points.length < 2) return null;
  const vertices = [];
  for (let index = 1; index < points.length; index += 1) {
    if (segmentStarts[index]) continue;
    vertices.push(...points[index - 1], ...points[index]);
  }
  if (!vertices.length) return null;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  return new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }),
  );
}

function addNativeCameraPath(group, centers) {
  const line = lineFromPoints(centers, 0xffb54a);
  if (line) {
    line.userData.kind = "native-camera-center path";
    line.userData.pathLayer = "raw";
    line.visible = visibleLayerNames.has("raw");
    group.add(line);
  }
}

function addOmegaPaths(group) {
  const paths = window.FPVViewer?.getSceneData?.().paths;
  if (!paths?.layers) return;
  Object.entries(PATH_COLORS).forEach(([name, color]) => {
    const line = lineFromPoints(
      paths.layers[name],
      color,
      paths.segment_boundaries || [],
    );
    if (!line) return;
    line.name = `omega-${name}-path`;
    line.userData.pathLayer = name;
    line.visible = window.FPVViewer?.getViewState?.().visibleLayers?.includes(name) ?? ["raw", "rts"].includes(name);
    group.add(line);
  });
}

function addOmegaCameraFrusta(group) {
  if (group.getObjectByName("recovered camera frusta")) return;
  const data = window.FPVViewer?.getSceneData?.();
  const samples = data?.cameras?.samples || [];
  if (!samples.length) return;
  const bounds = data.scene?.bounds;
  const span = bounds ? Math.max(...bounds.max.map((value, axis) => value - bounds.min[axis]), 1e-6) : 1;
  const depth = span * 0.035;
  const stride = Math.max(1, Math.floor(samples.length / 24));
  const vertices = [];
  const pushSegment = (start, end) => vertices.push(...start.toArray(), ...end.toArray());
  for (let index = 0; index < samples.length; index += stride) {
    const sample = samples[index];
    const position = new THREE.Vector3(...sample.position);
    const [qw, qx, qy, qz] = sample.quaternion_wxyz || [1, 0, 0, 0];
    const rotation = new THREE.Quaternion(qx, qy, qz, qw).normalize();
    const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(rotation);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(rotation);
    const down = new THREE.Vector3(0, 1, 0).applyQuaternion(rotation);
    const base = position.clone().addScaledVector(forward, depth);
    const corners = [
      base.clone().addScaledVector(right, depth * .58).addScaledVector(down, depth * .36),
      base.clone().addScaledVector(right, -depth * .58).addScaledVector(down, depth * .36),
      base.clone().addScaledVector(right, -depth * .58).addScaledVector(down, -depth * .36),
      base.clone().addScaledVector(right, depth * .58).addScaledVector(down, -depth * .36),
    ];
    corners.forEach((corner) => pushSegment(position, corner));
    corners.forEach((corner, cornerIndex) => pushSegment(corner, corners[(cornerIndex + 1) % corners.length]));
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  const frusta = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: 0xffb54a, transparent: true, opacity: .66 }));
  frusta.name = "recovered camera frusta";
  frusta.userData.referenceCameras = true;
  frusta.visible = referenceVisibility.cameras;
  group.add(frusta);
}

function addActiveCameraProxy(group) {
  const cone = new THREE.ConeGeometry(0.035, 0.1, 4);
  cone.rotateX(Math.PI / 2);
  const proxy = new THREE.Mesh(cone, new THREE.MeshBasicMaterial({ color: 0xffb54a, wireframe: true }));
  proxy.name = "active camera proxy";
  proxy.visible = referenceVisibility.cameras;
  group.add(proxy);
}

function updateActiveCameraProxy() {
  const group = renderGroups.get(activeFrameGroup);
  const proxy = group?.getObjectByName("active camera proxy");
  if (!proxy) return;
  const record = layerStates.get([...layerStates.keys()].find((id) => {
    const layer = manifest?.geometry_layers?.layers?.[id];
    return layer?.coordinate_frame === activeFrameGroup && layerStates.get(id)?.cameraCenters?.length;
  }));
  let centers = record?.cameraCenters;
  if (activeFrameGroup === "omega_shared") centers = window.FPVViewer?.getSceneData?.().paths?.layers?.raw;
  if (!centers?.length) return;
  const index = Math.max(0, Math.min(centers.length - 1, activeSample));
  proxy.position.fromArray(centers[index]);
  const nextIndex = Math.min(index + 1, centers.length - 1);
  const segmentStarts = window.FPVViewer?.getSceneData?.().paths?.segment_boundaries || [];
  const next = segmentStarts[nextIndex] ? centers[index] : centers[nextIndex];
  if (next && new THREE.Vector3(...next).distanceTo(proxy.position) > 1e-8) proxy.lookAt(...next);
}

async function loadLayer(id) {
  const layer = manifest.geometry_layers.layers[id];
  const current = layerState(id);
  if (current.status === "ready" || current.status === "loading") return current;
  current.status = "loading";
  current.error = null;
  renderLayerTray();
  try {
    validateRelativeLayer(layer);
    if (layer.status !== "available") throw new Error(layer.reason || "layer unavailable");
    const buffers = id === "vggt_omega"
      ? await reuseOmegaBuffers(layer)
      : await fetchBuffer(layer.positions, `${id} positions`).then(async (positions) => {
          let colors = null;
          if (layer.colors) {
            colors = new Uint8Array(await fetchBuffer(layer.colors, `${id} colors`));
          } else {
            uniformPointColor(layer);
          }
          return { positions: new Float32Array(positions), colors };
        });
    const count = validateBuffer(buffers.positions.buffer, layer.positions, `${id} positions`);
    if (buffers.colors) {
      const colorCount = validateBuffer(buffers.colors.buffer, layer.colors, `${id} colors`);
      if (colorCount !== count) throw new Error(`${id} color count mismatch`);
    }
    if (count !== layer.point_count) throw new Error(`${id} point_count mismatch`);
    const group = renderGroups.get(layer.coordinate_frame);
    const object = geometryFromBuffers(layer, buffers.positions, buffers.colors);
    object.name = id;
    group.add(object);
    current.object = object;
    current.budget = Math.min(count, globalPointBudget || (id === "vggt_omega" ? 80000 : 50000));
    object.visible = visibleLayerNames.has("points");
    object.geometry.setDrawRange(0, current.budget);
    current.cameraCenters = await fetchCameraCenters(layer).catch((error) => {
      current.cameraError = error.message;
      return [];
    });
    if (activeFrameGroup === "omega_shared" && id === "vggt_omega") {
      addOmegaPaths(group);
      addOmegaCameraFrusta(group);
    }
    if (current.cameraCenters.length) addNativeCameraPath(group, current.cameraCenters);
    if (!group.getObjectByName("active camera proxy")) addActiveCameraProxy(group);
    current.status = "ready";
    fitBounds("Orbit");
  } catch (error) {
    current.status = "error";
    current.error = error?.message || String(error);
  }
  renderLayerTray();
  updateActiveCameraProxy();
  return current;
}

function visibleLayersForFrame(frame) {
  return Object.values(manifest.geometry_layers.layers).filter((layer) => layer.coordinate_frame === frame);
}

function setGroundAvailability() {
  const supported = manifest.geometry_layers.coordinate_frames?.[activeFrameGroup]?.ground_display_supported === true && canShowGroundGrid();
  groundButton.disabled = !supported;
  groundButton.title = supported ? "Use the confidence-gated estimated display plane; scale remains relative" : "Estimated ground is unavailable in this method-native frame";
}

async function selectFrameGroup(frame, preferredLayer) {
  if (!EXCLUSIVE_FRAME_GROUPS.includes(frame) || !manifest) return;
  const available = visibleLayersForFrame(frame).filter((layer) => layer.status === "available");
  if (!available.length) {
    if (viewNote) viewNote.textContent = `${FRAME_LABELS[frame]} · no validated geometry layer available`;
    return;
  }
  const selectionToken = ++frameSelectionToken;
  activeFrameGroup = frame;
  renderGroups.forEach((group, key) => { group.visible = key === frame; });
  if (referenceGrid) referenceGrid.visible = referenceVisibility.grid && canShowGroundGrid();
  toolbar.querySelectorAll("[data-frame]").forEach((button) => {
    const active = button.dataset.frame === frame;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  setGroundAvailability();
  viewNote.textContent = `${FRAME_LABELS[frame]} · ${FRAME_HINTS[frame]} · relative only`;
  renderLayerTray();
  const target = preferredLayer ? available.find((layer) => layer.id === preferredLayer) : available[0];
  if (target) await loadLayer(target.id);
  if (selectionToken !== frameSelectionToken) return;
  document.dispatchEvent(new CustomEvent("fpv-frame-group-changed", { detail: { frameGroup: frame, layerId: target?.id } }));
  fitBounds("Orbit");
}

function updateFrameButtonAvailability() {
  if (!manifest || !toolbar) return;
  toolbar.querySelectorAll("[data-frame]").forEach((button) => {
    button.disabled = !visibleLayersForFrame(button.dataset.frame).some((layer) => layer.status === "available");
  });
}

function renderLayerTray() {
  if (!manifest || !layerTray) return;
  const rows = visibleLayersForFrame(activeFrameGroup).map((layer) => {
    const state = layerState(layer.id);
    const row = document.createElement("div");
    row.className = `method-scene3d-layer ${state.status === "error" ? "layer-error" : ""}`;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${layer.method.replaceAll("_", " ")} · ${layer.variant.replaceAll("_", " ")}`;
    button.disabled = layer.status !== "available" || state.status === "loading";
    button.addEventListener("click", () => loadLayer(layer.id));
    const status = document.createElement("span");
    status.textContent = state.status === "error" ? `error · ${state.error}` : layer.status !== "available" ? "unavailable" : state.status;
    row.append(button, status);
    if (state.status === "ready") {
      const range = document.createElement("input");
      range.type = "range";
      range.min = "1000";
      range.max = String(layer.point_count);
      range.step = String(Math.max(500, Math.round(layer.point_count / 100)));
      range.value = String(state.budget);
      range.setAttribute("aria-label", `${layer.method} point budget`);
      range.addEventListener("input", () => {
        state.budget = Number(range.value);
        state.object.geometry.setDrawRange(0, state.budget);
        status.textContent = `ready · ${state.budget.toLocaleString()} / ${layer.point_count.toLocaleString()}`;
      });
      row.append(range);
    }
    return row;
  });
  layerTray.replaceChildren(...rows);
}

function quantile(values, fraction) {
  values.sort((a, b) => a - b);
  return values[Math.max(0, Math.min(values.length - 1, Math.round((values.length - 1) * fraction)))];
}

function robustPointBounds(object) {
  const attribute = object?.geometry?.getAttribute?.("position");
  if (!attribute?.count) return new THREE.Box3().setFromObject(object);
  const axes = [[], [], []];
  const stride = Math.max(1, Math.ceil(attribute.count / 20000));
  for (let index = 0; index < attribute.count; index += stride) {
    axes[0].push(attribute.getX(index));
    axes[1].push(attribute.getY(index));
    axes[2].push(attribute.getZ(index));
  }
  const local = new THREE.Box3(
    new THREE.Vector3(...axes.map((axis) => quantile(axis, .01))),
    new THREE.Vector3(...axes.map((axis) => quantile(axis, .99))),
  );
  object.updateWorldMatrix(true, false);
  return local.applyMatrix4(object.matrixWorld);
}

function currentBounds() {
  const group = renderGroups.get(activeFrameGroup);
  group?.updateMatrixWorld(true);
  const box = new THREE.Box3().makeEmpty();
  layerStates.forEach((state) => {
    if (state.status === "ready" && state.object?.parent === group && state.object.visible) box.union(robustPointBounds(state.object));
  });
  if (box.isEmpty() && group) box.setFromObject(group);
  if (box.isEmpty()) return new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
  return box;
}

function activeCameraCenters() {
  if (activeFrameGroup === "omega_shared") {
    return window.FPVViewer?.getSceneData?.().paths?.layers?.raw || [];
  }
  const id = [...layerStates.keys()].find((candidate) => {
    const layer = manifest?.geometry_layers?.layers?.[candidate];
    return layer?.coordinate_frame === activeFrameGroup && layerStates.get(candidate)?.cameraCenters?.length;
  });
  return id ? layerStates.get(id).cameraCenters : [];
}

function canShowGroundGrid() {
  return activeFrameGroup === "omega_shared"
    && window.FPVViewer?.getAlignmentMode?.() === "estimated_ground"
    && Boolean(window.FPVViewer?.getGroundDisplayTransform?.());
}

function updateOmegaDisplayTransform() {
  const group = renderGroups.get("omega_shared");
  if (!group) return false;
  const mode = window.FPVViewer?.getAlignmentMode?.() || "raw";
  if (mode === omegaAlignmentMode) return false;
  group.matrixAutoUpdate = false;
  group.matrix.identity();
  const transform = window.FPVViewer?.getGroundDisplayTransform?.();
  if (mode === "estimated_ground" && transform) {
    const r = transform.rotation_3x3;
    const t = transform.translation;
    group.matrix.set(
      r[0][0], r[0][1], r[0][2], t[0],
      r[1][0], r[1][1], r[1][2], t[1],
      r[2][0], r[2][1], r[2][2], t[2],
      0, 0, 0, 1,
    );
  }
  group.updateMatrixWorld(true);
  omegaAlignmentMode = mode;
  if (referenceGrid) referenceGrid.visible = referenceVisibility.grid && canShowGroundGrid();
  return true;
}

function fitBounds(preset) {
  if (!camera || !controls) return;
  const box = currentBounds();
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 1e-4);
  if (referenceGrid) {
    referenceGrid.visible = referenceVisibility.grid && canShowGroundGrid();
    if (referenceGrid.visible) {
      referenceGrid.position.set(center.x, 0, center.z);
      referenceGrid.scale.setScalar(span / 2);
    }
  }
  controls.target.copy(center);
  if (preset === "Ground-relative") {
    const groundY = canShowGroundGrid() ? 0 : box.min.y;
    camera.position.set(center.x + span * 0.8, groundY + span * 0.08, center.z + span * 1.1);
    viewNote.textContent = "Estimated display ground · uncalibrated orientation only · relative scale preserved";
  } else if (preset === "Camera") {
    const centers = activeCameraCenters();
    const index = Math.min(activeSample, Math.max(centers.length - 1, 0));
    const position = centers[index];
    if (position) {
      const worldPosition = new THREE.Vector3(...position);
      if (activeFrameGroup === "omega_shared") renderGroups.get("omega_shared").localToWorld(worldPosition);
      camera.position.copy(worldPosition);
      const lookIndex = index < centers.length - 1 ? index + 1 : Math.max(0, index - 1);
      const lookAt = new THREE.Vector3(...centers[lookIndex]);
      if (activeFrameGroup === "omega_shared") renderGroups.get("omega_shared").localToWorld(lookAt);
      if (lookAt.distanceTo(worldPosition) > 1e-8) controls.target.copy(lookAt);
      viewNote.textContent = "Recovered camera-center path viewpoint · look direction proxy · relative only";
    } else {
      camera.position.set(center.x + span, center.y + span * 0.35, center.z + span);
      viewNote.textContent = "Camera path unavailable · synthetic scene-bounds viewpoint · relative only";
    }
  } else {
    camera.position.set(center.x + span * 0.95, center.y + span * 0.58, center.z + span * 0.95);
    viewNote.textContent = `${FRAME_LABELS[activeFrameGroup]} · ${FRAME_HINTS[activeFrameGroup]} · relative only`;
  }
  camera.near = Math.max(span / 10000, 1e-5);
  camera.far = span * 1000;
  camera.updateProjectionMatrix();
  controls.update();
}

function projectDisplayPoint(point, width, height) {
  const vector = new THREE.Vector3(...point).project(camera);
  return [(vector.x * 0.5 + 0.5) * width, (-vector.y * 0.5 + 0.5) * height];
}

function projectScenePoint(point, width, height) {
  const vector = new THREE.Vector3(...point);
  if (activeFrameGroup === "omega_shared") renderGroups.get("omega_shared")?.localToWorld(vector);
  vector.project(camera);
  return [(vector.x * 0.5 + 0.5) * width, (-vector.y * 0.5 + 0.5) * height];
}

function buildToolbar() {
  toolbar = document.createElement("div");
  toolbar.className = "method-scene3d-toolbar";
  const label = document.createElement("div");
  label.className = "method-scene3d-label";
  label.textContent = "METHOD FRAME · ONE COORDINATE SYSTEM AT A TIME";
  const frames = document.createElement("div");
  frames.className = "method-scene3d-frames";
  EXCLUSIVE_FRAME_GROUPS.forEach((frame) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.frame = frame;
    button.textContent = FRAME_LABELS[frame];
    button.disabled = true;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => selectFrameGroup(frame));
    frames.appendChild(button);
  });
  const presets = document.createElement("div");
  presets.className = "method-scene3d-presets";
  ["Orbit", "Ground-relative", "Camera"].forEach((name) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    if (name === "Ground-relative") groundButton = button;
    button.addEventListener("click", () => fitBounds(name));
    presets.appendChild(button);
  });
  layerDetails = document.createElement("details");
  layerDetails.className = "method-scene3d-layer-details";
  const layerSummary = document.createElement("summary");
  layerSummary.textContent = "Point budget and layer detail";
  layerTray = document.createElement("div");
  layerTray.className = "method-scene3d-layers";
  layerDetails.append(layerSummary, layerTray);
  viewNote = document.createElement("div");
  viewNote.className = "method-scene3d-note";
  viewNote.setAttribute("role", "status");
  toolbar.append(label, frames, presets, layerDetails, viewNote);
  root.appendChild(toolbar);
}

function initialiseRenderer() {
  if (!root || !window.WebGLRenderingContext) throw new Error("WebGL is not supported");
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer.setClearColor(0x02090d, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.domElement.setAttribute("aria-label", "WebGL relative reconstruction method viewer");
  renderer.domElement.setAttribute(
    "aria-description",
    "Use arrow keys to orbit, plus and minus to zoom, and zero to reset the view.",
  );
  renderer.domElement.tabIndex = 0;
  root.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(48, 1, 0.001, 10000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.addEventListener("change", () => {
    document.dispatchEvent(new CustomEvent("fpv-webgl-camera-changed"));
  });
  renderer.domElement.addEventListener("keydown", (event) => {
    const handled = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "_", "0"]);
    if (!handled.has(event.key)) return;
    event.preventDefault();
    if (event.key === "0") {
      fitBounds("Orbit");
      return;
    }
    const offset = camera.position.clone().sub(controls.target);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    if (event.key === "ArrowLeft") spherical.theta -= .08;
    if (event.key === "ArrowRight") spherical.theta += .08;
    if (event.key === "ArrowUp") spherical.phi = Math.max(.08, spherical.phi - .08);
    if (event.key === "ArrowDown") spherical.phi = Math.min(Math.PI - .08, spherical.phi + .08);
    if (event.key === "+" || event.key === "=") spherical.radius = Math.max(camera.near * 4, spherical.radius / 1.15);
    if (event.key === "-" || event.key === "_") spherical.radius = Math.min(camera.far * .25, spherical.radius * 1.15);
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical));
    controls.update();
  });
  EXCLUSIVE_FRAME_GROUPS.forEach((frame) => {
    const group = new THREE.Group();
    group.name = frame;
    group.visible = frame === activeFrameGroup;
    renderGroups.set(frame, group);
    scene.add(group);
  });
  referenceGrid = new THREE.GridHelper(2, 20, 0x31586a, 0x17303b);
  referenceGrid.name = "scale-free reference grid";
  referenceGrid.visible = false;
  scene.add(referenceGrid);
  const resize = () => {
    const width = Math.max(root.clientWidth, 1);
    const height = Math.max(root.clientHeight, 1);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(root);
  resize();
  const loop = () => {
    controls.update();
    renderer.render(scene, camera);
    animationFrame = requestAnimationFrame(loop);
  };
  loop();
}

async function boot() {
  try {
    initialiseRenderer();
    buildToolbar();
    const metaResponse = await fetch("scene_meta.json", { cache: "no-store" });
    if (!metaResponse.ok) throw new Error(`scene metadata request failed (${metaResponse.status})`);
    const meta = await metaResponse.json();
    if (meta.calibration?.state !== "relative_only" || meta.display_units !== "relative units") {
      throw new Error("3D method viewer requires the relative-only scene contract");
    }
    manifestUrl = new URL(meta.assets?.method_comparison?.path || "methods/method_comparison.json", window.location.href);
    const manifestResponse = await fetch(manifestUrl, { cache: "no-store" });
    if (!manifestResponse.ok) throw new Error(`method manifest request failed (${manifestResponse.status})`);
    manifest = await manifestResponse.json();
    if (!manifest.geometry_layers?.layers || !manifest.geometry_layers?.coordinate_frames) {
      throw new Error("method manifest lacks validated geometry_layers");
    }
    Object.values(manifest.geometry_layers.layers).forEach(validateRelativeLayer);
    const initialView = window.FPVViewer?.getViewState?.();
    pointSize = Number(initialView?.pointSize) || pointSize;
    globalPointBudget = Number(initialView?.pointBudget) || globalPointBudget;
    if (Array.isArray(initialView?.visibleLayers)) visibleLayerNames = new Set(initialView.visibleLayers);
    updateFrameButtonAvailability();
    updateOmegaDisplayTransform();
    await selectFrameGroup("omega_shared", "vggt_omega");
    const initial = layerState("vggt_omega");
    if (initial.status !== "ready") throw new Error(initial.error || "Omega geometry did not become ready");
    root.hidden = false;
    document.body.classList.add("webgl-ready");
    fitBounds("Orbit");
  } catch (error) {
    console.warn("WebGL method viewer retained the Canvas fallback", error);
    markFallback(error?.message || String(error));
  }
}

document.addEventListener("fpv-active-sample", (event) => {
  activeSample = Number(event.detail?.sampleIndex) || 0;
  updateActiveCameraProxy();
});
document.addEventListener("fpv-view-changed", (event) => {
  const alignmentChanged = updateOmegaDisplayTransform();
  if (alignmentChanged && activeFrameGroup === "omega_shared") fitBounds("Orbit");
  pointSize = Number(event.detail?.pointSize) || pointSize;
  const requestedBudget = Number(event.detail?.pointBudget);
  if (Number.isFinite(requestedBudget) && requestedBudget > 0) globalPointBudget = requestedBudget;
  const visible = event.detail?.visibleLayers;
  if (Array.isArray(visible)) visibleLayerNames = new Set(visible);
  layerStates.forEach((state, id) => {
    if (!state.object) return;
    state.object.material.size = pointSize;
    state.object.visible = visibleLayerNames.has("points");
    const layer = manifest?.geometry_layers?.layers?.[id];
    if (layer && globalPointBudget) {
      state.budget = Math.min(layer.point_count, globalPointBudget);
      state.object.geometry.setDrawRange(0, state.budget);
    }
  });
  renderGroups.forEach((group) => {
    group.children.forEach((child) => {
      if (child.userData.pathLayer) child.visible = visibleLayerNames.has(child.userData.pathLayer);
    });
  });
  renderLayerTray();
});
document.addEventListener("fpv-reference-visibility", (event) => {
  if (typeof event.detail?.grid === "boolean") referenceVisibility.grid = event.detail.grid;
  if (typeof event.detail?.cameras === "boolean") referenceVisibility.cameras = event.detail.cameras;
  if (referenceGrid) referenceGrid.visible = referenceVisibility.grid && canShowGroundGrid();
  renderGroups.forEach((group) => {
    group.traverse((object) => { if (object.userData.referenceCameras) object.visible = referenceVisibility.cameras; });
    const proxy = group.getObjectByName("active camera proxy");
    if (proxy) proxy.visible = referenceVisibility.cameras;
  });
});
document.addEventListener("fpv-select-geometry-layer", (event) => {
  const id = event.detail?.layerId;
  const layer = manifest?.geometry_layers?.layers?.[id];
  if (layer) selectFrameGroup(layer.coordinate_frame, id);
});

window.FPVViewer3D = {
  EXCLUSIVE_FRAME_GROUPS,
  getActiveFrameGroup: () => activeFrameGroup,
  getLayerState: (id) => ({ ...layerState(id), object: undefined }),
  projectDisplayPoint,
  projectScenePoint,
  selectFrameGroup,
  openLayer: (id) => {
    const layer = manifest?.geometry_layers?.layers?.[id];
    return layer ? selectFrameGroup(layer.coordinate_frame, id) : Promise.resolve();
  },
};

boot();

window.addEventListener("beforeunload", () => cancelAnimationFrame(animationFrame));
