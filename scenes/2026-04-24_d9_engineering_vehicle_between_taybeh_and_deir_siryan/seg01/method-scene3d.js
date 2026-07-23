import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const EXCLUSIVE_FRAME_GROUPS = Object.freeze([
  "omega_shared",
  "r3_native",
  "lingbot_native",
  "hloc_native",
]);

const FRAME_LABELS = {
  omega_shared: "Omega shared",
  r3_native: "R3 native",
  lingbot_native: "LingBot native",
  hloc_native: "HLoc native",
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

function geometryFromBuffers(layer, positions, colors) {
  const total = layer.point_count;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
  const material = new THREE.PointsMaterial({
    size: pointSize,
    sizeAttenuation: false,
    vertexColors: true,
    transparent: true,
    opacity: 0.94,
  });
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

function lineFromPoints(points, color) {
  if (!Array.isArray(points) || points.length < 2) return null;
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map((point) => new THREE.Vector3(...point)));
  return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }));
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
    const line = lineFromPoints(paths.layers[name], color);
    if (!line) return;
    line.name = `omega-${name}-path`;
    line.userData.pathLayer = name;
    line.visible = window.FPVViewer?.getViewState?.().visibleLayers?.includes(name) ?? ["raw", "rts"].includes(name);
    group.add(line);
  });
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
  const next = centers[Math.min(index + 1, centers.length - 1)];
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
      : await Promise.all([
          fetchBuffer(layer.positions, `${id} positions`),
          fetchBuffer(layer.colors, `${id} colors`),
        ]).then(([positions, colors]) => ({
          positions: new Float32Array(positions),
          colors: new Uint8Array(colors),
        }));
    const count = validateBuffer(buffers.positions.buffer, layer.positions, `${id} positions`);
    validateBuffer(buffers.colors.buffer, layer.colors, `${id} colors`);
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
    if (activeFrameGroup === "omega_shared" && id === "vggt_omega") addOmegaPaths(group);
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
  const supported = manifest.geometry_layers.coordinate_frames?.[activeFrameGroup]?.ground_display_supported === true;
  groundButton.disabled = !supported;
  groundButton.title = supported ? "Synthetic scale-free scene-bounds viewpoint" : "Ground-relative view is disabled in method-native frames";
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
  if (referenceGrid) referenceGrid.visible = frame === "omega_shared" && referenceVisibility.grid;
  toolbar.querySelectorAll("[data-frame]").forEach((button) => {
    const active = button.dataset.frame === frame;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  setGroundAvailability();
  viewNote.textContent = `${FRAME_LABELS[frame]} · relative only · one coordinate frame visible`;
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

function currentBounds() {
  const group = renderGroups.get(activeFrameGroup);
  const box = new THREE.Box3().setFromObject(group);
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
  return true;
}

function fitBounds(preset) {
  if (!camera || !controls) return;
  const box = currentBounds();
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 1e-4);
  if (referenceGrid && activeFrameGroup === "omega_shared") {
    const groundY = window.FPVViewer?.getAlignmentMode?.() === "estimated_ground" ? 0 : box.min.y;
    referenceGrid.position.set(center.x, groundY, center.z);
    referenceGrid.scale.setScalar(span / 2);
  }
  controls.target.copy(center);
  if (preset === "Ground-relative") {
    camera.position.set(center.x + span * 0.8, box.min.y + span * 0.08, center.z + span * 1.1);
    viewNote.textContent = "Ground-relative · synthetic scene-bounds viewpoint · relative only";
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
    viewNote.textContent = `${FRAME_LABELS[activeFrameGroup]} · Orbit · relative only`;
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
  layerTray = document.createElement("div");
  layerTray.className = "method-scene3d-layers";
  viewNote = document.createElement("div");
  viewNote.className = "method-scene3d-note";
  viewNote.setAttribute("role", "status");
  toolbar.append(frames, presets, layerTray, viewNote);
  root.appendChild(toolbar);
}

function initialiseRenderer() {
  if (!root || !window.WebGLRenderingContext) throw new Error("WebGL is not supported");
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer.setClearColor(0x02090d, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.domElement.setAttribute("aria-label", "WebGL relative reconstruction method viewer");
  root.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(48, 1, 0.001, 10000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.addEventListener("change", () => {
    document.dispatchEvent(new CustomEvent("fpv-webgl-camera-changed"));
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
  referenceGrid.visible = referenceVisibility.grid;
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
  if (referenceGrid) referenceGrid.visible = activeFrameGroup === "omega_shared" && referenceVisibility.grid;
  renderGroups.forEach((group) => {
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
