"use strict";

const state = {
  scene: null,
  paths: null,
  cameras: null,
  profiles: null,
  points: null,
  colors: null,
  activeSample: 0,
  visibleLayers: new Set(["points", "raw", "rts"]),
  yaw: -0.7,
  pitch: 0.48,
  zoom: 1,
  dragging: false,
  lastPointer: null,
  playing: false,
  reducedMotion: matchMedia("(prefers-reduced-motion: reduce)").matches,
};

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path} request failed (${response.status})`);
  return response.json();
}

async function fetchTyped(path, Type) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path} request failed (${response.status})`);
  return new Type(await response.arrayBuffer());
}

async function loadScene(url) {
  state.scene = await fetchJson(url);
  if (state.scene.calibration.state !== "relative_only" || state.scene.display_units !== "relative units") {
    throw new Error("This cockpit requires an explicitly relative-only uncalibrated scene");
  }
  const assets = state.scene.assets;
  [state.paths, state.profiles, state.cameras, state.points, state.colors] = await Promise.all([
    fetchJson(assets.camera_path.path),
    fetchJson(assets.trajectory_profiles.path),
    fetchJson(assets.cameras.path),
    fetchTyped(assets.points_preview.path, Float32Array),
    fetchTyped(assets.points_preview_colors.path, Uint8Array),
  ]);
  document.getElementById("scene-title").textContent = state.scene.title;
  document.getElementById("point-count").textContent = Number(state.scene.reconstruction.point_count_viewer).toLocaleString();
  document.getElementById("frame-count").textContent = Number(state.scene.reconstruction.frame_count).toLocaleString();
  document.getElementById("hero-score").textContent = state.scene.quality.hero_score.toFixed(3);
  const slider = document.getElementById("pose-slider");
  slider.max = String(state.paths.timestamps_sec.length - 1);
  renderProvenance();
  renderMethods();
  renderRawInspector(state.scene);
  renderWarnings();
  renderEditTimeline(state.scene.edit_segments, state);
  setActiveSample(0);
  document.body.classList.add("ready");
}

function canvasSize(canvas) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, dpr };
}

function sceneTransform(point, width, height) {
  const bounds = state.scene.bounds;
  const center = bounds.min.map((value, index) => (value + bounds.max[index]) * 0.5);
  const span = Math.max(...bounds.max.map((value, index) => value - bounds.min[index]), 1e-6);
  let x = point[0] - center[0], y = point[1] - center[1], z = point[2] - center[2];
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const rx = cy * x + sy * z;
  const rz = -sy * x + cy * z;
  const ry = cp * y - sp * rz;
  const depth = sp * y + cp * rz;
  const scale = Math.min(width, height) * 0.72 * state.zoom / span;
  return [width * 0.5 + rx * scale, height * 0.53 - ry * scale, depth / span];
}

function drawPointCloud(context, scene, viewState) {
  if (!viewState.visibleLayers.has("points")) return;
  const { width, height } = canvasSize(context.canvas);
  const pointCount = viewState.points.length / 3;
  const stride = Math.max(1, Math.ceil(pointCount / 52000));
  context.save();
  context.globalCompositeOperation = "lighter";
  for (let index = 0; index < pointCount; index += stride) {
    const offset = index * 3;
    const projected = sceneTransform(
      [viewState.points[offset], viewState.points[offset + 1], viewState.points[offset + 2]],
      width,
      height,
    );
    if (projected[0] < 0 || projected[0] >= width || projected[1] < 0 || projected[1] >= height) continue;
    const alpha = Math.max(0.08, Math.min(0.38, 0.22 - projected[2] * 0.08));
    context.fillStyle = `rgba(${viewState.colors[offset]},${viewState.colors[offset + 1]},${viewState.colors[offset + 2]},${alpha})`;
    context.fillRect(projected[0], projected[1], 1.4, 1.4);
  }
  context.restore();
}

function drawCameraPath(context, layer, viewState) {
  if (!viewState.visibleLayers.has(layer)) return;
  const points = viewState.paths.layers[layer];
  if (!points || points.length < 2) return;
  const palette = { raw: "rgba(243,183,95,.65)", bspline: "#7f9cff", kalman: "#b89bff", rts: "#54e5c2" };
  const { width, height } = canvasSize(context.canvas);
  context.save();
  context.strokeStyle = palette[layer] || "#dcecef";
  context.lineWidth = layer === "raw" ? 1.1 : 2.3;
  context.beginPath();
  points.forEach((point, index) => {
    const projected = sceneTransform(point, width, height);
    if (index === 0) context.moveTo(projected[0], projected[1]); else context.lineTo(projected[0], projected[1]);
  });
  context.stroke();
  context.restore();
}

function rotateByQuaternion(quaternion, vector) {
  const [qw, qx, qy, qz] = quaternion;
  const uv = [qy * vector[2] - qz * vector[1], qz * vector[0] - qx * vector[2], qx * vector[1] - qy * vector[0]];
  const uuv = [qy * uv[2] - qz * uv[1], qz * uv[0] - qx * uv[2], qx * uv[1] - qy * uv[0]];
  return vector.map((value, axis) => value + 2 * (qw * uv[axis] + uuv[axis]));
}

function drawCameraProxy(context, pose, viewState) {
  const { width, height } = canvasSize(context.canvas);
  const center = sceneTransform(pose.position, width, height);
  const span = Math.max(...viewState.scene.bounds.max.map((value, axis) => value - viewState.scene.bounds.min[axis]), 1e-6);
  const quaternion = pose.quaternion_wxyz || [1, 0, 0, 0];
  const forward = rotateByQuaternion(quaternion, [0, 0, 1]);
  const right = rotateByQuaternion(quaternion, [1, 0, 0]);
  const projectDirection = (direction, scale) => sceneTransform(pose.position.map((value, axis) => value + direction[axis] * span * scale), width, height);
  const forwardTip = projectDirection(forward, .055);
  const rightTip = projectDirection(right, .045);
  let forwardScreen = [forwardTip[0] - center[0], forwardTip[1] - center[1]];
  let rightScreen = [rightTip[0] - center[0], rightTip[1] - center[1]];
  const normalize = (vector, fallback) => {
    const length = Math.hypot(...vector);
    return length > 1e-6 ? vector.map(value => value / length) : fallback;
  };
  forwardScreen = normalize(forwardScreen, [1, 0]);
  rightScreen = normalize(rightScreen, [-forwardScreen[1], forwardScreen[0]]);
  const point = (forwardAmount, rightAmount) => [forwardScreen[0] * forwardAmount + rightScreen[0] * rightAmount, forwardScreen[1] * forwardAmount + rightScreen[1] * rightAmount];
  context.save();
  context.translate(center[0], center[1]);
  context.strokeStyle = "#f3b75f";
  context.fillStyle = "rgba(84,229,194,.18)";
  context.lineWidth = 1.6;
  const nose = point(14, 0), left = point(-5, -8), tail = point(-3, 0), rightWing = point(-5, 8);
  context.beginPath(); context.moveTo(...nose); context.lineTo(...left); context.lineTo(...tail); context.lineTo(...rightWing); context.closePath();
  context.fill(); context.stroke();
  const armA = point(7, 10), armB = point(-7, -10), armC = point(7, -10), armD = point(-7, 10);
  context.strokeStyle = "#dcecef"; context.beginPath(); context.moveTo(...armA); context.lineTo(...armB); context.moveTo(...armC); context.lineTo(...armD); context.stroke();
  [armA, armB, armC, armD].forEach(rotor => { context.beginPath(); context.arc(rotor[0], rotor[1], 2.7, 0, Math.PI * 2); context.stroke(); });
  context.restore();
  context.fillStyle = "rgba(220,236,239,.72)";
  context.font = "10px ui-monospace, monospace";
  context.fillText("camera pose proxy · recovered orientation", center[0] + 15, center[1] - 10);
}

function drawSixDofTraces(context, animation, viewState) {
  const width = context.canvas.width, height = context.canvas.height;
  const left = 38, right = width * 0.62, top = 24, bottom = height - 28;
  const series = [
    [animation.speed_relative, "#54e5c2", "speed"],
    [animation.acceleration_relative, "#f3b75f", "accel"],
    [animation.jerk_proxy, "#7f9cff", "jerk"],
  ];
  series.forEach(([values, color, label], seriesIndex) => {
    const finite = values.filter(Number.isFinite);
    const ceiling = Math.max(...finite, 1e-8);
    context.strokeStyle = color; context.lineWidth = 1.4; context.beginPath();
    values.forEach((value, index) => {
      const x = left + (right - left) * index / Math.max(values.length - 1, 1);
      const y = bottom - (bottom - top) * Math.min(value / ceiling, 1);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke(); context.fillStyle = color; context.font = "9px ui-monospace, monospace"; context.fillText(label, left + seriesIndex * 48, 12);
  });
  const cursorX = left + (right - left) * viewState.activeSample / Math.max(animation.speed_relative.length - 1, 1);
  context.strokeStyle = "rgba(255,255,255,.65)"; context.beginPath(); context.moveTo(cursorX, top); context.lineTo(cursorX, bottom); context.stroke();
}

function drawPhasePortrait(context, animation, viewState) {
  const width = context.canvas.width, height = context.canvas.height;
  const left = width * 0.69, right = width - 20, top = 24, bottom = height - 28;
  const positions = viewState.paths.layers.raw.map((point) => point[0]);
  const velocity = animation.speed_relative;
  const minX = Math.min(...positions), maxX = Math.max(...positions);
  const maxV = Math.max(...velocity, 1e-8);
  context.strokeStyle = "rgba(84,229,194,.7)"; context.lineWidth = 1.2; context.beginPath();
  positions.forEach((value, index) => {
    const x = left + (right - left) * (value - minX) / Math.max(maxX - minX, 1e-8);
    const y = bottom - (bottom - top) * velocity[index] / maxV;
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  const index = viewState.activeSample;
  const x = left + (right - left) * (positions[index] - minX) / Math.max(maxX - minX, 1e-8);
  const y = bottom - (bottom - top) * velocity[index] / maxV;
  context.fillStyle = "#fff"; context.beginPath(); context.arc(x, y, 3, 0, Math.PI * 2); context.fill();
  context.fillStyle = "#78909a"; context.font = "9px ui-monospace, monospace"; context.fillText("relative x / relative speed", left, 12);
}

function renderPov(frameRecord, overlays) {
  const image = document.getElementById("pov-frame");
  const empty = document.getElementById("pov-empty");
  if (!frameRecord) { image.style.display = "none"; empty.style.display = "grid"; return; }
  image.src = frameRecord.path;
  image.alt = `Recorded POV at ${frameRecord.timestamp_sec.toFixed(3)} source seconds`;
  image.dataset.overlayCount = String(Object.keys(overlays || {}).length);
  image.style.display = "block"; empty.style.display = "none";
  document.getElementById("pov-time").textContent = `${frameRecord.timestamp_sec.toFixed(3)} source s`;
}

function renderEditTimeline(segments, viewState) {
  const timeline = document.getElementById("edit-timeline");
  const time = viewState.paths ? viewState.paths.timestamps_sec[viewState.activeSample] : 0;
  const valid = (segments || []).filter((segment) => Number.isFinite(Number(segment.start_s)) && Number.isFinite(Number(segment.end_s)));
  const start = valid.length ? Math.min(...valid.map((segment) => Number(segment.start_s))) : 0;
  const end = valid.length ? Math.max(...valid.map((segment) => Number(segment.end_s))) : 1;
  timeline.replaceChildren();
  valid.forEach((segment) => {
    const element = document.createElement("span");
    element.className = `segment ${segment.start_type || segment.type || "other"}`;
    element.style.flexGrow = String(Math.max(Number(segment.end_s) - Number(segment.start_s), .001));
    if (Number(segment.start_s) <= time && time <= Number(segment.end_s)) element.classList.add("active");
    element.setAttribute("aria-label", `${segment.start_type || segment.type || "segment"}: ${segment.start_s} to ${segment.end_s} source seconds`);
    timeline.appendChild(element);
  });
  const active = valid.find((segment) => Number(segment.start_s) <= time && time <= Number(segment.end_s));
  const label = active ? `${active.start_type || active.type || "segment"} → ${active.end_type || "end"}` : "unclassified";
  document.getElementById("edit-readout").textContent = label;
  timeline.setAttribute("aria-label", `Edit state: ${label}`);
}

function renderRawInspector(scene) {
  const inspector = document.getElementById("raw-inspector");
  const items = Object.entries(scene.raw_assets || {}).map(([label, asset]) => {
    const row = document.createElement("div"); row.className = "raw-item";
    const name = document.createElement("b"); name.textContent = label;
    const path = document.createElement("span"); path.textContent = `${asset.bytes.toLocaleString()} bytes · sha256 ${asset.sha256.slice(0, 12)}…`;
    const source = document.createElement("span"); source.textContent = asset.path;
    row.append(name, path, source); return row;
  });
  inspector.replaceChildren(...items);
}

function renderProvenance() {
  const scene = state.scene;
  const rows = [
    ["annotation", scene.annotation.kind], ["source commit", scene.annotation.source_commit.slice(0, 12)],
    ["backend", scene.reconstruction.backend], ["backend commit", String(scene.reconstruction.backend_commit).slice(0, 12)],
    ["coordinate frame", scene.reconstruction.coordinate_frame], ["scale", scene.calibration.state],
  ];
  const list = document.getElementById("provenance");
  list.replaceChildren(...rows.flatMap(([term, value]) => {
    const dt = document.createElement("dt"); dt.textContent = term;
    const dd = document.createElement("dd"); dd.textContent = value;
    return [dt, dd];
  }));
}

function renderMethods() {
  const rows = Object.entries(state.scene.methods || {}).map(([name, detail]) => {
    const row = document.createElement("div"); row.className = "method-row";
    const label = document.createElement("span"); label.textContent = name.replaceAll("_", " ");
    const status = document.createElement("span"); status.className = `method-status ${detail.status}`; status.textContent = detail.status.replaceAll("_", " ");
    row.append(label, status); return row;
  });
  document.getElementById("method-list").replaceChildren(...rows);
}

function renderWarnings() {
  const rows = (state.scene.warnings || []).map((warning) => {
    const item = document.createElement("div"); item.textContent = `△ ${warning}`; return item;
  });
  document.getElementById("warning-list").replaceChildren(...rows);
}

function setActiveSample(index) {
  const count = state.paths.timestamps_sec.length;
  state.activeSample = Math.max(0, Math.min(count - 1, Number(index) || 0));
  document.getElementById("pose-slider").value = String(state.activeSample);
  const time = state.paths.timestamps_sec[state.activeSample];
  document.getElementById("pose-readout").textContent = `${Number(time).toFixed(3)} source s · relative only · camera pose proxy`;
  const pov = (state.scene.assets.pov.samples || []).reduce((best, item) => !best || Math.abs(item.sample_index - state.activeSample) < Math.abs(best.sample_index - state.activeSample) ? item : best, null);
  renderPov(pov, {});
  renderEditTimeline(state.scene.edit_segments, state);
  const p = state.profiles;
  document.getElementById("state-readout").textContent = `speed ${p.speed_relative[state.activeSample].toFixed(4)} · acceleration ${p.acceleration_relative[state.activeSample].toFixed(4)} · jerk proxy ${p.jerk_proxy[state.activeSample].toFixed(4)} · RTS residual ${p.rts_residual[state.activeSample].toFixed(5)}`;
  requestAnimationFrame(renderAll);
}

function renderAll() {
  if (!state.scene || !state.points) return;
  const sceneCanvas = document.getElementById("scene-canvas");
  const context = sceneCanvas.getContext("2d");
  const size = canvasSize(sceneCanvas);
  context.clearRect(0, 0, size.width, size.height);
  drawPointCloud(context, state.scene, state);
  ["raw", "bspline", "kalman", "rts"].forEach((layer) => drawCameraPath(context, layer, state));
  const raw = state.paths.layers.raw;
  const current = raw[state.activeSample];
  const next = raw[Math.min(state.activeSample + 1, raw.length - 1)] || current;
  const camera = state.cameras.samples[state.activeSample];
  drawCameraProxy(context, { position: current, next, quaternion_wxyz: camera?.quaternion_wxyz }, state);
  const stateCanvas = document.getElementById("state-canvas");
  const stateContext = stateCanvas.getContext("2d");
  const stateSize = canvasSize(stateCanvas);
  stateContext.clearRect(0, 0, stateSize.width, stateSize.height);
  drawSixDofTraces(stateContext, state.profiles, state);
  drawPhasePortrait(stateContext, state.profiles, state);
}

function wireControls() {
  document.querySelectorAll("[data-layer]").forEach((button) => button.addEventListener("click", () => {
    const layer = button.dataset.layer;
    if (["depth", "matches", "masks"].includes(layer)) {
      button.dataset.tooltip = "Evidence is shown in the method inspector; overlays remain separately gated.";
      return;
    }
    if (state.visibleLayers.has(layer)) state.visibleLayers.delete(layer); else state.visibleLayers.add(layer);
    const active = state.visibleLayers.has(layer); button.classList.toggle("active", active); button.setAttribute("aria-pressed", String(active));
    requestAnimationFrame(renderAll);
  }));
  document.getElementById("pose-slider").addEventListener("input", (event) => setActiveSample(event.target.value));
  document.getElementById("step-back").addEventListener("click", () => setActiveSample(state.activeSample - 1));
  document.getElementById("step-forward").addEventListener("click", () => setActiveSample(state.activeSample + 1));
  document.getElementById("play-toggle").addEventListener("click", (event) => {
    if (state.reducedMotion) return setActiveSample(state.activeSample + 1);
    state.playing = !state.playing; event.currentTarget.textContent = state.playing ? "Pause" : "Play";
  });
  const canvas = document.getElementById("scene-canvas");
  canvas.addEventListener("pointerdown", (event) => { state.dragging = true; state.lastPointer = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.dragging) return;
    state.yaw += (event.clientX - state.lastPointer[0]) * .007; state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch + (event.clientY - state.lastPointer[1]) * .007)); state.lastPointer = [event.clientX, event.clientY]; requestAnimationFrame(renderAll);
  });
  canvas.addEventListener("pointerup", () => { state.dragging = false; });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); state.zoom = Math.max(.25, Math.min(6, state.zoom * Math.exp(-event.deltaY * .001))); requestAnimationFrame(renderAll); }, { passive: false });
  canvas.addEventListener("dblclick", () => { state.yaw = -.7; state.pitch = .48; state.zoom = 1; requestAnimationFrame(renderAll); });
  addEventListener("resize", () => requestAnimationFrame(renderAll));
}

function playbackLoop() {
  if (state.playing && state.paths) {
    setActiveSample((state.activeSample + 1) % state.paths.timestamps_sec.length);
  }
  setTimeout(playbackLoop, 80);
}

wireControls();
loadScene("scene_meta.json").catch((error) => {
  document.getElementById("run-status").textContent = `bundle error: ${error.message}`;
  console.error(error);
});
playbackLoop();
