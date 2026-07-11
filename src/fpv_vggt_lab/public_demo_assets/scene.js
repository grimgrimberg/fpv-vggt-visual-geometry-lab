"use strict";

const state = {
  data: null,
  sceneJsonUrl: null,
  pointCloud: null,
  pointCloudRenderer: null,
  visibleLayers: new Set(["points", "raw", "rts"]),
  activeSample: 0,
  activeMethod: null,
  yaw: -0.72,
  pitch: 0.42,
  zoom: 1,
  dragging: false,
  lastPointer: null,
  playing: false,
  lastAdvance: 0,
  reducedMotion: matchMedia("(prefers-reduced-motion: reduce)").matches,
};

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

function projectPoint(point, width, height) {
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const x = point[0], y = point[1], z = point[2];
  const rx = cy * x + sy * z;
  const rz = -sy * x + cy * z;
  const ry = cp * y - sp * rz;
  const depth = sp * y + cp * rz;
  const perspective = 1 / Math.max(1.55 + depth * 0.18, 0.62);
  const scale = Math.min(width, height) * 0.58 * state.zoom * perspective;
  return [width * 0.5 + rx * scale, height * 0.51 - ry * scale, depth];
}

function quaternionRotate(quaternion, vector) {
  const [w, x, y, z] = quaternion;
  const [vx, vy, vz] = vector;
  const uv = [y * vz - z * vy, z * vx - x * vz, x * vy - y * vx];
  const uuv = [y * uv[2] - z * uv[1], z * uv[0] - x * uv[2], x * uv[1] - y * uv[0]];
  return [
    vx + 2 * (w * uv[0] + uuv[0]),
    vy + 2 * (w * uv[1] + uuv[1]),
    vz + 2 * (w * uv[2] + uuv[2]),
  ];
}

function addVector(left, right, scale = 1) {
  return [left[0] + right[0] * scale, left[1] + right[1] * scale, left[2] + right[2] * scale];
}

function drawSceneGrid(context, width, height) {
  context.save();
  context.strokeStyle = "rgba(120,144,153,.08)";
  context.lineWidth = 1;
  for (let value = -1; value <= 1.001; value += 0.25) {
    const a = projectPoint([-1, -1, value], width, height);
    const b = projectPoint([1, -1, value], width, height);
    const c = projectPoint([value, -1, -1], width, height);
    const d = projectPoint([value, -1, 1], width, height);
    context.beginPath(); context.moveTo(a[0], a[1]); context.lineTo(b[0], b[1]); context.stroke();
    context.beginPath(); context.moveTo(c[0], c[1]); context.lineTo(d[0], d[1]); context.stroke();
  }
  context.restore();
}

function drawGeometryDensity(context, sceneData, viewState) {
  if (!viewState.visibleLayers.has("density")) return;
  const { width, height } = canvasSize(context.canvas);
  const cells = sceneData.geometry_density.cells
    .map((cell) => ({ cell, projected: projectPoint(cell.position, width, height) }))
    .sort((left, right) => left.projected[2] - right.projected[2]);
  context.save();
  context.globalCompositeOperation = "lighter";
  for (const item of cells) {
    const density = item.cell.density;
    const size = 0.75 + density * 2.9;
    context.fillStyle = `rgba(84,229,194,${0.07 + density * 0.5})`;
    context.fillRect(item.projected[0] - size / 2, item.projected[1] - size / 2, size, size);
  }
  context.restore();
}

function drawCameraPath(context, layer, viewState) {
  if (!viewState.visibleLayers.has(layer)) return;
  const points = viewState.data.paths[layer];
  if (!points || points.length < 2) return;
  const { width, height } = canvasSize(context.canvas);
  const palette = {
    raw: "rgba(243,183,95,.68)",
    bspline: "rgba(150,168,255,.82)",
    kalman: "rgba(193,162,255,.74)",
    rts: "#e8f2f2",
  };
  context.save();
  context.strokeStyle = palette[layer] || "#e8f2f2";
  context.lineWidth = layer === "raw" ? 1.1 : 2;
  context.beginPath();
  points.forEach((point, index) => {
    const projected = projectPoint(point, width, height);
    if (index === 0) context.moveTo(projected[0], projected[1]);
    else context.lineTo(projected[0], projected[1]);
  });
  context.stroke();
  context.restore();
}

function drawCameraProxy(context, position, orientation, viewState) {
  const { width, height } = canvasSize(context.canvas);
  const center = projectPoint(position, width, height);
  const forward = quaternionRotate(orientation.quaternion_wxyz, [0, 0, -1]);
  const up = quaternionRotate(orientation.quaternion_wxyz, [0, 1, 0]);
  const tip = projectPoint(addVector(position, forward, 0.16), width, height);
  const wing = projectPoint(addVector(position, up, 0.07), width, height);
  const heading = Math.atan2(tip[1] - center[1], tip[0] - center[0]);
  const wingLength = Math.max(6, Math.hypot(wing[0] - center[0], wing[1] - center[1]));

  context.save();
  context.translate(center[0], center[1]);
  context.rotate(heading);
  context.strokeStyle = "#ffffff";
  context.fillStyle = "rgba(84,229,194,.16)";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(14, 0);
  context.lineTo(-7, -wingLength);
  context.lineTo(-3, 0);
  context.lineTo(-7, wingLength);
  context.closePath();
  context.fill();
  context.stroke();
  context.restore();
  context.fillStyle = "rgba(232,242,242,.74)";
  context.font = `${Math.max(9, width / 150)}px ui-monospace, monospace`;
  context.fillText("camera pose proxy", center[0] + 16, center[1] - 9);
}

function drawMatchOverlay(context, viewState) {
  if (!viewState.visibleLayers.has("matches")) return;
  const matrix = viewState.data.match_connectivity.matrix;
  const size = Math.min(context.canvas.width, context.canvas.height) * 0.23;
  const left = context.canvas.width - size - 24;
  const top = 42;
  const cellSize = size / matrix.length;
  context.save();
  context.fillStyle = "rgba(5,9,12,.82)";
  context.fillRect(left - 8, top - 20, size + 16, size + 28);
  context.fillStyle = "rgba(232,242,242,.64)";
  context.font = "9px ui-monospace, monospace";
  context.fillText("aggregate match connectivity", left, top - 7);
  matrix.forEach((row, rowIndex) => row.forEach((value, columnIndex) => {
    context.fillStyle = `rgba(84,229,194,${0.04 + value * 0.82})`;
    context.fillRect(left + columnIndex * cellSize, top + rowIndex * cellSize, Math.max(cellSize - 0.5, 0.5), Math.max(cellSize - 0.5, 0.5));
  }));
  context.restore();
}

function activateDensityFallback(error = null) {
  if (state.pointCloudRenderer) state.pointCloudRenderer.dispose();
  state.pointCloud = null;
  state.pointCloudRenderer = null;
  state.visibleLayers.delete("points");
  state.visibleLayers.add("density");
  document.body.classList.remove("point-cloud-loading", "point-cloud-ready");
  document.body.classList.add("point-cloud-fallback");
  const pointsButton = document.querySelector('[data-layer="points"]');
  const densityButton = document.querySelector('[data-layer="density"]');
  pointsButton.hidden = true;
  pointsButton.disabled = true;
  pointsButton.classList.remove("active");
  pointsButton.setAttribute("aria-pressed", "false");
  densityButton.classList.add("active");
  densityButton.setAttribute("aria-pressed", "true");
  const densityCount = state.data?.geometry_density?.cells?.length || 0;
  document.getElementById("point-cloud-readout").textContent =
    `${densityCount.toLocaleString()} density voxels · relative_only fallback`;
  document.getElementById("scene-status").textContent = error
    ? "point sample unavailable · density fallback"
    : "density fallback";
  refreshActiveLayerLabel();
  if (error) console.warn("Point-cloud rendering fell back to density", error);
  requestAnimationFrame(renderScene);
}

function activatePointCloud(cloud, renderer) {
  state.pointCloud = cloud;
  state.pointCloudRenderer = renderer;
  state.visibleLayers.add("points");
  state.visibleLayers.delete("density");
  document.body.classList.remove("point-cloud-loading", "point-cloud-fallback");
  document.body.classList.add("point-cloud-ready");
  const pointsButton = document.querySelector('[data-layer="points"]');
  const densityButton = document.querySelector('[data-layer="density"]');
  pointsButton.hidden = false;
  pointsButton.disabled = false;
  pointsButton.classList.add("active");
  pointsButton.setAttribute("aria-pressed", "true");
  densityButton.classList.remove("active");
  densityButton.setAttribute("aria-pressed", "false");
  document.getElementById("point-cloud-readout").textContent =
    `${cloud.pointCount.toLocaleString()} displayed / `
    + `${cloud.sourcePointCount.toLocaleString()} source points · relative_only`;
  document.getElementById("scene-status").textContent = "authorized point sample ready";
  refreshActiveLayerLabel();
}

async function initializePointCloud() {
  const declared = state.data.point_cloud
    && state.data.publication_boundary?.real_point_sample_published === true;
  if (!declared) {
    activateDensityFallback();
    return;
  }
  document.body.classList.add("point-cloud-loading");
  document.getElementById("scene-status").textContent = "authorized point sample loading";
  try {
    const api = window.FpvPointCloudWebGL;
    if (!api) throw new Error("Point-cloud WebGL runtime is unavailable");
    const cloud = await api.loadPointCloud(state.sceneJsonUrl, state.data.point_cloud);
    const renderer = api.createPointCloudRenderer(
      document.getElementById("point-cloud-canvas"),
      cloud,
      { onContextLost: (error) => activateDensityFallback(error) },
    );
    activatePointCloud(cloud, renderer);
  } catch (error) {
    activateDensityFallback(error);
  }
}

function renderPointCloud() {
  if (!state.pointCloudRenderer) return;
  try {
    if (!state.visibleLayers.has("points")) {
      state.pointCloudRenderer.clear();
      return;
    }
    state.pointCloudRenderer.drawPointCloud({
      yaw: state.yaw,
      pitch: state.pitch,
      anchor: [0.5, 0.51],
      base: 1.55,
      depth: 0.18,
      minimumPerspective: 0.62,
      scaleFactor: 0.58,
      zoom: state.zoom,
      pointSize: 1.45,
      alpha: 0.54,
    });
  } catch (error) {
    activateDensityFallback(error);
  }
}

function renderScene() {
  if (!state.data) return;
  renderPointCloud();
  const canvas = document.getElementById("scene-canvas");
  const context = canvas.getContext("2d");
  const { width, height } = canvasSize(canvas);
  context.clearRect(0, 0, width, height);
  drawSceneGrid(context, width, height);
  drawGeometryDensity(context, state.data, state);
  ["raw", "bspline", "kalman", "rts"].forEach((layer) => drawCameraPath(context, layer, state));
  const path = state.data.paths.rts || state.data.paths.raw;
  const position = path[state.activeSample];
  const orientation = state.data.orientations[state.activeSample];
  drawCameraProxy(context, position, orientation, state);
  drawMatchOverlay(context, state);
}

function drawTrajectoryProfiles() {
  if (!state.data) return;
  const canvas = document.getElementById("trajectory-canvas");
  const context = canvas.getContext("2d");
  const { width, height } = canvasSize(canvas);
  const pad = { left: 24, right: 10, top: 10, bottom: 15 };
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(120,144,153,.16)";
  context.beginPath();
  context.moveTo(pad.left, height - pad.bottom);
  context.lineTo(width - pad.right, height - pad.bottom);
  context.stroke();
  const series = [
    ["speed_relative", "#54e5c2"],
    ["acceleration_relative", "#f3b75f"],
    ["rts_residual", "#96a8ff"],
  ];
  for (const [key, color] of series) {
    const values = state.data.trajectory_profiles[key]?.values;
    if (!values) continue;
    context.strokeStyle = color;
    context.lineWidth = 1.35;
    context.beginPath();
    values.forEach((value, index) => {
      const x = pad.left + (width - pad.left - pad.right) * index / Math.max(values.length - 1, 1);
      const y = height - pad.bottom - (height - pad.top - pad.bottom) * value;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  }
  const cursor = pad.left + (width - pad.left - pad.right) * state.activeSample / Math.max(state.data.sample_count - 1, 1);
  context.strokeStyle = "rgba(255,255,255,.72)";
  context.beginPath(); context.moveTo(cursor, pad.top); context.lineTo(cursor, height - pad.bottom); context.stroke();
}

function drawConnectivityMatrix() {
  if (!state.data) return;
  const canvas = document.getElementById("connectivity-canvas");
  const context = canvas.getContext("2d");
  const { width, height } = canvasSize(canvas);
  context.clearRect(0, 0, width, height);
  const matrix = state.data.match_connectivity.matrix;
  const size = Math.min(width - 8, height - 8);
  const left = (width - size) / 2;
  const top = (height - size) / 2;
  const cell = size / matrix.length;
  matrix.forEach((row, rowIndex) => row.forEach((value, columnIndex) => {
    context.fillStyle = `rgba(84,229,194,${0.04 + value * 0.9})`;
    context.fillRect(left + columnIndex * cell, top + rowIndex * cell, Math.max(cell - 0.4, 0.5), Math.max(cell - 0.4, 0.5));
  }));
}

function renderMethodTabs(methods) {
  const tabs = document.getElementById("method-tabs");
  const buttons = methods.map((method, index) => {
    const button = document.createElement("button");
    button.className = `method-tab${index === 0 ? " active" : ""}`;
    button.type = "button";
    button.role = "tab";
    button.dataset.method = method.id;
    button.textContent = method.label.replace(" + ", "+");
    button.setAttribute("aria-selected", String(index === 0));
    button.addEventListener("click", () => selectMethod(method.id));
    return button;
  });
  tabs.replaceChildren(...buttons);
  state.activeMethod = methods[0]?.id || null;
  if (state.activeMethod) renderMethodPanel(state.activeMethod);
}

function selectMethod(methodId) {
  state.activeMethod = methodId;
  document.querySelectorAll(".method-tab").forEach((button) => {
    const active = button.dataset.method === methodId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  renderMethodPanel(methodId);
}

function renderMethodPanel(methodId) {
  const method = state.data.methods.find((row) => row.id === methodId);
  if (!method) return;
  const panel = document.getElementById("method-panel");
  panel.replaceChildren();
  const title = document.createElement("h3");
  title.textContent = `${method.label} · ${method.status.replaceAll("_", " ")}`;
  const role = document.createElement("p");
  role.textContent = `${method.role}. ${method.scale_status.replaceAll("_", " ")}.`;
  const columns = document.createElement("div");
  columns.className = "io-columns";
  for (const [label, rows] of [["Input", method.inputs], ["Published output", method.outputs]]) {
    const column = document.createElement("div");
    const heading = document.createElement("b");
    heading.textContent = label;
    const list = document.createElement("ul");
    list.replaceChildren(...rows.map((row) => {
      const item = document.createElement("li");
      item.textContent = row;
      return item;
    }));
    column.append(heading, list);
    columns.appendChild(column);
  }
  const metrics = document.createElement("div");
  metrics.className = "metric-line";
  metrics.replaceChildren(...method.metrics.map((metric) => {
    const item = document.createElement("span");
    item.textContent = `${metric.label} ${Number(metric.value).toLocaleString()}`;
    return item;
  }));
  const depth = state.data.depth_summaries[method.id];
  if (depth?.status === "summarized") {
    const item = document.createElement("span");
    item.textContent = `depth p50 ${depth.depth_normalized_to_scene_median.p50.toFixed(2)} relative`;
    metrics.appendChild(item);
  }
  panel.append(title, role, columns, metrics);
}

function renderProvenance() {
  const rows = [
    ["source", state.data.provenance.source_type],
    ["interval", state.data.provenance.interval_policy],
    ["derivation", state.data.provenance.public_derivation],
    ["coordinates", state.data.provenance.coordinate_frame],
    ["scale", state.data.scale_status.replaceAll("_", " ")],
    ["body attitude", state.data.body_attitude],
  ];
  if (state.data.point_cloud) {
    rows.splice(
      4,
      0,
      ["point sample", `${state.data.point_cloud.point_count.toLocaleString()} colored points`],
    );
  }
  const list = document.getElementById("provenance");
  const nodes = [];
  for (const [term, value] of rows) {
    const row = document.createElement("div");
    const dt = document.createElement("dt"); dt.textContent = term;
    const dd = document.createElement("dd"); dd.textContent = value;
    row.append(dt, dd); nodes.push(row);
  }
  list.replaceChildren(...nodes);
}

function renderMediaNotice() {
  const declared = state.data.point_cloud
    && state.data.publication_boundary?.real_point_sample_published === true;
  document.getElementById("media-notice-copy").textContent = declared
    ? (
      "original media is not redistributed. An authorized colored VGGT Omega point sample is "
      + "published because the scene contract records publication approval and attribution. "
      + "Video, source frames, recognizable reprojections, raw NPZ/PLY, full arrays, and machine "
      + "paths remain withheld."
    )
    : (
      "original media is not redistributed. No colored point sample is declared by this scene "
      + "contract, so the viewer uses the coarse density fallback. Video, source frames, "
      + "recognizable reprojections, raw NPZ/PLY, full arrays, and machine paths remain withheld."
    );
}

function renderFailureCase() {
  const failure = state.data.failure_case;
  document.getElementById("failure-status").textContent = failure.status.replaceAll("_", " ");
  document.getElementById("failure-copy").textContent = failure.interpretation;
  const facts = [
    ["category", failure.categories.join(", ") || "unclassified"],
    ["reconstruction retained", failure.successful_reconstruction_retained ? "yes" : "no"],
    ["registered summary", failure.declared_registered_images ?? "not reported"],
    ["max observed frame id", failure.registered_frame_max_observed ?? "not reported"],
    ["BA log events", failure.ba_log_events ?? "not reported"],
    ["fallback eligible", failure.fallback_eligible ? "yes" : "no"],
    ["secondary case", `${failure.secondary_method_failure.method}: ${failure.secondary_method_failure.status}`],
  ];
  const nodes = facts.map(([term, value]) => {
    const row = document.createElement("div");
    const dt = document.createElement("dt"); dt.textContent = term;
    const dd = document.createElement("dd"); dd.textContent = String(value);
    row.append(dt, dd); return row;
  });
  document.getElementById("failure-facts").replaceChildren(...nodes);
}

function setActiveSample(index) {
  if (!state.data) return;
  state.activeSample = Math.max(0, Math.min(state.data.sample_count - 1, Number(index) || 0));
  document.getElementById("pose-slider").value = String(state.activeSample);
  document.getElementById("pose-readout").textContent = `${state.activeSample + 1} / ${state.data.sample_count}`;
  const orientation = state.data.orientations[state.activeSample];
  document.getElementById("trajectory-readout").textContent =
    `t ${orientation.time_normalized.toFixed(3)} · confidence ${orientation.confidence.toFixed(2)}`;
  requestAnimationFrame(renderScene);
  requestAnimationFrame(drawTrajectoryProfiles);
}

function refreshActiveLayerLabel() {
  const labels = {
    points: "Points",
    density: "Density fallback",
    raw: "Raw",
    bspline: "B-spline",
    kalman: "Kalman",
    rts: "RTS",
    matches: "Matches",
  };
  document.getElementById("active-layer-label").textContent =
    [...state.visibleLayers].map((value) => labels[value] || value).join(" · ");
}

function wireControls() {
  document.querySelectorAll("[data-layer]").forEach((button) => {
    button.addEventListener("click", () => {
      const layer = button.dataset.layer;
      if (state.visibleLayers.has(layer)) state.visibleLayers.delete(layer);
      else state.visibleLayers.add(layer);
      const active = state.visibleLayers.has(layer);
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      refreshActiveLayerLabel();
      requestAnimationFrame(renderScene);
    });
  });
  document.getElementById("pose-slider").addEventListener("input", (event) => {
    state.playing = false;
    document.getElementById("play-toggle").textContent = "Play";
    setActiveSample(event.target.value);
  });
  document.getElementById("step-back").addEventListener("click", () => setActiveSample(state.activeSample - 1));
  document.getElementById("step-forward").addEventListener("click", () => setActiveSample(state.activeSample + 1));
  document.getElementById("play-toggle").addEventListener("click", (event) => {
    if (state.reducedMotion) return setActiveSample(state.activeSample + 1);
    state.playing = !state.playing;
    event.currentTarget.textContent = state.playing ? "Pause" : "Play";
  });

  const canvas = document.getElementById("scene-canvas");
  canvas.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    state.lastPointer = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.dragging) return;
    state.yaw += (event.clientX - state.lastPointer[0]) * 0.007;
    state.pitch = Math.max(-1.35, Math.min(1.35, state.pitch + (event.clientY - state.lastPointer[1]) * 0.007));
    state.lastPointer = [event.clientX, event.clientY];
    requestAnimationFrame(renderScene);
  });
  canvas.addEventListener("pointerup", () => { state.dragging = false; });
  canvas.addEventListener("pointercancel", () => { state.dragging = false; });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.zoom = Math.max(0.35, Math.min(4.5, state.zoom * Math.exp(-event.deltaY * 0.001)));
    requestAnimationFrame(renderScene);
  }, { passive: false });
  canvas.addEventListener("dblclick", () => {
    state.yaw = -0.72; state.pitch = 0.42; state.zoom = 1;
    requestAnimationFrame(renderScene);
  });
}

function playbackLoop(timestamp) {
  if (state.playing && state.data && timestamp - state.lastAdvance > 90) {
    state.lastAdvance = timestamp;
    setActiveSample((state.activeSample + 1) % state.data.sample_count);
  }
  requestAnimationFrame(playbackLoop);
}

async function initializeScene() {
  state.sceneJsonUrl = document.body.dataset.sceneJson;
  const response = await fetch(state.sceneJsonUrl, { cache: "no-store" });
  if (!response.ok) throw new Error(`Scene contract request failed (${response.status})`);
  state.data = await response.json();
  if (state.data.scale_status !== "relative_only" || state.data.pose_semantics !== "camera_pose_proxy") {
    throw new Error("Public scene contract must remain relative_only with camera_pose_proxy semantics");
  }
  document.getElementById("scene-title").textContent = state.data.title;
  document.getElementById("hero-score").textContent = state.data.quality.hero_score.toFixed(3);
  document.getElementById("pose-slider").max = String(state.data.sample_count - 1);
  document.getElementById("connectivity-readout").textContent =
    `${state.data.match_connectivity.node_count} nodes · ${state.data.match_connectivity.edge_count.toLocaleString()} edges`;
  renderMethodTabs(state.data.methods);
  renderProvenance();
  renderMediaNotice();
  renderFailureCase();
  wireControls();
  await initializePointCloud();
  setActiveSample(0);
  drawConnectivityMatrix();
  document.body.classList.add("ready");
  requestAnimationFrame(playbackLoop);
}

addEventListener("resize", () => {
  requestAnimationFrame(renderScene);
  requestAnimationFrame(drawTrajectoryProfiles);
  requestAnimationFrame(drawConnectivityMatrix);
});

initializeScene().catch((error) => {
  document.getElementById("scene-status").textContent = "bundle error";
  const message = document.createElement("div");
  message.className = "error-banner";
  message.textContent = `Public scene unavailable: ${error.message}`;
  document.body.appendChild(message);
  console.error(error);
});
