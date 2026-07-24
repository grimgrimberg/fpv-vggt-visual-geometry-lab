const $ = (id) => document.getElementById(id);
const MANIFEST_URL = "./manifest.json";
const LAYERS = ["raw", "bspline", "kalman", "rts"];
const PROJECTIONS = {
  xz: { axes: [0, 2], label: "X/Z coordinate projection · no ground plane inferred" },
  xy: { axes: [0, 1], label: "X/Y coordinate projection · no ground plane inferred" },
  yz: { axes: [1, 2], label: "Y/Z coordinate projection · no ground plane inferred" },
};
const PROFILE_SERIES = [
  ["speed_relative", "#43e0cd"],
  ["acceleration_relative", "#f4bf5f"],
  ["jerk_proxy", "#ff7474"],
  ["curvature", "#be96ff"],
];
const COHORT_COLORS = [
  "#43e0cd", "#be96ff", "#80adff", "#f4bf5f", "#ff7474", "#66d59a",
  "#dc8eff", "#70c7ff", "#f59e78", "#aacb6c", "#8ee0d7", "#e9a6c9",
];

const state = {
  manifest: null,
  records: [],
  cache: new Map(),
  activeIndex: 0,
  layer: "rts",
  projection: "xz",
  view: "trajectory",
  sampleIndex: 0,
  playing: false,
  animationFrame: 0,
  lastAdvance: 0,
};

function textNode(tag, value, className) {
  const node = document.createElement(tag);
  node.textContent = value;
  if (className) node.className = className;
  return node;
}

function safeToken(value) {
  return typeof value === "string" && /^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(value);
}

function safeJsonName(value) {
  return safeToken(value) && value.toLowerCase().endsWith(".json");
}

function sceneBase(record) {
  if (!safeToken(record.slug) || !safeToken(record.scene_id)) {
    throw new Error("Scene manifest contains an unsafe identifier");
  }
  const url = new URL(`../../scenes/${record.slug}/${record.scene_id}/`, window.location.href);
  if (url.origin !== window.location.origin) throw new Error("Scene asset escaped the public origin");
  return url;
}

async function fetchJson(url) {
  const resolved = new URL(url, window.location.href);
  if (resolved.origin !== window.location.origin) throw new Error("Cross-origin artifact rejected");
  const response = await fetch(resolved, { credentials: "same-origin" });
  if (!response.ok) throw new Error(`Artifact request failed (${response.status})`);
  return response.json();
}

function finiteArray(value) {
  return Array.isArray(value) ? value.map(Number).filter(Number.isFinite) : [];
}

function finitePath(value) {
  if (!Array.isArray(value)) return [];
  return value
    .map((point) => Array.isArray(point) ? point.slice(0, 3).map(Number) : [])
    .filter((point) => point.length === 3 && point.every(Number.isFinite));
}

function validateScene(record, camera, profiles, meta) {
  if (camera?.scale_status !== "relative_only") {
    throw new Error("Only relative-only camera paths are accepted");
  }
  const layers = {};
  for (const name of LAYERS) layers[name] = finitePath(camera?.layers?.[name]);
  if (layers.raw.length < 2) throw new Error("Raw camera path is unavailable");
  const signals = {};
  for (const [name] of PROFILE_SERIES) signals[name] = finiteArray(profiles?.[name]);
  signals.bspline_residual = finiteArray(profiles?.bspline_residual);
  signals.rts_residual = finiteArray(profiles?.rts_residual);
  signals.innovation_norm = finiteArray(profiles?.innovation_norm);
  signals.pose_jump = finiteArray(profiles?.pose_jump);
  return {
    record,
    camera,
    profiles: signals,
    meta: meta && typeof meta === "object" ? meta : {},
    layers,
  };
}

async function loadScene(record) {
  const key = `${record.slug}::${record.scene_id}`;
  if (state.cache.has(key)) return state.cache.get(key);
  const promise = (async () => {
    const base = sceneBase(record);
    const assets = {
      camera_path: record?.assets?.camera_path || "camera_path.json",
      trajectory_profiles: record?.assets?.trajectory_profiles || "trajectory_profiles.json",
      scene_meta: record?.assets?.scene_meta || "scene_meta.json",
    };
    if (!Object.values(assets).every(safeJsonName)) {
      throw new Error("Scene manifest declared an unsafe artifact name");
    }
    const [camera, profiles, meta] = await Promise.all([
      fetchJson(new URL(assets.camera_path, base)),
      fetchJson(new URL(assets.trajectory_profiles, base)),
      fetchJson(new URL(assets.scene_meta, base)),
    ]);
    return validateScene(record, camera, profiles, meta);
  })();
  state.cache.set(key, promise);
  return promise;
}

function annotationKind(record, data) {
  return data?.meta?.annotation?.kind || record.annotation_kind || "not_available";
}

function annotationLabel(kind) {
  if (kind === "manual_ground_truth") return "manual review";
  if (kind === "auto_generated" || kind === "auto_candidate") return "auto candidate";
  return "provenance limited";
}

function sceneTitle(record, data) {
  return data?.meta?.title || record.title || record.slug.replaceAll("_", " ");
}

function addEvidence(label, value) {
  const row = document.createElement("div");
  row.append(textNode("dt", label), textNode("dd", value));
  $("evidence-list").append(row);
}

function methodSummary(data) {
  const methods = data?.meta?.methods;
  const output = [];
  if (String(data?.meta?.reconstruction?.backend || "").toLowerCase().includes("vggt")) {
    output.push("Omega");
  }
  if (!methods || typeof methods !== "object") {
    const declared = Array.isArray(data?.record?.methods) ? data.record.methods : [];
    output.push(...declared);
    return output.length ? [...new Set(output)].join(" · ") : "not reported";
  }
  const labels = {
    vggt: "Omega",
    vggt_omega: "Omega",
    r3: "R3",
    lingbot_map: "LingBot",
    hloc_lightglue: "HLoc",
    hloc_lightglue_colmap: "HLoc/COLMAP",
  };
  for (const [name, detail] of Object.entries(methods)) {
    const status = typeof detail === "string" ? detail : detail?.status;
    if (status === "done" || status === "available" || status === "completed") {
      output.push(labels[name] || name);
    }
  }
  return output.length ? [...new Set(output)].join(" · ") : "not reported";
}

function formatRelative(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const numeric = Number(value);
  const magnitude = Math.abs(numeric);
  if (magnitude >= 1000 || (magnitude > 0 && magnitude < 0.001)) return `${numeric.toExponential(2)} rel`;
  return `${numeric.toFixed(magnitude >= 100 ? 1 : magnitude >= 10 ? 2 : 4)} rel`;
}

function scoreLabel(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "not ranked";
}

function drawSceneRail() {
  const list = $("scene-list");
  list.replaceChildren();
  state.records.forEach((record, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `scene-record${index === state.activeIndex ? " active" : ""}`;
    button.dataset.index = String(index);
    button.setAttribute("role", "listitem");
    const kind = record.annotation_kind || "not_available";
    button.append(
      textNode("strong", record.title || record.slug.replaceAll("_", " ")),
      textNode("span", `${annotationLabel(kind)} · ${record.scene_id}`, kind === "manual_ground_truth" ? "manual" : kind.startsWith("auto") ? "auto" : "unknown"),
    );
    button.addEventListener("click", () => selectScene(index));
    list.append(button);
  });
}

function setLoading(message) {
  $("load-status").textContent = message;
  $("evidence-status").textContent = message;
}

function setEmpty(reason) {
  $("empty-state").hidden = false;
  $("empty-reason").textContent = reason;
  setLoading("unavailable");
}

function activeRecord() {
  return state.records[state.activeIndex] || null;
}

async function selectScene(index) {
  state.activeIndex = Math.max(0, Math.min(state.records.length - 1, index));
  state.sampleIndex = 0;
  $("time-slider").value = "0";
  drawSceneRail();
  setLoading("loading");
  $("empty-state").hidden = true;
  const record = activeRecord();
  if (!record) {
    setEmpty("No scene records are available.");
    return;
  }
  try {
    const data = await loadScene(record);
    const maxSamples = Math.max(1, (data.layers[state.layer].length || data.layers.raw.length) - 1);
    $("time-slider").max = String(maxSamples);
    $("open-scene").href = new URL(`../../${record.route}`, window.location.href).href;
    updateEvidence(data);
    drawAll(data);
    setLoading("ready");
  } catch (error) {
    setEmpty(error instanceof Error ? error.message : "Scene load failed");
  }
}

function canvasContext(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(320, Math.round(rect.width));
  const height = Math.max(180, Math.round(rect.height));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function projectionPoint(point, projection = state.projection) {
  const axes = PROJECTIONS[projection]?.axes || PROJECTIONS.xz.axes;
  return [point[axes[0]], point[axes[1]]];
}

function normalizeShape(path) {
  if (!path.length) return [];
  const points = path.map((point) => projectionPoint(point));
  const center = points.reduce((sum, point) => [sum[0] + point[0], sum[1] + point[1]], [0, 0]).map((value) => value / points.length);
  const centered = points.map((point) => [point[0] - center[0], point[1] - center[1]]);
  const extent = Math.max(...centered.flatMap((point) => point.map(Math.abs)), 1e-9);
  return centered.map((point) => [point[0] / extent, point[1] / extent]);
}

function fitTransform(paths, width, height, padding = 52) {
  const points = paths.flat();
  if (!points.length) return () => [width / 2, height / 2];
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1e-9);
  const spanY = Math.max(maxY - minY, 1e-9);
  const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
  const offsetX = (width - spanX * scale) / 2;
  const offsetY = (height - spanY * scale) / 2;
  return (point) => [offsetX + (point[0] - minX) * scale, height - offsetY - (point[1] - minY) * scale];
}

function drawGrid(context, width, height) {
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#030709";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(40, 97, 106, 0.24)";
  context.lineWidth = 1;
  for (let index = 1; index < 10; index += 1) {
    const x = index * width / 10;
    const y = index * height / 10;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
}

function strokePath(context, path, transform, color, width = 1.5, alpha = 1) {
  if (path.length < 2) return;
  context.save();
  context.globalAlpha = alpha;
  context.strokeStyle = color;
  context.lineWidth = width;
  context.beginPath();
  path.forEach((point, index) => {
    const [x, y] = transform(point);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
  context.restore();
}

function drawActiveSample(context, path, transform) {
  if (!path.length) return;
  const index = Math.min(path.length - 1, state.sampleIndex);
  const point = transform(path[index]);
  const before = transform(path[Math.max(0, index - 1)]);
  const after = transform(path[Math.min(path.length - 1, index + 1)]);
  const angle = Math.atan2(after[1] - before[1], after[0] - before[0]);
  context.save();
  context.translate(point[0], point[1]);
  context.rotate(angle);
  context.fillStyle = "#f4bf5f";
  context.beginPath();
  context.moveTo(11, 0);
  context.lineTo(-7, -5);
  context.lineTo(-4, 0);
  context.lineTo(-7, 5);
  context.closePath();
  context.fill();
  context.restore();
  context.fillStyle = "#43e0cd";
  context.beginPath();
  context.arc(point[0], point[1], 3.5, 0, Math.PI * 2);
  context.fill();
}

function profilePair(data) {
  const speed = data.profiles.speed_relative;
  const curvature = data.profiles.curvature;
  const count = Math.min(speed.length, curvature.length);
  return Array.from({ length: count }, (_, index) => [speed[index], curvature[index]]);
}

function quantile(values, fraction) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return 1;
  return sorted[Math.min(sorted.length - 1, Math.round((sorted.length - 1) * fraction))];
}

function clippedPairs(pairs) {
  if (!pairs.length) return [];
  const xMax = Math.max(quantile(pairs.map((point) => Math.abs(point[0])), 0.98), 1e-9);
  const yMax = Math.max(quantile(pairs.map((point) => Math.abs(point[1])), 0.98), 1e-9);
  return pairs.map((point) => [
    Math.max(-1, Math.min(1, point[0] / xMax)),
    Math.max(-1, Math.min(1, point[1] / yMax)),
  ]);
}

async function drawTrajectory(data) {
  const { context, width, height } = canvasContext($("trajectory-canvas"));
  drawGrid(context, width, height);
  $("compare-legend").hidden = state.view !== "compare";
  if (state.view === "phase") {
    const pairs = clippedPairs(profilePair(data));
    const transform = fitTransform([pairs], width, height);
    strokePath(context, pairs, transform, "#be96ff", 1.8);
    drawActiveSample(context, pairs.map((pair) => [pair[0], 0, pair[1]]), (point) => transform([point[0], point[2]]));
    $("stage-kicker").textContent = "Signal phase";
    $("stage-title").textContent = "Relative speed / curvature portrait";
    $("plane-note").textContent = "Each axis clipped at its own 98th percentile · display comparison only";
    return;
  }
  if (state.view === "compare") {
    setLoading("loading cohort");
    const settled = await Promise.allSettled(state.records.map(loadScene));
    const cohort = settled
      .map((result, index) => result.status === "fulfilled"
        ? { data: result.value, index, path: normalizeShape(result.value.layers[state.layer].length ? result.value.layers[state.layer] : result.value.layers.raw) }
        : null)
      .filter((item) => item && item.path.length);
    const transform = fitTransform(cohort.map((item) => item.path), width, height);
    cohort.forEach((item) => {
      const active = item.index === state.activeIndex;
      strokePath(context, item.path, transform, active ? "#43e0cd" : COHORT_COLORS[item.index % COHORT_COLORS.length], active ? 2.5 : 1.1, active ? 1 : 0.48);
    });
    const activeItem = cohort.find((item) => item.index === state.activeIndex);
    if (activeItem) drawActiveSample(context, activeItem.path, transform);
    $("stage-kicker").textContent = "Cohort compare";
    $("stage-title").textContent = "Shape-normalized path comparison";
    $("plane-note").textContent = "Each path centered and scaled independently · shapes are not spatially co-registered";
    setLoading(`${cohort.length} scenes`);
    return;
  }
  const raw = data.layers.raw.map((point) => projectionPoint(point));
  const selected3d = data.layers[state.layer].length ? data.layers[state.layer] : data.layers.raw;
  const selected = selected3d.map((point) => projectionPoint(point));
  const transform = fitTransform([raw, selected], width, height);
  if (state.layer !== "raw") strokePath(context, raw, transform, "#607b81", 1.1, 0.65);
  strokePath(context, selected, transform, "#43e0cd", 2.2);
  drawActiveSample(context, selected, transform);
  $("stage-kicker").textContent = "Trajectory field";
  $("stage-title").textContent = sceneTitle(data.record, data);
  $("plane-note").textContent = PROJECTIONS[state.projection].label;
}

function normalizedSeries(values) {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return [];
  const low = quantile(finite, 0.02);
  const high = quantile(finite, 0.98);
  const span = Math.max(high - low, 1e-9);
  return values.map((value) => Math.max(0, Math.min(1, (value - low) / span)));
}

function drawSignals(data) {
  const { context, width, height } = canvasContext($("signal-canvas"));
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#030709";
  context.fillRect(0, 0, width, height);
  const padding = { left: 30, right: 18, top: 18, bottom: 22 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  context.strokeStyle = "rgba(40, 97, 106, 0.26)";
  context.lineWidth = 1;
  for (let row = 0; row <= 4; row += 1) {
    const y = padding.top + row * plotHeight / 4;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
  }
  PROFILE_SERIES.forEach(([name, color]) => {
    const values = normalizedSeries(data.profiles[name]);
    if (values.length < 2) return;
    context.strokeStyle = color;
    context.lineWidth = 1.25;
    context.beginPath();
    values.forEach((value, index) => {
      const x = padding.left + index / (values.length - 1) * plotWidth;
      const y = padding.top + (1 - value) * plotHeight;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  });
  const count = Math.max(data.profiles.speed_relative.length, 2);
  const cursorX = padding.left + Math.min(count - 1, state.sampleIndex) / (count - 1) * plotWidth;
  context.strokeStyle = "#edf7f6";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(cursorX, padding.top);
  context.lineTo(cursorX, height - padding.bottom);
  context.stroke();
}

function sampleValue(values, index) {
  if (!values.length) return null;
  return values[Math.min(values.length - 1, index)];
}

function tangentDegrees(path, index) {
  if (path.length < 2) return null;
  const before = path[Math.max(0, index - 1)];
  const after = path[Math.min(path.length - 1, index + 1)];
  const dx = after[0] - before[0];
  const dz = after[2] - before[2];
  if (!Number.isFinite(dx) || !Number.isFinite(dz) || Math.hypot(dx, dz) < 1e-12) return null;
  return Math.atan2(dz, dx) * 180 / Math.PI;
}

function updateProxyState(data) {
  const path = data.layers[state.layer].length ? data.layers[state.layer] : data.layers.raw;
  const index = Math.min(path.length - 1, state.sampleIndex);
  const point = path[index] || [];
  const heading = tangentDegrees(path, index);
  $("state-x").textContent = formatRelative(point[0]);
  $("state-y").textContent = formatRelative(point[1]);
  $("state-z").textContent = formatRelative(point[2]);
  $("state-heading").textContent = Number.isFinite(heading) ? `${heading.toFixed(1)}° proxy` : "unavailable";
  $("proxy-arrow").style.transform = `rotate(${Number.isFinite(heading) ? heading : 0}deg)`;
  $("state-speed").textContent = formatRelative(sampleValue(data.profiles.speed_relative, index));
  $("state-accel").textContent = formatRelative(sampleValue(data.profiles.acceleration_relative, index));
  $("state-jerk").textContent = formatRelative(sampleValue(data.profiles.jerk_proxy, index));
  $("state-curvature").textContent = formatRelative(sampleValue(data.profiles.curvature, index));
  const denominator = Math.max(path.length - 1, 1);
  $("time-readout").textContent = (index / denominator).toFixed(3);
  $("sample-readout").textContent = `${index + 1}/${path.length}`;
}

function updateEvidence(data) {
  const record = data.record;
  const kind = annotationKind(record, data);
  const quality = data.meta?.quality || {};
  $("evidence-list").replaceChildren();
  addEvidence("Annotation", annotationLabel(kind));
  addEvidence("Scale", "relative_only");
  addEvidence("Pose semantics", data.meta?.pose_semantics || "camera pose proxy");
  addEvidence("Hero score", scoreLabel(quality.hero_score ?? record.hero_score));
  addEvidence("Methods", methodSummary(data));
  addEvidence("Active layer", state.layer);
  addEvidence("Samples", String((data.layers[state.layer].length || data.layers.raw.length)));
  $("evidence-status").textContent = kind === "manual_ground_truth" ? "manual provenance" : "limited provenance";
}

async function drawAll(data) {
  await drawTrajectory(data);
  drawSignals(data);
  updateProxyState(data);
}

async function redraw() {
  const record = activeRecord();
  if (!record) return;
  try {
    const data = await loadScene(record);
    updateEvidence(data);
    await drawAll(data);
  } catch (error) {
    setEmpty(error instanceof Error ? error.message : "Scene render failed");
  }
}

function setView(view) {
  if (!["trajectory", "phase", "compare"].includes(view)) return;
  state.view = view;
  document.querySelectorAll("[data-view]").forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  redraw();
}

function animationStep(timestamp) {
  if (!state.playing) return;
  if (timestamp - state.lastAdvance >= 66) {
    state.lastAdvance = timestamp;
    const slider = $("time-slider");
    const max = Number(slider.max) || 1;
    state.sampleIndex = state.sampleIndex >= max ? 0 : state.sampleIndex + 1;
    slider.value = String(state.sampleIndex);
    redraw();
  }
  state.animationFrame = requestAnimationFrame(animationStep);
}

function togglePlayback() {
  state.playing = !state.playing;
  $("play-toggle").textContent = state.playing ? "Pause" : "Play";
  if (state.playing) {
    state.lastAdvance = 0;
    state.animationFrame = requestAnimationFrame(animationStep);
  } else {
    cancelAnimationFrame(state.animationFrame);
  }
}

function bindControls() {
  $("scene-select").addEventListener("change", (event) => selectScene(Number(event.target.value)));
  $("layer-select").addEventListener("change", (event) => {
    state.layer = LAYERS.includes(event.target.value) ? event.target.value : "raw";
    state.sampleIndex = 0;
    $("time-slider").value = "0";
    redraw();
  });
  $("projection-select").addEventListener("change", (event) => {
    state.projection = PROJECTIONS[event.target.value] ? event.target.value : "xz";
    redraw();
  });
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  $("time-slider").addEventListener("input", (event) => {
    state.sampleIndex = Number(event.target.value) || 0;
    redraw();
  });
  $("play-toggle").addEventListener("click", togglePlayback);
  window.addEventListener("resize", redraw, { passive: true });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && state.playing) togglePlayback();
  });
}

function populateSceneControl() {
  const select = $("scene-select");
  select.replaceChildren();
  state.records.forEach((record, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = record.title || record.slug.replaceAll("_", " ");
    select.append(option);
  });
}

async function initialize() {
  bindControls();
  try {
    const manifest = await fetchJson(MANIFEST_URL);
    if (manifest?.scale_state !== "relative_only" || !Array.isArray(manifest?.scenes)) {
      throw new Error("Observatory manifest failed the relative-only contract");
    }
    state.manifest = manifest;
    state.records = manifest.scenes.filter((record) => safeToken(record.slug) && safeToken(record.scene_id));
    $("scene-count").textContent = String(state.records.length);
    $("ready-count").textContent = `${state.records.length} ready`;
    if (!state.records.length) throw new Error("No public scenes expose the required trajectory artifacts");
    populateSceneControl();
    drawSceneRail();
    await selectScene(0);
  } catch (error) {
    $("scene-select").replaceChildren(new Option("No scenes available", ""));
    setEmpty(error instanceof Error ? error.message : "Observatory initialization failed");
  }
}

initialize();
