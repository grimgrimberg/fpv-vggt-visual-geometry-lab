"use strict";

const galleryState = {
  data: null,
  sceneJsonUrl: null,
  pointCloud: null,
  pointCloudRenderer: null,
  densityFallback: false,
  yaw: -0.72,
  pitch: 0.38,
  reducedMotion: matchMedia("(prefers-reduced-motion: reduce)").matches,
};

function galleryCanvasSize(canvas) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, dpr };
}

function galleryProject(point, width, height, yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x = point[0], y = point[1], z = point[2];
  const rx = cy * x + sy * z;
  const rz = -sy * x + cy * z;
  const ry = cp * y - sp * rz;
  const depth = sp * y + cp * rz;
  const perspective = 1 / Math.max(1.35 + depth * 0.2, 0.6);
  const scale = Math.min(width, height) * 0.48 * perspective;
  return [width * 0.67 + rx * scale, height * 0.5 - ry * scale, depth];
}

function activateGalleryDensityFallback(error = null) {
  if (galleryState.pointCloudRenderer) galleryState.pointCloudRenderer.dispose();
  galleryState.pointCloud = null;
  galleryState.pointCloudRenderer = null;
  galleryState.densityFallback = true;
  document.body.classList.remove("point-cloud-loading", "point-cloud-ready");
  document.body.classList.add("point-cloud-fallback");
  const densityCount = galleryState.data.geometry_density.cells.length;
  document.getElementById("hero-cell-count").textContent =
    `${densityCount.toLocaleString()} voxels · density fallback`;
  if (error && galleryState.data.point_cloud) {
    document.getElementById("hero-geometry-deck").textContent =
      "The authorized colored point sample is published, but this browser could not render it; "
      + "the coarse density fallback remains available with the same relative camera path.";
    document.getElementById("proof-title").textContent = "The fallback stays explicit.";
    document.getElementById("proof-copy").textContent =
      "The scene contract still records the authorized sample and attribution. This browser is "
      + "showing coarse normalized density because point-array loading or WebGL failed.";
    console.warn("Gallery point-cloud rendering fell back to density", error);
  }
}

function activateGalleryPointCloud(cloud, renderer) {
  galleryState.pointCloud = cloud;
  galleryState.pointCloudRenderer = renderer;
  galleryState.densityFallback = false;
  document.body.classList.remove("point-cloud-loading", "point-cloud-fallback");
  document.body.classList.add("point-cloud-ready");
  document.getElementById("hero-cell-count").textContent =
    `${cloud.pointCount.toLocaleString()} points · relative_only`;
  document.getElementById("hero-geometry-deck").textContent =
    "A completed historical clip reduced to relative camera motion, an authorized colored VGGT "
    + "Omega point sample, model summaries, and visible failure states.";
  document.getElementById("proof-title").textContent = "The published sample is the artifact.";
  document.getElementById("proof-copy").textContent =
    `This authorized colored VGGT Omega point sample publishes ${cloud.pointCount.toLocaleString()} `
    + "derived points in relative-only coordinates. Video, source frames, recognizable "
    + "reprojections, raw NPZ/PLY, full arrays, and machine paths remain withheld.";
}

function renderGalleryPointCloudCredit() {
  const credit = document.getElementById("gallery-point-cloud-credit");
  const pointCloud = galleryState.data.point_cloud;
  const authorized = pointCloud
    && galleryState.data.publication_boundary?.real_point_sample_published === true;
  if (!authorized) {
    credit.hidden = true;
    document.getElementById("gallery-point-cloud-attribution").textContent = "";
    document.getElementById("gallery-point-cloud-authorization").textContent = "";
    return;
  }
  document.getElementById("gallery-point-cloud-attribution").textContent =
    pointCloud.attribution;
  document.getElementById("gallery-point-cloud-authorization").textContent =
    pointCloud.authorization_provenance;
  credit.hidden = false;
}

async function initializeGalleryPointCloud() {
  const declared = galleryState.data.point_cloud
    && galleryState.data.publication_boundary?.real_point_sample_published === true;
  if (!declared) {
    activateGalleryDensityFallback();
    return;
  }
  document.body.classList.add("point-cloud-loading");
  try {
    const api = window.FpvPointCloudWebGL;
    if (!api) throw new Error("Point-cloud WebGL runtime is unavailable");
    const cloud = await api.loadPointCloud(
      galleryState.sceneJsonUrl,
      galleryState.data.point_cloud,
    );
    const renderer = api.createPointCloudRenderer(
      document.getElementById("hero-point-cloud-canvas"),
      cloud,
      { onContextLost: (error) => activateGalleryDensityFallback(error) },
    );
    activateGalleryPointCloud(cloud, renderer);
  } catch (error) {
    activateGalleryDensityFallback(error);
  }
}

function drawGalleryDensity(context, width, height, yaw) {
  const cells = galleryState.data.geometry_density.cells
    .map((cell) => ({
      cell,
      projected: galleryProject(cell.position, width, height, yaw, galleryState.pitch),
    }))
    .sort((left, right) => left.projected[2] - right.projected[2]);
  context.globalCompositeOperation = "lighter";
  for (const item of cells) {
    const density = item.cell.density;
    const radius = 0.7 + density * 2.8;
    context.fillStyle = `rgba(84,229,194,${0.06 + density * 0.45})`;
    context.beginPath();
    context.arc(item.projected[0], item.projected[1], radius, 0, Math.PI * 2);
    context.fill();
  }
  context.globalCompositeOperation = "source-over";
}

function drawGalleryHero(timestamp) {
  if (!galleryState.data) return;
  const canvas = document.getElementById("hero-canvas");
  const context = canvas.getContext("2d");
  const { width, height } = galleryCanvasSize(canvas);
  const motion = galleryState.reducedMotion ? 0 : timestamp * 0.000035;
  const yaw = galleryState.yaw + motion;
  context.clearRect(0, 0, width, height);

  if (galleryState.pointCloudRenderer) {
    try {
      galleryState.pointCloudRenderer.drawPointCloud({
        yaw,
        pitch: galleryState.pitch,
        anchor: [0.67, 0.5],
        base: 1.35,
        depth: 0.2,
        minimumPerspective: 0.6,
        scaleFactor: 0.48,
        zoom: 1,
        pointSize: 1.4,
        alpha: 0.5,
      });
    } catch (error) {
      activateGalleryDensityFallback(error);
    }
  }
  if (galleryState.densityFallback) drawGalleryDensity(context, width, height, yaw);

  const path = galleryState.data.paths.rts || galleryState.data.paths.raw;
  context.strokeStyle = "rgba(232,242,242,.72)";
  context.lineWidth = Math.max(1.2, width / 1300);
  context.beginPath();
  path.forEach((point, index) => {
    const projected = galleryProject(point, width, height, yaw, galleryState.pitch);
    if (index === 0) context.moveTo(projected[0], projected[1]);
    else context.lineTo(projected[0], projected[1]);
  });
  context.stroke();

  const first = galleryProject(path[0], width, height, yaw, galleryState.pitch);
  const last = galleryProject(path[path.length - 1], width, height, yaw, galleryState.pitch);
  context.fillStyle = "#f3b75f";
  for (const point of [first, last]) {
    context.beginPath();
    context.arc(point[0], point[1], 3, 0, Math.PI * 2);
    context.fill();
  }
  if (!galleryState.reducedMotion) requestAnimationFrame(drawGalleryHero);
}

function renderGalleryMethods(methods) {
  const ledger = document.getElementById("gallery-method-list");
  const rows = methods.map((method) => {
    const row = document.createElement("div");
    row.className = "method-ledger-row";
    const name = document.createElement("strong");
    name.textContent = method.label;
    const state = document.createElement("span");
    state.className = `method-state ${method.status}`;
    state.textContent = method.status.replaceAll("_", " ");
    const role = document.createElement("p");
    role.textContent = method.role;
    const alignment = document.createElement("small");
    alignment.textContent = method.alignment.replaceAll("_", " ");
    row.append(name, state, role, alignment);
    return row;
  });
  ledger.replaceChildren(...rows);
}

async function initializeGallery() {
  galleryState.sceneJsonUrl = document.body.dataset.sceneData;
  const response = await fetch(galleryState.sceneJsonUrl, { cache: "no-store" });
  if (!response.ok) throw new Error(`Scene summary request failed (${response.status})`);
  galleryState.data = await response.json();
  document.getElementById("hero-pose-count").textContent = String(galleryState.data.sample_count);
  document.getElementById("hero-connectivity").textContent =
    `${galleryState.data.match_connectivity.connected_component_count} component`;
  renderGalleryPointCloudCredit();
  renderGalleryMethods(galleryState.data.methods);
  await initializeGalleryPointCloud();
  document.body.classList.add("ready");
  requestAnimationFrame(drawGalleryHero);
}

addEventListener("resize", () => {
  if (galleryState.data && galleryState.reducedMotion) requestAnimationFrame(drawGalleryHero);
});

initializeGallery().catch((error) => {
  const message = document.createElement("div");
  message.className = "error-banner";
  message.textContent = `Public scene unavailable: ${error.message}`;
  document.body.appendChild(message);
  console.error(error);
});
