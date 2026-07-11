"use strict";

const galleryState = {
  data: null,
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

function drawGalleryHero(timestamp) {
  if (!galleryState.data) return;
  const canvas = document.getElementById("hero-canvas");
  const context = canvas.getContext("2d");
  const { width, height } = galleryCanvasSize(canvas);
  const motion = galleryState.reducedMotion ? 0 : timestamp * 0.000035;
  const yaw = galleryState.yaw + motion;
  context.clearRect(0, 0, width, height);

  const cells = galleryState.data.geometry_density.cells
    .map((cell) => ({ cell, projected: galleryProject(cell.position, width, height, yaw, galleryState.pitch) }))
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
  const url = document.body.dataset.sceneData;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Scene summary request failed (${response.status})`);
  galleryState.data = await response.json();
  document.getElementById("hero-pose-count").textContent = String(galleryState.data.sample_count);
  document.getElementById("hero-connectivity").textContent =
    `${galleryState.data.match_connectivity.connected_component_count} component`;
  document.getElementById("hero-cell-count").textContent =
    `${galleryState.data.geometry_density.cells.length} voxels`;
  renderGalleryMethods(galleryState.data.methods);
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
