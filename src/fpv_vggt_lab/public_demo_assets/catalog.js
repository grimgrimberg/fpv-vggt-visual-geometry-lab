"use strict";

const catalogState = {
  data: null,
  observer: null,
};

const BLOCKED_LOCAL_HOSTNAMES = new Set(["localhost", "localdomain", "home.arpa"]);
const BLOCKED_LOCAL_HOST_SUFFIXES = Object.freeze([
  ".localhost",
  ".localdomain",
  ".local",
  ".internal",
  ".home.arpa",
  ".lan",
  ".home",
  ".corp",
  ".invalid",
  ".test",
  ".example",
]);

function catalogElement(tag, className = "", text = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function safeSourceRecordUrl(record) {
  const value = record.source_record_url;
  if (typeof value !== "string") return null;
  const match = value.match(
    /^https:\/\/www\.itamarweiss\.com\/fpv\/video\/([a-z0-9]+(?:[_-][a-z0-9]+)*)\/$/,
  );
  return match && match[1] === record.slug ? value : null;
}

function isPublicIpv4(hostname) {
  const octets = hostname.split(".").map(Number);
  if (
    octets.length !== 4
    || octets.some((value) => !Number.isInteger(value) || value < 0 || value > 255)
  ) return false;
  const [first, second, third] = octets;
  if (first === 0 || first === 10 || first === 127 || first >= 224) return false;
  if (first === 100 && second >= 64 && second <= 127) return false;
  if (first === 169 && second === 254) return false;
  if (first === 172 && second >= 16 && second <= 31) return false;
  if (first === 192 && second === 168) return false;
  if (first === 192 && second === 0 && third <= 2) return false;
  if (first === 198 && (second === 18 || second === 19 || second === 51)) return false;
  if (first === 203 && second === 0 && third === 113) return false;
  return true;
}

function isPublicSceneHostname(hostname) {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (normalized.endsWith(".")) return false;
  if (normalized.includes(":")) {
    const first = Number.parseInt(normalized.split(":", 1)[0] || "0", 16);
    if (!Number.isFinite(first) || first < 0x2000 || first > 0x3fff) return false;
    if (normalized === "2001:db8" || normalized.startsWith("2001:db8:")) return false;
    return true;
  }
  if (/^\d+\.\d+\.\d+\.\d+$/.test(normalized)) {
    return isPublicIpv4(normalized);
  }
  return /^[a-z0-9.-]+$/.test(normalized)
    && normalized.includes(".")
    && !BLOCKED_LOCAL_HOSTNAMES.has(normalized)
    && !BLOCKED_LOCAL_HOST_SUFFIXES.some((suffix) => normalized.endsWith(suffix));
}

function safePublicSceneUrl(record) {
  const value = record.research?.public_scene_url;
  if (typeof value !== "string" || /[\\%?#]/.test(value)) return null;
  if (/^scenes\/[a-z0-9]+(?:[_-][a-z0-9]+)*\/$/.test(value)) return value;
  try {
    const parsed = new URL(value);
    if (
      parsed.protocol !== "https:"
      || parsed.username
      || parsed.password
      || parsed.search
      || parsed.hash
      || !isPublicSceneHostname(parsed.hostname)
    ) return null;
    return value;
  } catch (_error) {
    return null;
  }
}

window.FpvCatalogSafety = Object.freeze({ safePublicSceneUrl });

function segmentRanges(record) {
  const segments = record.edit_segmentation?.segments || [];
  if (!segments.length) return [];
  if (segments.every((segment) => Number.isFinite(segment.start_s) && Number.isFinite(segment.end_s))) {
    const end = Math.max(...segments.map((segment) => segment.end_s), 1);
    return segments.map((segment) => ({
      start: segment.start_s / end,
      end: segment.end_s / end,
      type: segment.type,
    }));
  }
  const times = segments.map((segment) => Number(segment.time) || 0);
  const end = Math.max(times[times.length - 1] || 0, 1);
  return segments.map((segment, index) => ({
    start: times[index] / end,
    end: (times[index + 1] ?? end) / end,
    type: segment.type,
  }));
}

function drawCatalogDiagram(canvas, record) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const width = Math.max(220, Math.round(canvas.clientWidth * dpr));
  const height = Math.round(112 * dpr);
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#071015";
  context.fillRect(0, 0, width, height);

  const pad = 14 * dpr;
  const railY = 74 * dpr;
  const railWidth = width - pad * 2;
  context.strokeStyle = "rgba(183,211,214,.16)";
  context.lineWidth = Math.max(1, dpr);
  context.beginPath();
  context.moveTo(pad, railY);
  context.lineTo(width - pad, railY);
  context.stroke();

  const ranges = segmentRanges(record);
  ranges.forEach((range, index) => {
    const left = pad + railWidth * Math.max(0, Math.min(1, range.start));
    const right = pad + railWidth * Math.max(0, Math.min(1, range.end));
    const alpha = range.type.includes("flight") ? 0.9 : 0.26 + (index % 3) * 0.1;
    context.fillStyle = `rgba(84,229,194,${alpha})`;
    context.fillRect(left, railY - 4 * dpr, Math.max(right - left - dpr, 2 * dpr), 8 * dpr);
  });

  context.strokeStyle = "rgba(84,229,194,.35)";
  context.beginPath();
  for (let index = 0; index < 16; index += 1) {
    const x = pad + railWidth * index / 15;
    const phase = (record.slug.length * 0.19) + index * 0.72;
    const y = 38 * dpr + Math.sin(phase) * 9 * dpr;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();
  if (record.research?.public_scene_available) {
    context.fillStyle = "#54e5c2";
    context.beginPath();
    context.arc(width - pad, 24 * dpr, 3 * dpr, 0, Math.PI * 2);
    context.fill();
  }
}

function observeCatalogDiagram(canvas, record) {
  canvas._catalogRecord = record;
  if (catalogState.observer) {
    catalogState.observer.observe(canvas);
  } else {
    drawCatalogDiagram(canvas, record);
  }
}

function catalogLink(label, href, primary = false) {
  const link = catalogElement("a", primary ? "catalog-link primary" : "catalog-link", label);
  link.href = href;
  if (/^[a-z]+:/i.test(href)) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  return link;
}

function catalogCard(record) {
  const card = catalogElement("article", "catalog-card");
  const diagram = catalogElement("canvas", "catalog-diagram");
  diagram.setAttribute("role", "img");
  diagram.setAttribute(
    "aria-label",
    `Edit-segment and research-availability diagram for ${record.description}`,
  );
  const meta = catalogElement("div", "catalog-card-meta");
  meta.append(
    catalogElement("time", "", record.date),
    catalogElement("span", "", record.town),
  );
  const title = catalogElement("h3", "", record.description);
  const states = catalogElement("div", "catalog-card-states");
  states.append(
    catalogElement(
      "span",
      "annotation-state",
      record.annotation_state.replaceAll("_", " "),
    ),
    catalogElement(
      "span",
      record.research?.public_scene_available ? "scene-state ready" : "scene-state",
      record.research?.public_scene_available ? "3D scene" : "metadata only",
    ),
  );
  const actions = catalogElement("div", "catalog-card-actions");
  const sourceUrl = safeSourceRecordUrl(record);
  const sceneUrl = safePublicSceneUrl(record);
  if (sourceUrl) actions.append(catalogLink("Open source player ↗", sourceUrl));
  if (record.research?.public_scene_available && sceneUrl) {
    actions.append(catalogLink("Open 3D scene →", sceneUrl, true));
  }
  card.append(diagram, meta, title, states, actions);
  observeCatalogDiagram(diagram, record);
  return card;
}

function filteredCatalogRecords() {
  const query = document.getElementById("catalog-search").value.trim().toLowerCase();
  const annotation = document.getElementById("catalog-annotation").value;
  const scenesOnly = document.getElementById("catalog-scenes-only").checked;
  const sort = document.getElementById("catalog-sort").value;
  const records = catalogState.data.records.filter((record) => {
    const haystack = [record.date, record.town, record.description, record.slug]
      .join(" ")
      .toLowerCase();
    return (!query || haystack.includes(query))
      && (annotation === "all" || record.annotation_state === annotation)
      && (!scenesOnly || record.research?.public_scene_available === true);
  });
  records.sort((left, right) => {
    const order = left.date.localeCompare(right.date) || left.slug.localeCompare(right.slug);
    return sort === "date-asc" ? order : -order;
  });
  return records;
}

function renderCatalog() {
  const grid = document.getElementById("catalog-grid");
  if (catalogState.observer) catalogState.observer.disconnect();
  const records = filteredCatalogRecords();
  grid.replaceChildren(...records.map(catalogCard));
  document.getElementById("catalog-count").textContent =
    `${records.length.toLocaleString()} of ${catalogState.data.records.length.toLocaleString()} records`;
  if (!records.length) {
    grid.append(catalogElement("p", "catalog-empty", "No records match these filters."));
  }
}

function wireCatalogControls() {
  for (const id of [
    "catalog-search",
    "catalog-sort",
    "catalog-scenes-only",
    "catalog-annotation",
  ]) {
    const control = document.getElementById(id);
    control.addEventListener(control.type === "search" ? "input" : "change", renderCatalog);
  }
}

async function initializeCatalog() {
  const response = await fetch("catalog.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Catalog request failed (${response.status})`);
  const data = await response.json();
  if (data.schema_version !== "wow-public-videos-v1" || !Array.isArray(data.records)) {
    throw new Error("Catalog contract is invalid");
  }
  catalogState.data = data;
  if ("IntersectionObserver" in window) {
    catalogState.observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting || entry.target.dataset.drawn === "true") return;
        drawCatalogDiagram(entry.target, entry.target._catalogRecord);
        entry.target.dataset.drawn = "true";
        catalogState.observer.unobserve(entry.target);
      });
    }, { rootMargin: "160px" });
  }
  wireCatalogControls();
  renderCatalog();
}

if (document.getElementById("catalog-grid")) {
  initializeCatalog().catch((error) => {
    document.getElementById("catalog-count").textContent = "catalog unavailable";
    const grid = document.getElementById("catalog-grid");
    grid.replaceChildren(catalogElement("p", "catalog-empty", error.message));
    console.error(error);
  });
}
