"use strict";

const app = {
  catalog: null,
  records: [],
  active: null,
  filter: "all",
  routeNotice: "",
};

const $ = (id) => document.getElementById(id);

function element(tagName, className, text) {
  const node = document.createElement(tagName);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function methodLabel(method) {
  return {
    vggt_omega: "VGGT Ω",
    r3: "R3",
    lingbot_map: "LingBot",
    hloc_lightglue_colmap: "HLoc",
    mast3r_sfm: "MASt3R",
    colmap: "COLMAP",
  }[method] || String(method).replaceAll("_", " ");
}

function annotationLabel(state) {
  if (state === "manual_ground_truth") return "Manual edit map";
  if (state === "auto") return "Auto edit map";
  return "Annotation pending";
}

function capabilityLabel(capability) {
  return {
    rendered_3d: "3D",
    artifact_complete: "artifact",
    numerical_only: "numerical",
    metadata_only: "metadata",
    failed: "failed",
    unavailable: "unavailable",
    unknown: "unknown",
  }[capability] || "unknown";
}

function capabilityTone(capability) {
  if (capability === "rendered_3d") return "hot";
  if (capability === "numerical_only") return "data";
  if (capability === "metadata_only" || capability === "artifact_complete") return "meta";
  if (capability === "failed" || capability === "unavailable") return "warn";
  return "";
}

function methodEvidence(record) {
  const evidence = record.research?.method_evidence;
  if (Array.isArray(evidence) && evidence.length) return evidence;
  const fullNative = Boolean(record.research?.hub_inventory?.full_native);
  return (record.research?.methods || []).map((method) => ({
    method,
    status: "reported",
    capability: fullNative ? "rendered_3d" : "artifact_complete",
    available_3d: fullNative,
    note: "",
  }));
}

function methodSummary(record) {
  const declared = record.research?.method_summary;
  if (declared && Number.isFinite(Number(declared.rendered_3d))) return declared;
  return methodEvidence(record).reduce((summary, row) => {
    summary[row.capability] = (summary[row.capability] || 0) + 1;
    return summary;
  }, {});
}

function isDetailed(record) {
  return Boolean(record.research && record.research.detail_route);
}

function isEnsemble(record) {
  const inventory = record.research?.hub_inventory || {};
  const rendered = Number(methodSummary(record).rendered_3d || 0);
  return Boolean(inventory.full_native || rendered >= 3);
}

async function main() {
  const response = await fetch("cockpit_catalog.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`catalog request returned ${response.status}`);
  app.catalog = await response.json();
  app.records = Array.isArray(app.catalog.records) ? app.catalog.records : [];
  wire();
  renderHeader();
  renderGallery();
  routeFromUrl();
}

function wire() {
  $("search").addEventListener("input", renderGallery);
  document.querySelectorAll("[data-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      app.filter = button.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("active", active);
        candidate.setAttribute("aria-pressed", String(active));
      });
      renderGallery();
    });
  });
  $("back-to-gallery").addEventListener("click", () => showGallery(true));
  $("scene-frame").addEventListener("load", () => {
    if ($("scene-frame").getAttribute("src")) $("viewer-loading").hidden = true;
  });
  $("video").addEventListener("loadedmetadata", () => {
    if (app.active && !isDetailed(app.active)) {
      const duration = Number.isFinite($("video").duration) ? $("video").duration : 0;
      renderTimeline(app.active, duration);
    }
  });
  $("retry-load").addEventListener("click", () => location.reload());
  window.addEventListener("popstate", routeFromUrl);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.dataset.view === "detail") {
      showGallery(true);
    }
    if (event.key === "/" && document.body.dataset.view === "gallery" && document.activeElement !== $("search")) {
      event.preventDefault();
      $("search").focus();
    }
  });
}

function renderHeader() {
  const counts = app.catalog.counts || {};
  const detailed = app.records.filter(isDetailed).length;
  const manual = app.records.filter((record) => record.annotation?.state === "manual_ground_truth").length;
  const vggt = Number(counts.vggt_done_records)
    || app.records.filter((record) => record.research?.vggt_omega?.status === "done").length;
  $("catalog-count").textContent = `${app.records.length} scenes · ${detailed} full WebGL cockpits`;
  const values = [
    [app.records.length, "catalog scenes"],
    [detailed, "full 3D bundles"],
    [vggt, "VGGT complete"],
    [manual, "manual edit maps"],
  ];
  $("summary-stats").replaceChildren(...values.map(([value, label]) => {
    const stat = element("div", "summary-stat");
    stat.append(element("strong", "", String(value)), element("span", "", label));
    return stat;
  }));
}

function routeFromUrl() {
  const slug = new URL(location.href).searchParams.get("scene");
  if (!slug) {
    showGallery(false);
    return;
  }
  const record = app.records.find((candidate) => candidate.slug === slug);
  if (!record) {
    app.routeNotice = `Scene “${slug}” is not in this catalog. Showing all scenes.`;
    showGallery(false);
    renderGallery();
    return;
  }
  app.routeNotice = "";
  openDetail(record, false);
}

function passes(record) {
  const query = $("search").value.trim().toLowerCase();
  if (app.filter === "detailed" && !isDetailed(record)) return false;
  if (app.filter === "ensemble" && !isEnsemble(record)) return false;
  if (app.filter === "manual" && record.annotation?.state !== "manual_ground_truth") return false;
  if (!query) return true;
  return [
    record.slug,
    record.title,
    record.town,
    record.date,
    ...(record.research?.methods || []),
    ...methodEvidence(record).flatMap((row) => [
      row.method,
      row.status,
      row.capability,
    ]),
  ].join(" ").toLowerCase().includes(query);
}

function featuredRank(record) {
  const summary = methodSummary(record);
  const rendered = Number(summary.rendered_3d || 0);
  const evidence = methodEvidence(record).length;
  return (isEnsemble(record) ? 100 : 0) + rendered * 10 + evidence;
}

function renderGallery() {
  const rows = app.records.filter(passes);
  const featured = rows
    .filter(isDetailed)
    .sort((left, right) => featuredRank(right) - featuredRank(left));
  const archive = rows.filter((record) => !isDetailed(record));

  $("featured-list").replaceChildren(...featured.map((record, index) => buildCard(record, {
    featured: true,
    lead: index === 0,
  })));
  $("scene-list").replaceChildren(...archive.map((record) => buildCard(record, {
    featured: false,
    lead: false,
  })));

  $("featured-section").hidden = featured.length === 0;
  $("archive-section").hidden = archive.length === 0;
  $("no-results").hidden = rows.length !== 0;
  $("featured-count").textContent = `${featured.length} scene${featured.length === 1 ? "" : "s"}`;
  $("archive-count").textContent = `${archive.length} scene${archive.length === 1 ? "" : "s"}`;
  $("results-count").textContent = app.routeNotice || `Showing ${rows.length} of ${app.records.length} scenes`;
}

function buildCard(record, options) {
  const detailed = isDetailed(record);
  const button = element("button", [
    "scene-card",
    options.featured ? "scene-card--featured" : "",
    options.lead ? "scene-card--lead" : "",
  ].filter(Boolean).join(" "));
  button.type = "button";
  button.dataset.slug = record.slug;
  button.setAttribute(
    "aria-label",
    `${detailed ? "Open full 3D viewer for" : "Open video record for"} ${record.title}`,
  );

  const thumb = element("div", "scene-card__thumb");
  const image = element("img");
  image.src = record.thumbnail_url;
  image.alt = "";
  image.decoding = "async";
  image.loading = options.featured && options.lead ? "eager" : "lazy";
  if (options.featured && options.lead) image.fetchPriority = "high";
  image.addEventListener("error", () => {
    image.remove();
    thumb.classList.add("is-missing");
  }, { once: true });
  const evidence = methodEvidence(record);
  const summary = methodSummary(record);
  const renderedCount = Number(summary.rendered_3d || 0);
  const availabilityText = detailed
    ? renderedCount >= 3
      ? "Full multi-method 3D"
      : "Omega WebGL · partial ensemble"
    : "Video record";
  const availability = element(
    "span",
    `scene-card__availability${detailed ? " is-ready" : ""}`,
    availabilityText,
  );
  thumb.append(image, availability);

  const body = element("div", "scene-card__body");
  body.append(
    element("p", "scene-card__meta", `${record.date || "Undated"} · ${record.town || "Town not reported"}`),
    element("h3", "", record.title),
  );

  const tags = element("div", "scene-card__tags");
  if (isEnsemble(record)) {
    tags.append(tag("Full method ensemble", "hot"));
  } else if (detailed) {
    tags.append(tag("3D geometry", "hot"));
  } else {
    const vggt = record.research?.vggt_omega || {};
    const text = vggt.status === "done"
      ? "VGGT complete"
      : vggt.status === "partial_error"
        ? "VGGT partial"
        : "Catalog record";
    tags.append(tag(text, vggt.status === "partial_error" ? "warn" : ""));
  }
  if (record.annotation?.state === "manual_ground_truth") tags.append(tag("Manual edits", ""));
  const methodLimit = options.featured ? 5 : 1;
  evidence
    .filter((row) => row.capability !== "unknown")
    .slice(0, methodLimit)
    .forEach((row) => {
      const label = `${methodLabel(row.method)} · ${capabilityLabel(row.capability)}`;
      if (!Array.from(tags.children).some((node) => node.textContent === label)) {
        const chip = tag(label, capabilityTone(row.capability));
        if (row.note) chip.title = row.note;
        tags.append(chip);
      }
    });
  body.append(tags);
  if (detailed && evidence.length) {
    const coverageParts = [
      summary.rendered_3d ? `${summary.rendered_3d} rendered` : "",
      summary.numerical_only ? `${summary.numerical_only} numerical` : "",
      summary.metadata_only ? `${summary.metadata_only} metadata` : "",
      summary.failed ? `${summary.failed} failed` : "",
      summary.unavailable ? `${summary.unavailable} unavailable` : "",
    ].filter(Boolean);
    body.append(element("p", "method-coverage", coverageParts.join(" · ")));
  }

  const action = element("span", "scene-card__action");
  action.append(
    element("span", "", detailed ? "Open full research cockpit" : "Open video and edit map"),
    element("span", "", "→"),
  );
  body.append(action);

  button.append(thumb, body);
  button.addEventListener("click", () => openDetail(record, true));
  return button;
}

function tag(text, tone) {
  return element("span", `tag${tone ? ` ${tone}` : ""}`, text);
}

function openDetail(record, push) {
  app.active = record;
  if (push) {
    const url = new URL(location.href);
    url.searchParams.set("scene", record.slug);
    url.hash = "";
    history.pushState({ scene: record.slug }, "", url);
  }
  document.body.dataset.view = "detail";
  $("gallery-view").hidden = true;
  $("load-error").hidden = true;
  $("detail-view").hidden = false;
  renderDetail(record);
  window.scrollTo(0, 0);
  requestAnimationFrame(() => $("title").focus({ preventScroll: true }));
}

function showGallery(push) {
  const previousSlug = app.active?.slug;
  if (push) {
    const url = new URL(location.href);
    url.searchParams.delete("scene");
    url.hash = "";
    history.pushState({ view: "gallery" }, "", url);
  }
  clearMedia();
  app.active = null;
  document.body.dataset.view = "gallery";
  $("detail-view").hidden = true;
  $("load-error").hidden = true;
  $("gallery-view").hidden = false;
  document.title = "FPV Visual Geometry Lab V2";
  if (previousSlug) {
    requestAnimationFrame(() => {
      document.querySelector(`[data-slug="${CSS.escape(previousSlug)}"]`)?.focus({ preventScroll: true });
    });
  }
}

function renderDetail(record) {
  $("kicker").textContent = `${record.date || "Undated"} · ${record.town || "Town not reported"}`;
  $("title").textContent = record.title;
  document.title = `${record.title} · FPV Geometry Lab`;
  renderBadges(record);

  if (isDetailed(record)) {
    $("video-detail").hidden = true;
    $("viewer-panel").hidden = false;
    $("empty-viewer").hidden = true;
    $("viewer-loading").hidden = false;
    const frame = $("scene-frame");
    frame.title = `${record.title} — full reconstruction cockpit`;
    frame.src = record.research.detail_route;
    $("detail-action").href = record.research.detail_route;
    $("detail-action").querySelector("span:first-child").textContent = "Open standalone";
  } else {
    $("viewer-panel").hidden = true;
    $("viewer-loading").hidden = true;
    $("scene-frame").removeAttribute("src");
    $("video-detail").hidden = false;
    renderCatalogVideo(record);
    $("detail-action").href = record.source_player_url || record.source_record_url;
    $("detail-action").querySelector("span:first-child").textContent = "Open source page";
  }
}

function renderBadges(record) {
  const evidence = methodEvidence(record);
  const summary = methodSummary(record);
  const rendered = Number(summary.rendered_3d || 0);
  const coverage = isDetailed(record)
    ? `${rendered}/${Math.max(evidence.length, rendered)} method lanes in 3D`
    : "Video record";
  const values = [
    ["Relative only", "warn"],
    [coverage, rendered >= 3 ? "good" : "partial"],
    [annotationLabel(record.annotation?.state), ""],
  ];
  $("badges").replaceChildren(...values.map(([text, tone]) => (
    element("span", `badge${tone ? ` ${tone}` : ""}`, text)
  )));
}

function renderCatalogVideo(record) {
  const video = $("video");
  video.poster = record.thumbnail_url;
  video.src = record.video_url;
  renderTimeline(record, 0);
  renderVideoContext(record);
}

function renderTimeline(record, duration) {
  const segments = record.annotation?.segments || [];
  const lastStart = Math.max(...segments.map((segment) => Number(segment.time) || 0), 1);
  const total = Math.max(duration || 0, lastStart, 1);
  $("timeline").replaceChildren(...segments.map((segment, index) => {
    const next = segments[index + 1];
    const start = Number(segment.time) || 0;
    const end = Number(next?.time) || total;
    const span = element("span", `seg ${segment.type || "other"}`);
    span.style.flexGrow = String(Math.max(end - start, 0.1));
    span.title = `${String(segment.type || "segment").replaceAll("_", " ")} · ${start.toFixed(2)}s`;
    return span;
  }));
}

function renderVideoContext(record) {
  const vggt = record.research?.vggt_omega || {};
  const rows = [
    ["Town", record.town || "Not reported"],
    ["Annotation", annotationLabel(record.annotation?.state)],
    ["VGGT", vggt.status === "done" ? "Complete" : vggt.status === "partial_error" ? "Partial" : "Not packaged"],
    ["Scale", record.research?.scale_state || "relative_only"],
  ];
  $("state-table").replaceChildren(...rows.flatMap(([key, value]) => [
    element("dt", "", key),
    element("dd", "", value),
  ]));

  const evidence = methodEvidence(record);
  const fallback = ["Video + edit segmentation"];
  $("methods").replaceChildren(...(evidence.length ? evidence : fallback).map((row) => {
    if (typeof row === "string") return element("span", "method", row);
    const chip = element(
      "span",
      `method ${row.capability === "rendered_3d" ? "live" : row.capability}`,
      `${methodLabel(row.method)} · ${capabilityLabel(row.capability)}`,
    );
    if (row.note) chip.title = row.note;
    return chip;
  }));
  $("boundary").textContent = app.catalog.publication_boundary?.media_copied === false
    ? "Video and thumbnail are referenced from the public CDN. Extracted frames are not copied; detailed geometry is published only for approved scene bundles."
    : "Review the publication manifest before sharing this record.";
}

function clearMedia() {
  const video = $("video");
  video.pause();
  video.removeAttribute("src");
  video.removeAttribute("poster");
  video.load();
  $("scene-frame").removeAttribute("src");
  $("viewer-loading").hidden = true;
}

function showLoadError(error) {
  console.error(error);
  clearMedia();
  document.body.dataset.view = "error";
  $("gallery-view").hidden = true;
  $("detail-view").hidden = true;
  $("load-error").hidden = false;
  $("load-error-message").textContent = `The public catalog could not be loaded: ${error.message}`;
}

main().catch(showLoadError);
