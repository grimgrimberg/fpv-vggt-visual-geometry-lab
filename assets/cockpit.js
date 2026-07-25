"use strict";

const app = {
  catalog: null,
  records: [],
  active: null,
  activeScene: null,
  filter: "all",
  routeNotice: "",
};

const FEATURED_LIMIT = 3;

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
  if (["auto", "auto_generated", "auto_candidate"].includes(state)) {
    return "Auto edit candidate";
  }
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

function sceneOptions(record) {
  const research = record.research || {};
  const routes = Array.isArray(research.detail_routes) && research.detail_routes.length
    ? research.detail_routes.filter((route) => typeof route === "string" && route)
    : research.detail_route
      ? [research.detail_route]
      : [];
  const sceneIds = Array.isArray(research.scene_ids) ? research.scene_ids : [];
  return routes.map((route, index) => {
    const routeParts = route.split("/").filter(Boolean);
    const fallbackId = routeParts.at(-2) || `scene-${index + 1}`;
    const sceneId = String(sceneIds[index] || fallbackId);
    return {
      route,
      sceneId,
      label: routes.length > 1 ? `Scene ${index + 1} · ${sceneId}` : sceneId,
    };
  });
}

function selectedScene(record, requestedSceneId = "") {
  const options = sceneOptions(record);
  return options.find((option) => option.sceneId === requestedSceneId)
    || options.find((option) => option.route === record.research?.detail_route)
    || options[0]
    || null;
}

function sceneHref(record, sceneId = "") {
  const url = new URL(location.href);
  url.searchParams.set("scene", record.slug);
  url.hash = "";
  const selected = selectedScene(record, sceneId);
  if (selected && sceneOptions(record).length > 1) {
    url.searchParams.set("segment", selected.sceneId);
  } else {
    url.searchParams.delete("segment");
  }
  return `${url.pathname}${url.search}`;
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
  $("segment-select").addEventListener("change", () => {
    if (app.active) openDetail(app.active, true, $("segment-select").value);
  });
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
  const url = new URL(location.href);
  const slug = url.searchParams.get("scene");
  const requestedSceneId = url.searchParams.get("segment") || "";
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
  openDetail(record, false, requestedSceneId);
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

function isHeroEligible(record) {
  return isDetailed(record) && record.research?.promotion_eligible === true;
}

function isExperimentalCase(record) {
  return record.research?.publication_role === "experimental_case_study";
}

function curatedHeroRank(record) {
  const rank = Number(record.research?.hero_rank);
  return Number.isInteger(rank) && rank > 0 ? rank : Number.MAX_SAFE_INTEGER;
}

function featuredRank(record) {
  const summary = methodSummary(record);
  const rendered = Number(summary.rendered_3d || 0);
  const evidence = methodEvidence(record).length;
  const quality = Number(record.research?.hero_score || 0);
  const confidence = Number(record.research?.score_confidence || 0);
  return quality * 100 + confidence * 20
    + (isEnsemble(record) ? 100 : 0) + rendered * 10 + evidence;
}

function renderGallery() {
  const rows = app.records.filter(passes);
  const featured = rows
    .filter(isHeroEligible)
    .sort((left, right) => (
      curatedHeroRank(left) - curatedHeroRank(right)
      || featuredRank(right) - featuredRank(left)
    ))
    .slice(0, FEATURED_LIMIT);
  const featuredSlugs = new Set(featured.map((record) => record.slug));
  const detailed = rows.filter(
    (record) => isDetailed(record) && !featuredSlugs.has(record.slug),
  );
  const archive = rows.filter((record) => !isDetailed(record));

  $("featured-list").replaceChildren(...featured.map((record, index) => buildCard(record, {
    featured: true,
    lead: index === 0,
  })));
  $("detailed-list").replaceChildren(...detailed.map((record) => buildCard(record, {
    featured: false,
    lead: false,
  })));
  $("scene-list").replaceChildren(...archive.map((record) => buildCard(record, {
    featured: false,
    lead: false,
  })));

  $("featured-section").hidden = featured.length === 0;
  $("detailed-section").hidden = detailed.length === 0;
  $("archive-section").hidden = archive.length === 0;
  $("no-results").hidden = rows.length !== 0;
  $("featured-count").textContent = `${featured.length} reviewed scene${featured.length === 1 ? "" : "s"}`;
  $("detailed-count").textContent = `${detailed.length} full 3D scene${detailed.length === 1 ? "" : "s"}`;
  $("archive-count").textContent = `${archive.length} video record${archive.length === 1 ? "" : "s"}`;
  $("results-count").textContent = app.routeNotice || `Showing ${rows.length} of ${app.records.length} scenes`;
}

function buildCard(record, options) {
  const detailed = isDetailed(record);
  const defaultScene = selectedScene(record);
  const link = element("a", [
    "scene-card",
    options.featured ? "scene-card--featured" : "",
    options.lead ? "scene-card--lead" : "",
  ].filter(Boolean).join(" "));
  link.href = sceneHref(record, defaultScene?.sceneId || "");
  link.dataset.slug = record.slug;
  link.setAttribute(
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
  let availabilityText = "Video record";
  if (detailed && isExperimentalCase(record)) {
    availabilityText = "Experimental 3D · unreviewed";
  } else if (detailed && renderedCount >= 3) {
    availabilityText = isHeroEligible(record)
      ? "Full multi-method 3D"
      : "Full multi-method 3D · archive";
  } else if (detailed) {
    availabilityText = isHeroEligible(record)
      ? "Omega WebGL · partial ensemble"
      : "3D reconstruction · archive";
  }
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
  const semanticTags = [];
  if (isEnsemble(record)) {
    semanticTags.push(["Multi-method 3D", "hot"]);
  } else if (detailed) {
    semanticTags.push(["3D geometry", "hot"]);
  } else {
    const vggt = record.research?.vggt_omega || {};
    const text = vggt.status === "done"
      ? "VGGT complete"
      : vggt.status === "partial_error"
        ? "VGGT partial"
        : "Catalog record";
    semanticTags.push([text, vggt.status === "partial_error" ? "warn" : ""]);
  }
  if (detailed && isExperimentalCase(record)) {
    semanticTags.push(["Experimental case study", "warn"]);
  } else if (detailed && !isHeroEligible(record)) {
    semanticTags.push(["Archive reconstruction", "data"]);
  } else if (record.annotation?.state === "manual_ground_truth") {
    semanticTags.push(["Manual edits", ""]);
  }
  if (
    record.annotation?.state === "manual_ground_truth"
    && semanticTags.length < 2
  ) {
    semanticTags.push(["Manual edits", ""]);
  }
  semanticTags
    .slice(0, 2)
    .forEach(([text, tone]) => tags.append(tag(text, tone)));
  body.append(tags);

  if (detailed && evidence.length) {
    const available = evidence.filter((row) => row.capability !== "unknown");
    if (available.length) {
      const roster = element("div", "method-roster");
      const accessible = available.map(
        (row) => `${methodLabel(row.method)}: ${capabilityLabel(row.capability)}`,
      );
      roster.setAttribute("aria-label", `Method evidence: ${accessible.join(", ")}`);
      roster.title = accessible.join(" · ");
      roster.append(...available.map((row) => {
        const item = element("span", "", methodLabel(row.method));
        item.dataset.capability = row.capability;
        return item;
      }));
      body.append(roster);
      body.append(
        element(
          "p",
          "method-coverage",
          `${renderedCount}/${available.length} method lanes rendered in 3D`,
        ),
      );
    }
  }
  const action = element("span", "scene-card__action");
  action.append(
    element("span", "", detailed ? "Open full research cockpit" : "Open video and edit map"),
    element("span", "", "→"),
  );
  body.append(action);

  link.append(thumb, body);
  link.addEventListener("click", (event) => {
    if (
      event.defaultPrevented
      || event.button !== 0
      || event.metaKey
      || event.ctrlKey
      || event.shiftKey
      || event.altKey
    ) return;
    event.preventDefault();
    openDetail(record, true, defaultScene?.sceneId || "");
  });
  return link;
}

function tag(text, tone) {
  return element("span", `tag${tone ? ` ${tone}` : ""}`, text);
}

function openDetail(record, push, requestedSceneId = "") {
  const selected = selectedScene(record, requestedSceneId);
  app.active = record;
  app.activeScene = selected;
  if (push) {
    const url = new URL(sceneHref(record, selected?.sceneId || ""), location.href);
    history.pushState(
      { scene: record.slug, segment: selected?.sceneId || null },
      "",
      url,
    );
  }
  document.body.dataset.view = "detail";
  $("gallery-view").hidden = true;
  $("load-error").hidden = true;
  $("detail-view").hidden = false;
  renderDetail(record, selected);
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
  app.activeScene = null;
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

function embeddedRoute(route) {
  const url = new URL(route, window.location.href);
  url.searchParams.set("embed", "1");
  return url.href;
}

function renderDetail(record, selected) {
  $("kicker").textContent = `${record.date || "Undated"} · ${record.town || "Town not reported"}`;
  $("title").textContent = record.title;
  document.title = `${record.title} · FPV Geometry Lab`;
  renderBadges(record);
  renderSegmentControl(record, selected);

  if (isDetailed(record) && selected) {
    $("video-detail").hidden = true;
    $("viewer-panel").hidden = false;
    $("empty-viewer").hidden = true;
    $("viewer-loading").hidden = false;
    const frame = $("scene-frame");
    frame.title = `${record.title} — ${selected.label} reconstruction cockpit`;
    frame.src = embeddedRoute(selected.route);
    $("detail-action").href = selected.route;
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

function renderSegmentControl(record, selected) {
  const options = sceneOptions(record);
  const control = $("segment-control");
  const select = $("segment-select");
  control.hidden = options.length <= 1;
  select.replaceChildren(...options.map((option) => {
    const node = element("option", "", option.label);
    node.value = option.sceneId;
    return node;
  }));
  if (selected) select.value = selected.sceneId;
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
