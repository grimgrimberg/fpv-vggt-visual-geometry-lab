"use strict";

(async function installVideoPanel() {
  const panel = document.querySelector(".pov-panel");
  const label = panel?.querySelector(".panel-label");
  if (!panel || !label) return;
  const tabs = document.createElement("div");
  tabs.className = "pov-tabs";
  const videoButton = document.createElement("button");
  videoButton.type = "button"; videoButton.textContent = "Original video"; videoButton.className = "active";
  const frameButton = document.createElement("button");
  frameButton.type = "button"; frameButton.textContent = "Pose frame";
  tabs.append(videoButton, frameButton); panel.appendChild(tabs);

  let meta;
  try { meta = await (await fetch("scene_meta.json", { cache: "no-store" })).json(); }
  catch { return; }
  const asset = meta.assets?.source_video;
  let video = null;
  if (asset?.path) {
    video = document.createElement("video");
    video.id = "source-video"; video.controls = true; video.preload = "metadata"; video.src = asset.path;
    video.setAttribute("aria-label", "Original historical recorded clip synchronized to reconstructed camera poses");
    label.insertAdjacentElement("afterend", video);
  } else {
    const pending = document.createElement("div"); pending.className = "video-pending"; pending.textContent = "Original clip has not been attached to this bundle.";
    label.insertAdjacentElement("afterend", pending);
  }
  panel.classList.add("video-mode");

  function select(mode) {
    panel.classList.toggle("video-mode", mode === "video"); panel.classList.toggle("frame-mode", mode === "frame");
    videoButton.classList.toggle("active", mode === "video"); frameButton.classList.toggle("active", mode === "frame");
    if (mode === "frame" && video && !video.paused) video.pause();
  }
  videoButton.addEventListener("click", () => select("video"));
  frameButton.addEventListener("click", () => select("frame"));

  if (!video) return;
  const slider = document.getElementById("pose-slider");
  let timestamps = [];
  try { timestamps = (await (await fetch(meta.assets.camera_path.path, { cache: "no-store" })).json()).timestamps_sec || []; }
  catch { return; }
  let syncingFromVideo = false;
  let syncingFromSlider = false;
  let sliderSeekTimer = null;
  slider.addEventListener("input", () => {
    if (syncingFromVideo || !timestamps.length || !Number.isFinite(video.duration)) return;
    const time = Number(timestamps[Number(slider.value)]);
    if (Number.isFinite(time) && Math.abs(video.currentTime - time) > .08) {
      syncingFromSlider = true;
      clearTimeout(sliderSeekTimer);
      video.currentTime = Math.min(Math.max(time, 0), video.duration);
      sliderSeekTimer = setTimeout(() => { syncingFromSlider = false; }, 1200);
    }
  });
  function syncSliderFromVideo(force = false) {
    if (!timestamps.length || video.seeking) return;
    if (syncingFromSlider) return;
    if (!force && video.paused) return;
    let best = 0, distance = Infinity;
    for (let index = 0; index < timestamps.length; index += 1) {
      const candidate = Math.abs(Number(timestamps[index]) - video.currentTime);
      if (candidate < distance) { distance = candidate; best = index; }
    }
    if (Number(slider.value) !== best) {
      syncingFromVideo = true; slider.value = String(best); slider.dispatchEvent(new Event("input", { bubbles: true })); syncingFromVideo = false;
    }
  }
  video.addEventListener("timeupdate", () => syncSliderFromVideo(false));
  video.addEventListener("seeked", () => {
    if (syncingFromSlider) {
      syncingFromSlider = false; clearTimeout(sliderSeekTimer); return;
    }
    syncSliderFromVideo(true);
  });
})();
