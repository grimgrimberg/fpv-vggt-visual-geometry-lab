"use strict";

(async function installMethodComparisonDock(){
  const workspace=document.querySelector(".scene-workspace"),slider=document.getElementById("pose-slider");
  if(!workspace||!slider)return;
  const meta=await fetch("scene_meta.json",{cache:"no-store"}).then(r=>r.json());
  const path=meta.assets?.method_comparison?.path||"methods/method_comparison.json";
  let manifest=null;
  try{
    const response=await fetch(path,{cache:"no-store"});
    if(response.ok)manifest=await response.json();
  }catch(error){console.warn("Method comparison unavailable",error);}
  if(!manifest){
    const dock=document.createElement("section");dock.className="method-dock";dock.hidden=true;dock.setAttribute("aria-label","Method comparison unavailable");
    dock.innerHTML=`<div class="method-dock-head"><strong>Method-native I/O</strong><span>not packaged for this scene</span><button class="method-close" type="button" aria-label="Close method comparison">×</button></div><div class="method-status-copy"><h3>Comparison lane unavailable</h3><p>This scene does not currently include a public R3/LingBot/HLoc method comparison manifest. The primary VGGT geometry, camera path, original video, and scale-free trajectory panels remain available.</p></div>`;
    workspace.appendChild(dock);
    const open=()=>{dock.hidden=false;};
    document.querySelector('[data-layer="depth"]')?.addEventListener("click",open);
    document.querySelector('[data-layer="matches"]')?.addEventListener("click",open);
    document.querySelector('[data-layer="masks"]')?.addEventListener("click",open);
    dock.querySelector(".method-close").addEventListener("click",()=>{dock.hidden=true;});
    return;
  }
  if(manifest.alignment_policy!=="method_native_not_aligned")console.warn("Unexpected method alignment policy",manifest.alignment_policy);
  let consistency=null;
  const consistencyPath=manifest.diagnostics?.geometry_consistency;
  if(typeof consistencyPath==="string"){
    try{
      const url=new URL(consistencyPath,location.href);
      if(url.origin===location.origin&&url.pathname.endsWith("/diagnostics/geometry_consistency.json")){
        const response=await fetch(url,{cache:"no-store"});
        if(response.ok)consistency=await response.json();
      }
    }catch(_error){consistency=null;}
  }
  const labels={input:"Input",vggt_omega:"VGGT Omega",r3:"R3",lingbot_map:"LingBot",hloc_lightglue_colmap:"LightGlue / COLMAP",mast3r_sfm:"MASt3R"};
  const dock=document.createElement("section");dock.className="method-dock";dock.hidden=true;dock.setAttribute("aria-label","Method input and output comparison");
  dock.innerHTML=`<div class="method-dock-head"><strong>Method-native I/O</strong><span>same sample · independent outputs</span><div class="method-tabs"><button type="button" data-view="depth" class="active">Depth outputs</button><button type="button" data-view="consistency">Consistency</button><button type="button" data-view="matches">Matches / COLMAP</button><button type="button" data-view="masks">Mask status</button></div><button class="method-close" type="button" aria-label="Close method comparison">×</button></div><div class="method-content"></div><div class="method-evidence"></div>`;
  workspace.appendChild(dock);
  const content=dock.querySelector(".method-content"),evidence=dock.querySelector(".method-evidence");let view="depth";
  function nativeSupport(layerId){
    const alignment=manifest.geometry_layers?.layers?.[layerId]?.alignment;
    if(!alignment)return"native frame · alignment evidence unavailable";
    const matched=Number(alignment.matched_count),inliers=Number(alignment.inlier_count);
    const support=Number.isFinite(matched)&&Number.isFinite(inliers)?`${inliers}/${matched} camera-center inliers`:"camera support unavailable";
    return alignment.status==="available"?`aligned · ${support}`:`native only · ${support} · Omega overlay withheld`;
  }
  const reliabilityObserver=new MutationObserver(()=>{
    [["R3","r3_native"],["LingBot","lingbot_map_native"]].forEach(([label,layerId])=>{
      const target=[...content.querySelectorAll(".method-pane")].find(item=>item.querySelector("header b")?.textContent===label);
      if(!target)return;
      let note=target.querySelector(".method-reliability");
      if(!note){note=document.createElement("p");note.className="method-reliability";target.appendChild(note);}
      const message=nativeSupport(layerId);
      if(note.textContent!==message)note.textContent=message;
    });
  });
  reliabilityObserver.observe(content,{childList:true,subtree:true});
  function assetHref(value){return typeof value==="string"?value:null;}
  function link(label,href){if(!href)return;const a=document.createElement("a");a.textContent=label;a.href=href;a.target="_blank";a.rel="noopener";evidence.appendChild(a);}
  link("Geometry consistency",manifest.diagnostics?.geometry_consistency);link("Match graph",manifest.methods.hloc_lightglue_colmap?.assets?.match_graph);link("BA doctor",manifest.methods.hloc_lightglue_colmap?.assets?.ba_diagnostics);link("Failure taxonomy",manifest.methods.hloc_lightglue_colmap?.assets?.failure_taxonomy);link("Pose EKF metrics",manifest.diagnostics?.pose_ekf_metrics);link("Edit segmentation",manifest.diagnostics?.edit_segmentation);link("6DoF state space",meta.assets?.sixdof_state_space?.path);
  const shortcuts=document.createElement("div");shortcuts.className="method-native-shortcuts";
  [["r3_native","Open R3 native 3D"],["lingbot_native","Open LingBot native 3D"],["hloc_native","Open HLoc sparse 3D"]].forEach(([frame,label])=>{
    const layer=Object.values(manifest.geometry_layers?.layers||{}).find(candidate=>candidate.coordinate_frame===frame&&candidate.status==="available");
    if(!layer)return;const button=document.createElement("button");button.type="button";button.textContent=label;button.addEventListener("click",()=>{dock.hidden=true;document.dispatchEvent(new CustomEvent("fpv-select-geometry-layer",{detail:{layerId:layer.id}}));});shortcuts.appendChild(button);
  });
  if(shortcuts.childElementCount)evidence.appendChild(shortcuts);
  const caution=document.createElement("span");caution.className="warning";caution.textContent="relative only · native frames are intentionally not overlaid";evidence.appendChild(caution);
  function nearestSample(index){return manifest.samples.reduce((best,item)=>!best||Math.abs(item.frame_index-index)<Math.abs(best.frame_index-index)?item:best,null);}
  function nearestOmega(index){const samples=manifest.methods.vggt_omega?.reprojection_samples||[];return samples.reduce((best,item)=>!best||Math.abs(item.sample_index-index)<Math.abs(best.sample_index-index)?item:best,null);}
  function pane(label,status,image,note){const section=document.createElement("section");section.className=`method-pane${image?"":" empty"}`;const head=document.createElement("header"),title=document.createElement("b"),state=document.createElement("span");title.textContent=label;state.textContent=status;head.append(title,state);section.appendChild(head);if(image){const img=document.createElement("img");img.src=image;img.alt=`${label} output at the selected source sample`;section.appendChild(img);}const copy=document.createElement("p");copy.textContent=note;section.appendChild(copy);return section;}
  function renderDepth(){const index=Number(slider.value)||0,sample=nearestSample(index),omega=nearestOmega(index);if(!sample){const summary=document.createElement("div");summary.className="method-status-copy";const title=document.createElement("h3");title.textContent="Method evidence is present; comparable image samples are not published";const copy=document.createElement("p");copy.textContent="The table below separates renderable 3D geometry from numerical-only, metadata-only, and missing lanes. No source-frame image or method geometry is synthesized.";const table=document.createElement("dl");table.className="method-status-grid";[["VGGT Omega",manifest.methods.vggt_omega],["R3",manifest.methods.r3],["LingBot",manifest.methods.lingbot_map],["HLoc / COLMAP",manifest.methods.hloc_lightglue_colmap]].forEach(([label,method])=>{const term=document.createElement("dt"),value=document.createElement("dd");term.textContent=label;const state=String(method?.status||"missing").replaceAll("_"," ");const reason=method?.reason||method?.status_reason||method?.geometry_status||"";value.textContent=reason?`${state} · ${reason}`:state;table.append(term,value);});summary.append(title,copy,table);content.replaceChildren(summary);return;}const publishedVisuals=[sample.input?.image,omega?.render,manifest.methods.vggt_omega?.viewer_preview,sample.r3?.depth_visualization,sample.lingbot_map?.depth_visualization].some(assetHref);if(!publishedVisuals){const summary=document.createElement("div");summary.className="method-status-copy";const title=document.createElement("h3");title.textContent="Native depth geometry is available in 3D";const copy=document.createElement("p");copy.textContent="Rendered source-frame/depth images are intentionally not copied into this public package. Use the R3 depth and LingBot depth buttons in the compact Method Frame control above; those native point layers are real and remain separate from VGGT Omega.";const facts=document.createElement("p");facts.textContent=`Selected sample ${sample.frame_index??index} · R3 ${(sample.r3?.native_shape||[]).join("×")||"shape unavailable"} · LingBot ${(sample.lingbot_map?.native_shape||[]).join("×")||"shape unavailable"} · relative only.`;summary.append(title,copy,facts);content.replaceChildren(summary);return;}const grid=document.createElement("div");grid.className="method-compare-grid";grid.appendChild(pane(labels.input,"shared input",assetHref(sample.input?.image),sample.input?.provenance||"Source-aligned input preview unavailable."));grid.appendChild(pane(labels.vggt_omega,"primary geometry",assetHref(omega?.render)||assetHref(manifest.methods.vggt_omega?.viewer_preview),omega?"Point-cloud reprojection from the recovered Omega camera.":"Scene-level Omega point-cloud preview; per-frame reprojection unavailable."));grid.appendChild(pane(labels.r3,manifest.methods.r3?.status||"missing",assetHref(sample.r3?.depth_visualization),sample.r3?`Native R3 depth display · shape ${(sample.r3.native_shape||[]).join("×")} · relative only.`:"No R3 output at this sampled frame."));grid.appendChild(pane(labels.lingbot_map,manifest.methods.lingbot_map?.status||"missing",assetHref(sample.lingbot_map?.depth_visualization),sample.lingbot_map?`Native LingBot depth display · shape ${(sample.lingbot_map.native_shape||[]).join("×")} · confidence available · relative only.`:"No LingBot output at this sampled frame."));content.replaceChildren(grid);}
  function finiteRelative(value,digits=4){const number=Number(value);return Number.isFinite(number)?`${number.toFixed(digits)} rel`:"unavailable";}
  function renderConsistency(){
    const methods=consistency?.methods;
    if(consistency?.scale_state!=="relative_only"||!methods){
      const missing=document.createElement("div");missing.className="method-status-copy";missing.innerHTML="<h3>Quantitative consistency unavailable</h3><p>No validated relative-only comparison report is packaged for this scene. Method-native layers remain separate and no agreement score is invented.</p>";content.replaceChildren(missing);return;
    }
    const section=document.createElement("section");section.className="consistency-lab";
    const intro=document.createElement("header");intro.className="consistency-intro";
    const copy=document.createElement("div"),kicker=document.createElement("span"),title=document.createElement("h3"),note=document.createElement("p");
    kicker.textContent="CPU-only existing-artifact comparison";title.textContent="Geometry consistency lab";note.textContent="Sim(3) fits use recovered camera centers. Point residuals use a deterministic capped sample with the largest 10% of symmetric residuals trimmed. Every value stays scale-free.";
    copy.append(kicker,title,note);const hash=document.createElement("code");hash.textContent=`run ${String(consistency.parameter_hash||"unknown").slice(0,12)}`;intro.append(copy,hash);section.appendChild(intro);
    const grid=document.createElement("div");grid.className="consistency-grid";
    [["r3","R3"],["lingbot_map","LingBot"]].forEach(([name,label])=>{
      const entry=methods[name]||{},trajectory=entry.metrics?.trajectory,pointBlock=entry.metrics?.points,points=pointBlock?.status==="validated"?pointBlock.metrics:null;
      const card=document.createElement("article");card.className=`consistency-card ${entry.status==="validated"?"validated":"unavailable"}`;
      const head=document.createElement("header"),heading=document.createElement("h4"),state=document.createElement("span");heading.textContent=label;state.textContent=String(entry.status||"unavailable").replaceAll("_"," ");head.append(heading,state);card.appendChild(head);
      if(!trajectory){
        const reason=document.createElement("p");reason.textContent=entry.reason||"Quantitative camera correspondence is unavailable for this method.";card.appendChild(reason);grid.appendChild(card);return;
      }
      const inliers=Number(trajectory.alignment?.inlier_count)||0,total=Number(trajectory.alignment?.sample_count)||0,support=total?inliers/total:0;
      const supportRow=document.createElement("div");supportRow.className="consistency-support";const supportLabel=document.createElement("span");supportLabel.textContent=`trajectory support ${inliers}/${total}`;const track=document.createElement("i"),fill=document.createElement("b");fill.style.width=`${Math.max(0,Math.min(100,support*100)).toFixed(1)}%`;track.appendChild(fill);supportRow.append(supportLabel,track);card.appendChild(supportRow);
      const facts=document.createElement("dl");
      [["ATE median",finiteRelative(trajectory.ate?.median)],["RPE median",finiteRelative(trajectory.rpe?.translation_relative?.median)],["held-out median",finiteRelative(trajectory.held_out?.translation_residual?.median)],["point median",finiteRelative(points?.distance?.median)],["point p95",finiteRelative(points?.distance?.p95)]].forEach(([term,value])=>{const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=term;dd.textContent=value;facts.append(dt,dd);});
      card.appendChild(facts);grid.appendChild(card);
    });
    section.appendChild(grid);const boundary=document.createElement("p");boundary.className="consistency-boundary";boundary.textContent="Lower residuals indicate closer agreement after per-method similarity alignment; they do not carry physical units, establish accuracy, or calibrate scenes.";section.appendChild(boundary);content.replaceChildren(section);
  }
  function renderMatches(){const method=manifest.methods.hloc_lightglue_colmap||{},doctor=method.ba_doctor||{},doctorStatus=String(doctor.status||"unavailable").replaceAll("_"," "),warnings=(doctor.categories||[]).map(value=>String(value).replaceAll("_"," ")).join(", ")||"none reported",layout=document.createElement("div");layout.className="method-match-layout";const matrix=assetHref(method.assets?.match_matrix);if(matrix){const image=document.createElement("img");image.src=matrix;image.alt="LightGlue and COLMAP match matrix";layout.appendChild(image);}else{layout.classList.add("no-image");}const status=String(method.status||"missing").replaceAll("_"," "),hasSparse=Number(method.registered_images)>0&&Number(method.points3D)>0,availability=hasSparse?"Rendered match matrices are not copied publicly; the validated sparse 3D layer and numeric reconstruction diagnostics remain available.":`No validated HLoc sparse bundle is published for this scene. Status: ${status}.`;const copy=document.createElement("div");copy.className="method-match-copy";copy.innerHTML=`<h3>${labels.hloc_lightglue_colmap}</h3><p>${method.pipeline||"Sparse feature reconstruction"}</p>${matrix?"":`<p>${availability}</p>`}<dl><dt>Method status</dt><dd>${status}</dd><dt>Input frames</dt><dd>${method.input_frames??"—"}</dd><dt>Registered images</dt><dd>${method.registered_images??"—"}</dd><dt>Selected pairs</dt><dd>${method.pairs??"—"}</dd><dt>Sparse points</dt><dd>${method.points3D??"—"}</dd><dt>BA doctor</dt><dd>${doctorStatus}</dd><dt>Warnings</dt><dd>${warnings}</dd><dt>Reconstruction</dt><dd>${doctor.successful_reconstruction===true?"successful model retained":hasSparse?"sparse model retained":"unavailable"}</dd><dt>Fallback</dt><dd>${doctor.fallback_eligible?"bounded retry eligible":"not eligible"}</dd><dt>Omega alignment</dt><dd>${hasSparse?"not aligned":"unavailable"}</dd></dl>`;layout.appendChild(copy);content.replaceChildren(layout);}
  function renderMasks(){const status=document.createElement("div");status.className="method-status-copy";status.innerHTML=`<h3>Artifact-mask lane: ${manifest.masks?.status||"not run"}</h3><p>${manifest.masks?.reason||"No mask artifact is available."}</p><p>SAM2 masks will remain a separate offline explanation layer for sky, smoke/dust, overlays, title cards, and dynamic regions. This panel will never substitute a fabricated mask for a missing output.</p>`;content.replaceChildren(status);}
  function render(){dock.querySelectorAll("[data-view]").forEach(button=>button.classList.toggle("active",button.dataset.view===view));if(view==="consistency")renderConsistency();else if(view==="matches")renderMatches();else if(view==="masks")renderMasks();else renderDepth();}
  function open(next){view=next;dock.hidden=false;render();content.scrollTop=0;}
  dock.querySelector(".method-close").addEventListener("click",()=>{dock.hidden=true;});dock.querySelector(".method-tabs").addEventListener("click",event=>{if(event.target.dataset.view)open(event.target.dataset.view);});
  document.querySelector('[data-layer="depth"]')?.addEventListener("click",()=>open("depth"));document.querySelector('[data-layer="matches"]')?.addEventListener("click",()=>open("matches"));document.querySelector('[data-layer="masks"]')?.addEventListener("click",()=>open("masks"));document.querySelector('[data-layer="points"]')?.addEventListener("click",()=>{dock.hidden=true;});slider.addEventListener("input",()=>{if(!dock.hidden&&view==="depth")renderDepth();});
})();
