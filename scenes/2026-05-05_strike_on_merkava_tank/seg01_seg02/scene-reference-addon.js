"use strict";

(async function installReferenceSceneControls(){
  const workspace=document.querySelector(".scene-workspace"),timeline=document.querySelector(".timeline-panel"),head=document.querySelector(".timeline-head");
  if(!workspace||!timeline||!head)return;
  const [meta,pathData,cameraData,profiles]=await Promise.all([
    fetch("scene_meta.json",{cache:"no-store"}).then(r=>r.json()),
    fetch("camera_path.json",{cache:"no-store"}).then(r=>r.json()),
    fetch("cameras.json",{cache:"no-store"}).then(r=>r.json()),
    fetch("trajectory_profiles.json",{cache:"no-store"}).then(r=>r.json()),
  ]);
  const reprojection=(meta.assets?.reprojection?.samples||[]).filter(sample=>["overlay","render","actual"].some(mode=>typeof sample?.[mode]==="string"&&sample[mode]));
  const availableModes=["overlay","render","actual"].filter(mode=>reprojection.some(sample=>typeof sample?.[mode]==="string"&&sample[mode]));
  const overlay=document.createElement("canvas");overlay.id="scene-overlay-canvas";overlay.setAttribute("aria-label","Canvas fallback camera references");workspace.appendChild(overlay);
  const legend=document.createElement("div");legend.className="camera-legend";legend.textContent="canvas fallback camera references";workspace.appendChild(legend);
  const groundAvailable=meta.display_alignment?.estimated_ground_available===true&&meta.assets?.ground_alignment?.status==="estimated";
  const settings={cameras:true,grid:groundAvailable,frames:reprojection.length>0,yaw:-.7,pitch:.48,zoom:1};
  const controls=document.createElement("div");controls.className="reference-controls";
  const toggleDefinitions=[["points","Points",true],["path","Path",true],["cameras","Cameras",true]];
  if(groundAvailable)toggleDefinitions.push(["grid","Ground plane",true]);
  if(reprojection.length)toggleDefinitions.push(["frames","Reprojection",true]);
  toggleDefinitions.forEach(([key,label,checked])=>{
    const wrap=document.createElement("label"),input=document.createElement("input");input.type="checkbox";input.checked=checked;input.dataset.referenceToggle=key;wrap.append(input,document.createTextNode(label));controls.appendChild(wrap);
  });
  head.appendChild(controls);
  controls.addEventListener("change",event=>{
    const key=event.target.dataset.referenceToggle,checked=event.target.checked;
    if(key==="points"){const button=document.querySelector('[data-layer="points"]');if(button&&(button.getAttribute("aria-pressed")==="true")!==checked)button.click();}
    else if(key==="path"){["raw","rts"].forEach(layer=>{const button=document.querySelector(`[data-layer="${layer}"]`);if(button&&(button.getAttribute("aria-pressed")==="true")!==checked)button.click();});}
    else if(key==="frames"){settings.frames=checked;document.querySelector(".reprojection-inset")?.toggleAttribute("hidden",!checked);}
    else{settings[key]=checked;document.dispatchEvent(new CustomEvent("fpv-reference-visibility",{detail:{cameras:settings.cameras,grid:settings.grid}}));drawOverlay();}
  });

  let activeMode=availableModes[0]||"overlay",inset=null,tabs=null,counter=null,image=null;
  if(reprojection.length){
    inset=document.createElement("div");inset.className="reprojection-inset";
    tabs=document.createElement("div");tabs.className="reprojection-tabs";
    availableModes.forEach(mode=>{const button=document.createElement("button");button.type="button";button.textContent=mode[0].toUpperCase()+mode.slice(1);button.dataset.mode=mode;button.classList.toggle("active",mode===activeMode);tabs.appendChild(button);});
    counter=document.createElement("span");counter.className="reprojection-count";tabs.appendChild(counter);
    image=document.createElement("img");image.id="reprojection-image";image.alt="VGGT reprojection comparison at the active camera pose";inset.append(tabs,image);workspace.appendChild(inset);
    tabs.addEventListener("click",event=>{if(!event.target.dataset.mode)return;activeMode=event.target.dataset.mode;tabs.querySelectorAll("button").forEach(button=>button.classList.toggle("active",button.dataset.mode===activeMode));updateInset();});
  }
  function nearestSample(index){return reprojection.reduce((best,item)=>!best||Math.abs(item.sample_index-index)<Math.abs(best.sample_index-index)?item:best,null);}
  function updateInset(){if(!inset)return;const index=Number(document.getElementById("pose-slider").value),sample=nearestSample(index),source=sample?.[activeMode];if(!sample||!source){inset.hidden=true;return;}inset.hidden=!settings.frames;image.src=source;image.alt=`${activeMode} view at ${Number(sample.timestamp_sec).toFixed(3)} source seconds`;counter.textContent=`${sample.sample_index+1}/${pathData.timestamps_sec.length}`;}

  function sizeCanvas(canvas){const dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*dpr)),h=Math.max(1,Math.round(canvas.clientHeight*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}return{w,h};}
  function displayBounds(){return window.FPVViewer?.getDisplayBounds?.()||meta.bounds;}
  function displayPoint(point){return window.FPVViewer?.toDisplayPoint?.(point)||point.map(Number);}
  function projectDisplay(point,w,h){const b=displayBounds(),c=b.min.map((v,i)=>(v+b.max[i])*.5),span=Math.max(...b.max.map((v,i)=>v-b.min[i]),1e-6);let x=point[0]-c[0],y=point[1]-c[1],z=point[2]-c[2];const cy=Math.cos(settings.yaw),sy=Math.sin(settings.yaw),cp=Math.cos(settings.pitch),sp=Math.sin(settings.pitch),rx=cy*x+sy*z,rz=-sy*x+cy*z,ry=cp*y-sp*rz,scale=Math.min(w,h)*.72*settings.zoom/span;return[w*.5+rx*scale,h*.53-ry*scale];}
  function project(point,w,h){return projectDisplay(displayPoint(point),w,h);}
  function groundFrameActive(){return window.FPVViewer?.getAlignmentMode?.()==="estimated_ground"&&Boolean(window.FPVViewer?.getGroundDisplayTransform?.());}
  function updateFrameLabels(){const estimated=groundFrameActive();const frame=estimated?"estimated_ground_y_up":"raw_reconstruction";const label=estimated?"estimated-ground display · Y up · plane Y=0":"raw reconstruction axes · no ground plane";legend.dataset.coordinateFrame=frame;legend.textContent=`canvas fallback · ${label} · camera references`;const key=document.querySelector(".axis-key");if(key){key.dataset.coordinateFrame=frame;key.removeAttribute("aria-hidden");key.setAttribute("aria-label",label);let frameLabel=key.querySelector(".axis-frame-label");if(!frameLabel){frameLabel=document.createElement("span");frameLabel.className="axis-frame-label";key.appendChild(frameLabel);}frameLabel.textContent=` · ${label}`;}const toggle=controls.querySelector('[data-reference-toggle="grid"]');if(toggle){toggle.disabled=!estimated;toggle.closest("label")?.toggleAttribute("data-unavailable",!estimated);}}
  function rotate(q,v){const [qw,qx,qy,qz]=q,uv=[qy*v[2]-qz*v[1],qz*v[0]-qx*v[2],qx*v[1]-qy*v[0]],uuv=[qy*uv[2]-qz*uv[1],qz*uv[0]-qx*uv[2],qx*uv[1]-qy*uv[0]];return[v[0]+2*(qw*uv[0]+uuv[0]),v[1]+2*(qw*uv[1]+uuv[1]),v[2]+2*(qw*uv[2]+uuv[2])];}
  function add(a,b,s=1){return[a[0]+b[0]*s,a[1]+b[1]*s,a[2]+b[2]*s];}
  function drawOverlay(){if(document.body.classList.contains("webgl-ready"))return;const ctx=overlay.getContext("2d"),{w,h}=sizeCanvas(overlay);ctx.clearRect(0,0,w,h);const b=displayBounds(),span=Math.max(...b.max.map((v,i)=>v-b.min[i]),1e-6);
    if(settings.grid&&groundFrameActive()){ctx.strokeStyle="rgba(70,105,119,.34)";ctx.lineWidth=1;const steps=14;for(let i=0;i<=steps;i++){const t=i/steps,x=b.min[0]+(b.max[0]-b.min[0])*t,z=b.min[2]+(b.max[2]-b.min[2])*t;[[[x,0,b.min[2]],[x,0,b.max[2]]],[[b.min[0],0,z],[b.max[0],0,z]]].forEach(line=>{const p0=projectDisplay(line[0],w,h),p1=projectDisplay(line[1],w,h);ctx.beginPath();ctx.moveTo(...p0);ctx.lineTo(...p1);ctx.stroke();});}}
    if(settings.cameras){ctx.strokeStyle="rgba(243,183,95,.74)";ctx.lineWidth=1;const stride=Math.max(1,Math.floor(cameraData.samples.length/18)),depth=span*.045;for(let i=0;i<cameraData.samples.length;i+=stride){const s=cameraData.samples[i],p=s.position,q=s.quaternion_wxyz,forward=rotate(q,[0,0,1]),right=rotate(q,[1,0,0]),down=rotate(q,[0,1,0]),base=add(p,forward,depth),corners=[add(add(base,right,depth*.58),down,depth*.36),add(add(base,right,-depth*.58),down,depth*.36),add(add(base,right,-depth*.58),down,-depth*.36),add(add(base,right,depth*.58),down,-depth*.36)],pc=project(p,w,h),screen=corners.map(c=>project(c,w,h));screen.forEach(c=>{ctx.beginPath();ctx.moveTo(...pc);ctx.lineTo(...c);ctx.stroke();});ctx.beginPath();ctx.moveTo(...screen[0]);screen.slice(1).forEach(c=>ctx.lineTo(...c));ctx.closePath();ctx.stroke();}}
  }
  const sceneCanvas=document.getElementById("scene-canvas");sceneCanvas.addEventListener("pointerdown",e=>{settings.last=[e.clientX,e.clientY];});sceneCanvas.addEventListener("pointermove",e=>{if(!settings.last||!(e.buttons&1))return;settings.yaw+=(e.clientX-settings.last[0])*.007;settings.pitch=Math.max(-1.45,Math.min(1.45,settings.pitch+(e.clientY-settings.last[1])*.007));settings.last=[e.clientX,e.clientY];drawOverlay();});sceneCanvas.addEventListener("pointerup",()=>{settings.last=null;});sceneCanvas.addEventListener("wheel",e=>{settings.zoom=Math.max(.25,Math.min(6,settings.zoom*Math.exp(-e.deltaY*.001)));drawOverlay();},{passive:true});sceneCanvas.addEventListener("dblclick",()=>{settings.yaw=-.7;settings.pitch=.48;settings.zoom=1;drawOverlay();});

  const speedWrap=document.createElement("div");speedWrap.className="speed-mini";const speedLabel=document.createElement("div");speedLabel.className="speed-mini-label";speedLabel.textContent="Speed (relative)";const speedCanvas=document.createElement("canvas");speedWrap.append(speedLabel,speedCanvas);timeline.appendChild(speedWrap);timeline.classList.add("reference-expanded");
  function drawSpeed(){const ctx=speedCanvas.getContext("2d"),{w,h}=sizeCanvas(speedCanvas),values=profiles.speed_relative,max=Math.max(...values,1e-8),pad=8;ctx.clearRect(0,0,w,h);ctx.strokeStyle="rgba(120,144,154,.25)";ctx.beginPath();ctx.moveTo(pad,h-10);ctx.lineTo(w-pad,h-10);ctx.stroke();ctx.strokeStyle="#54e5c2";ctx.lineWidth=1;ctx.beginPath();values.forEach((v,i)=>{const x=pad+(w-pad*2)*i/Math.max(values.length-1,1),y=h-10-(h-20)*v/max;if(i===0||profiles.segment_boundaries?.[i])ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();const index=Number(document.getElementById("pose-slider").value),x=pad+(w-pad*2)*index/Math.max(values.length-1,1);ctx.strokeStyle="#f3b75f";ctx.beginPath();ctx.moveTo(x,5);ctx.lineTo(x,h-7);ctx.stroke();}
  const slider=document.getElementById("pose-slider");slider.addEventListener("input",()=>{updateInset();drawSpeed();});document.addEventListener("fpv-scene-ready",()=>{updateFrameLabels();drawOverlay();});document.addEventListener("fpv-view-changed",()=>{updateFrameLabels();drawOverlay();});addEventListener("resize",()=>{drawOverlay();drawSpeed();});updateFrameLabels();updateInset();drawOverlay();drawSpeed();
})();
