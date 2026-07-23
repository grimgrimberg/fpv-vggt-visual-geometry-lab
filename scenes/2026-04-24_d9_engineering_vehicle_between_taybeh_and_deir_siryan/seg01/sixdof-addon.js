"use strict";

function quaternionMultiply(a,b){
  const[aw,ax,ay,az]=a,[bw,bx,by,bz]=b;
  return[aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw];
}
function normalizeQuaternion(q){
  const values=Array.isArray(q)?q.slice(0,4).map(Number):[];
  if(values.length!==4||values.some(value=>!Number.isFinite(value)))return[1,0,0,0];
  const norm=Math.hypot(...values);
  return norm>1e-12?values.map(value=>value/norm):[1,0,0,0];
}
function cameraToDisplayQuaternion(q){return normalizeQuaternion(quaternionMultiply(normalizeQuaternion(q),[0,1,0,0]));}
function orientationForDisplay(q,convention){return convention==="camera_cv_right_down_forward"?cameraToDisplayQuaternion(q):normalizeQuaternion(q);}
function quaternionEuler(q){const[w,x,y,z]=q,sinp=Math.max(-1,Math.min(1,2*(w*y-z*x)));return{roll:Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))*180/Math.PI,pitch:Math.asin(sinp)*180/Math.PI,yaw:Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))*180/Math.PI};}
function matrixQuaternion(r){
  const trace=r[0][0]+r[1][1]+r[2][2];let w,x,y,z,s;
  if(trace>0){s=Math.sqrt(trace+1)*2;w=.25*s;x=(r[2][1]-r[1][2])/s;y=(r[0][2]-r[2][0])/s;z=(r[1][0]-r[0][1])/s;}
  else if(r[0][0]>r[1][1]&&r[0][0]>r[2][2]){s=Math.sqrt(1+r[0][0]-r[1][1]-r[2][2])*2;w=(r[2][1]-r[1][2])/s;x=.25*s;y=(r[0][1]+r[1][0])/s;z=(r[0][2]+r[2][0])/s;}
  else if(r[1][1]>r[2][2]){s=Math.sqrt(1+r[1][1]-r[0][0]-r[2][2])*2;w=(r[0][2]-r[2][0])/s;x=(r[0][1]+r[1][0])/s;y=.25*s;z=(r[1][2]+r[2][1])/s;}
  else{s=Math.sqrt(1+r[2][2]-r[0][0]-r[1][1])*2;w=(r[1][0]-r[0][1])/s;x=(r[0][2]+r[2][0])/s;y=(r[1][2]+r[2][1])/s;z=.25*s;}
  return normalizeQuaternion([w,x,y,z]);
}
function activeDisplayTransform(){return window.FPVViewer?.getAlignmentMode?.()==="estimated_ground"?window.FPVViewer?.getGroundDisplayTransform?.():null;}
function transformPosition(position){const transform=activeDisplayTransform();if(!transform)return position.map(Number);const r=transform.rotation_3x3,t=transform.translation;return r.map((row,axis)=>row.reduce((sum,value,index)=>sum+value*Number(position[index]),Number(t[axis])));}
function displayPose(pose){const orientation=orientationForDisplay(pose.quaternion_wxyz,pose.orientation_convention),transform=activeDisplayTransform();return{...pose,position:transformPosition(pose.position),quaternion_wxyz:transform?normalizeQuaternion(quaternionMultiply(matrixQuaternion(transform.rotation_3x3),orientation)):orientation};}

(async function installSixDofLab(){
  const strip=document.querySelector(".analysis-strip");if(!strip)return;
  const [meta,cameras,paths,profiles]=await Promise.all([
    fetch("scene_meta.json",{cache:"no-store"}).then(r=>r.json()),
    fetch("cameras.json",{cache:"no-store"}).then(r=>r.json()),
    fetch("camera_path.json",{cache:"no-store"}).then(r=>r.json()),
    fetch("trajectory_profiles.json",{cache:"no-store"}).then(r=>r.json()),
  ]);
  let ekf=null;
  if(meta.layers?.pose_ekf?.path){try{const text=await fetch(meta.layers.pose_ekf.path,{cache:"no-store"}).then(r=>r.text()),rows=text.trim().split(/\r?\n/),head=rows.shift().split(",");ekf=rows.map(row=>{const values=row.split(","),item={};head.forEach((key,i)=>item[key]=values[i]);return item;});}catch(error){console.warn("pose EKF layer unavailable",error);}}
  const panel=document.createElement("article");panel.className="sixdof-panel";panel.innerHTML=`<div class="sixdof-head"><span>6DoF camera-proxy lab</span><select id="sixdof-layer" aria-label="6DoF translation layer"><option value="raw">Raw</option><option value="bspline">B-spline</option><option value="kalman">Kalman</option><option value="rts" selected>RTS</option>${ekf?'<option value="ekf">Pose EKF</option>':''}</select></div><div class="sixdof-body"><div class="attitude-wrap"><canvas id="attitude-canvas" aria-label="Camera orientation artificial horizon and proxy drone"></canvas><div class="proxy-label">camera pose proxy</div></div><div class="sixdof-values"><div class="sixdof-value"><span>X relative</span><b id="six-x">—</b><div class="sixdof-bar"><i id="bar-x"></i></div></div><div class="sixdof-value"><span>Y relative</span><b id="six-y">—</b><div class="sixdof-bar"><i id="bar-y"></i></div></div><div class="sixdof-value"><span>Z relative</span><b id="six-z">—</b><div class="sixdof-bar"><i id="bar-z"></i></div></div><div class="sixdof-value"><span>Roll</span><b id="six-roll">—</b></div><div class="sixdof-value"><span>Pitch</span><b id="six-pitch">—</b></div><div class="sixdof-value"><span>Yaw</span><b id="six-yaw">—</b></div><div class="sixdof-value"><span>Speed</span><b id="six-speed">—</b></div><div class="sixdof-value"><span>Acceleration</span><b id="six-accel">—</b></div><div class="sixdof-value"><span>Jerk proxy</span><b id="six-jerk">—</b></div><div class="sixdof-value"><span>Angular rate</span><b id="six-angular">—</b></div><div class="sixdof-value wide"><span>Pose / filter state</span><b id="six-state">—</b></div></div></div><div class="sixdof-foot"><strong>Relative only.</strong> Translation layers never overwrite raw poses. Camera-CV orientation is converted once, then the canonical scene display transform is applied to translation and orientation together. Raw stored poses remain unchanged. Body attitude is still unavailable without camera-to-body calibration.</div>`;
  strip.appendChild(panel);strip.classList.add("sixdof-enabled");
  const slider=document.getElementById("pose-slider"),selector=panel.querySelector("#sixdof-layer"),canvas=panel.querySelector("#attitude-canvas");
  function angleDelta(a,b){let d=a-b;while(d>180)d-=360;while(d<-180)d+=360;return d;}
  function range(values){const lo=Math.min(...values),hi=Math.max(...values);return[lo,hi,Math.max(hi-lo,1e-9)];}
  function selectedPose(index){
    const sample=cameras.samples[index],layer=selector.value;
    if(layer==="ekf"&&ekf?.[index]){
      const row=ekf[index];
      return{
        position:[Number(row.x_rel),Number(row.y_rel),Number(row.z_rel)],
        quaternion_wxyz:[Number(row.qw),Number(row.qx),Number(row.qy),Number(row.qz)],
        orientation_convention:"display_right_up_back",
        timestamp_sec:Number(row.timestamp_sec??sample.timestamp_sec),
      };
    }
    return{
      position:(paths.layers[layer]||paths.layers.raw)[index],
      quaternion_wxyz:sample.quaternion_wxyz,
      orientation_convention:"camera_cv_right_down_forward",
      timestamp_sec:Number(sample.timestamp_sec),
    };
  }
  function setBar(axis,value,ranges){const[lo,,span]=ranges[axis],ratio=Math.max(0,Math.min(1,(value-lo)/span));document.getElementById(`bar-${"xyz"[axis]}`).style.width=`${ratio*100}%`;}
  function drawAttitude(euler){const dpr=Math.min(devicePixelRatio||1,2),w=Math.max(1,canvas.clientWidth),h=Math.max(1,canvas.clientHeight);canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext("2d");c.scale(dpr,dpr);c.clearRect(0,0,w,h);c.save();c.translate(w/2,h/2);c.rotate(-euler.roll*Math.PI/180);const offset=Math.max(-h*.42,Math.min(h*.42,euler.pitch*1.05));c.fillStyle="#183744";c.fillRect(-w,-h+offset,w*2,h);c.fillStyle="#3a3021";c.fillRect(-w,offset,w*2,h);c.strokeStyle="#dcecef";c.lineWidth=1.5;c.beginPath();c.moveTo(-w,offset);c.lineTo(w,offset);c.stroke();[-20,-10,10,20].forEach(deg=>{const y=offset-deg*1.05,half=deg%20===0?30:20;c.beginPath();c.moveTo(-half,y);c.lineTo(half,y);c.stroke();});c.restore();c.strokeStyle="#54e5c2";c.lineWidth=2;c.beginPath();c.moveTo(w/2-22,h/2);c.lineTo(w/2-6,h/2);c.lineTo(w/2,h/2+6);c.lineTo(w/2+6,h/2);c.lineTo(w/2+22,h/2);c.stroke();const heading=((euler.yaw%360)+360)%360;c.fillStyle="#dcecef";c.font="9px ui-monospace,monospace";c.textAlign="center";c.fillText(`${heading.toFixed(0)}°`,w/2,13);c.save();c.translate(w/2,h/2+40);c.rotate(euler.yaw*Math.PI/180);c.strokeStyle="#f3b75f";c.lineWidth=1.4;c.beginPath();c.moveTo(0,-16);c.lineTo(-10,10);c.lineTo(0,6);c.lineTo(10,10);c.closePath();c.stroke();c.beginPath();c.moveTo(-12,-7);c.lineTo(12,7);c.moveTo(12,-7);c.lineTo(-12,7);c.stroke();c.restore();}
  function update(){const index=Math.max(0,Math.min(cameras.samples.length-1,Number(slider.value)||0)),pose=displayPose(selectedPose(index)),position=pose.position,euler=quaternionEuler(pose.quaternion_wxyz),previousPose=index?displayPose(selectedPose(index-1)):pose,previous=quaternionEuler(previousPose.quaternion_wxyz),dt=index?Math.max(pose.timestamp_sec-previousPose.timestamp_sec,1e-6):1,angular=Math.hypot(angleDelta(euler.roll,previous.roll),angleDelta(euler.pitch,previous.pitch),angleDelta(euler.yaw,previous.yaw))/dt,displayPositions=cameras.samples.map((_,sampleIndex)=>displayPose(selectedPose(sampleIndex)).position),xyzRanges=[0,1,2].map(axis=>range(displayPositions.map(point=>point[axis]))),displayFrame=activeDisplayTransform()?"estimated-ground display":"raw reconstruction display";
    ["x","y","z"].forEach((axis,i)=>{document.getElementById(`six-${axis}`).textContent=position[i].toFixed(4);setBar(i,position[i],xyzRanges);});document.getElementById("six-roll").textContent=`${euler.roll.toFixed(1)}°`;document.getElementById("six-pitch").textContent=`${euler.pitch.toFixed(1)}°`;document.getElementById("six-yaw").textContent=`${euler.yaw.toFixed(1)}°`;document.getElementById("six-speed").textContent=Number(profiles.speed_relative[index]).toFixed(4);document.getElementById("six-accel").textContent=Number(profiles.acceleration_relative[index]).toFixed(4);document.getElementById("six-jerk").textContent=Number(profiles.jerk_proxy[index]).toFixed(4);document.getElementById("six-angular").textContent=`${angular.toFixed(1)}°/s`;document.getElementById("six-state").textContent=`${selector.options[selector.selectedIndex].text} · ${displayFrame} · sample ${index+1}/${cameras.samples.length} · ${pose.timestamp_sec.toFixed(3)} source s`;drawAttitude(euler);
  }
  slider.addEventListener("input",update);selector.addEventListener("change",update);document.addEventListener("fpv-scene-ready",update);document.addEventListener("fpv-view-changed",update);addEventListener("resize",update);update();
})();
