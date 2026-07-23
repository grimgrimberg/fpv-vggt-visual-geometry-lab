"use strict";

// Core playback updates the range input's value property directly. Browsers do
// not emit an input event for programmatic value changes, so sidecar panels
// would otherwise freeze while Play/step keeps moving the main canvas.
(function bridgeProgrammaticPlayback(){
  const slider=document.getElementById("pose-slider");
  if(!slider)return;
  let last=slider.value;
  function poll(){
    if(slider.value!==last){
      last=slider.value;
      slider.dispatchEvent(new Event("input",{bubbles:false}));
    }
    requestAnimationFrame(poll);
  }
  requestAnimationFrame(poll);
})();
