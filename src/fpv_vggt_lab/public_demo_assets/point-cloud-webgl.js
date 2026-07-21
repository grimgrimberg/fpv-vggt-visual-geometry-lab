"use strict";

(function installPointCloudWebGL(global) {
  const POSITION_DTYPE = "float32_le";
  const COLOR_DTYPE = "uint8";
  const COMPONENT_COUNT = 3;

  function requireValue(condition, message) {
    if (!condition) throw new Error(`Point-cloud contract error: ${message}`);
  }

  function finiteNumber(value, fallback) {
    return Number.isFinite(Number(value)) ? Number(value) : fallback;
  }

  function sceneDocumentUrl(sceneJsonUrl) {
    const documentBase = global.document?.baseURI || global.location?.href;
    requireValue(documentBase, "a document base URL is required");
    return new URL(sceneJsonUrl, documentBase);
  }

  function assetUrl(sceneDirectoryUrl, path, label) {
    requireValue(typeof path === "string" && path.length > 0, `${label}.path is required`);
    requireValue(!/^[A-Za-z][A-Za-z0-9+.-]*:/.test(path), `${label}.path must not use a scheme`);
    requireValue(!path.startsWith("/"), `${label}.path must not use an authority or root path`);
    requireValue(!path.includes("\\"), `${label}.path must use forward slashes`);
    requireValue(!path.includes("?") && !path.includes("#"), `${label}.path must not alias a URL`);
    const rawSegments = path.split("/");
    requireValue(rawSegments.every((segment) => segment.length > 0), `${label}.path must be canonical`);
    for (const segment of rawSegments) {
      let decoded;
      try {
        decoded = decodeURIComponent(segment);
      } catch (_error) {
        requireValue(false, `${label}.path contains malformed percent encoding`);
      }
      requireValue(decoded !== "." && decoded !== "..", `${label}.path has a dot segment`);
      requireValue(
        !decoded.includes("/") && !decoded.includes("\\"),
        `${label}.path has an encoded path separator`,
      );
      requireValue(
        !decoded.includes("?") && !decoded.includes("#"),
        `${label}.path has an encoded URL alias`,
      );
    }
    const resolved = new URL(path, sceneDirectoryUrl);
    requireValue(
      resolved.origin === sceneDirectoryUrl.origin,
      `${label}.path must remain same-origin`,
    );
    requireValue(
      resolved.href.startsWith(sceneDirectoryUrl.href),
      `${label}.path must remain inside the scene directory`,
    );
    requireValue(!resolved.search && !resolved.hash, `${label}.path must not alias a URL`);
    return resolved;
  }

  function validateAssetRecord(record, label, dtype, bytesPerComponent, pointCount) {
    requireValue(record && typeof record === "object", `${label} record is required`);
    requireValue(record.dtype === dtype, `${label}.dtype must be ${dtype}`);
    requireValue(record.components === COMPONENT_COUNT, `${label}.components must be 3`);
    requireValue(
      typeof record.sha256 === "string" && /^[0-9a-f]{64}$/.test(record.sha256),
      `${label}.sha256 must be a lowercase SHA-256 digest`,
    );
    const expectedBytes = pointCount * COMPONENT_COUNT * bytesPerComponent;
    requireValue(
      Number.isSafeInteger(record.bytes) && record.bytes === expectedBytes,
      `${label}.bytes must equal ${expectedBytes}`,
    );
    return expectedBytes;
  }

  async function sha256Hex(buffer) {
    const subtle = global.crypto?.subtle;
    requireValue(subtle && typeof subtle.digest === "function", "Web Crypto SHA-256 is unavailable");
    const digest = await subtle.digest("SHA-256", buffer);
    return [...new Uint8Array(digest)]
      .map((value) => value.toString(16).padStart(2, "0"))
      .join("");
  }

  async function fetchBuffer(url, expectedBytes, expectedSha256, label) {
    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`Point-cloud ${label} request failed (${response.status})`);
    }
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength !== expectedBytes) {
      throw new Error(
        `Point-cloud ${label} byte length mismatch: expected ${expectedBytes}, received ${buffer.byteLength}`,
      );
    }
    const actualSha256 = await sha256Hex(buffer);
    if (actualSha256 !== expectedSha256) {
      throw new Error(`Point-cloud ${label} SHA-256 mismatch`);
    }
    return buffer;
  }

  function validateDisplayTransform(displayTransform) {
    requireValue(
      displayTransform && typeof displayTransform === "object",
      "display_transform is required",
    );
    requireValue(
      Array.isArray(displayTransform.center)
        && displayTransform.center.length === COMPONENT_COUNT
        && displayTransform.center.every(Number.isFinite),
      "display_transform.center must contain three finite values",
    );
    requireValue(
      Number.isFinite(displayTransform.scale) && displayTransform.scale > 0,
      "display_transform.scale must be positive",
    );
  }

  async function loadPointCloud(sceneJsonUrl, pointCloud) {
    requireValue(pointCloud && typeof pointCloud === "object", "point_cloud is required");
    requireValue(pointCloud.scale_status === "relative_only", "scale_status must be relative_only");
    requireValue(
      Number.isSafeInteger(pointCloud.point_count) && pointCloud.point_count > 0,
      "point_count must be a positive integer",
    );
    requireValue(
      Number.isSafeInteger(pointCloud.source_point_count)
        && pointCloud.source_point_count >= pointCloud.point_count,
      "source_point_count must cover point_count",
    );
    validateDisplayTransform(pointCloud.display_transform);

    const pointCount = pointCloud.point_count;
    const positionBytes = validateAssetRecord(
      pointCloud.positions,
      "positions",
      POSITION_DTYPE,
      Float32Array.BYTES_PER_ELEMENT,
      pointCount,
    );
    const colorBytes = validateAssetRecord(
      pointCloud.colors,
      "colors",
      COLOR_DTYPE,
      Uint8Array.BYTES_PER_ELEMENT,
      pointCount,
    );
    const sceneUrl = sceneDocumentUrl(sceneJsonUrl);
    const sceneDirectoryUrl = new URL("./", sceneUrl);
    const positionsUrl = assetUrl(
      sceneDirectoryUrl,
      pointCloud.positions.path,
      "positions",
    );
    const colorsUrl = assetUrl(sceneDirectoryUrl, pointCloud.colors.path, "colors");
    const [positionBuffer, colorBuffer] = await Promise.all([
      fetchBuffer(positionsUrl, positionBytes, pointCloud.positions.sha256, "positions"),
      fetchBuffer(colorsUrl, colorBytes, pointCloud.colors.sha256, "colors"),
    ]);

    const endianProbe = new Uint8Array(new Uint32Array([0x01020304]).buffer);
    requireValue(endianProbe[0] === 0x04, "float32_le requires a little-endian browser");
    return Object.freeze({
      contract: pointCloud,
      pointCount,
      sourcePointCount: pointCloud.source_point_count,
      displayTransform: pointCloud.display_transform,
      positions: new Float32Array(positionBuffer),
      colors: new Uint8Array(colorBuffer),
    });
  }

  const VERTEX_SHADER = `
    attribute vec3 aPosition;
    attribute vec3 aColor;
    uniform vec3 uCenter;
    uniform float uNormalizationScale;
    uniform float uYaw;
    uniform float uPitch;
    uniform vec2 uAnchor;
    uniform vec2 uViewport;
    uniform float uBase;
    uniform float uDepthFactor;
    uniform float uMinimumPerspective;
    uniform float uScaleFactor;
    uniform float uZoom;
    uniform float uPointSize;
    uniform float uPixelRatio;
    varying vec3 vColor;

    void main() {
      vec3 point = (aPosition - uCenter) / uNormalizationScale;
      float cy = cos(uYaw);
      float sy = sin(uYaw);
      float cp = cos(uPitch);
      float sp = sin(uPitch);
      float rx = cy * point.x + sy * point.z;
      float rz = -sy * point.x + cy * point.z;
      float ry = cp * point.y - sp * rz;
      float viewDepth = sp * point.y + cp * rz;
      float perspective = 1.0 / max(uBase + viewDepth * uDepthFactor, uMinimumPerspective);
      float pixelScale = min(uViewport.x, uViewport.y) * uScaleFactor * uZoom * perspective;
      vec2 screen = vec2(
        uAnchor.x * uViewport.x + rx * pixelScale,
        uAnchor.y * uViewport.y - ry * pixelScale
      );
      vec2 clip = vec2(
        screen.x / uViewport.x * 2.0 - 1.0,
        1.0 - screen.y / uViewport.y * 2.0
      );
      gl_Position = vec4(clip, clamp(viewDepth * 0.02, -0.95, 0.95), 1.0);
      gl_PointSize = clamp(uPointSize * uPixelRatio * perspective, 1.0, 4.5);
      vColor = aColor;
    }
  `;

  const FRAGMENT_SHADER = `
    precision mediump float;
    uniform float uAlpha;
    varying vec3 vColor;

    void main() {
      vec2 centered = gl_PointCoord * 2.0 - 1.0;
      float radius = dot(centered, centered);
      if (radius > 1.0) discard;
      float softEdge = 1.0 - smoothstep(0.62, 1.0, radius);
      gl_FragColor = vec4(vColor, uAlpha * softEdge);
    }
  `;

  function compileShader(gl, type, source, label) {
    const shader = gl.createShader(type);
    if (!shader) throw new Error(`Point-cloud ${label} shader allocation failed`);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const detail = gl.getShaderInfoLog(shader) || "unknown compiler error";
      gl.deleteShader(shader);
      throw new Error(`Point-cloud ${label} shader compilation failed: ${detail}`);
    }
    return shader;
  }

  function linkProgram(gl) {
    let vertex = null;
    let fragment = null;
    let program = null;
    try {
      vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER, "vertex");
      fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER, "fragment");
      program = gl.createProgram();
      if (!program) throw new Error("Point-cloud shader program allocation failed");
      gl.attachShader(program, vertex);
      gl.attachShader(program, fragment);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const detail = gl.getProgramInfoLog(program) || "unknown linker error";
        throw new Error(`Point-cloud shader program link failed: ${detail}`);
      }
      return program;
    } catch (error) {
      if (program) gl.deleteProgram(program);
      throw error;
    } finally {
      if (vertex) gl.deleteShader(vertex);
      if (fragment) gl.deleteShader(fragment);
    }
  }

  function createBuffer(gl, target, data, label) {
    const buffer = gl.createBuffer();
    if (!buffer) throw new Error(`Point-cloud ${label} buffer allocation failed`);
    try {
      gl.bindBuffer(target, buffer);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      return buffer;
    } catch (error) {
      gl.deleteBuffer(buffer);
      throw error;
    }
  }

  function requireLocation(gl, program, name, attribute = false) {
    const location = attribute
      ? gl.getAttribLocation(program, name)
      : gl.getUniformLocation(program, name);
    if (location === null || location < 0) {
      throw new Error(`Point-cloud shader location missing: ${name}`);
    }
    return location;
  }

  function createPointCloudRenderer(canvas, cloud, options = {}) {
    if (!canvas || typeof canvas.getContext !== "function") {
      throw new Error("Point-cloud renderer requires a canvas");
    }
    if (!cloud || !(cloud.positions instanceof Float32Array) || !(cloud.colors instanceof Uint8Array)) {
      throw new Error("Point-cloud renderer requires loaded position and color arrays");
    }
    const gl = canvas.getContext("webgl", {
      alpha: true,
      antialias: false,
      depth: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error("WebGL is unavailable for the authorized point sample");

    let program = null;
    let positionBuffer = null;
    let colorBuffer = null;
    let locations;
    try {
      program = linkProgram(gl);
      positionBuffer = createBuffer(gl, gl.ARRAY_BUFFER, cloud.positions, "position");
      colorBuffer = createBuffer(gl, gl.ARRAY_BUFFER, cloud.colors, "color");
      locations = {
        position: requireLocation(gl, program, "aPosition", true),
        color: requireLocation(gl, program, "aColor", true),
        center: requireLocation(gl, program, "uCenter"),
        normalizationScale: requireLocation(gl, program, "uNormalizationScale"),
        yaw: requireLocation(gl, program, "uYaw"),
        pitch: requireLocation(gl, program, "uPitch"),
        anchor: requireLocation(gl, program, "uAnchor"),
        viewport: requireLocation(gl, program, "uViewport"),
        base: requireLocation(gl, program, "uBase"),
        depthFactor: requireLocation(gl, program, "uDepthFactor"),
        minimumPerspective: requireLocation(gl, program, "uMinimumPerspective"),
        scaleFactor: requireLocation(gl, program, "uScaleFactor"),
        zoom: requireLocation(gl, program, "uZoom"),
        pointSize: requireLocation(gl, program, "uPointSize"),
        pixelRatio: requireLocation(gl, program, "uPixelRatio"),
        alpha: requireLocation(gl, program, "uAlpha"),
      };
    } catch (error) {
      if (colorBuffer) gl.deleteBuffer(colorBuffer);
      if (positionBuffer) gl.deleteBuffer(positionBuffer);
      if (program) gl.deleteProgram(program);
      throw error;
    }
    let disposed = false;
    let contextLost = false;

    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    gl.clearColor(0, 0, 0, 0);

    function handleContextLost(event) {
      event.preventDefault();
      contextLost = true;
      if (typeof options.onContextLost === "function") {
        options.onContextLost(new Error("Point-cloud WebGL context was lost"));
      }
    }

    canvas.addEventListener("webglcontextlost", handleContextLost, false);

    function resize() {
      const pixelRatio = Math.min(Math.max(finiteNumber(global.devicePixelRatio, 1), 1), 2);
      const cssWidth = Math.max(1, canvas.clientWidth || canvas.width || 1);
      const cssHeight = Math.max(1, canvas.clientHeight || canvas.height || 1);
      const width = Math.max(1, Math.round(cssWidth * pixelRatio));
      const height = Math.max(1, Math.round(cssHeight * pixelRatio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      gl.viewport(0, 0, width, height);
      return { width, height, pixelRatio };
    }

    function drawRendererPointCloud(view = {}) {
      if (disposed) throw new Error("Point-cloud renderer has been disposed");
      if (contextLost || gl.isContextLost()) throw new Error("Point-cloud WebGL context is lost");
      const { width, height, pixelRatio } = resize();
      const anchor = Array.isArray(view.anchor) && view.anchor.length === 2
        ? view.anchor
        : [0.5, 0.51];
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.enableVertexAttribArray(locations.position);
      gl.vertexAttribPointer(locations.position, COMPONENT_COUNT, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.enableVertexAttribArray(locations.color);
      gl.vertexAttribPointer(locations.color, COMPONENT_COUNT, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.uniform3fv(locations.center, cloud.displayTransform.center);
      gl.uniform1f(locations.normalizationScale, cloud.displayTransform.scale);
      gl.uniform1f(locations.yaw, finiteNumber(view.yaw, -0.72));
      gl.uniform1f(locations.pitch, finiteNumber(view.pitch, 0.42));
      gl.uniform2f(locations.anchor, finiteNumber(anchor[0], 0.5), finiteNumber(anchor[1], 0.51));
      gl.uniform2f(locations.viewport, width, height);
      gl.uniform1f(locations.base, finiteNumber(view.base, 1.55));
      gl.uniform1f(locations.depthFactor, finiteNumber(view.depth, 0.18));
      gl.uniform1f(locations.minimumPerspective, finiteNumber(view.minimumPerspective, 0.62));
      gl.uniform1f(locations.scaleFactor, finiteNumber(view.scaleFactor, 0.58));
      gl.uniform1f(locations.zoom, finiteNumber(view.zoom, 1));
      gl.uniform1f(locations.pointSize, finiteNumber(view.pointSize, 1.45));
      gl.uniform1f(locations.pixelRatio, pixelRatio);
      gl.uniform1f(locations.alpha, finiteNumber(view.alpha, 0.54));
      gl.drawArrays(gl.POINTS, 0, cloud.pointCount);
      return cloud.pointCount;
    }

    function clear() {
      if (disposed || contextLost || gl.isContextLost()) return;
      resize();
      gl.clear(gl.COLOR_BUFFER_BIT);
    }

    function dispose() {
      if (disposed) return;
      disposed = true;
      canvas.removeEventListener("webglcontextlost", handleContextLost, false);
      if (!contextLost && !gl.isContextLost()) {
        gl.deleteBuffer(positionBuffer);
        gl.deleteBuffer(colorBuffer);
        gl.deleteProgram(program);
      }
    }

    return Object.freeze({
      pointCount: cloud.pointCount,
      drawPointCloud: drawRendererPointCloud,
      clear,
      dispose,
    });
  }

  function drawPointCloud(renderer, view) {
    if (!renderer || typeof renderer.drawPointCloud !== "function") {
      throw new Error("drawPointCloud requires a point-cloud renderer");
    }
    return renderer.drawPointCloud(view);
  }

  window.FpvPointCloudWebGL = Object.freeze({
    loadPointCloud,
    createPointCloudRenderer,
    drawPointCloud,
  });
})(window);
