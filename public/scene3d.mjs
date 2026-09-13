// Small WebGL renderer. No framework, runtime CDN, or server computation.
const COLORS = {
  EPG: [0.77, 0.91, 0.52],
  PEG: [0.45, 0.75, 0.67],
  PENa: [0.48, 0.7, 0.92],
  PENb: [0.67, 0.61, 0.89],
  Delta7: [0.91, 0.64, 0.4],
};
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export function cameraMatrix(yaw, pitch, distance, aspect) {
  const cy = Math.cos(yaw),
    sy = Math.sin(yaw),
    cx = Math.cos(pitch),
    sx = Math.sin(pitch);
  const model = [cy, sx * sy, -cx * sy, 0, 0, cx, sx, 0, sy, -sx * cy, cx * cy, 0, 0, 0, -distance, 1];
  const f = 1 / Math.tan(Math.PI / 8),
    near = 0.1,
    far = 100;
  const projection = [
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) / (near - far),
    -1,
    0,
    0,
    (2 * far * near) / (near - far),
    0,
  ];
  const result = new Float32Array(16);
  for (let col = 0; col < 4; col++)
    for (let row = 0; row < 4; row++)
      for (let k = 0; k < 4; k++) result[col * 4 + row] += projection[k * 4 + row] * model[col * 4 + k];
  return result;
}
export function project(point, matrix, width, height) {
  const [x, y, z] = point;
  const w = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
  if (w <= 0) return null;
  return [
    (((matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) / w) * width) / 2 + width / 2,
    height / 2 - (((matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) / w) * height) / 2,
    w,
  ];
}
export function decodeAnatomy(buffer, manifest, neurons) {
  if (
    manifest.schemaVersion !== 1 ||
    manifest.format !== 'float32-le-segments-xyzxyz' ||
    buffer.byteLength !== manifest.segmentCount * 24
  )
    throw new Error('Anatomy data do not match their manifest.');
  const ids = new Map(neurons.map((n, i) => [n.id, i]));
  const data = new Float32Array(buffer.byteLength / 4),
    view = new DataView(buffer);
  for (let i = 0; i < data.length; i++) {
    data[i] = view.getFloat32(i * 4, true);
    if (!Number.isFinite(data[i])) throw new Error('Invalid anatomy coordinate.');
  }
  let expected = 0;
  const cells = manifest.neurons.map((n) => {
    if (!ids.has(n.id) || n.start !== expected || !Number.isInteger(n.count) || n.count <= 0)
      throw new Error('Anatomy neuron mapping is invalid.');
    expected += n.count;
    return { ...n, index: ids.get(n.id) };
  });
  if (expected !== manifest.segmentCount || new Set(cells.map((n) => n.id)).size !== cells.length)
    throw new Error('Anatomy segment totals are invalid.');
  return { data, cells, manifest };
}

export class Scene3D {
  constructor(canvas, neurons, onPick = () => {}, onContext = () => {}) {
    this.canvas = canvas;
    this.neurons = neurons;
    this.onPick = onPick;
    this.onContext = onContext;
    this.gl = canvas.getContext('webgl', { alpha: true, antialias: true, powerPreference: 'low-power' });
    if (!this.gl) throw new Error('This browser does not support WebGL. The 2D view is still available.');
    this.view = 'ring';
    this.filter = 'all';
    this.selected = -1;
    this.lost = false;
    this.geometry = {};
    this.ratePixels = new Uint8Array(256 * 4);
    this.points = [];
    this.pointers = new Map();
    this.cleanups = [];
    this.initGL();
    this.buildRing();
    this.resetCamera();
    this.bindControls();
    this.listen(canvas, 'webglcontextlost', (event) => {
      event.preventDefault();
      this.lost = true;
      onContext(false);
    });
    this.listen(canvas, 'webglcontextrestored', () => {
      this.geometry = {};
      this.initGL();
      this.buildRing();
      if (this.anatomy) this.buildAnatomy();
      this.lost = false;
      onContext(true);
    });
  }
  listen(target, event, callback, options) {
    target.addEventListener(event, callback, options);
    this.cleanups.push(() => target.removeEventListener(event, callback, options));
  }
  initGL() {
    const gl = this.gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const vs = compile(
      gl.VERTEX_SHADER,
      `
      precision highp float;
      attribute vec3 aPosition; attribute vec3 aColor; attribute float aCell;
      uniform mat4 uMatrix; uniform sampler2D uRates; uniform float uPointSize; uniform float uSelected;
      varying vec3 vColor; varying float vRate; varying float vSelected;
      void main(){
        vec4 p=uMatrix*vec4(aPosition,1.0);gl_Position=p;
        vRate=aCell<0.0?0.0:texture2D(uRates,vec2((aCell+0.5)/256.0,0.5)).r;
        vSelected=abs(aCell-uSelected)<0.1?1.0:0.0;
        vColor=aColor;gl_PointSize=clamp(uPointSize*(0.75+vRate*0.65+vSelected*0.3)*4.0/p.w,2.0,60.0);
      }`,
    );
    const fs = compile(
      gl.FRAGMENT_SHADER,
      `
      precision mediump float; varying vec3 vColor; varying float vRate; varying float vSelected;
      uniform float uPoints; uniform float uOpacity;
      void main(){
        float alpha;
        if(uPoints>0.5){float d=length(gl_PointCoord-0.5)*2.0;if(d>1.0)discard;
          alpha=(exp(-d*d*8.0)*0.75+smoothstep(0.23,0.0,d)*0.8)*(0.12+0.88*vRate+vSelected*0.3);
        } else {alpha=uOpacity*(0.10+vRate*0.9+vSelected*0.7);}
        gl_FragColor=vec4(mix(vColor,vec3(1.0),vSelected*0.4),alpha);
      }`,
    );
    this.program = gl.createProgram();
    gl.attachShader(this.program, vs);
    gl.attachShader(this.program, fs);
    gl.linkProgram(this.program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error('Unable to initialize 3D shader.');
    this.attributes = Object.fromEntries(
      ['Position', 'Color', 'Cell'].map((n) => [n, gl.getAttribLocation(this.program, 'a' + n)]),
    );
    this.uniforms = Object.fromEntries(
      ['Matrix', 'Rates', 'PointSize', 'Selected', 'Points', 'Opacity'].map((n) => [
        n,
        gl.getUniformLocation(this.program, 'u' + n),
      ]),
    );
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, this.ratePixels);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.disable(gl.DEPTH_TEST);
  }
  upload(name, vertices) {
    const gl = this.gl;
    if (this.geometry[name]) gl.deleteBuffer(this.geometry[name].buffer);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    this.geometry[name] = { buffer, count: vertices.length / 7 };
  }
  vertex(out, point, index, color) {
    out.push(...point, ...(color || COLORS[this.neurons[index]?.type] || [0.25, 0.32, 0.22]), index);
  }
  buildRing() {
    const radius = { EPG: 1, PEG: 0.75, PENa: 0.86, PENb: 0.86, Delta7: 0.59 };
    const depth = { EPG: 0.16, PEG: 0, PENa: -0.17, PENb: 0.37, Delta7: -0.32 };
    this.ringPoints = this.neurons.map((n, i) => {
      const r = radius[n.type] + ((i % 3) - 1) * 0.018;
      return [Math.sin(n.angle) * r, Math.cos(n.angle) * r, depth[n.type]];
    });
    const vertices = [],
      guides = [];
    this.ringPoints.forEach((p, i) => this.vertex(vertices, p, i));
    for (const type of Object.keys(radius))
      for (let j = 0; j < 100; j++) {
        for (const a of [(j / 100) * Math.PI * 2, ((j + 1) / 100) * Math.PI * 2])
          this.vertex(
            guides,
            [Math.sin(a) * radius[type], Math.cos(a) * radius[type], depth[type]],
            -1,
            [0.34, 0.44, 0.26],
          );
      }
    for (let j = 0; j < 72; j++)
      for (const r of [1.12, 1.12 + (j % 6 === 0 ? 0.045 : 0.018)])
        this.vertex(guides, [Math.sin((j / 72) * Math.PI * 2) * r, Math.cos((j / 72) * Math.PI * 2) * r, 0.16], -1);
    this.upload('ringPoints', vertices);
    this.upload('guides', guides);
    if (this.edges) this.setEdges(this.edges);
  }
  setEdges(edges) {
    this.edges = edges;
    const vertices = [];
    // All edges are simulated. A subset is drawn to keep the ring legible.
    for (let i = 0; i < edges.length; i += 7) {
      const e = edges[i];
      this.vertex(vertices, this.ringPoints[e.source], e.source);
      this.vertex(vertices, this.ringPoints[e.target], e.source);
    }
    this.upload('edges', vertices);
  }
  setAnatomy(anatomy) {
    this.anatomy = anatomy;
    this.buildAnatomy();
  }
  buildAnatomy() {
    const vertices = [],
      points = [],
      pickPoints = [];
    const { data, cells } = this.anatomy;
    for (const cell of cells) {
      if (this.filter !== 'all' && this.neurons[cell.index].type !== this.filter) continue;
      let sx = 0,
        sy = 0,
        sz = 0;
      for (let i = cell.start; i < cell.start + cell.count; i++) {
        const a = Array.from(data.subarray(i * 6, i * 6 + 3)),
          b = Array.from(data.subarray(i * 6 + 3, i * 6 + 6));
        this.vertex(vertices, a, cell.index);
        this.vertex(vertices, b, cell.index);
        sx += a[0];
        sy += a[1];
        sz += a[2];
      }
      const p = [sx / cell.count, sy / cell.count, sz / cell.count];
      pickPoints.push({ point: p, index: cell.index });
      this.vertex(points, p, cell.index);
    }
    this.anatomyPickPoints = pickPoints;
    this.upload('anatomyLines', vertices);
    this.upload('anatomyPoints', points);
  }
  setFilter(filter) {
    this.filter = filter;
    this.selected = -1;
    if (this.anatomy) this.buildAnatomy();
  }
  setView(view) {
    this.view = view;
    this.selected = -1;
    this.resetCamera();
  }
  resetCamera() {
    this.yaw = this.view === 'ring' ? -0.25 : 0;
    this.pitch = this.view === 'ring' ? 0.43 : 0;
    this.distance = 4.25;
  }
  zoom(factor) {
    this.distance = clamp(this.distance * factor, 2.1, 8);
  }
  resize(width, height) {
    this.width = width;
    this.height = height;
    this.dpr = Math.min(devicePixelRatio || 1, matchMedia('(pointer:coarse)').matches ? 1.5 : 2);
    this.canvas.width = Math.round(width * this.dpr);
    this.canvas.height = Math.round(height * this.dpr);
  }
  bindControls() {
    const canvas = this.canvas;
    this.listen(canvas, 'pointerdown', (e) => {
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      canvas.setPointerCapture(e.pointerId);
      this.start = { x: e.clientX, y: e.clientY };
      this.dragged = false;
    });
    this.listen(canvas, 'pointermove', (e) => {
      const previous = this.pointers.get(e.pointerId);
      if (!previous) return;
      const other = [...this.pointers.entries()].find(([id]) => id !== e.pointerId)?.[1];
      if (other) {
        const old = Math.hypot(previous.x - other.x, previous.y - other.y),
          now = Math.hypot(e.clientX - other.x, e.clientY - other.y);
        if (now > 5) this.zoom(old / now);
        this.dragged = true;
      } else {
        this.yaw += (e.clientX - previous.x) * 0.008;
        this.pitch = clamp(this.pitch + (e.clientY - previous.y) * 0.008, -1.5, 1.5);
        if (Math.hypot(e.clientX - this.start.x, e.clientY - this.start.y) > 5) this.dragged = true;
      }
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    });
    const end = (e) => {
      if (e.type === 'pointerup' && !this.dragged && this.pointers.size === 1) this.pick(e.clientX, e.clientY);
      this.pointers.delete(e.pointerId);
    };
    this.listen(canvas, 'pointerup', end);
    this.listen(canvas, 'pointercancel', end);
    this.listen(canvas, 'lostpointercapture', (e) => this.pointers.delete(e.pointerId));
    // Do not trap ordinary page scrolling. Ctrl+wheel zoom or use +/- buttons.
    this.listen(
      canvas,
      'wheel',
      (e) => {
        if (e.ctrlKey) {
          e.preventDefault();
          this.zoom(Math.exp(e.deltaY * 0.002));
        }
      },
      { passive: false },
    );
    this.listen(canvas, 'keydown', (e) => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', 'Home'].includes(e.key)) return;
      e.preventDefault();
      if (e.key === 'ArrowLeft') this.yaw -= 0.15;
      if (e.key === 'ArrowRight') this.yaw += 0.15;
      if (e.key === 'ArrowUp') this.pitch = clamp(this.pitch - 0.15, -1.5, 1.5);
      if (e.key === 'ArrowDown') this.pitch = clamp(this.pitch + 0.15, -1.5, 1.5);
      if (e.key === '+' || e.key === '=') this.zoom(0.9);
      if (e.key === '-') this.zoom(1.1);
      if (e.key === 'Home') this.resetCamera();
    });
  }
  pick(x, y) {
    if (!this.matrix) return;
    const rect = this.canvas.getBoundingClientRect();
    x -= rect.left;
    y -= rect.top;
    let nearest = -1,
      best = 28;
    const points =
      this.view === 'ring' ? this.ringPoints.map((point, index) => ({ point, index })) : this.anatomyPickPoints || [];
    for (const { point, index } of points) {
      const p = project(point, this.matrix, this.width, this.height);
      if (!p) continue;
      const d = Math.hypot(x - p[0], y - p[1]);
      if (d < best) {
        best = d;
        nearest = index;
      }
    }
    this.selected = nearest;
    this.onPick(nearest);
  }
  drawGeometry(name, points = false, opacity = 1, size = 14) {
    const gl = this.gl,
      geometry = this.geometry[name];
    if (!geometry) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, geometry.buffer);
    for (const [name, count, offset] of [
      ['Position', 3, 0],
      ['Color', 3, 12],
      ['Cell', 1, 24],
    ]) {
      gl.enableVertexAttribArray(this.attributes[name]);
      gl.vertexAttribPointer(this.attributes[name], count, gl.FLOAT, false, 28, offset);
    }
    gl.uniform1f(this.uniforms.Points, points ? 1 : 0);
    gl.uniform1f(this.uniforms.Opacity, opacity);
    gl.uniform1f(this.uniforms.PointSize, size * this.dpr);
    gl.drawArrays(points ? gl.POINTS : gl.LINES, 0, geometry.count);
  }
  draw(rates) {
    if (this.lost || !this.width || !this.height) return;
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);
    this.matrix = cameraMatrix(this.yaw, this.pitch, this.distance, this.width / this.height);
    gl.uniformMatrix4fv(this.uniforms.Matrix, false, this.matrix);
    gl.uniform1f(this.uniforms.Selected, this.selected);
    for (let i = 0; i < rates.length; i++) this.ratePixels[i * 4] = Math.round(rates[i] * 255);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.ratePixels);
    gl.uniform1i(this.uniforms.Rates, 0);
    if (this.view === 'ring') {
      this.drawGeometry('guides', false, 0.65);
      this.drawGeometry('edges', false, 0.15);
      this.drawGeometry('ringPoints', true, 1, 23);
    } else if (this.anatomy) {
      this.drawGeometry('anatomyLines', false, 0.65);
      this.drawGeometry('anatomyPoints', true, 1, 9);
    }
  }
  dispose() {
    this.cleanups.forEach((fn) => fn());
    for (const { buffer } of Object.values(this.geometry)) this.gl.deleteBuffer(buffer);
    this.gl.deleteTexture(this.texture);
    this.gl.deleteProgram(this.program);
  }
}
