// dev/creator=tubakhxn
const MP_VERSION = '0.10.18';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const G = 7;
const FOCAL = 4.5;

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const damp = (rate, dt) => 1 - Math.exp(-rate * dt);

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

function orthonormalize(f) {
  const ey = norm(f.ey);
  const d = dot(f.ez, ey);
  const ez = norm([f.ez[0] - ey[0] * d, f.ez[1] - ey[1] * d, f.ez[2] - ey[2] * d]);
  f.ey = ey; f.ez = ez; f.ex = cross(ey, ez);
}

function eulerFrame(pitch, yaw, roll) {
  const [cx, sx] = [Math.cos(pitch), Math.sin(pitch)];
  const [cy, sy] = [Math.cos(yaw), Math.sin(yaw)];
  const [cz, sz] = [Math.cos(roll), Math.sin(roll)];
  const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
  const Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]];
  const Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]];
  const mm = (A, B) => A.map((r) => [0, 1, 2].map((c) => r[0] * B[0][c] + r[1] * B[1][c] + r[2] * B[2][c]));
  const R = mm(Rz, mm(Rx, Ry));
  return { ex: [R[0][0], R[1][0], R[2][0]], ey: [R[0][1], R[1][1], R[2][1]], ez: [R[0][2], R[1][2], R[2][2]] };
}

const video = document.getElementById('cam');
const glc = document.getElementById('gl');
const c2 = document.getElementById('c');
const ctx = c2.getContext('2d');
const elParams = document.getElementById('params');
const elHand = document.getElementById('hand');
const elStatus = document.getElementById('status');

let W = innerWidth, H = innerHeight, DPR = 1;

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let info = { rho0: 0, radius: 0.052, sigma: 0.04, damping: 0.059, maxSpeed: 7, spacing: 0.085 };
let pending = null;
let pendingN = 0;
let liveN = 0;
worker.onmessage = (e) => {
  const m = e.data;
  if (m.t === 'info') { info = m; elParams.textContent = paramsText(); }
  else if (m.t === 'pos') {
    if (pending) worker.postMessage({ t: 'ret', buf: pending }, [pending.buffer]);
    pending = m.buf; pendingN = m.n;
  }
};

const gl = glc.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, powerPreference: 'high-performance' });
let fbo = null, fboTex = null, fboW = 0, fboH = 0, useFloat = true;
let splatProg, resolveProg, posBuf, vao, resolveVao;
const U = {};
const UD = {};
let dotProg;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(vs, fs) {
  const p = gl.createProgram();
  gl.bindAttribLocation(p, 0, 'aCorner');
  gl.bindAttribLocation(p, 1, 'aPos');
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

const SPLAT_VS = `#version 300 es
in vec2 aCorner;
in vec3 aPos;
uniform mat3 uR; uniform vec2 uCenter; uniform vec2 uRes;
uniform float uSc; uniform float uRad; uniform float uFocal;
out vec2 vUv; out float vZ;
void main() {
  vec3 w = uR * aPos;
  float p = uFocal / (uFocal - w.z);
  vec2 s = uCenter + vec2(w.x, -w.y) * uSc * p + aCorner * uRad * uSc * p;
  gl_Position = vec4(s.x / uRes.x * 2.0 - 1.0, 1.0 - s.y / uRes.y * 2.0, 0.0, 1.0);
  vUv = aCorner; vZ = w.z;
}`;
const SPLAT_FS = `#version 300 es
precision highp float;
in vec2 vUv; in float vZ;
uniform float uW;
out vec4 o;
void main() {
  float d2 = dot(vUv, vUv);
  if (d2 > 1.0) discard;
  float k = 1.0 - d2; k *= k;
  float z01 = clamp(vZ * 0.5 + 0.5, 0.0, 1.0);
  o = vec4(k * uW, k * uW * z01, 0.0, 0.0);
}`;
const DOT_FS = `#version 300 es
precision highp float;
in vec2 vUv; in float vZ;
out vec4 o;
void main() {
  float d = length(vUv);
  if (d > 1.0) discard;
  float ring = smoothstep(0.5, 0.78, d) * (1.0 - smoothstep(0.84, 1.0, d));
  float core = 1.0 - smoothstep(0.0, 0.7, d);
  float z = clamp(vZ * 0.5 + 0.5, 0.0, 1.0);
  float a = (0.22 * core + 0.6 * ring) * (0.3 + 0.7 * z);
  vec3 c = mix(vec3(0.8, 0.07, 0.03), vec3(1.0, 0.27, 0.09), z);
  o = vec4(c * a, a);
}`;
const RES_VS = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const RES_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex; uniform vec2 uTexel; uniform float uTime;
out vec4 o;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  vec4 t = texture(uTex, vUv);
  float f = t.r;
  float zc = t.g / max(f, 1e-4);

  vec2 e = uTexel * 2.0;
  float fx = texture(uTex, vUv + vec2(e.x, 0.0)).r - texture(uTex, vUv - vec2(e.x, 0.0)).r;
  float fy = texture(uTex, vUv + vec2(0.0, e.y)).r - texture(uTex, vUv - vec2(0.0, e.y)).r;
  vec2 gr = vec2(fx, fy) / (f + 0.25);
  vec3 N = normalize(vec3(-gr.x * 1.6, gr.y * 1.6, 1.0));
  vec3 L = normalize(vec3(-0.45, 0.6, 0.65));
  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(reflect(-L, N), vec3(0.0, 0.0, 1.0)), 0.0), 38.0);

  float mask = smoothstep(0.16, 0.26, f);
  float thick = 1.0 - exp(-f * 0.30);
  vec3 col = mix(vec3(1.0, 0.26, 0.07), vec3(0.55, 0.02, 0.01), thick);
  col *= 0.72 + 0.5 * zc;
  col *= 0.62 + 0.75 * diff;
  col += spec * vec3(1.0, 0.45, 0.3) * 0.4;
  float rim = (1.0 - smoothstep(0.22, 0.75, f)) * mask;
  col += vec3(1.0, 0.22, 0.06) * rim * 0.35;
  col *= 0.96 + 0.08 * hash(gl_FragCoord.xy + uTime);

  float a = mask * (0.72 + 0.22 * thick);
  float halo = smoothstep(0.0, 0.12, f) * (1.0 - mask) * 0.12;
  o = vec4(col * a + vec3(1.0, 0.22, 0.07) * halo, a + halo * 0.6);
}`;

function makeFbo() {
  fboW = Math.max(16, Math.floor(glc.width * 0.5));
  fboH = Math.max(16, Math.floor(glc.height * 0.5));
  if (fboTex) gl.deleteTexture(fboTex);
  if (fbo) gl.deleteFramebuffer(fbo);
  fboTex = gl.createTexture();
  fbo = gl.createFramebuffer();
  gl.bindTexture(gl.TEXTURE_2D, fboTex);
  const tryFmt = (internal, type) => {
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, fboW, fboH, 0, gl.RGBA, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fboTex, 0);
    return gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  };
  useFloat = tryFmt(gl.RGBA16F, gl.HALF_FLOAT);
  if (!useFloat) tryFmt(gl.RGBA8, gl.UNSIGNED_BYTE);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function initGL() {
  gl.getExtension('EXT_color_buffer_float');
  gl.getExtension('EXT_color_buffer_half_float');
  splatProg = program(SPLAT_VS, SPLAT_FS);
  resolveProg = program(RES_VS, RES_FS);
  dotProg = program(SPLAT_VS, DOT_FS);
  for (const n of ['uR', 'uCenter', 'uRes', 'uSc', 'uRad', 'uFocal']) UD[n] = gl.getUniformLocation(dotProg, n);
  for (const [p, names] of [[splatProg, ['uR', 'uCenter', 'uRes', 'uSc', 'uRad', 'uFocal', 'uW']], [resolveProg, ['uTex', 'uTexel', 'uTime']]]) {
    for (const n of names) U[n] = gl.getUniformLocation(p, n);
  }
  vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const cb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, cb);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const aCorner = gl.getAttribLocation(splatProg, 'aCorner');
  gl.enableVertexAttribArray(aCorner);
  gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);
  posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, 3600 * 3 * 4, gl.DYNAMIC_DRAW);
  const aPos = gl.getAttribLocation(splatProg, 'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(aPos, 1);
  gl.bindVertexArray(null);
  resolveVao = gl.createVertexArray();
}

function resize() {
  W = innerWidth; H = innerHeight;
  DPR = Math.min(devicePixelRatio || 1, 1.5);
  glc.width = Math.round(W * DPR); glc.height = Math.round(H * DPR);
  c2.width = Math.round(W * DPR); c2.height = Math.round(H * DPR);
  if (gl && splatProg) makeFbo();
}
addEventListener('resize', resize);

const frame = eulerFrame(0.3, 0.5, 0.2);
const target = { ex: [...frame.ex], ey: [...frame.ey], ez: [...frame.ez] };
const center = { x: W * 0.5, y: H * 0.5 };
const centerT = { x: W * 0.5, y: H * 0.5 };
const baseSc = () => Math.min(W, H) * 0.2;
let sc = baseSc(), scT = sc;
let time = 0;

const hand = { n: 0, lastSeen: -99, twoUntil: -99, a: null, b: null, ez: [null, null], dist: 0 };
const mouse = { x: W / 2, y: H / 2, down: false, last: -99, roll: 0.3, pitch: 0.3, zoom: 1 };
let mode = 'mouse';
let landmarker = null;
let lastVT = -1;

function mapPt(nx, ny) {
  const vw = video.videoWidth || W, vh = video.videoHeight || H;
  const s = Math.max(W / vw, H / vh);
  const dw = vw * s, dh = vh * s;
  return [W - ((W - dw) / 2 + nx * dw), (H - dh) / 2 + ny * dh];
}

function analyze(lms, slot) {
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const aspect = vw / vh;
  const P = lms.map((l) => [-(l.x - 0.5) * aspect, -(l.y - 0.5), -(l.z || 0) * aspect]);
  const ey = norm(sub(P[9], P[0]));
  let ez = norm(cross(sub(P[17], P[5]), ey));
  const prev = hand.ez[slot];
  if (prev ? dot(ez, prev) < 0 : ez[2] < 0) ez = mul(ez, -1);
  hand.ez[slot] = ez;
  let sx = 0, sy = 0;
  for (const i of [0, 5, 9, 13, 17]) { const p = mapPt(lms[i].x, lms[i].y); sx += p[0]; sy += p[1]; }
  return {
    ey, ez, cx: sx / 5, cy: sy / 5, size: len(sub(P[9], P[0])),
    rot: ((Math.atan2(ey[0], ey[1]) * 180) / Math.PI + 360) % 360,
    px: 1 - lms[9].x, py: lms[9].y, pz: lms[0].z || 0,
  };
}

function onHands(list) {
  if (!list.length) return;
  const hs = list.slice(0, 2).map((l, i) => analyze(l, i)).sort((p, q) => p.cx - q.cx);
  hand.lastSeen = time;

  if (hs.length === 2) {
    const [A, B] = hs;
    const sA = A.size * H, sB = B.size * H;
    const axis = [B.cx - A.cx, -(B.cy - A.cy), 3 * (sB - sA)];
    const ex = norm(axis);
    const upHint = norm(add(A.ey, B.ey));
    const ey = norm(sub(upHint, mul(ex, dot(upHint, ex))));
    target.ex = ex; target.ey = ey; target.ez = cross(ex, ey);
    centerT.x = (A.cx + B.cx) / 2; centerT.y = (A.cy + B.cy) / 2;
    hand.dist = Math.hypot(B.cx - A.cx, B.cy - A.cy);
    const t = clamp((hand.dist / Math.min(W, H) - 0.3) / 0.6, 0, 1);
    scT = baseSc() * (1 + t * t * (3 - 2 * t));
    hand.n = 2; hand.a = A; hand.b = B;
    hand.twoUntil = time + 0.45;
  } else if (time > hand.twoUntil) {
    const A = hs[0];
    const f = { ex: [0, 0, 0], ey: A.ey, ez: A.ez };
    orthonormalize(f);
    target.ex = f.ex; target.ey = f.ey; target.ez = f.ez;
    centerT.x = A.cx; centerT.y = A.cy;
    scT = baseSc();
    hand.n = 1; hand.a = A; hand.b = null;
  }
}

function detect() {
  if (!landmarker || video.readyState < 2 || video.currentTime === lastVT) return;
  lastVT = video.currentTime;
  let res;
  try { res = landmarker.detectForVideo(video, performance.now()); } catch (e) { return; }
  if (res && res.landmarks && res.landmarks.length) onHands(res.landmarks);
  else if (hand.n && time - hand.lastSeen > 0.5) { hand.n = 0; hand.a = hand.b = null; hand.ez = [null, null]; }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    video.classList.add('on');
  } catch (e) {
    elStatus.textContent = 'NO CAMERA · MOUSE MODE (DRAG ROTATE · WHEEL ZOOM · R RESET)';
    return;
  }
  elStatus.textContent = 'LOADING HAND TRACKING…';
  try {
    const vision = await import(`${MP_BASE}/vision_bundle.mjs`);
    const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    const make = (delegate) => vision.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate }, runningMode: 'VIDEO', numHands: 2,
      minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    try { landmarker = await make('GPU'); } catch (e) { landmarker = await make('CPU'); }
    mode = 'hand';
    elStatus.textContent = 'HAND TRACKING · ONE HAND STEERS · TWO HANDS APART = ZOOM';
  } catch (e) {
    console.warn('Hand tracker failed to load', e);
    elStatus.textContent = 'HAND TRACKER FAILED · MOUSE MODE (DRAG ROTATE · WHEEL ZOOM)';
  }
}

addEventListener('pointermove', (e) => {
  if (mouse.down) { mouse.roll += (e.clientX - mouse.x) * 0.012; mouse.pitch += (e.clientY - mouse.y) * 0.01; mouse.last = time; }
  mouse.x = e.clientX; mouse.y = e.clientY;
});
addEventListener('pointerdown', (e) => { mouse.down = true; mouse.x = e.clientX; mouse.y = e.clientY; mouse.last = time; });
addEventListener('pointerup', () => { mouse.down = false; });
addEventListener('wheel', (e) => { mouse.zoom = clamp(mouse.zoom * Math.exp(-e.deltaY * 0.0012), 0.3, 4); mouse.last = time; }, { passive: true });
addEventListener('keydown', (e) => { if (e.key === 'r' || e.key === 'R') { worker.postMessage({ t: 'reset' }); mouse.roll = 0.3; mouse.pitch = 0.3; mouse.zoom = 1; } });

function paramsText() {
  const f = (v, d = 1) => Number(v).toFixed(d);
  return [
    ['radius', info.radius], ['dfriction', '0.0'], ['sfriction', '0.0'], ['pfriction', '0.0'],
    ['rest', f(info.rho0, 4)], ['adhesion', '0.0'], ['sleepthresh', '0.0'], ['clampspeed', '0'],
    ['maxspeed', f(info.maxSpeed)], ['clampaccel', '1'], ['maxaccel', '100.0'], ['diss', '0.0'],
    ['damping', info.damping], ['cohesion', '0.0'], ['surftension', '0.0'], ['viscosity', info.sigma],
    ['buoyancy', '0.0'], ['colldist', '0.1'], ['scollmargin', '0.1'], ['smoothing', '0.0'], ['vortconf', '90.0'],
    ['particles', liveN],
  ].map(([k, v]) => `${k}  ${v}`).join('\n');
}
elParams.textContent = paramsText();
let hudT = 0;
function updateHud() {
  const rot = ((Math.atan2(frame.ey[0], frame.ey[1]) * 180) / Math.PI + 360) % 360;
  const A = hand.a;
  const px = A ? A.px : 1 - center.x / W, py = A ? A.py : center.y / H;
  let s = `+ HAND_1_ROTATION:${rot.toFixed(1)}\n+ HAND_1_POSITION_X:${px.toFixed(2)}\n+ HAND_1_POSITION_Y:${py.toFixed(2)}\n+ HAND_1_POSITION_Z:${(A ? A.pz : 0).toFixed(2)}`;
  if (hand.b) s += `\n+ HAND_2_POSITION_X:${hand.b.px.toFixed(2)}\n+ HAND_2_POSITION_Y:${hand.b.py.toFixed(2)}\n+ HANDS_DISTANCE:${(hand.dist / Math.min(W, H)).toFixed(2)}`;
  elHand.textContent = s;
  hudT = 0;
  elParams.textContent = paramsText();
}

const corners = [];
for (let i = 0; i < 8; i++) corners.push([i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1]);
const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
const prevC = { x: center.x, y: center.y };
const velW = [0, 0], accW = [0, 0];

function update(dt) {
  time += dt;
  detect();

  const handActive = mode === 'hand' && hand.n > 0;
  const mouseActive = !handActive && mode !== 'hand' && time - mouse.last < 8;
  if (!handActive) {
    let f;
    if (mouseActive) {
      f = eulerFrame(mouse.pitch, 0, mouse.roll);
      centerT.x = mouse.x; centerT.y = mouse.y;
    } else {
      f = eulerFrame(Math.sin(time * 0.33) * 0.45 + 0.2, time * 0.28, Math.sin(time * 0.5) * 1.2);
      centerT.x = W * 0.5; centerT.y = H * 0.5;
    }
    scT = baseSc() * (mouseActive ? clamp(mouse.zoom, 1, 2) : 1);
    target.ex = f.ex; target.ey = f.ey; target.ez = f.ez;
  }

  const k = damp(handActive ? 7 : 3, dt);
  for (const a of ['ex', 'ey', 'ez']) for (let i = 0; i < 3; i++) frame[a][i] = lerp(frame[a][i], target[a][i], k);
  orthonormalize(frame);
  const kp = damp(handActive ? 8 : 3, dt);
  center.x = lerp(center.x, centerT.x, kp);
  center.y = lerp(center.y, centerT.y, kp);
  sc = lerp(sc, scT, damp(3.5, dt));

  center.x = clamp(center.x, W * 0.18, W * 0.82);
  center.y = clamp(center.y, H * 0.22, H * 0.78);

  const vx = (center.x - prevC.x) / dt / sc, vy = -(center.y - prevC.y) / dt / sc;
  prevC.x = center.x; prevC.y = center.y;
  const ax = (vx - velW[0]) / dt, ay = (vy - velW[1]) / dt;
  velW[0] = vx; velW[1] = vy;
  accW[0] = lerp(accW[0], clamp(ax, -40, 40), 0.25);
  accW[1] = lerp(accW[1], clamp(ay, -40, 40), 0.25);
  const pw = [-accW[0] * 0.5, -accW[1] * 0.5 - G, 0];
  worker.postMessage({ t: 'g', g: [dot(frame.ex, pw), dot(frame.ey, pw), dot(frame.ez, pw)] });
}

function project(lx, ly, lz) {
  const { ex, ey, ez } = frame;
  const wx = ex[0] * lx + ey[0] * ly + ez[0] * lz;
  const wy = ex[1] * lx + ey[1] * ly + ez[1] * lz;
  const wz = ex[2] * lx + ey[2] * ly + ez[2] * lz;
  const p = FOCAL / (FOCAL - wz);
  return [center.x + wx * sc * p, center.y - wy * sc * p];
}

function renderLiquid() {
  if (pending) {
    liveN = pendingN;
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, pending, 0, pendingN * 3);
    worker.postMessage({ t: 'ret', buf: pending }, [pending.buffer]);
    pending = null;
  }
  if (!liveN) return;
  const { ex, ey, ez } = frame;

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.viewport(0, 0, fboW, fboH);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);
  gl.useProgram(splatProg);
  gl.uniformMatrix3fv(U.uR, false, [...ex, ...ey, ...ez]);
  gl.uniform2f(U.uCenter, center.x, center.y);
  gl.uniform2f(U.uRes, W, H);
  gl.uniform1f(U.uSc, sc);
  gl.uniform1f(U.uRad, info.spacing * 1.35);
  gl.uniform1f(U.uFocal, FOCAL);
  gl.uniform1f(U.uW, useFloat ? 0.07 : 0.02);
  gl.bindVertexArray(vao);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, liveN);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, glc.width, glc.height);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.useProgram(resolveProg);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, fboTex);
  gl.uniform1i(U.uTex, 0);
  gl.uniform2f(U.uTexel, 1 / fboW, 1 / fboH);
  gl.uniform1f(U.uTime, (time * 7) % 13);
  gl.bindVertexArray(resolveVao);
  gl.drawArrays(gl.TRIANGLES, 0, 3);

  gl.useProgram(dotProg);
  gl.uniformMatrix3fv(UD.uR, false, [...ex, ...ey, ...ez]);
  gl.uniform2f(UD.uCenter, center.x, center.y);
  gl.uniform2f(UD.uRes, W, H);
  gl.uniform1f(UD.uSc, sc);
  gl.uniform1f(UD.uRad, info.spacing * 0.42);
  gl.uniform1f(UD.uFocal, FOCAL);
  gl.bindVertexArray(vao);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, liveN);
}

function renderCube() {
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const pr = corners.map(([x, y, z]) => project(x, y, z));
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (const [a, b] of edges) { ctx.moveTo(pr[a][0], pr[a][1]); ctx.lineTo(pr[b][0], pr[b][1]); }
  ctx.strokeStyle = 'rgba(255,150,40,0.22)'; ctx.lineWidth = 5; ctx.stroke();
  ctx.strokeStyle = 'rgba(255,184,70,0.96)'; ctx.lineWidth = 1.6; ctx.stroke();
  if (hand.n && hand.a) {
    ctx.fillStyle = 'rgba(255,190,90,0.55)';
    for (const h of [hand.a, hand.b]) if (h) { ctx.beginPath(); ctx.arc(h.cx, h.cy, 4, 0, Math.PI * 2); ctx.fill(); }
  }
}

let last = performance.now();
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min((now - last) / 1000, 1 / 20);
  last = now;
  if (dt <= 0) return;
  update(dt);
  renderLiquid();
  renderCube();
  hudT += dt;
  if (hudT > 0.1) updateHud();
}

if (!gl) {
  elStatus.textContent = 'WEBGL2 NOT SUPPORTED IN THIS BROWSER';
} else {
  initGL();
  resize();
  window.__fluid = { frame, target, hand, onHands, update, forceHand: () => { mode = 'hand'; }, state: () => ({ sc, scT, center, liveN, useFloat }) };
  requestAnimationFrame((t) => { last = t; tick(t); });
  startCamera();
}
