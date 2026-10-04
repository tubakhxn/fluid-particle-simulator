// dev/creator=tubakhxn
import { Fluid } from './sim.js';

const DT = 1 / 80;
const sim = new Fluid(3600);
let g = [0, -7, 0];
let acc = 0;
let last = performance.now();
let behind = 0;
const free = [new Float32Array(sim.max * 3), new Float32Array(sim.max * 3), new Float32Array(sim.max * 3)];

postMessage({ t: 'info', rho0: sim.rho0, radius: sim.radius, sigma: sim.sigma, damping: sim.damping, maxSpeed: sim.maxSpeed, spacing: sim.spacing });

onmessage = (e) => {
  const m = e.data;
  if (m.t === 'g') g = m.g;
  else if (m.t === 'reset') sim.reset();
  else if (m.t === 'ret') free.push(m.buf);
};

function loop() {
  const now = performance.now();
  acc += Math.min((now - last) / 1000, 0.1);
  last = now;
  let steps = 0;
  while (acc >= DT && steps < 3) { sim.step(DT, g[0], g[1], g[2]); acc -= DT; steps++; }
  if (acc > DT * 2) { acc = DT * 2; behind++; } else behind = Math.max(0, behind - 1);
  if (behind > 40 && sim.n > 1800) { sim.setCount(sim.n * 0.88); behind = 0; }
  if (steps && free.length) {
    const b = free.pop();
    b.set(sim.x.subarray(0, sim.n * 3));
    postMessage({ t: 'pos', buf: b, n: sim.n }, [b.buffer]);
  }
  setTimeout(loop, 3);
}
loop();
