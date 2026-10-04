// dev/creator=tubakhxn
export class Fluid {
  constructor(n = 2400) {
    this.max = n;
    this.n = n;
    this.spacing = Math.cbrt(2.2 / n);
    this.h = this.spacing * 2;
    this.dt2c = (1 / 120) * (1 / 120) * (this.spacing / 0.1);
    this.radius = 0.052 * 1.0;
    this.k = 140;
    this.kNear = 320;
    this.sigma = 0.04;
    this.damping = 0.059;
    this.maxSpeed = 7;
    this.margin = 0.02;
    this.x = new Float32Array(n * 3);
    this.v = new Float32Array(n * 3);
    this.p = new Float32Array(n * 3);
    this.rho = new Float32Array(n);
    this.rhoN = new Float32Array(n);
    const g = Math.ceil(2 / this.h) + 1;
    this.gd = g;
    this.cellStart = new Int32Array(g * g * g + 1);
    this.cellOf = new Int32Array(n);
    this.sorted = new Int32Array(n);
    this.rho0 = this._latticeDensity();
    this._init();
  }

  _latticeDensity() {
    const s = this.spacing, h = this.h, m = Math.ceil(h / s);
    let d = 0;
    for (let i = -m; i <= m; i++) for (let j = -m; j <= m; j++) for (let k = -m; k <= m; k++) {
      if (!i && !j && !k) continue;
      const r = Math.hypot(i, j, k) * s;
      if (r < h) { const q = 1 - r / h; d += q * q; }
    }
    return d;
  }

  _init() {
    const s = this.spacing, nx = Math.floor(1.9 / s);
    let i = 0;
    for (let y = 0; i < this.n; y++) {
      for (let z = 0; z < nx && i < this.n; z++) {
        for (let x = 0; x < nx && i < this.n; x++) {
          this.x[i * 3] = -0.95 + x * s + Math.random() * 0.004;
          this.x[i * 3 + 1] = -0.97 + y * s + Math.random() * 0.004;
          this.x[i * 3 + 2] = -0.95 + z * s + Math.random() * 0.004;
          i++;
        }
      }
    }
    this.v.fill(0);
  }

  reset() { this.n = this.max; this._init(); }

  setCount(n) { this.n = Math.max(600, Math.min(this.max, n | 0)); }

  _grid() {
    const { n, x, h, gd, cellStart, cellOf, sorted } = this;
    cellStart.fill(0);
    for (let i = 0; i < n; i++) {
      const cx = Math.min(gd - 1, Math.max(0, ((x[i * 3] + 1) / h) | 0));
      const cy = Math.min(gd - 1, Math.max(0, ((x[i * 3 + 1] + 1) / h) | 0));
      const cz = Math.min(gd - 1, Math.max(0, ((x[i * 3 + 2] + 1) / h) | 0));
      const c = cx + gd * (cy + gd * cz);
      cellOf[i] = c;
      cellStart[c + 1]++;
    }
    for (let c = 0; c < cellStart.length - 1; c++) cellStart[c + 1] += cellStart[c];
    const fill = this._fill || (this._fill = new Int32Array(cellStart.length));
    fill.set(cellStart);
    for (let i = 0; i < n; i++) sorted[fill[cellOf[i]]++] = i;
  }

  step(dt, gx, gy, gz) {
    const { n, x, v, p, h, gd, cellStart, sorted, rho, rhoN } = this;
    const h2 = h * h, dt2 = this.dt2c;
    const damp = Math.exp(-this.damping * dt);
    const maxS = this.maxSpeed;

    for (let i = 0; i < n; i++) {
      const o = i * 3;
      v[o] += gx * dt; v[o + 1] += gy * dt; v[o + 2] += gz * dt;
      v[o] *= damp; v[o + 1] *= damp; v[o + 2] *= damp;
      const sp = Math.hypot(v[o], v[o + 1], v[o + 2]);
      if (sp > maxS) { const f = maxS / sp; v[o] *= f; v[o + 1] *= f; v[o + 2] *= f; }
      p[o] = x[o]; p[o + 1] = x[o + 1]; p[o + 2] = x[o + 2];
      x[o] += v[o] * dt; x[o + 1] += v[o + 1] * dt; x[o + 2] += v[o + 2] * dt;
    }

    this._grid();

    let np = 0;
    rho.fill(0, 0, n); rhoN.fill(0, 0, n);
    let pi = this.pi, pj = this.pj, pm = this.pm;
    if (!pi || pi.length < n * 40) {
      pi = this.pi = new Int32Array(n * 40); pj = this.pj = new Int32Array(n * 40); pm = this.pm = new Float32Array(n * 40);
    }
    const cap = pi.length;
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      const xi = x[o], yi = x[o + 1], zi = x[o + 2];
      const cx = Math.min(gd - 1, Math.max(0, ((xi + 1) / h) | 0));
      const cy = Math.min(gd - 1, Math.max(0, ((yi + 1) / h) | 0));
      const cz = Math.min(gd - 1, Math.max(0, ((zi + 1) / h) | 0));
      let d = 0, dn = 0;
      for (let oz = -1; oz <= 1; oz++) {
        const zz = cz + oz; if (zz < 0 || zz >= gd) continue;
        for (let oy = -1; oy <= 1; oy++) {
          const yy = cy + oy; if (yy < 0 || yy >= gd) continue;
          for (let ox = -1; ox <= 1; ox++) {
            const xx = cx + ox; if (xx < 0 || xx >= gd) continue;
            const c = xx + gd * (yy + gd * zz);
            for (let e = cellStart[c]; e < cellStart[c + 1]; e++) {
              const j = sorted[e]; if (j <= i) continue;
              const q = j * 3;
              const dx = x[q] - xi, dy = x[q + 1] - yi, dz = x[q + 2] - zi;
              const r2 = dx * dx + dy * dy + dz * dz;
              if (r2 >= h2 || r2 < 1e-12) continue;
              const m = 1 - Math.sqrt(r2) / h, m2 = m * m, m3 = m2 * m;
              d += m2; dn += m3; rho[j] += m2; rhoN[j] += m3;
              if (np < cap) { pi[np] = i; pj[np] = j; pm[np] = m; np++; }
            }
          }
        }
      }
      rho[i] += d; rhoN[i] += dn;
    }

    const k = this.k, kn = this.kNear, rho0 = this.rho0, sigma = this.sigma, maxD = this.h * 0.12;
    for (let e = 0; e < np; e++) {
      const i = pi[e], j = pj[e], m = pm[e];
      const o = i * 3, q = j * 3;
      let dx = x[q] - x[o], dy = x[q + 1] - x[o + 1], dz = x[q + 2] - x[o + 2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (r < 1e-6) continue;
      dx /= r; dy /= r; dz /= r;
      const press = Math.max(k * (rho[i] - rho0), 0) + Math.max(k * (rho[j] - rho0), 0);
      let Dm = 0.25 * dt2 * (press * m + (kn * (rhoN[i] + rhoN[j])) * m * m);
      if (Dm > maxD) Dm = maxD;
      x[o] -= dx * Dm; x[o + 1] -= dy * Dm; x[o + 2] -= dz * Dm;
      x[q] += dx * Dm; x[q + 1] += dy * Dm; x[q + 2] += dz * Dm;
      const u = (v[o] - v[q]) * dx + (v[o + 1] - v[q + 1]) * dy + (v[o + 2] - v[q + 2]) * dz;
      if (u > 0) {
        const I = 0.5 * dt * m * sigma * u * 30;
        v[o] -= dx * I; v[o + 1] -= dy * I; v[o + 2] -= dz * I;
        v[q] += dx * I; v[q + 1] += dy * I; v[q + 2] += dz * I;
      }
    }

    const lo = -1 + this.margin, hi = 1 - this.margin, inv = 1 / dt;
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      for (let a = 0; a < 3; a++) {
        let val = x[o + a];
        if (val < lo) val = lo; else if (val > hi) val = hi;
        x[o + a] = val;
        v[o + a] = (val - p[o + a]) * inv;
      }
    }
  }
}
