// Fine raster products: radar reflectivity (with single-radar beam physics),
// echo tops and a satellite view. Computed on a raster of up to ~900 px
// across, independent of the coarser simulation grid.

import { makeSimplex, fbm, hash2, mulberry32 } from './noise.js';
import { offsetLL } from './geo.js';

const D2R = Math.PI / 180, KE = 4 / 3 * 6371;
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export const beamHeightKm = (r, elevDeg, h0) => Math.sqrt(r * r + KE * KE + 2 * r * KE * Math.sin(elevDeg * D2R)) - KE + h0;

const NAZ = 720, DR = 0.5;

// Real-data texture atlases (see README): rank-equalised 0..255 grids.
//   rain: real Uljenje radar patterns, 1 km/px, periodic.
//   cloud: real MTG infrared cloud scene, 1.6 km/px, mirror-wrapped.
export const ATLAS = { rain: null, cloud: null, rainQ: null };
function sampleWrap(A, x, y) {
  const n = A.n, xf = ((x % n) + n) % n, yf = ((y % n) + n) % n;
  const i = xf | 0, j = yf | 0, fx = xf - i, fy = yf - j, i1 = (i + 1) % n, j1 = (j + 1) % n, d = A.data;
  return ((d[j * n + i] * (1 - fx) + d[j * n + i1] * fx) * (1 - fy) + (d[j1 * n + i] * (1 - fx) + d[j1 * n + i1] * fx) * fy) / 255;
}
function sampleMirror(A, x, y) {
  const n = A.n, m = 2 * n;
  let xf = ((x % m) + m) % m, yf = ((y % m) + m) % m;
  if (xf >= n) xf = m - 1 - xf; if (yf >= n) yf = m - 1 - yf;
  const i = Math.min(n - 2, xf | 0), j = Math.min(n - 2, yf | 0), fx = xf - i, fy = yf - j, d = A.data;
  return ((d[j * n + i] * (1 - fx) + d[j * n + i + 1] * fx) * (1 - fy) + (d[(j + 1) * n + i] * (1 - fx) + d[(j + 1) * n + i + 1] * fx) * fy) / 255;
}
// Wet-pixel dBZ distribution of real Uljenje rain (quantiles).
function realDbz(v) {
  const q = ATLAS.rainQ, x = clamp(v, 0, 1) * (q.length - 1), i = Math.min(q.length - 2, x | 0);
  return q[i] + (q[i + 1] - q[i]) * (x - i);
}
// Fraction of an area that is actually wet for a given areal-mean rain rate.
// (Real radar: light areal rain is sparse showers; heavy areal rain is a
// nearly continuous shield.)
const wetFraction = r => clamp((r - 0.04) / (r + 1.2), 0, 0.8);
// The atlas's own wet fraction: a pixel's dBZ comes from its absolute place in
// the real pattern, so sparse light rain stays weak and only heavy areal rain
// reaches the cores.
const F0 = 0.3;

export class Raster {
  constructor(view, dem, sim) {
    this.view = view; this.dem = dem;
    const k = this.k = Math.max(1, Math.ceil(Math.max(view.w, view.h) / 900));
    const W = this.W = Math.ceil(view.w / k), H = this.H = Math.ceil(view.h / k);
    const N = W * H;
    this.gx = new Float32Array(N); this.gy = new Float32Array(N);
    this.X = new Float32Array(N); this.Y = new Float32Array(N);
    this.elev = new Float32Array(N); this.water = new Uint8Array(N);
    this.rowLat = new Float32Array(H); this.mpp = new Float32Array(H);
    for (let y = 0; y < H; y++) {
      const la = view.lat((y + 0.5) * k);
      this.rowLat[y] = la; this.mpp[y] = view.metersPerPx(la) * k;
      for (let x = 0; x < W; x++) {
        const p = y * W + x, vx = (x + 0.5) * k, vy = (y + 0.5) * k;
        this.gx[p] = vx / sim.cpx - 0.5; this.gy[p] = vy / sim.cpy - 0.5;
        const lo = view.lon(vx);
        this.X[p] = lo * 111.32 * Math.cos(43 * D2R); this.Y[p] = la * 111.32;
        const di = Math.min(dem.h - 1, vy | 0) * dem.w + Math.min(dem.w - 1, vx | 0);
        this.water[p] = dem.water[di]; this.elev[p] = dem.water[di] ? 0 : Math.max(0, dem.elev[di]);
      }
    }
    this.zlin = new Float32Array(N); this.dbz = new Float32Array(N); this.top = new Float32Array(N);
    this.mask = new Uint8Array(N); // 1 = inside radar coverage (or everywhere without a site)
    this.nA = makeSimplex(11); this.nB = makeSimplex(23); this.nC = makeSimplex(37);
    this.offX = 0; this.offY = 0; this.frame = 0;
    this.site = null;
    this.mask.fill(1);
  }

  advance(sim, dt) {
    let su = 0, sv = 0, n = 0;
    for (let k = 0; k < sim.N; k += 7) { su += sim.us[k]; sv += sim.vs[k]; n++; }
    this.offX += su / n * dt / 1000; this.offY += sv / n * dt / 1000;
    this.frame++;
  }
  // Unit vector of the mean steering flow: rain bands stretch along it.
  _flowDir(sim) {
    let su = 0, sv = 0;
    for (let k = 0; k < sim.N; k += 11) { su += sim.us[k]; sv += sim.vs[k]; }
    const l = Math.hypot(su, sv) || 1;
    return [su / l, sv / l];
  }

  demAt(lat, lon) {
    const v = this.view, x = v.px(lon) | 0, y = v.py(lat) | 0;
    if (x < 0 || y < 0 || x >= this.dem.w || y >= this.dem.h) return 0;
    const i = y * this.dem.w + x;
    return this.dem.water[i] ? 0 : Math.max(0, this.dem.elev[i]);
  }

  setSite(site) {
    this.site = site;
    const { W, H } = this;
    if (!site) { this.mask.fill(1); return; }
    const v = this.view, NR = this.NR = Math.ceil(site.rangeKm / DR);
    // Radars sit on summits: use the highest ground within ~2 km of the site.
    let top = 0;
    for (let a = 0; a < 16; a++) for (const rk of [0, 0.7, 1.4, 2.1]) {
      const [la, lo] = offsetLL(site.lat, site.lon, Math.sin(a * Math.PI / 8) * rk * 1000, Math.cos(a * Math.PI / 8) * rk * 1000);
      top = Math.max(top, this.demAt(la, lo));
    }
    const hA = this.hA = (top + 20) / 1000;
    this.block = new Float32Array(NAZ * NR); this.terr = new Float32Array(NAZ * NR);
    for (let a = 0; a < NAZ; a++) {
      const az = a * 360 / NAZ * D2R, se = Math.sin(az), ce = Math.cos(az);
      let mx = -10;
      for (let rb = 0; rb < NR; rb++) {
        const r = (rb + 0.5) * DR;
        const [la, lo] = offsetLL(site.lat, site.lon, se * r * 1000, ce * r * 1000);
        const h = this.demAt(la, lo) / 1000 - r * r / (2 * KE);
        const ang = Math.atan2(h - hA, r) / D2R;
        this.terr[a * NR + rb] = ang;
        this.block[a * NR + rb] = mx; // horizon angle in front of this bin
        if (ang > mx) mx = ang;
      }
    }
    this.az = new Uint16Array(W * H); this.rb = new Uint16Array(W * H); this.rkm = new Float32Array(W * H);
    const sx = v.px(site.lon) / this.k, sy = v.py(site.lat) / this.k;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = y * W + x, m = this.mpp[y] / 1000;
      const dx = (x + 0.5 - sx) * m, dy = -(y + 0.5 - sy) * m;
      const r = Math.hypot(dx, dy);
      this.rkm[p] = r;
      this.mask[p] = r <= site.rangeKm ? 1 : 0;
      this.az[p] = Math.floor(((Math.atan2(dx, dy) / D2R + 360) % 360) / 360 * NAZ) % NAZ;
      this.rb[p] = Math.min(NR - 1, Math.floor(r / DR));
    }
    // Ray -> raster pixel lookup for path-integrated attenuation.
    this.rayPix = new Int32Array(NAZ * NR);
    for (let a = 0; a < NAZ; a++) {
      const az = (a + 0.5) * 360 / NAZ * D2R;
      for (let rb = 0; rb < NR; rb++) {
        const r = (rb + 0.5) * DR;
        const [la, lo] = offsetLL(site.lat, site.lon, Math.sin(az) * r * 1000, Math.cos(az) * r * 1000);
        const x = Math.floor(v.px(lo) / this.k), y = Math.floor(v.py(la) / this.k);
        this.rayPix[a * NR + rb] = x >= 0 && y >= 0 && x < W && y < H ? y * W + x : -1;
      }
    }
  }

  // Reflectivity (dBZ, -99 = no echo) and echo tops (km) from the sim state.
  compute(sim, opts = {}) {
    const { W, H, zlin, top } = this;
    const NX = sim.NX, NY = sim.NY, R = sim.R, T = sim.T;
    const tt = (sim.time / 3.6e6) % 100000;
    zlin.fill(0); top.fill(0);
    const ph = sim.texPhase(), wA = Math.sin(Math.PI * ph) ** 2, wB = 1 - wA, wN = 1 / Math.sqrt(wA * wA + wB * wB);
    const tAx = sim.tAx, tAy = sim.tAy, tBx = sim.tBx, tBy = sim.tBy;
    const atlas = ATLAS.rain;
    for (let p = 0; p < W * H; p++) {
      let gx = this.gx[p], gy = this.gy[p];
      gx = gx < 0 ? 0 : gx > NX - 1.001 ? NX - 1.001 : gx;
      gy = gy < 0 ? 0 : gy > NY - 1.001 ? NY - 1.001 : gy;
      const i = gx | 0, j = gy | 0, fx = gx - i, fy = gy - j, k = j * NX + i;
      const r = (R[k] * (1 - fx) + R[k + 1] * fx) * (1 - fy) + (R[k + NX] * (1 - fx) + R[k + NX + 1] * fx) * fy;
      if (r < 0.02) continue;
      const bil = f => (f[k] * (1 - fx) + f[k + 1] * fx) * (1 - fy) + (f[k + NX] * (1 - fx) + f[k + NX + 1] * fx) * fy;
      // Where in the real rain pattern this pixel sits (flow-following coordinates).
      let a;
      if (atlas) {
        a = 0;
        if (wA > 0.01) a += wA * (sampleWrap(atlas, bil(tAx), bil(tAy)) - 0.5);
        if (wB > 0.01) a += wB * (sampleWrap(atlas, bil(tBx), bil(tBy)) - 0.5);
        a = clamp(0.5 + a * wN, 0, 1);
      } else {
        a = clamp(0.5 + 0.5 * fbm(this.nA, bil(tAx) / 30, bil(tAy) / 30, 5), 0, 1);
      }
      // Coarse real pattern (5x scale) breaks rain-area outlines and makes
      // heavy areas lumpy instead of smooth blobs following the forcing.
      let env = 1;
      if (atlas) {
        let cz = 0;
        if (wA > 0.01) cz += wA * (sampleWrap(atlas, bil(tAx) / 5 + 333, bil(tAy) / 5 + 777) - 0.5);
        if (wB > 0.01) cz += wB * (sampleWrap(atlas, bil(tBx) / 5 + 333, bil(tBy) / 5 + 777) - 0.5);
        const u = clamp(0.5 + cz * wN, 0, 1);
        env = 0.2 + 1.6 * u * u;
      }
      const f = wetFraction(r * env);
      if (a < 1 - f) continue;
      // Real wet-pixel dBZ distribution, shifted by how heavy the rain is.
      const v = Math.min(0.985, (a - (1 - f)) / Math.max(f, F0));
      const d = (ATLAS.rainQ ? realDbz(v) : 8 + 30 * v) + 4 * Math.log10(Math.max(r * env, 0.05) / 1.5);
      if (d < 4) continue;
      zlin[p] = Math.pow(10, d / 10);
      top[p] = clamp(this.elev[p] / 1000 + T[k] / 6.5 + 2.2 + 2 * v, 2, 9);
    }
    this._cells(sim);
    const dbz = this.dbz;
    for (let p = 0; p < W * H; p++) dbz[p] = zlin[p] > 1 ? 10 * Math.log10(zlin[p]) : -99;
    if (this.site) this._siteEffects(opts);
    if (opts.proj) this._projections();
  }

  _cells(sim) {
    const { W, H, view, k: K, zlin, top } = this;
    for (const c of sim.cells) {
      const dbzC = c.dbz ?? sim.cellDbz(c);
      if (dbzC < 8) continue;
      const cx = view.px(c.lon) / K, cy = view.py(c.lat) / K;
      const mpp = view.metersPerPx(c.lat) * K / 1000; // km per raster px
      const rr = c.r * 2.3;
      const ext = rr * (c.type === 'super' ? 2.6 : 1.8) / mpp;
      if (cx < -ext || cy < -ext || cx > W + ext || cy > H + ext) continue;
      const th = c.th ?? Math.atan2(c.mvy || 0.01, c.mvx || 0.01), ct = Math.cos(th), st = Math.sin(th);
      const e = c.type === 'super' ? 1.45 : c.type === 'multi' ? 1.25 : c.type === 'line' ? 0.7 : 1.12;
      const rnd = mulberry32(c.seed);
      const amp = [0, 0, 0.06 + 0.1 * rnd(), 0.04 + 0.08 * rnd(), 0.03 + 0.06 * rnd(), 0.02 + 0.05 * rnd()];
      const ph = [0, 0, rnd() * 6.28, rnd() * 6.28, rnd() * 6.28, rnd() * 6.28];
      const sxo = rnd() * 1000, syo = rnd() * 1000;
      const ctop = sim.cellTop(c);
      const x0 = Math.max(0, Math.floor(cx - ext)), x1 = Math.min(W - 1, Math.ceil(cx + ext));
      const y0 = Math.max(0, Math.floor(cy - ext)), y1 = Math.min(H - 1, Math.ceil(cy + ext));
      const RA = ATLAS.rain;
      if (c.type === 'super') { this._supercell(c, dbzC, cx, cy, mpp, ct, st, sxo, syo, ctop); continue; }
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const dx = (x + 0.5 - cx) * mpp, dy = -(y + 0.5 - cy) * mpp;
        const a = dx * ct + dy * st, b = -dx * st + dy * ct;
        const ang = Math.atan2(b, a);
        let hh = 1 + amp[2] * Math.cos(2 * ang + ph[2]) + amp[3] * Math.cos(3 * ang + ph[3]) + amp[4] * Math.cos(4 * ang + ph[4]) + amp[5] * Math.cos(5 * ang + ph[5]);
        hh += 0.22 * fbm(this.nB, (a + sxo) / 3.5, (b + syo) / 3.5, 3);
        const dn = Math.sqrt((a / e) ** 2 + b * b) / (rr * hh);
        if (dn >= 1.25) continue;
        const n2 = fbm(this.nC, (a + sxo) / 2.4, (b + syo) / 2.4, 3);
        // Real-radar texture makes the outer echo ragged, with holes and fingers.
        const tex = RA ? sampleWrap(RA, dx * 1.3 + sxo, dy * 1.3 + syo) : 0.5;
        const d = dbzC - (dbzC - 11) * Math.pow(dn, 1.35) + (3 + 3.5 * dn) * n2 + 16 * (tex - 0.5) * Math.min(1, dn * 1.4);
        if (d > 5) {
          const p = y * W + x;
          zlin[p] += Math.pow(10, d / 10);
          if (d > 15) { const t = ctop * (1 - 0.55 * Math.min(1, dn * dn)); if (t > top[p]) top[p] = t; }
        }
      }
    }
  }

  // Classic supercell reflectivity: compact core with a tight gradient on the
  // inflow (right-front) side, a ragged forward-flank echo spreading downwind
  // and left, and a hook wrapping cyclonically around a weak-echo hole at the
  // rear-right. a = along motion (km), b = left of motion (km).
  _supercell(c, dbzC, cx, cy, mpp, ct, st, sxo, syo, ctop) {
    const { W, H, zlin, top } = this;
    const RA = ATLAS.rain;
    const Rc = c.r * 1.25;
    const ext = Rc * 6 / mpp;
    const x0 = Math.max(0, Math.floor(cx - ext)), x1 = Math.min(W - 1, Math.ceil(cx + ext));
    const y0 = Math.max(0, Math.floor(cy - ext)), y1 = Math.min(H - 1, Math.ceil(cy + ext));
    const Pa = -0.7 * Rc, Pb = -0.9 * Rc;                 // mesocyclone (hook pivot)
    const phi0 = Math.atan2(-Pb, -Pa), SWEEP = 4.3;          // hook arc: from the core, around the rear
    const grow = clamp((dbzC - 20) / 30, 0.25, 1);           // structure fills in as the storm matures
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const dx = (x + 0.5 - cx) * mpp, dy = -(y + 0.5 - cy) * mpp;
      const a = dx * ct + dy * st, b = -dx * st + dy * ct;
      // Two scales of real-radar texture: cell-scale lumps and fine fingers.
      const tex = RA ? 0.6 * sampleWrap(RA, dx * 2.2 + sxo, dy * 2.2 + syo) + 0.4 * sampleWrap(RA, dx * 6 + syo, dy * 6 + sxo) : 0.5;
      const n2 = fbm(this.nC, (a + sxo) / 1.6, (b + syo) / 1.6, 3);
      // Core.
      const sa = a / (a > 0 ? (b < 0 ? 0.7 : 1.15) : 1.0), sb = b / (b < 0 ? 0.75 : 1.25);
      const dc = Math.hypot(sa, sb) / Rc;
      let dCore = dbzC - 30 * Math.pow(dc, 1.4) + 3 * n2 + 10 * (tex - 0.5) * Math.min(1, dc);
      // Forward flank: downwind and rotated ~25 deg to the left.
      const fa = (a * 0.906 + b * 0.423) - 1.6 * Rc, fb = (-a * 0.423 + b * 0.906) - 0.25 * Rc;
      const df = Math.hypot(fa / (3.2 * Rc * grow + 0.5 * Rc), fb / (1.3 * Rc));
      let dFF = (dbzC - 15) - 24 * Math.pow(df, 1.6) + 18 * (tex - 0.5) + 2.5 * n2;
      // Hook and weak-echo hole.
      const ha = a - Pa, hb = b - Pb, rho = Math.hypot(ha, hb) / Rc;
      let sArc = (Math.atan2(hb, ha) - phi0 + 4 * Math.PI) % (2 * Math.PI) / SWEEP;
      let dHook = -99;
      if (sArc < 1 && grow > 0.5) {
        const wdt = 0.28 * (1 - 0.5 * sArc), off = (rho - 0.6) / wdt;
        dHook = dbzC - 9 - 20 * sArc - 14 * off * off + 6 * (tex - 0.5);
      }
      if (rho < 0.42) { const k = 28 * (1 - rho / 0.42) * grow; dCore -= k; dFF -= k; }
      let z = 0, best = -99;
      for (const d of [dCore, dFF, dHook]) if (d > 5) { z += Math.pow(10, d / 10); if (d > best) best = d; }
      if (z <= 0) continue;
      const p = y * W + x;
      zlin[p] += z;
      if (best > 15) { const t = ctop * (1 - 0.5 * Math.min(1, dc * dc / 4)); if (t > top[p]) top[p] = t; }
    }
  }

  _siteEffects(opts) {
    const { W, H, dbz, top, NR } = this;
    const site = this.site;
    // C-band path-integrated attenuation along each ray (two-way).
    if (opts.attenuation !== false) {
      const pia = this.pia || (this.pia = new Float32Array(NAZ * NR));
      for (let a = 0; a < NAZ; a++) {
        let acc = 0;
        for (let rb = 0; rb < NR; rb++) {
          const p = this.rayPix[a * NR + rb];
          pia[a * NR + rb] = acc;
          if (p >= 0 && dbz[p] > 20) acc += 2 * 2.5e-5 * Math.pow(Math.pow(10, Math.min(dbz[p], 56) / 10), 0.78) * DR;
        }
      }
      for (let p = 0; p < W * H; p++) if (dbz[p] > -99 && this.mask[p]) dbz[p] -= pia[this.az[p] * NR + this.rb[p]];
    }
    const fr = this.frame;
    for (let p = 0; p < W * H; p++) {
      if (!this.mask[p]) { dbz[p] = -99; continue; }
      const r = this.rkm[p], a = this.az[p], rb = this.rb[p];
      const blk = this.block[a * NR + rb];
      let d = dbz[p];
      if (d > -99) {
        // Lowest usable beam: 0.5 deg or just above the terrain horizon
        // (half the 1 deg beam may clip it), plus a partial-blockage loss.
        const hmin = beamHeightKm(r, Math.max(0.5, blk - 0.2), this.hA);
        if (blk > 0) d -= 4 * Math.min(1, blk / 1.5);
        const t = top[p];
        if (hmin > t) d = -99;
        else if (hmin > t - 1.6) d -= 15 * (hmin - (t - 1.6)) / 1.6;
        const mds = 20 * Math.log10(Math.max(r, 1)) - 36;
        if (d < mds) d = -99;
      }
      if (opts.clutter !== false) {
        const x = p % W, y = (p / W) | 0;
        const ta = this.terr[a * NR + rb];
        if (r < 110 && !this.water[p] && ta >= blk - 0.05 && ta > -0.25) {
          const hsh = hash2(x, y, 3);
          if (hsh < 0.55 - r / 260) {
            const flick = hash2(x, y, fr) < 0.25 ? -8 : 0;
            const cd = 14 + 34 * hash2(x, y, 9) * (1 - r / 140) + flick;
            if (cd > d) d = cd;
          }
        } else if (r < 70 && this.water[p] && hash2(x, y, fr * 7 + 1) < 0.035 * (1 - r / 70)) {
          const cd = 3 + 6 * hash2(x, y, fr);
          if (cd > d) d = cd;
        } else if (d < 2 && hash2(x, y, fr * 13 + 5) < 0.0025) {
          d = 2 + 7 * hash2(x, y, fr + 99);
        }
      }
      dbz[p] = d;
    }
  }

  // Max-reflectivity side views (E-W above the map, N-S to the right).
  _projections() {
    const { W, H, dbz, top } = this;
    const NZ = this.NZ = 56, DZ = 0.25;
    const pt = this.projTop || (this.projTop = new Float32Array(NZ * W));
    const ps = this.projSide || (this.projSide = new Float32Array(NZ * H));
    const tt = this.terrTop || (this.terrTop = new Float32Array(W));
    const ts = this.terrSide || (this.terrSide = new Float32Array(H));
    pt.fill(-99); ps.fill(-99); tt.fill(0); ts.fill(0);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const p = y * W + x;
      if (!this.mask[p]) continue;
      const e = this.elev[p] / 1000;
      if (e > tt[x]) tt[x] = e;
      if (e > ts[y]) ts[y] = e;
      const D = dbz[p];
      if (D < 2) continue;
      const Tp = Math.max(top[p], e + 1);
      let zmin = e;
      if (this.site) zmin = Math.max(e, beamHeightKm(this.rkm[p], Math.max(0.5, this.block[this.az[p] * this.NR + this.rb[p]] - 0.2), this.hA) - 0.4);
      const z0 = Math.max(0, Math.floor(zmin / DZ)), z1 = Math.min(NZ - 1, Math.floor(Tp / DZ));
      for (let zi = z0; zi <= z1; zi++) {
        const z = zi * DZ;
        const dz = z < 0.6 * Tp ? D : D - 20 * (z - 0.6 * Tp) / (0.4 * Tp);
        if (dz > pt[zi * W + x]) pt[zi * W + x] = dz;
        if (dz > ps[zi * H + y]) ps[zi * H + y] = dz;
      }
    }
  }

  // Satellite: 'ir', 'ir-enh' or 'vis'. base = Uint8ClampedArray RGBA (terrain basemap at raster size).
  satellite(sim, mode, base, out) {
    const { W, H } = this;
    const NX = sim.NX, cf = sim.cf, C = sim.C, T = sim.T;
    const tt = (sim.time / 3.6e6) % 100000;
    const ph = sim.texPhase(), wA = Math.sin(Math.PI * ph) ** 2, wB = 1 - wA, wN = 1 / Math.sqrt(wA * wA + wB * wB);
    const cloud = this._cloud || (this._cloud = new Float32Array(W * H));
    const ztop = this._ztop || (this._ztop = new Float32Array(W * H));
    for (let p = 0; p < W * H; p++) {
      const gx = clamp(this.gx[p], 0, sim.NX - 1.001), gy = clamp(this.gy[p], 0, sim.NY - 1.001);
      const i = gx | 0, j = gy | 0, fx = gx - i, fy = gy - j, k = j * NX + i;
      const bil = f => (f[k] * (1 - fx) + f[k + 1] * fx) * (1 - fy) + (f[k + NX] * (1 - fx) + f[k + NX + 1] * fx) * fy;
      const c0 = bil(cf), cw = bil(C);
      // Real cloud scene texture (or noise), thresholded to the cloud fraction.
      let n = 0;
      const CA = ATLAS.cloud;
      if (CA) {
        if (wA > 0.01) n += wA * (sampleMirror(CA, bil(sim.tAx) / 1.6, bil(sim.tAy) / 1.6) - 0.5);
        if (wB > 0.01) n += wB * (sampleMirror(CA, bil(sim.tBx) / 1.6, bil(sim.tBy) / 1.6) - 0.5);
        n = clamp(n * wN, -0.5, 0.5);
      } else {
        if (wA > 0.01) n += wA * fbm(this.nB, bil(sim.tAx) / 48 + tt * 0.03, bil(sim.tAy) / 48, 5);
        if (wB > 0.01) n += wB * fbm(this.nB, bil(sim.tBx) / 48 + tt * 0.03, bil(sim.tBy) / 48, 5);
        n = 0.5 * n * wN;
      }
      const a = n + 0.5, fr = clamp(0.85 * c0, 0, 0.95), thr = 1 - fr;
      // Crisp cloud edges at the cloud-fraction threshold; thin cloud near
      // the edge, optically thick toward the textures' bright structures.
      const e = fr <= 0 ? 0 : (a - thr) / Math.max(0.05, fr);
      const o = e <= 0 ? 0 : clamp(e / 0.18, 0, 1) * clamp(0.45 + 0.75 * e, 0, 1);
      cloud[p] = o;
      // Cloud-top height varies with the real texture: deep tops where it
      // rains, low/mid cloud elsewhere (deeper when mid-levels are moist).
      const Mk = bil(sim.M);
      ztop[p] = cw > 0.03 ? clamp(4 + 5 * e + 2 * cw, 3, 10.5) : 1 + (2 + 6 * clamp((Mk - 0.5) / 0.4, 0, 1)) * clamp(e, 0, 1);
    }
    // Anvils and overshooting tops.
    const view = this.view, K = this.k;
    for (const c of sim.cells) {
      const dbzC = c.dbz ?? sim.cellDbz(c);
      if (dbzC < 25) continue;
      const mpp = view.metersPerPx(c.lat) * K / 1000;
      const ra = c.r * 5.5 * clamp((dbzC - 20) / 30, 0.2, 1.2) / mpp;
      // Anvil: an oval cold canopy pushed downwind by upper winds, with a
      // sharp upwind edge, ragged sides and an overshooting top over the core.
      const ang = Math.atan2(c.mvy, c.mvx);
      const ux = c.mvx + 0.6 * sim.P.shear * Math.cos(ang), uy = c.mvy + 0.6 * sim.P.shear * Math.sin(ang);
      const ul = Math.hypot(ux, uy) + 1e-6, ex = ux / ul, ey = -uy / ul;
      const L = ra * (1 + Math.min(0.8, ul / 30)), Wd = ra * 0.9, Up = ra * 0.65;
      const cx0 = view.px(c.lon) / K, cy0 = view.py(c.lat) / K;
      const ctop = sim.cellTop(c);
      const ext = L * 1.3 + 2;
      const x0 = Math.max(0, Math.floor(cx0 - ext)), x1 = Math.min(W - 1, Math.ceil(cx0 + ext));
      const y0 = Math.max(0, Math.floor(cy0 - ext)), y1 = Math.min(H - 1, Math.ceil(cy0 + ext));
      const CA = ATLAS.cloud;
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const dx = x - cx0, dy = y - cy0;
        const al = dx * ex + dy * ey, ac = -dx * ey + dy * ex;
        const up = al < 0;
        const da = up ? -al / Up : al / L, dc = ac / Wd;
        const r = Math.sqrt(da * da + dc * dc);
        if (r >= 1.35) continue;
        const p = y * W + x;
        const tex = CA ? sampleMirror(CA, (x * mpp + c.seed * 37) / 1.6, (y * mpp + c.seed * 11) / 1.6) : 0.5;
        const o = clamp((1 - r + (up ? 0.12 : 0.4) * (tex - 0.5) * 2) / (up ? 0.1 : 0.3), 0, 1);
        if (o <= 0) continue;
        if (o > cloud[p]) cloud[p] = o;
        const dd = Math.sqrt(dx * dx + dy * dy) / ra;
        const zt = ctop * (1 - 0.12 * r) + 0.6 * (tex - 0.5) + (dd < 0.12 ? 1.4 * (1 - dd / 0.12) : 0);
        if (zt > ztop[p]) ztop[p] = zt;
      }
    }
    const sunMid = clamp(sim.cosz[(sim.NY >> 1) * NX + (NX >> 1)] * 1.6 + 0.08, 0, 1);
    for (let p = 0; p < W * H; p++) {
      const o = cloud[p], q = p * 4;
      const gx = clamp(this.gx[p], 0, sim.NX - 1), gy = clamp(this.gy[p], 0, sim.NY - 1);
      const k = (gy | 0) * NX + (gx | 0);
      const ts = T[k];
      if (mode === 'vis') {
        const sun = clamp(sim.cosz[k] * 1.6 + 0.08, 0, 1);
        const b = 0.2 + 0.8 * sun;
        const cb = (0.62 + 0.38 * Math.min(1, ztop[p] / 10)) * 255 * sun;
        const a = Math.pow(o, 0.85);
        out[q] = base[q] * b * (1 - a) + cb * a; out[q + 1] = base[q + 1] * b * (1 - a) + cb * a; out[q + 2] = base[q + 2] * b * (1 - a) + cb * a;
      } else {
        const tc = ts - 6.5 * Math.max(0, ztop[p] - this.elev[p] / 1000);
        const tb = ts + (tc - ts) * o;
        const g = Math.pow(clamp((34 - tb) / 94, 0, 1), 0.95) * 255;
        if (mode === 'ir-enh' && tb < -32) {
          const e = clamp((-32 - tb) / 40, 0, 1);
          const col = e < 0.25 ? [80, 160, 255] : e < 0.5 ? [40, 220, 80] : e < 0.75 ? [250, 220, 30] : e < 0.9 ? [240, 60, 30] : [255, 255, 255];
          out[q] = col[0]; out[q + 1] = col[1]; out[q + 2] = col[2];
        } else { out[q] = g; out[q + 1] = g; out[q + 2] = g; }
      }
      out[q + 3] = 255;
    }
    return sunMid;
  }
}
