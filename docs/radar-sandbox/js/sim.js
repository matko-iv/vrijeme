// The weather engine. It is not a forecast model: a cheap, physically
// motivated sandbox on a ~200-cell grid covering the visible map.
//
//   * Surface pressure = background gradient (from the steering wind) plus
//     user-placed lows/highs. Wind = gradient wind, turned toward low
//     pressure and slowed by friction (more over land), blocked by slopes,
//     accelerated downslope (bora), plus a diurnal sea/land breeze.
//   * Temperature and moisture are advected by the surface wind and relaxed
//     toward an equilibrium set by the air mass, terrain height, sea surface
//     temperature and the sun (diurnal cycle with cloud damping).
//   * Stratiform rain: lift from convergence, terrain (upslope) and warm
//     advection condenses moisture into a cloud-water reservoir that rains
//     out over ~15 min and drifts with the steering wind.
//   * Convection: Lagrangian storm cells (single, multicell, supercell,
//     squall line) with life cycles, triggered where CAPE (parcel theory
//     from T, q and the 500 hPa temperature) meets a lifting mechanism.
//     Cells rain, cool the surface (cold pools), spawn daughters along gust
//     fronts and make lightning.

import { mulberry32, makeSimplex, fbm } from './noise.js';

const RHO = 1.2, OMEGA = 7.292e-5, D2R = Math.PI / 180, G = 9.81;
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

export const esat = Tc => 6.112 * Math.exp(17.67 * Tc / (Tc + 243.5));
export const qsat = (Tc, p = 1013) => { const e = esat(Tc); return 622 * e / (p - 0.378 * e); };
export function dewpoint(q, p = 1013) {
  const e = Math.max(1e-3, q * p / (622 + 0.378 * q));
  const l = Math.log(e / 6.112);
  return 243.5 * l / (17.67 - l);
}

function thetaE(Tk, p, r) {
  const e = Math.max(0.01, r * p / (622 + r));
  const TL = 2840 / (3.5 * Math.log(Tk) - Math.log(e) - 4.805) + 55;
  return Tk * Math.pow(1000 / p, 0.2854 * (1 - 0.00028 * r)) *
    Math.exp((3.376 / TL - 0.00254) * r * (1 + 0.00081 * r));
}
function thetaEsat500(Tk) {
  const es = esat(Tk - 273.15), r = 622 * es / (500 - es);
  return Tk * Math.pow(2, 0.2854 * (1 - 0.00028 * r)) * Math.exp((3.376 / Tk - 0.00254) * r * (1 + 0.00081 * r));
}
// Temperature (K) of a saturated parcel at 500 hPa with the given theta-e,
// from an inverted lookup table (theta-e 220..480 K in 0.05 K steps).
const TH_MIN = 220, TH_STEP = 0.05, TH_N = 5201;
const PT500 = (() => {
  const t = new Float32Array(TH_N);
  let T = 180;
  for (let n = 0; n < TH_N; n++) {
    const target = TH_MIN + n * TH_STEP;
    while (T < 340 && thetaEsat500(T + 0.01) < target) T += 0.01;
    t[n] = T;
  }
  return t;
})();
function parcelT500(the) {
  const x = (the - TH_MIN) / TH_STEP;
  if (x <= 0) return PT500[0];
  if (x >= TH_N - 1) return PT500[TH_N - 1];
  const n = x | 0, f = x - n;
  return PT500[n] * (1 - f) + PT500[n + 1] * f;
}

// Wind vector from meteorological direction (degrees the wind blows FROM).
export const windFrom = (dir, spd) => [-spd * Math.sin(dir * D2R), -spd * Math.cos(dir * D2R)];
export const dirFrom = (u, v) => (Math.atan2(-u, -v) / D2R + 360) % 360;

export const DEFAULT_PARAMS = {
  steerDir: 225, steerSpd: 12, shear: 14,
  t850: 9, t500: -21, rh: 70, sst: 22,
  sun: 1, trigger: 1, breeze: 1, variability: 1,
};

const TEX_PERIOD = 3 * 3600 * 1000; // texture coordinate layers reset every 3 h (staggered)

let nextId = 1;

export class Sim {
  constructor(view, dem, params, timeMs, seed = 7) {
    this.view = view;
    this.P = params;
    this.time = timeMs;
    this.rng = mulberry32(seed);
    this.cells = [];
    this.systems = [];
    this.strikes = [];
    this.lineSeq = 1;

    const target = 200;
    const cp = Math.max(view.w, view.h) / target;
    const NX = this.NX = Math.max(40, Math.round(view.w / cp));
    const NY = this.NY = Math.max(30, Math.round(view.h / cp));
    this.cpx = view.w / NX; this.cpy = view.h / NY;
    const N = this.N = NX * NY;

    this.lat = new Float32Array(NY); this.dx = new Float32Array(NY); this.dy = new Float32Array(NY);
    this.f = new Float32Array(NY); this.lon = new Float32Array(NX);
    for (let j = 0; j < NY; j++) {
      const la = view.lat((j + 0.5) * this.cpy);
      this.lat[j] = la;
      const m = view.metersPerPx(la);
      this.dx[j] = m * this.cpx; this.dy[j] = m * this.cpy;
      this.f[j] = Math.max(2e-5, 2 * OMEGA * Math.sin(la * D2R));
    }
    for (let i = 0; i < NX; i++) this.lon[i] = view.lon((i + 0.5) * this.cpx);

    const F = () => new Float32Array(N);
    this.elev = F(); this.land = F(); this.rough = F();
    this.T = F(); this.q = F(); this.C = F(); this.Fz = F(); this.acc = F();
    this.R = F(); this.Rc = F(); this.out = F(); this.anvil = F();
    this.p = F(); this.u = F(); this.v = F(); this.us = F(); this.vs = F();
    this.w = F(); this.cape = F(); this.cf = F(); this.t500 = F(); this.gust = F();
    this.cosz = F();
    // Free atmosphere: a slowly evolving pattern of troughs, ridges and eddies
    // (stream function psi) plus mesoscale lift noise, so the flow varies
    // across the map even without user-placed lows and highs.
    this.psi = F(); this.pu = F(); this.pv = F(); this.meso = F();
    this.pn = makeSimplex(seed * 7 + 3); this.mn = makeSimplex(seed * 13 + 5);
    this.pox = 0; this.poy = 0;
    this.Xk = new Float32Array(NX); this.Yk = new Float32Array(NY);
    for (let i = 0; i < NX; i++) this.Xk[i] = this.lon[i] * 111.32 * Math.cos(43 * D2R);
    for (let j = 0; j < NY; j++) this.Yk[j] = this.lat[j] * 111.32;
    // Two flow-following texture coordinate layers (km) for the radar and
    // satellite texture: advected by the steering wind, reset alternately.
    this.tAx = F(); this.tAy = F(); this.tBx = F(); this.tBy = F();
    this.texOff = { A: [0, 0], B: [5000, 3000] };
    this._resetTex('A'); this._resetTex('B');
    this._phase = (timeMs % TEX_PERIOD) / TEX_PERIOD;
    this.M = F(); this.syn = F(); // mid-level humidity (0..1) and synoptic lift factor (lows +, highs -)
    this._tmp = F();

    this._initTerrain(dem);
    this.diagnose();
    this._initState();
    this.diagnose();
  }

  _resetTex(layer, keep) {
    const off = this.texOff[layer] = keep ? keep.slice() : [this.rng() * 20000, this.rng() * 20000];
    const X = layer === 'A' ? this.tAx : this.tBx, Y = layer === 'A' ? this.tAy : this.tBy;
    for (let j = 0; j < this.NY; j++) for (let i = 0; i < this.NX; i++) {
      X[j * this.NX + i] = this.Xk[i] + off[0]; Y[j * this.NX + i] = this.Yk[j] + off[1];
    }
  }
  // Inflow edges would stretch the texture; extrapolate the coordinates
  // from the interior with an undistorted (identity) gradient instead.
  _texEdges() {
    const { NX, NY } = this, E = 3;
    for (const [X, Y] of [[this.tAx, this.tAy], [this.tBx, this.tBy]]) {
      for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
        if (i >= E && j >= E && i < NX - E && j < NY - E) continue;
        const ii = Math.min(NX - 1 - E, Math.max(E, i)), jj = Math.min(NY - 1 - E, Math.max(E, j));
        const k = j * NX + i, kk = jj * NX + ii;
        X[k] = X[kk] + this.Xk[i] - this.Xk[ii];
        Y[k] = Y[kk] + this.Yk[j] - this.Yk[jj];
      }
    }
  }
  texPhase() { return (this.time % TEX_PERIOD) / TEX_PERIOD; }

  _pattern() {
    const { NX, NY } = this, V = this.P.variability ?? 1;
    const L = 480, A = V * 6.5 * L * 1000 / 1.6, tt = (this.time / 3.6e6) % 100000;
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      const x = this.Xk[i] - this.pox, y = this.Yk[j] - this.poy, k = j * NX + i;
      // Low gain: small eddies are weak, so the flow meanders instead of swirling.
      this.psi[k] = A * fbm(this.pn, x / L + tt * 0.011, y / L - tt * 0.008, 3, 2.03, 0.3);
      this.meso[k] = V * 0.06 * fbm(this.mn, x / 75 + tt * 0.07, y / 75 - tt * 0.05, 3);
    }
    for (let j = 0; j < NY; j++) {
      const jm = Math.max(0, j - 1), jp = Math.min(NY - 1, j + 1);
      for (let i = 0; i < NX; i++) {
        const im = Math.max(0, i - 1), ip = Math.min(NX - 1, i + 1), k = j * NX + i;
        this.pu[k] = -(this.psi[jm * NX + i] - this.psi[jp * NX + i]) / (this.dy[j] * (jp - jm));
        this.pv[k] = (this.psi[j * NX + ip] - this.psi[j * NX + im]) / (this.dx[j] * (ip - im));
      }
    }
    this._pA = A || 1;
  }

  // ---------- grid helpers ----------
  idx(i, j) { return j * this.NX + i; }
  // Fractional grid coordinates of a lon/lat point.
  gridXY(lon, lat) { return [this.view.px(lon) / this.cpx - 0.5, this.view.py(lat) / this.cpy - 0.5]; }
  sample(field, gx, gy) {
    const NX = this.NX, NY = this.NY;
    gx = clamp(gx, 0, NX - 1.001); gy = clamp(gy, 0, NY - 1.001);
    const i = gx | 0, j = gy | 0, fx = gx - i, fy = gy - j, k = j * NX + i;
    return (field[k] * (1 - fx) + field[k + 1] * fx) * (1 - fy) + (field[k + NX] * (1 - fx) + field[k + NX + 1] * fx) * fy;
  }
  sampleLL(field, lon, lat) { const [gx, gy] = this.gridXY(lon, lat); return this.sample(field, gx, gy); }
  inside(lon, lat, padCells = 0) {
    const [gx, gy] = this.gridXY(lon, lat);
    return gx >= -padCells && gy >= -padCells && gx <= this.NX - 1 + padCells && gy <= this.NY - 1 + padCells;
  }

  _initTerrain(dem) {
    const { NX, NY, cpx, cpy } = this;
    const sum = new Float64Array(this.N), sum2 = new Float64Array(this.N), cnt = new Float64Array(this.N), wat = new Float64Array(this.N);
    for (let y = 0; y < dem.h; y++) {
      const j = Math.min(NY - 1, Math.floor(y / cpy));
      for (let x = 0; x < dem.w; x++) {
        const i = Math.min(NX - 1, Math.floor(x / cpx)), k = j * NX + i, d = y * dem.w + x;
        const e = dem.water[d] ? 0 : Math.max(0, dem.elev[d]);
        sum[k] += e; sum2[k] += e * e; cnt[k]++; wat[k] += dem.water[d] ? 1 : 0;
      }
    }
    for (let k = 0; k < this.N; k++) {
      const n = Math.max(1, cnt[k]), m = sum[k] / n;
      this.elev[k] = m;
      this.land[k] = 1 - wat[k] / n;
      this.rough[k] = Math.sqrt(Math.max(0, sum2[k] / n - m * m));
    }
    // Smoothed elevation (for slope / lift) and smoothed land fraction (for
    // the sea breeze), both over ~12 km.
    const kmCell = this.dx[NY >> 1] / 1000;
    const rad = Math.max(1, Math.round(12 / kmCell));
    this.elevS = this._blur(this.elev, Math.min(rad, 4));
    this.landS = this._blur(this.land, Math.min(Math.max(rad, 1), 6));
    this.hx = new Float32Array(this.N); this.hy = new Float32Array(this.N);
    this.lx = new Float32Array(this.N); this.ly = new Float32Array(this.N);
    for (let j = 0; j < NY; j++) {
      const jm = Math.max(0, j - 1), jp = Math.min(NY - 1, j + 1);
      for (let i = 0; i < NX; i++) {
        const im = Math.max(0, i - 1), ip = Math.min(NX - 1, i + 1), k = j * NX + i;
        const ddx = this.dx[j] * (ip - im), ddy = this.dy[j] * (jp - jm);
        this.hx[k] = (this.elevS[j * NX + ip] - this.elevS[j * NX + im]) / ddx;
        this.hy[k] = (this.elevS[jm * NX + i] - this.elevS[jp * NX + i]) / ddy; // north-positive
        this.lx[k] = (this.landS[j * NX + ip] - this.landS[j * NX + im]) / (ip - im);
        this.ly[k] = (this.landS[jm * NX + i] - this.landS[jp * NX + i]) / (jp - jm);
      }
    }
  }

  _blur(src, r) {
    const { NX, NY } = this;
    const a = new Float32Array(src), b = new Float32Array(this.N);
    for (let pass = 0; pass < 2; pass++) {
      for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
        let s = 0, n = 0;
        for (let d = -r; d <= r; d++) { const ii = i + d; if (ii >= 0 && ii < NX) { s += a[j * NX + ii]; n++; } }
        b[j * NX + i] = s / n;
      }
      for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
        let s = 0, n = 0;
        for (let d = -r; d <= r; d++) { const jj = j + d; if (jj >= 0 && jj < NY) { s += b[jj * NX + i]; n++; } }
        a[j * NX + i] = s / n;
      }
    }
    return a;
  }

  // ---------- climatology ----------
  // Daily-mean sea-level temperature of the air mass.
  tBase(j) { return this.P.t850 + 8 + 0.65 * (43 - this.lat[j]); }
  sstAt(j) { return this.P.sst + 0.55 * (43 - this.lat[j]); }

  _sun() {
    const d = new Date(this.time);
    const start = Date.UTC(d.getUTCFullYear(), 0, 0);
    const n = (this.time - start) / 86400000;
    const decl = 23.44 * Math.sin(2 * Math.PI * (284 + n) / 365) * D2R;
    const hUTC = d.getUTCHours() + d.getUTCMinutes() / 60 + d.getUTCSeconds() / 3600;
    const { NX, NY } = this;
    for (let j = 0; j < NY; j++) {
      const la = this.lat[j] * D2R, a = Math.sin(la) * Math.sin(decl), b = Math.cos(la) * Math.cos(decl);
      for (let i = 0; i < NX; i++) {
        const H = (15 * (hUTC + this.lon[i] / 15 - 12)) * D2R;
        this.cosz[j * NX + i] = a + b * Math.cos(H);
      }
    }
  }

  tEq(k, j) {
    const P = this.P, tb = this.tBase(j) - 0.0065 * this.elev[k];
    const cz = Math.max(0, this.cosz[k]), cf = this.cf[k];
    const tl = tb + P.sun * (13 * Math.pow(cz, 0.8) * (1 - 0.7 * cf) - 4.5 * (1 - 0.6 * cf));
    const ts = 0.62 * this.sstAt(j) + 0.38 * tb + 0.6 * cz;
    const lf = this.land[k];
    return lf * tl + (1 - lf) * ts;
  }

  qEq(k, j) {
    const lf = this.land[k];
    const qs = 0.76 * qsat(this.sstAt(j));
    // Daily-mean temperature, so afternoon heating lowers RH instead of adding moisture.
    const ql = this.P.rh / 100 * qsat(this.tBase(j)) * Math.exp(-this.elev[k] / 4000);
    return lf * ql + (1 - lf) * qs;
  }

  _initState() {
    this._sun();
    for (let j = 0; j < this.NY; j++) for (let i = 0; i < this.NX; i++) {
      const k = j * this.NX + i;
      this.T[k] = this.tEq(k, j);
      this.q[k] = this.qEq(k, j);
      this.M[k] = clamp(this.P.rh / 100 + 0.2 * this.syn[k], 0.05, 0.98);
    }
  }

  // ---------- diagnostics: pressure, wind, lift, CAPE ----------
  diagnose() {
    const { NX, NY, P } = this;
    this._sun();
    const [su, sv] = windFrom(P.steerDir, P.steerSpd);
    // Surface geostrophic background: 70% of the steering flow, backed 15 deg.
    const cb = Math.cos(15 * D2R), sb = Math.sin(15 * D2R);
    const ubg = 0.6 * (su * cb - sv * sb), vbg = 0.6 * (su * sb + sv * cb);
    const latC = this.lat[NY >> 1], fC = 2 * OMEGA * Math.sin(latC * D2R);
    const lonC = this.lon[NX >> 1];
    const sys = this.systems;
    this._pattern();

    for (let j = 0; j < NY; j++) {
      const la = this.lat[j], f = this.f[j], cl = Math.cos(la * D2R);
      const ym = (la - latC) * 111320;
      for (let i = 0; i < NX; i++) {
        const k = j * NX + i, lo = this.lon[i];
        const xm = (lo - lonC) * 111320 * cl;
        // Background gradient, tapered beyond ~700 km so continent-wide maps
        // keep realistic pressures (the wind itself stays uniform).
        let p = 1013 + RHO * fC * (vbg * 7e5 * Math.tanh(xm / 7e5) - ubg * 7e5 * Math.tanh(ym / 7e5)) / 100;
        let ug = 0, vg = 0, t5 = 0, syn = -0.3 * this.psi[k] / this._pA;
        p += RHO * f * this.psi[k] / 100;
        t5 += 2.5 * this.psi[k] / this._pA; // troughs a little colder aloft
        for (let s = 0; s < sys.length; s++) {
          const S = sys[s];
          const dxm = (lo - S.lon) * 111320 * cl, dym = (la - S.lat) * 111320;
          const R = S.R * 1000, r2 = dxm * dxm + dym * dym, e = Math.exp(-r2 / (R * R));
          p += S.dp * e;
          // Pressure gradient (Pa/m) of the Gaussian, then geostrophic wind.
          const gx = S.dp * 100 * e * (-2 * dxm / (R * R)), gy = S.dp * 100 * e * (-2 * dym / (R * R));
          let gu = -gy / (RHO * f), gv = gx / (RHO * f);
          const vgs = Math.hypot(gu, gv), r = Math.sqrt(r2) + 1;
          if (vgs > 0.01) {
            const ro = 4 * vgs / (f * r);
            let vgr;
            if (S.dp < 0) vgr = 2 * vgs / (1 + Math.sqrt(1 + ro));
            else vgr = ro < 1 ? 2 * vgs / (1 + Math.sqrt(1 - ro)) : Math.min(2 * vgs, f * r / 2);
            const ratio = vgr / vgs;
            gu *= ratio; gv *= ratio;
          }
          ug += gu; vg += gv;
          // Lows are cold aloft, highs warm.
          t5 += (S.dp < 0 ? 0.3 : 0.15) * S.dp * Math.exp(-r2 / (1.4 * R * 1.4 * R));
          syn += -S.dp / 20 * Math.exp(-r2 / (1.2 * R * 1.2 * R));
        }
        this.syn[k] = clamp(syn, -1, 1);
        this.p[k] = p;
        this.t500[k] = P.t500 + 0.75 * (43 - la) + t5;
        // Steering: background plus most of the systems' circulation aloft.
        this.us[k] = su + 0.6 * ug + this.pu[k]; this.vs[k] = sv + 0.6 * vg + this.pv[k];
        // Surface: friction turns the wind toward low pressure and slows it.
        let u = ubg + ug + 0.75 * this.pu[k], v = vbg + vg + 0.75 * this.pv[k];
        const spdg = Math.hypot(u, v);
        if (spdg > 55) { u *= 55 / spdg; v *= 55 / spdg; }
        const lf = this.land[k];
        const a = (14 + 22 * lf) * D2R, ca = Math.cos(a), sa = Math.sin(a);
        const red = (0.72 - 0.25 * lf) / (1 + this.rough[k] / 350);
        let us = (u * ca - v * sa) * red, vs = (u * sa + v * ca) * red;
        // Terrain: slopes block flow that tries to climb them and speed up
        // flow that descends them.
        const hx = this.hx[k], hy = this.hy[k], s2 = hx * hx + hy * hy;
        if (s2 > 1e-7) {
          const comp = us * hx + vs * hy, sl = Math.sqrt(s2);
          if (comp > 0) {
            const block = clamp(sl * 9, 0, 0.75);
            us -= block * comp * hx / s2; vs -= block * comp * hy / s2;
          } else {
            const spd = Math.hypot(us, vs) + 0.1;
            const boost = 1 + clamp(-comp / spd * sl * 14, 0, 0.9);
            us *= boost; vs *= boost;
          }
        }
        this.u[k] = us; this.v[k] = vs;
      }
    }
    this._seaBreeze();

    // Lift (m/s at ~1.5 km).
    const tau = 600;
    for (let j = 0; j < NY; j++) {
      const jm = Math.max(0, j - 1), jp = Math.min(NY - 1, j + 1);
      for (let i = 0; i < NX; i++) {
        const im = Math.max(0, i - 1), ip = Math.min(NX - 1, i + 1), k = j * NX + i;
        const dudx = (this.u[j * NX + ip] - this.u[j * NX + im]) / (this.dx[j] * (ip - im));
        const dvdy = (this.v[jm * NX + i] - this.v[jp * NX + i]) / (this.dy[j] * (jp - jm));
        const conv = -1000 * (dudx + dvdy);
        const gx = i - this.u[k] * tau / this.dx[j], gy = j + this.v[k] * tau / this.dy[j];
        const oro = clamp(0.8 * (this.u[k] * this.sample(this.hx, gx, gy) + this.v[k] * this.sample(this.hy, gx, gy)), -1.5, 1.5);
        const dTdx = (this.T[j * NX + ip] - this.T[j * NX + im]) / (this.dx[j] * (ip - im));
        const dTdy = (this.T[jm * NX + i] - this.T[jp * NX + i]) / (this.dy[j] * (jp - jm));
        const wa = clamp(-150 * (this.us[k] * dTdx + this.vs[k] * dTdy) * 0.7, -0.12, 0.12);
        // Synoptic ascent near lows (and subsidence under highs).
        this._tmp[k] = clamp(conv, -0.4, 0.4) + oro + wa + 0.12 * this.syn[k] + this.meso[k] + this.Fz[k];
      }
    }
    // Light smoothing removes grid-scale noise in the lift.
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      let s = 0, n = 0;
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const ii = i + di, jj = j + dj;
        if (ii >= 0 && jj >= 0 && ii < NX && jj < NY) { const wgt = di === 0 && dj === 0 ? 2 : 1; s += this._tmp[jj * NX + ii] * wgt; n += wgt; }
      }
      this.w[j * NX + i] = s / n;
    }

    // CAPE (parcel theory with a lightly mixed surface parcel) and cloud.
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      const k = j * NX + i;
      const ps = this.p[k] * Math.exp(-this.elev[k] / 8000);
      const Tc = this.T[k], qs = qsat(Tc, ps), rh = clamp(this.q[k] / qs, 0, 1.05);
      // Mixed-layer-ish parcel: a bit cooler and drier than the surface.
      const r = this.q[k] * 0.88;
      const the = thetaE(Tc - 0.5 + 273.15, ps, r);
      const li = (this.t500[k] + 273.15) - parcelT500(the);
      const cin = clamp((0.4 - rh) / 0.25, 0, 0.8);
      this.cape[k] = clamp(-li * 160, 0, 6000) * (1 - cin);
      const rhe = 0.45 * rh + 0.55 * this.M[k];
      this.cf[k] = clamp(clamp((rhe - 0.55) / 0.35, 0, 1) * 0.85 + 3 * this.C[k] + this.anvil[k], 0, 1);
      const spd = Math.hypot(this.u[k], this.v[k]);
      this.gust[k] = spd * (1.25 + 0.2 * this.land[k] + 0.25 * clamp(this.rough[k] / 300, 0, 1)) + this.out[k];
    }
  }

  _seaBreeze() {
    const { NX, NY, P } = this;
    if (!P.breeze) return;
    for (let j = 0; j < NY; j++) {
      const sst = this.sstAt(j);
      for (let i = 0; i < NX; i++) {
        const k = j * NX + i;
        const g = Math.hypot(this.lx[k], this.ly[k]);
        if (g < 0.01) continue;
        // Land warmer than the sea -> onshore (toward land, up the land gradient).
        const dT = clamp(this.T[k] - sst, -6, 10);
        const mag = P.breeze * 0.55 * dT * clamp(g * 4, 0, 1);
        this.u[k] += mag * this.lx[k] / g;
        this.v[k] += mag * this.ly[k] / g;
      }
    }
  }

  // ---------- advection ----------
  _advect(field, u, v, dt) {
    const { NX, NY } = this, src = this._tmp;
    src.set(field);
    for (let j = 0; j < NY; j++) {
      const ddx = dt / this.dx[j], ddy = dt / this.dy[j];
      for (let i = 0; i < NX; i++) {
        const k = j * NX + i;
        field[k] = this.sample(src, i - u[k] * ddx, j + v[k] * ddy);
      }
    }
  }

  // ---------- time stepping ----------
  step(seconds) {
    const n = Math.max(1, Math.ceil(seconds / 300));
    const dt = seconds / n;
    for (let s = 0; s < n; s++) this._substep(dt);
  }

  _substep(dt) {
    this.diagnose();
    this._advect(this.T, this.u, this.v, dt);
    this._advect(this.q, this.u, this.v, dt);
    this._advect(this.out, this.u, this.v, dt);
    this._advect(this.C, this.us, this.vs, dt);
    this._advect(this.Fz, this.us, this.vs, dt);
    this._advect(this.M, this.us, this.vs, dt);
    for (const f of [this.tAx, this.tAy, this.tBx, this.tBy]) this._advect(f, this.us, this.vs, dt);
    this._texEdges();
    const [su, sv] = windFrom(this.P.steerDir, this.P.steerSpd);
    this.pox += su * 0.8 * dt / 1000; this.poy += sv * 0.8 * dt / 1000;
    this._physics(dt);
    this._convection(dt);
    this._systems(dt);
    this.time += dt * 1000;
    const ph = this.texPhase();
    if (ph < this._phase) this._resetTex('A');               // A's weight is 0 at phase 0
    else if (this._phase < 0.5 && ph >= 0.5) this._resetTex('B'); // B's weight is 0 at phase 0.5
    this._phase = ph;
    const keep = this.time - 3 * 3600 * 1000;
    if (this.strikes.length && this.strikes[0].t < keep) {
      let c = 0; while (c < this.strikes.length && this.strikes[c].t < keep) c++;
      this.strikes.splice(0, c);
    }
  }

  _physics(dt) {
    const { NX, NY } = this;
    const fDecay = Math.exp(-dt / 10800), oDecay = Math.exp(-dt / 1800);
    for (let j = 0; j < NY; j++) {
      const edge = j < 2 || j >= NY - 2;
      for (let i = 0; i < NX; i++) {
        const k = j * NX + i, lf = this.land[k];
        const teq = this.tEq(k, j);
        const tauT = (lf * 3 + (1 - lf) * 10) * 3600;
        this.T[k] += (teq - this.T[k]) * Math.min(1, dt / tauT);
        if (this.T[k] < teq - 8) this.T[k] = teq - 8; // cold pools bottom out
        const qe = this.qEq(k, j);
        const spd = Math.hypot(this.u[k], this.v[k]);
        const tauQ = (lf * 30 + (1 - lf) * 8 / (1 + spd / 8)) * 3600;
        this.q[k] += (qe - this.q[k]) * Math.min(1, dt / tauQ);
        const mt = clamp(this.P.rh / 100 + 0.2 * this.syn[k], 0.05, 0.98);
        this.M[k] += (mt - this.M[k]) * Math.min(1, dt / (8 * 3600));
        if (edge || i < 2 || i >= NX - 2) {
          this.M[k] += (mt - this.M[k]) * Math.min(1, dt / 1800);
          this.T[k] += (teq - this.T[k]) * Math.min(1, dt / 1800);
          this.q[k] += (qe - this.q[k]) * Math.min(1, dt / 1800);
          this.C[k] *= Math.exp(-dt / 1800);
        }
        const ps = this.p[k] * Math.exp(-this.elev[k] / 8000);
        const qs = qsat(this.T[k], ps);
        if (this.q[k] > 1.03 * qs) this.q[k] = 1.03 * qs;
        if (this.q[k] < 0.3) this.q[k] = 0.3;
        const rh = this.q[k] / qs;
        const w = this.w[k];
        let C = this.C[k];
        // Condensation in rising air; a deep moist layer (surface RH and
        // mid-level humidity together) is needed for widespread rain.
        if (w > 0) {
          const sat = clamp((0.45 * rh + 0.55 * this.M[k] - 0.7) / 0.22, 0, 1);
          const cond = RHO * this.q[k] * 1e-3 * w * sat * 0.5 * dt;
          C += cond; this.q[k] -= cond * 0.33;
        } else if (C > 0) {
          const ev = C * Math.min(1, -w * dt / 700);
          C -= ev; this.q[k] += ev * 0.33;
        }
        if (C > 0 && rh < 0.6) {
          const ev = C * Math.min(1, dt / 1800 * (0.6 - rh) / 0.3);
          C -= ev; this.q[k] += ev * 0.33;
        }
        const pr = C * Math.min(1, dt / 900);
        C -= pr;
        this.C[k] = C;
        const R = pr / dt * 3600;
        this.R[k] = R;
        this.acc[k] += pr;
        if (R > 0.05) this.T[k] -= R * dt / 3600 * 0.15;
        this.Fz[k] *= fDecay;
        this.out[k] *= oDecay;
      }
    }
  }

  // ---------- pressure systems ----------
  _systems(dt) {
    const [su, sv] = windFrom(this.P.steerDir, this.P.steerSpd);
    for (const S of this.systems) {
      const k = S.dp < 0 ? 0.75 : 0.45;
      S.lat += sv * k * dt / 111320;
      S.lon += su * k * dt / (111320 * Math.cos(S.lat * D2R));
      if (S.trend) S.dp += S.trend * dt / 3600;
      if (this.inside(S.lon, S.lat) && this.sampleLL(this.land, S.lon, S.lat) > 0.6 && S.dp < 0) S.dp += 0.08 * dt / 3600;
    }
    const kmCell = this.dx[this.NY >> 1] / 1000;
    this.systems = this.systems.filter(S => Math.abs(S.dp) >= 1 && Math.abs(S.lat) < 80 &&
      !(S.auto && !this.inside(S.lon, S.lat, 2.5 * S.R / kmCell)));
  }

  // ---------- convection ----------
  cellDbz(c) {
    const a = c.age / c.life;
    let env;
    if (c.type === 'super') env = a < 0.1 ? Math.sin(Math.PI / 2 * a / 0.1) : a < 0.82 ? 1 : Math.max(0, Math.cos(Math.PI / 2 * (a - 0.82) / 0.18));
    else env = a < 0.25 ? Math.sin(Math.PI / 2 * a / 0.25) : a < 0.55 ? 1 : Math.max(0, Math.cos(Math.PI / 2 * (a - 0.55) / 0.45));
    const pulse = c.type === 'super' ? 1.5 * Math.sin(c.age / 420 + c.seed) : 0;
    return 12 + (c.peak - 12) * env + pulse * env;
  }
  cellTop(c) {
    const a = c.age / c.life;
    const env = a < 0.3 ? 0.35 + 0.65 * a / 0.3 : 1 - 0.3 * Math.max(0, a - 0.7) / 0.3;
    return c.top * env;
  }

  addCell(lat, lon, opts = {}) {
    const P = this.P;
    const capeHere = this.inside(lon, lat) ? this.sampleLL(this.cape, lon, lat) : 800;
    const type = opts.type || 'single';
    const r = this.rng;
    const c = {
      id: nextId++, lat, lon, age: opts.age ?? 0, type,
      peak: opts.peak ?? clamp(38 + 15 * Math.log10(1 + capeHere / 250) + (r() * 8 - 4) + (type === 'super' ? 7 : 0), 30, 70),
      r: opts.r ?? (type === 'super' ? 5 + r() * 2.5 : 2.4 + capeHere / 1200 + r() * 1.6),
      top: opts.top ?? clamp(6.5 + capeHere / 450 + r() * 1.5 + (type === 'super' ? 3 : 0), 5, 16),
      life: opts.life ?? (type === 'super' ? (150 + r() * 120) * 60 : (40 + r() * 30) * 60),
      gen: opts.gen ?? 0, seed: Math.floor(r() * 1e6), lineId: opts.lineId || 0,
      devx: 0, devy: 0, mvx: 0, mvy: 0, spawned: false, forced: !!opts.forced,
    };
    if (opts.peak == null && opts.boost) c.peak = clamp(c.peak + opts.boost, 30, 70);
    const [su, sv] = windFrom(P.steerDir, P.steerSpd);
    const sl = Math.hypot(su, sv) + 1e-6;
    if (type === 'super') {
      // Right-mover: ~7.5 m/s to the right of the (steering-parallel) shear.
      c.devx = 7.5 * sv / sl; c.devy = -7.5 * su / sl;
    } else if (type === 'line') {
      c.devx = 3.5 * su / sl; c.devy = 3.5 * sv / sl;
    }
    if (opts.dev) { c.devx = opts.dev[0]; c.devy = opts.dev[1]; }
    this.cells.push(c);
    return c;
  }

  addLine(lat1, lon1, lat2, lon2, opts = {}) {
    const lineId = this.lineSeq++;
    const lenKm = Math.hypot((lat2 - lat1) * 111.32, (lon2 - lon1) * 111.32 * Math.cos(lat1 * D2R));
    const n = Math.max(2, Math.round(lenKm / 9));
    for (let s = 0; s <= n; s++) {
      const t = s / n + (this.rng() - 0.5) * 0.3 / n;
      this.addCell(lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t,
        { type: 'line', lineId, age: this.rng() * 900, life: (50 + this.rng() * 25) * 60, ...opts });
    }
  }

  _deposit(c, dbz, dt) {
    const { NX, NY } = this;
    const [gx, gy] = this.gridXY(c.lon, c.lat);
    if (gx < -10 || gy < -10 || gx > NX + 10 || gy > NY + 10) return;
    const j0 = clamp(Math.round(gy), 0, NY - 1);
    const dxk = this.dx[j0] / 1000, dyk = this.dy[j0] / 1000;
    const re = c.r * 1.4;
    const Rcore = Math.min(150, Math.pow(Math.pow(10, Math.min(dbz, 56) / 10) / 300, 1 / 1.4));
    if (Rcore < 0.1) return;
    const rx = re / dxk, ry = re / dyk;
    const ext = Math.ceil(Math.max(rx, ry) * 3 + 1);
    const ia = Math.max(0, Math.floor(gx - ext)), ib = Math.min(NX - 1, Math.ceil(gx + ext));
    const ja = Math.max(0, Math.floor(gy - ext)), jb = Math.min(NY - 1, Math.ceil(gy + ext));
    let wsum = 0;
    for (let j = ja; j <= jb; j++) for (let i = ia; i <= ib; i++) {
      const ddx = (i - gx) / Math.max(rx, 0.5), ddy = (j - gy) / Math.max(ry, 0.5);
      wsum += Math.max(0, Math.exp(-(ddx * ddx + ddy * ddy)) - 0.003);
    }
    if (wsum <= 0) return;
    const vol = Rcore * Math.PI * re * re * 0.5; // mm/h * km2
    const cellA = dxk * dyk;
    const [su, sv] = [c.mvx, c.mvy], sl = Math.hypot(su, sv) + 1e-6;
    const env = this.cells.length; void env;
    // Trailing/anvil stratiform fallout behind the motion.
    const stratK = c.type === 'line' ? 1 : c.type === 'multi' ? 0.45 : c.type === 'super' ? 0.5 : 0.2;
    const bgx = gx - su / sl * 1.6 * re / dxk, bgy = gy + sv / sl * 1.6 * re / dyk;
    for (let j = ja; j <= jb; j++) for (let i = ia; i <= ib; i++) {
      const k = j * NX + i;
      const ddx = (i - gx) / Math.max(rx, 0.5), ddy = (j - gy) / Math.max(ry, 0.5);
      const wgt = Math.max(0, Math.exp(-(ddx * ddx + ddy * ddy)) - 0.003) / wsum;
      if (wgt <= 0) continue;
      const R = vol * wgt / cellA;
      this.Rc[k] += R;
      this.acc[k] += R * dt / 3600;
      this.T[k] -= dt * 0.0014 * Math.min(1, R / 25);
      this.q[k] -= dt * 0.0002 * Math.min(1, R / 25);
      this.M[k] += (0.92 - this.M[k]) * Math.min(1, dt / 2400) * R / (R + 8); // detrainment moistens mid-levels
      const og = Math.min(17, 2.2 * Math.sqrt(R)) * (c.type === 'super' ? 1.35 : 1);
      if (og > this.out[k]) this.out[k] += (og - this.out[k]) * Math.min(1, dt / 300);
    }
    // Fallout gets its own box around the rear point so the Gaussian is not cut off.
    const fr = Math.max(rx * 1.8, 0.7), frY = Math.max(ry * 1.8, 0.7), fext = Math.ceil(Math.max(fr, frY) * 2.6 + 1);
    for (let j = Math.max(0, Math.floor(bgy - fext)); j <= Math.min(NY - 1, Math.ceil(bgy + fext)); j++) {
      for (let i = Math.max(0, Math.floor(bgx - fext)); i <= Math.min(NX - 1, Math.ceil(bgx + fext)); i++) {
        const bx = (i - bgx) / fr, by = (j - bgy) / frY;
        const wb = Math.exp(-(bx * bx + by * by));
        if (wb > 0.002) this.C[j * NX + i] += stratK * 0.0016 * Math.min(1, Rcore / 40) * wb * dt;
      }
    }
    // Anvil cloud for the satellite view and cloud fraction.
    const ar = c.r * 5 / dxk;
    const aext = Math.ceil(ar * 1.6 + 1);
    for (let j = Math.max(0, Math.floor(gy - aext)); j <= Math.min(NY - 1, Math.ceil(gy + aext)); j++) {
      for (let i = Math.max(0, Math.floor(gx - aext)); i <= Math.min(NX - 1, Math.ceil(gx + aext)); i++) {
        const d2 = ((i - gx) ** 2 + (j - gy) ** 2) / Math.max(ar * ar, 0.25);
        const a = Math.exp(-d2) * clamp((dbz - 20) / 25, 0, 1);
        const k = j * NX + i;
        if (a > this.anvil[k]) this.anvil[k] = a;
      }
    }
  }

  _convection(dt) {
    const { NX, NY, P } = this;
    const rnd = this.rng;
    this.Rc.fill(0);
    for (let k = 0; k < this.N; k++) this.anvil[k] *= Math.exp(-dt / 2400);
    const [su, sv] = windFrom(P.steerDir, P.steerSpd);
    const born = [];

    for (const c of this.cells) {
      const inGrid = this.inside(c.lon, c.lat, 2);
      const capeHere = inGrid ? this.sampleLL(this.cape, c.lon, c.lat) : 600;
      const wHere = inGrid ? this.sampleLL(this.w, c.lon, c.lat) : 0;
      let ux = su, vy = sv;
      if (inGrid) { ux = this.sampleLL(this.us, c.lon, c.lat); vy = this.sampleLL(this.vs, c.lon, c.lat); }
      c.mvx = ux + c.devx; c.mvy = vy + c.devy;
      c.lat += c.mvy * dt / 111320;
      c.lon += c.mvx * dt / (111320 * Math.cos(c.lat * D2R));
      // Stable air ages a cell faster; persistent lift keeps it going.
      let rate = 1;
      const stable = c.forced ? 0.5 : 1;
      if (capeHere < 150) rate += 1.6 * stable;
      else if (capeHere < 400) rate += 0.5 * stable;
      if (wHere > 0.15) rate -= 0.25;
      c.age += dt * Math.max(0.5, rate);
      const dbz = this.cellDbz(c);
      c.dbz = dbz;
      if (inGrid) this._deposit(c, dbz, dt);

      // Lightning.
      const top = this.cellTop(c);
      if (dbz > 40 && top > 6) {
        const perMin = Math.min(45, 0.45 * Math.exp((dbz - 45) / 4.2) * (top / 10) * (c.type === 'super' ? 1.8 : 1));
        let lam = perMin * dt / 60;
        while (lam > 0) {
          if (rnd() < Math.min(1, lam)) {
            const far = rnd() < 0.12;
            const rr = c.r * (far ? 2.5 + rnd() * 2 : 0.9 * Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * 0.6);
            const th = rnd() * 2 * Math.PI;
            const la = c.lat + rr * Math.sin(th) / 111.32;
            const lo = c.lon + rr * Math.cos(th) / (111.32 * Math.cos(c.lat * D2R));
            this.strikes.push({ lat: la, lon: lo, t: this.time + rnd() * dt * 1000, cg: rnd() < 0.3 });
          }
          lam -= 1;
        }
      }

      // Daughter cells.
      const a = c.age / c.life;
      const capeF = clamp((capeHere - 200) / 900, 0, 2.5);
      const speed = Math.hypot(c.mvx, c.mvy) + 1e-6;
      const th0 = Math.atan2(c.mvy, c.mvx);
      if ((c.type === 'multi' || c.type === 'super') && a > 0.18 && a < 0.7 && c.gen < 60) {
        // About one daughter per cell (more with big CAPE), so multicell
        // clusters persist without exploding.
        const expected = capeHere < 400 ? 0 : c.type === 'multi' ? 0.5 + 0.35 * Math.min(capeF, 1.5) : 0.18 * Math.min(capeF, 2) * c.life / 3600;
        const perMin = expected / (0.52 * c.life / 60);
        if (rnd() < perMin * dt / 60) {
          let th = th0 - 50 * D2R + (rnd() - 0.5) * 70 * D2R;
          const oro = inGrid ? this.sampleLL(this.w, c.lon, c.lat) : 0;
          if (oro > 0.12 && speed < 11 && rnd() < 0.55) th = th0 + Math.PI + (rnd() - 0.5) * 60 * D2R;
          const dist = (1.8 + rnd() * 1.2) * c.r * (c.type === 'super' ? 1.6 : 1);
          born.push({ parent: c.id, lat: c.lat + dist * Math.sin(th) / 111.32, lon: c.lon + dist * Math.cos(th) / (111.32 * Math.cos(c.lat * D2R)),
            opts: { type: 'multi', gen: c.gen + 1, peak: clamp((c.type === 'super' ? c.peak - 12 : c.peak) + rnd() * 7 - 4.5 + (capeF - 1) * 2, 30, 66), r: c.r * (0.8 + rnd() * 0.35) * (c.type === 'super' ? 0.6 : 1) } });
        }
      }
      if (c.type === 'line' && !c.spawned && a > 0.42) {
        c.spawned = true;
        if (capeHere > 180 || wHere > 0.2 || (c.forced && c.gen < 6)) {
          const dist = 0.8 * c.r;
          const lat = c.lat + (dist * Math.sin(th0)) / 111.32 + (rnd() - 0.5) * 0.4 * c.r / 111.32;
          const lon = c.lon + (dist * Math.cos(th0)) / (111.32 * Math.cos(c.lat * D2R));
          born.push({ parent: c.id, lat, lon, opts: { type: 'line', lineId: c.lineId, gen: c.gen + 1, forced: c.forced, r: c.r * (0.9 + rnd() * 0.2), dev: [c.devx, c.devy],
            peak: clamp(38 + 12 * Math.log10(1 + capeHere / 300) + rnd() * 8 - 3 + (wHere > 0.2 ? 3 : 0), 30, 64), life: (45 + rnd() * 25) * 60 } });
        }
      }
    }
    this.cells = this.cells.filter(c => c.age < c.life && this.inside(c.lon, c.lat, 25));
    for (const b of born) if (!this._crowded(b.lat, b.lon, 1.3 * (b.opts.r || 3), b.parent)) this.addCell(b.lat, b.lon, b.opts);

    this._initiate(dt);
  }

  _crowded(lat, lon, km, except = -1) {
    const cl = Math.cos(lat * D2R);
    for (const c of this.cells) {
      if (c.id === except) continue;
      const dy = (c.lat - lat) * 111.32, dx = (c.lon - lon) * 111.32 * cl;
      if (dx * dx + dy * dy < km * km && c.age / c.life < 0.75) return true;
    }
    return false;
  }

  _initiate(dt) {
    const { NX, NY, P } = this;
    const MAX = 420;
    if (this.cells.length >= MAX || P.trigger <= 0) return;
    const rnd = this.rng;
    for (let j = 1; j < NY - 1; j++) {
      const areaK = this.dx[j] * this.dy[j] / 1e6;
      const sst = this.sstAt(j), t850 = this.tBase(j) - 8;
      for (let i = 1; i < NX - 1; i++) {
        const k = j * NX + i, cape = this.cape[k];
        if (cape < 150) continue;
        const lf = this.land[k];
        let trig = clamp((this.w[k] - 0.04) / 0.22, 0, 1.2);
        const cz = this.cosz[k];
        if (lf > 0.5 && cz > 0.25) trig += P.sun * (cz - 0.25) * 1.8 * (1 - this.cf[k]) * (this.elev[k] > 250 ? 1 + this.elev[k] / 1400 : 0.45);
        const go = Math.abs(this.out[k + 1] - this.out[k - 1]) + Math.abs(this.out[k + NX] - this.out[k - NX]);
        trig += clamp(go / 6, 0, 1.4);
        // Cold air over a warm sea (SST - T850 > ~10 K): the classic autumn
        // and winter Adriatic/Mediterranean shower and waterspout setup.
        const seaTrig = lf < 0.5 ? clamp((sst - t850 - 10) / 5, 0, 1.6) : 0;
        if (trig <= 0.02 && seaTrig <= 0) continue;
        const capeF = clamp((cape - 150) / 700, 0, 3);
        const rate = (0.05 * trig + 0.11 * seaTrig) * capeF * P.trigger; // per 1000 km2 per hour
        if (rnd() < rate * areaK / 1000 * dt / 3600) {
          const lon = this.lon[i] + (rnd() - 0.5) * (this.lon[1] - this.lon[0]);
          const lat = this.lat[j] + (rnd() - 0.5) * (this.lat[Math.min(j + 1, NY - 1)] - this.lat[j]);
          if (this._crowded(lat, lon, 11)) continue;
          const S = P.shear;
          let type = 'single';
          const u = rnd();
          if (S < 10) type = u < 0.3 * Math.min(1, capeF) ? 'multi' : 'single';
          else if (S < 18) type = u < 0.75 ? 'multi' : 'single';
          else type = u < 0.28 * Math.min(1, cape / 1500) ? 'super' : 'multi';
          this.addCell(lat, lon, { type });
          if (this.cells.length >= MAX) return;
        }
      }
    }
  }

  // ---------- user actions ----------
  _brush(lat, lon, km, fn) {
    const { NX, NY } = this;
    const [gx, gy] = this.gridXY(lon, lat);
    const j0 = clamp(Math.round(gy), 0, NY - 1);
    const rx = km * 1000 / this.dx[j0], ry = km * 1000 / this.dy[j0];
    for (let j = Math.max(0, Math.floor(gy - ry)); j <= Math.min(NY - 1, Math.ceil(gy + ry)); j++) {
      for (let i = Math.max(0, Math.floor(gx - rx)); i <= Math.min(NX - 1, Math.ceil(gx + rx)); i++) {
        const d2 = ((i - gx) / rx) ** 2 + ((j - gy) / ry) ** 2;
        if (d2 > 1) continue;
        const wgt = 0.5 * (1 + Math.cos(Math.PI * Math.sqrt(d2)));
        fn(j * NX + i, wgt, j);
      }
    }
  }

  paintRain(lat, lon, km, strength) {
    this._brush(lat, lon, km, (k, w) => {
      this.Fz[k] = Math.min(1.2, this.Fz[k] + 0.12 * strength * w);
      this.C[k] = Math.min(12, this.C[k] + 0.6 * strength * w);
      this.R[k] = Math.max(this.R[k], this.C[k] * 4);
      this.M[k] += (0.97 - this.M[k]) * Math.min(1, 0.5 * w);
      const qs = qsat(this.T[k]);
      this.q[k] += (0.95 * qs - this.q[k]) * Math.min(1, 0.4 * w);
    });
  }
  paintMoisture(lat, lon, km, sign) {
    this._brush(lat, lon, km, (k, w) => {
      const qs = qsat(this.T[k]);
      this.q[k] = clamp(this.q[k] + sign * 0.9 * w, 0.3, 1.02 * qs);
      this.M[k] = clamp(this.M[k] + sign * 0.07 * w, 0.05, 0.99);
    });
  }
  paintHeat(lat, lon, km, sign) {
    this._brush(lat, lon, km, (k, w) => { this.T[k] += sign * 1.2 * w; });
  }
  erase(lat, lon, km) {
    this._brush(lat, lon, km, (k, w) => {
      this.C[k] *= 1 - w; this.Fz[k] *= 1 - w; this.R[k] *= 1 - w; this.Rc[k] *= 1 - w; this.out[k] *= 1 - w;
    });
    const cl = Math.cos(lat * D2R), near = (la, lo) => Math.hypot((la - lat) * 111.32, (lo - lon) * 111.32 * cl) < km;
    this.cells = this.cells.filter(c => !near(c.lat, c.lon));
    this.systems = this.systems.filter(s => !near(s.lat, s.lon));
    this.strikes = this.strikes.filter(s => !near(s.lat, s.lon));
  }
  addSystem(type, lat, lon, dp, R) {
    const S = { id: nextId++, type, lat, lon, dp: type === 'L' ? -Math.abs(dp) : Math.abs(dp), R, trend: 0 };
    this.systems.push(S);
    return S;
  }

  // Fronts: a temperature step across the line plus a lift band. The cold
  // side of a cold front is the side the steering wind comes from; a warm
  // front lifts air broadly on its cold (downwind) side.
  addFront(kind, lat1, lon1, lat2, lon2) {
    const { NX, NY } = this;
    const [su, sv] = windFrom(this.P.steerDir, this.P.steerSpd);
    const cl = Math.cos(((lat1 + lat2) / 2) * D2R);
    const ax = (lon2 - lon1) * 111.32 * cl, ay = (lat2 - lat1) * 111.32;
    const L = Math.hypot(ax, ay) || 1;
    let nx = -ay / L, ny = ax / L; // left normal
    if (nx * su + ny * sv > 0) { nx = -nx; ny = -ny; } // normal now points upwind (cold side for a cold front)
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      const k = j * NX + i;
      const px = (this.lon[i] - lon1) * 111.32 * cl, py = (this.lat[j] - lat1) * 111.32;
      const t = (px * ax + py * ay) / (L * L);
      const along = t < 0 ? -t * L : t > 1 ? (t - 1) * L : 0;
      const endFade = Math.exp(-((along / 120) ** 2));
      if (endFade < 0.02) continue;
      const d = px * nx + py * ny; // km toward the upwind side
      const band = Math.exp(-((d / 350) ** 2)) * endFade;
      if (kind === 'cold') {
        const s = 0.5 * (1 + Math.tanh(d / 15));
        this.T[k] -= 7 * s * band;
        this.q[k] -= this.q[k] * 0.3 * s * band;
        const lift = Math.exp(-(((d + 5) / 22) ** 2)) * endFade;
        this.Fz[k] = Math.min(1.2, this.Fz[k] + 0.35 * lift);
        this.M[k] += (0.9 - this.M[k]) * Math.min(1, lift);
        this.C[k] += 1.2 * lift;
      } else {
        const s = 0.5 * (1 - Math.tanh(d / 30)); // 1 ahead of the front (downwind, cold), 0 behind it
        this.T[k] += 5 * (1 - s) * band;
        this.q[k] += Math.max(0, 0.95 * qsat(this.T[k]) - this.q[k]) * 0.5 * band * (1 - s);
        const shield = Math.exp(-(((d + 120) / 110) ** 2)) * endFade; // cloud/rain shield ahead (downwind)
        this.Fz[k] = Math.min(1.0, this.Fz[k] + 0.16 * shield);
        this.M[k] += (0.93 - this.M[k]) * Math.min(1, 1.3 * shield);
        this.C[k] += 0.8 * shield;
      }
    }
    if (kind === 'cold') {
      const capeMid = this.inside((lon1 + lon2) / 2, (lat1 + lat2) / 2) ? this.sampleLL(this.cape, (lon1 + lon2) / 2, (lat1 + lat2) / 2) : 0;
      if (capeMid > 250) this.addLine(lat1, lon1, lat2, lon2);
    }
  }

  // Drag weather: shift fields within the brush and carry cells along.
  smudge(lat, lon, km, eastM, northM) {
    const fields = [this.T, this.q, this.C, this.Fz, this.out, this.anvil, this.M];
    const copies = fields.map(f => new Float32Array(f));
    const { NX } = this;
    this._brush(lat, lon, km, (k, w, j) => {
      const i = k - j * NX;
      const gx = i - w * eastM / this.dx[j], gy = j + w * northM / this.dy[j];
      for (let f = 0; f < fields.length; f++) fields[f][k] = this.sample(copies[f], gx, gy);
    });
    const cl = Math.cos(lat * D2R);
    for (const c of this.cells) {
      const d = Math.hypot((c.lat - lat) * 111.32, (c.lon - lon) * 111.32 * cl);
      if (d < km) {
        const w = 0.5 * (1 + Math.cos(Math.PI * d / km));
        c.lat += w * northM / 111320; c.lon += w * eastM / (111320 * Math.cos(c.lat * D2R));
      }
    }
  }

  // ---------- history & region change ----------
  snapshot() {
    return {
      time: this.time,
      f: [this.T, this.q, this.C, this.Fz, this.acc, this.out, this.anvil, this.R, this.Rc, this.M, this.tAx, this.tAy, this.tBx, this.tBy].map(a => new Float32Array(a)),
      pox: this.pox, poy: this.poy, texOff: { A: this.texOff.A.slice(), B: this.texOff.B.slice() }, phase: this._phase,
      cells: this.cells.map(c => ({ ...c })),
      systems: this.systems.map(s => ({ ...s })),
      strikes: this.strikes.slice(),
    };
  }
  restore(s) {
    this.time = s.time;
    [this.T, this.q, this.C, this.Fz, this.acc, this.out, this.anvil, this.R, this.Rc, this.M, this.tAx, this.tAy, this.tBx, this.tBy].forEach((a, i) => a.set(s.f[i]));
    this.pox = s.pox; this.poy = s.poy; this.texOff = { A: s.texOff.A.slice(), B: s.texOff.B.slice() }; this._phase = s.phase;
    this.cells = s.cells.map(c => ({ ...c }));
    this.systems = s.systems.map(x => ({ ...x }));
    this.strikes = s.strikes.slice();
    this.diagnose();
  }
  adoptFrom(old) {
    this.time = old.time;
    this.cells = old.cells; this.systems = old.systems; this.strikes = old.strikes; this.lineSeq = old.lineSeq;
    this.pox = old.pox; this.poy = old.poy; this._phase = old._phase;
    this._resetTex('A', old.texOff.A); this._resetTex('B', old.texOff.B);
    const { NX, NY } = this;
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      const [gx, gy] = old.gridXY(this.lon[i], this.lat[j]);
      if (gx < 0 || gy < 0 || gx > old.NX - 1 || gy > old.NY - 1) continue;
      const k = j * NX + i;
      const de = this.elev[k] - old.sample(old.elev, gx, gy);
      this.T[k] = old.sample(old.T, gx, gy) - 0.0065 * de;
      this.q[k] = old.sample(old.q, gx, gy);
      this.C[k] = old.sample(old.C, gx, gy);
      this.Fz[k] = old.sample(old.Fz, gx, gy);
      this.acc[k] = old.sample(old.acc, gx, gy);
      this.out[k] = old.sample(old.out, gx, gy);
      this.anvil[k] = old.sample(old.anvil, gx, gy);
      this.M[k] = old.sample(old.M, gx, gy);
      this.tAx[k] = old.sample(old.tAx, gx, gy); this.tAy[k] = old.sample(old.tAy, gx, gy);
      this.tBx[k] = old.sample(old.tBx, gx, gy); this.tBy[k] = old.sample(old.tBy, gx, gy);
    }
    this.diagnose();
  }
}
