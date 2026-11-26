// Basemaps, colour scales and vector overlays.

import { distKm, offsetLL } from './geo.js';
import { frontGeom } from './sim.js';

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

// ---------- colour scales ----------
function lerpStops(stops, v) {
  if (v <= stops[0][0]) return stops[0].slice(1);
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const a = stops[i - 1], b = stops[i], t = (v - a[0]) / (b[0] - a[0]);
      return [a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t, (a[4] ?? 1) + ((b[4] ?? 1) - (a[4] ?? 1)) * t];
    }
  }
  return stops[stops.length - 1].slice(1);
}

// Builds a 1024-entry RGBA lookup for fast per-pixel colouring.
export function makeLUT(stops, min, max, discrete = false) {
  const lut = new Uint8ClampedArray(1024 * 4);
  for (let i = 0; i < 1024; i++) {
    const v = min + (max - min) * i / 1023;
    let c;
    if (discrete) {
      c = [0, 0, 0, 0];
      for (const s of stops) if (v >= s[0]) c = [s[1], s[2], s[3], s[4] ?? 1];
    } else c = lerpStops(stops, v);
    lut[i * 4] = c[0]; lut[i * 4 + 1] = c[1]; lut[i * 4 + 2] = c[2]; lut[i * 4 + 3] = Math.round((c[3] ?? 1) * 255);
  }
  return { lut, min, max, stops, discrete };
}

export const RADAR_PALETTES = {
  dhmz: [[2, 165, 165, 165], [10, 0, 150, 219], [15, 0, 85, 190], [20, 0, 78, 128], [25, 0, 150, 10], [30, 0, 192, 39],
    [35, 0, 232, 10], [40, 255, 255, 0], [45, 255, 187, 0], [50, 255, 131, 0], [55, 255, 0, 0], [60, 161, 0, 0],
    [65, 115, 0, 112], [70, 200, 120, 230]],
  nws: [[5, 4, 233, 231], [10, 1, 159, 244], [15, 3, 0, 244], [20, 2, 253, 2], [25, 1, 197, 1], [30, 0, 142, 0],
    [35, 253, 248, 2], [40, 229, 188, 0], [45, 253, 149, 0], [50, 253, 0, 0], [55, 212, 0, 0], [60, 188, 0, 0],
    [65, 248, 0, 253], [70, 152, 84, 198], [75, 253, 253, 253]],
  opera: [[0, 10, 130, 200, 0.85], [8, 10, 185, 175], [12, 5, 205, 170], [18, 140, 230, 20], [24, 240, 240, 20],
    [30, 255, 205, 20], [34, 255, 150, 50], [40, 255, 80, 60], [45, 250, 120, 255], [50, 190, 255, 255]],
};

export const SCALES = {
  temp: { unit: '°C', min: -30, max: 42, stops: [[-30, 145, 0, 200], [-20, 80, 30, 190], [-10, 30, 90, 225], [-0.01, 70, 170, 245],
    [0, 60, 195, 225], [4, 40, 200, 170], [8, 90, 205, 100], [12, 165, 220, 70], [16, 230, 230, 60], [20, 255, 200, 40], [24, 255, 150, 30],
    [28, 245, 95, 30], [32, 220, 40, 40], [36, 170, 0, 60], [42, 110, 0, 90]], ticks: [-20, -10, 0, 10, 20, 30, 40] },
  dew: { unit: '°C', min: -15, max: 28, stops: [[-15, 140, 100, 60], [0, 205, 175, 115], [8, 175, 215, 120], [13, 70, 185, 85],
    [17, 0, 145, 110], [21, 0, 95, 165], [25, 70, 40, 170], [28, 140, 30, 160]], ticks: [-10, 0, 8, 13, 17, 21, 25] },
  rh: { unit: '%', min: 0, max: 100, stops: [[0, 165, 105, 55], [35, 220, 185, 120], [55, 235, 228, 190], [75, 120, 195, 120],
    [90, 30, 140, 170], [100, 20, 80, 160]], ticks: [20, 40, 60, 80, 100] },
  wind: { unit: 'km/h', min: 0, max: 160, stops: [[0, 50, 80, 150, 0.15], [8, 60, 115, 195, 0.55], [18, 40, 170, 190, 0.7], [30, 60, 190, 110, 0.75],
    [42, 185, 210, 60, 0.8], [55, 250, 200, 30, 0.85], [68, 250, 130, 20, 0.9], [85, 230, 45, 40, 0.92], [105, 180, 20, 120, 0.95],
    [130, 120, 40, 175], [160, 235, 205, 255]], ticks: [10, 30, 50, 70, 90, 120, 150] },
  rain: { unit: 'mm/h', min: 0, max: 100, log: true, stops: [[0.05, 150, 200, 255, 0], [0.1, 150, 200, 255, 0.6], [0.5, 80, 150, 240, 0.8], [1, 30, 110, 230, 0.9],
    [2, 20, 170, 90], [5, 130, 210, 30], [10, 255, 220, 0], [20, 255, 140, 0], [40, 230, 30, 30], [80, 170, 0, 140]], ticks: [0.1, 1, 2, 5, 10, 20, 50] },
  acc: { unit: 'mm', min: 0, max: 300, log: true, stops: [[0.1, 170, 210, 255, 0], [0.3, 170, 210, 255, 0.55], [1, 110, 170, 245, 0.75], [3, 50, 120, 230, 0.85],
    [7, 30, 170, 100, 0.9], [15, 140, 210, 40], [25, 255, 225, 0], [40, 255, 150, 0], [60, 235, 50, 30], [100, 175, 0, 120], [180, 110, 0, 170], [300, 230, 190, 255]], ticks: [1, 5, 10, 25, 50, 100, 200] },
  cape: { unit: 'J/kg', min: 0, max: 5000, stops: [[0, 0, 0, 0, 0], [100, 120, 190, 120, 0.25], [400, 180, 220, 90, 0.55], [800, 250, 220, 50, 0.7],
    [1500, 255, 150, 30, 0.8], [2500, 230, 40, 30, 0.85], [3500, 170, 0, 120, 0.9], [5000, 240, 180, 255, 0.95]], ticks: [100, 500, 1000, 2000, 3000, 4000] },
  mslp: { unit: 'hPa', min: 960, max: 1050, stops: [[960, 110, 40, 150], [980, 70, 110, 200], [995, 120, 180, 220], [1005, 200, 225, 235],
    [1013, 240, 240, 230], [1020, 245, 215, 160], [1030, 230, 150, 100], [1050, 180, 60, 60]], ticks: [970, 990, 1000, 1010, 1020, 1030, 1040] },
  top: { unit: 'km', min: 0, max: 17, stops: [[0, 0, 0, 0, 0], [1.5, 120, 150, 200, 0.6], [4, 60, 160, 220], [6, 40, 190, 110], [8, 230, 230, 40],
    [10, 250, 150, 30], [12, 230, 40, 40], [14, 200, 60, 200], [17, 255, 255, 255]], ticks: [2, 4, 6, 8, 10, 12, 14, 16] },
};
for (const s of Object.values(SCALES)) {
  if (s.log) {
    // log scale LUT indexed by log10(v)
    const lmin = Math.log10(s.stops[0][0]), lmax = Math.log10(s.max);
    const ls = s.stops.map(x => [Math.log10(x[0]), ...x.slice(1)]);
    s.lut = makeLUT(ls, lmin, lmax);
    s.lut.log = true;
  } else s.lut = makeLUT(s.stops, s.min, s.max);
}

export function lutColor(L, v, out, o) {
  let x = L.log ? (v > 0 ? Math.log10(v) : -9) : v;
  let i = Math.round((x - L.min) / (L.max - L.min) * 1023);
  i = i < 0 ? 0 : i > 1023 ? 1023 : i;
  out[o] = L.lut[i * 4]; out[o + 1] = L.lut[i * 4 + 1]; out[o + 2] = L.lut[i * 4 + 2]; out[o + 3] = L.lut[i * 4 + 3];
}

// ---------- basemap ----------
const STYLE = {
  terrain: {
    land: [[0, 104, 148, 86], [150, 132, 166, 98], [400, 170, 178, 112], [800, 190, 170, 118], [1300, 164, 136, 98], [1900, 146, 122, 104], [2500, 185, 180, 175], [3200, 245, 245, 245]],
    sea: [[-3000, 22, 58, 118], [-1000, 36, 86, 152], [-200, 62, 122, 182], [-30, 100, 158, 205], [0, 125, 176, 214]],
    shade: 0.75, coast: [50, 70, 90, 0.45],
  },
  radar: {
    land: [[0, 247, 240, 200], [300, 247, 230, 189], [800, 239, 214, 189], [1400, 230, 205, 160], [2100, 212, 175, 120]],
    sea: [[-5000, 189, 189, 255], [0, 189, 189, 255]],
    landOut: [[0, 138, 161, 132], [500, 170, 165, 120], [1000, 212, 155, 95], [1800, 149, 108, 66]],
    seaOut: [[-5000, 132, 132, 186], [0, 132, 132, 186]],
    shade: 0.35, coast: [40, 40, 70, 0.75],
  },
  dark: {
    land: [[0, 48, 52, 50], [800, 66, 66, 62], [2000, 88, 86, 82], [3000, 120, 120, 120]],
    sea: [[-4000, 12, 18, 30], [0, 22, 30, 46]],
    shade: 0.6, coast: [130, 150, 170, 0.5],
  },
  light: {
    land: [[0, 236, 234, 226], [800, 220, 214, 202], [2000, 196, 188, 176], [3000, 230, 230, 230]],
    sea: [[-4000, 178, 200, 222], [0, 206, 222, 236]],
    shade: 0.45, coast: [90, 110, 130, 0.5],
  },
};

export function computeShade(dem, view) {
  const { w, h, elev, water } = dem;
  const shade = new Float32Array(w * h);
  const az = 315 * Math.PI / 180, alt = 40 * Math.PI / 180;
  const lx = Math.sin(az) * Math.cos(alt), ly = Math.cos(az) * Math.cos(alt), lz = Math.sin(alt);
  for (let y = 1; y < h - 1; y++) {
    const m = view.metersPerPx(view.lat(y));
    const exag = 1 + m / 140;
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (water[i]) { shade[i] = 0; continue; }
      const e = k => (water[k] ? 0 : elev[k]);
      const dzdx = (e(i + 1) - e(i - 1)) / (2 * m) * exag;
      const dzdy = (e(i - w) - e(i + w)) / (2 * m) * exag;
      const nz = 1 / Math.sqrt(dzdx * dzdx + dzdy * dzdy + 1);
      const dot = (-dzdx * lx - dzdy * ly + lz) * nz;
      shade[i] = clamp(dot - lz, -1, 1); // 0 on flat ground
    }
  }
  return shade;
}

export function renderBasemap(dem, view, styleName, shade, coverage) {
  const st = STYLE[styleName] || STYLE.terrain;
  const { w, h, elev, water } = dem;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  const img = g.createImageData(w, h), d = img.data;
  const lutL = makeLUT(st.land, 0, 3500), lutS = makeLUT(st.sea, -4000, 0);
  const lutLo = st.landOut && makeLUT(st.landOut, 0, 3500), lutSo = st.seaOut && makeLUT(st.seaOut, -4000, 0);
  const tmp = [0, 0, 0, 0];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, o = i * 4;
      const out = coverage && !coverage(x, y);
      if (water[i]) {
        lutColor(out && lutSo ? lutSo : lutS, Math.min(0, elev[i]), tmp, 0);
      } else {
        lutColor(out && lutLo ? lutLo : lutL, Math.max(0, elev[i]), tmp, 0);
        const s = 1 + shade[i] * st.shade * 1.6;
        tmp[0] *= s; tmp[1] *= s; tmp[2] *= s;
      }
      // Coastline: water pixel next to land.
      if (water[i] && x > 0 && y > 0 && x < w - 1 && y < h - 1 && (!water[i - 1] || !water[i + 1] || !water[i - w] || !water[i + w])) {
        const [cr, cg, cb, ca] = st.coast;
        tmp[0] += (cr - tmp[0]) * ca; tmp[1] += (cg - tmp[1]) * ca; tmp[2] += (cb - tmp[2]) * ca;
      }
      d[o] = tmp[0]; d[o + 1] = tmp[1]; d[o + 2] = tmp[2]; d[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  return c;
}

// ---------- gridded field -> image ----------
export function gridImage(sim, valueAt, scale) {
  const { NX, NY } = sim;
  const c = document.createElement('canvas');
  c.width = NX; c.height = NY;
  const g = c.getContext('2d');
  const img = g.createImageData(NX, NY), d = img.data;
  for (let k = 0; k < NX * NY; k++) {
    const v = valueAt(k);
    if (v == null || Number.isNaN(v)) { d[k * 4 + 3] = 0; continue; }
    lutColor(scale.lut, v, d, k * 4);
  }
  g.putImageData(img, 0, 0);
  return c;
}

// ---------- lines ----------
export function drawBorders(ctx, view, lines, color, width) {
  ctx.save();
  ctx.translate(view.ox, view.oy);
  ctx.beginPath();
  for (const line of lines) {
    let prevIn = false;
    for (let i = 0; i < line.length; i++) {
      const x = view.px(line[i][0]), y = view.py(line[i][1]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      prevIn = prevIn || (x > -50 && y > -50 && x < view.w + 50 && y < view.h + 50);
    }
  }
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
  ctx.stroke();
  ctx.restore();
}

export function drawCities(ctx, view, cities, opts) {
  const { maxRank, style, dpr, values } = opts;
  ctx.save();
  ctx.translate(view.ox, view.oy);
  const fs = Math.round(11 * dpr);
  ctx.font = `${style === 'radar' ? '' : '600 '}${fs}px ${style === 'radar' ? 'Verdana, sans-serif' : 'Inter, system-ui, sans-serif'}`;
  ctx.textBaseline = 'middle';
  const placed = [];
  for (const c of cities) {
    if (c[4] > maxRank) continue;
    const x = view.px(c[3]), y = view.py(c[2]);
    if (x < 4 || y < 4 || x > view.w - 4 || y > view.h - 4) continue;
    if (placed.some(p => Math.abs(p[0] - x) < 34 * dpr && Math.abs(p[1] - y) < 14 * dpr)) continue;
    placed.push([x, y]);
    const label = style === 'radar' ? c[1] : c[0];
    if (style === 'radar') {
      ctx.strokeStyle = '#222'; ctx.lineWidth = 1 * dpr;
      ctx.beginPath(); ctx.moveTo(x, y - 3 * dpr); ctx.lineTo(x + 3 * dpr, y); ctx.lineTo(x, y + 3 * dpr); ctx.lineTo(x - 3 * dpr, y); ctx.closePath(); ctx.stroke();
      ctx.fillStyle = '#222'; ctx.fillText(label, x + 3 * dpr, y - 8 * dpr);
    } else {
      const dark = style === 'dark' || style === 'satellite';
      ctx.fillStyle = dark ? '#fff' : '#1b1f24';
      ctx.beginPath(); ctx.arc(x, y, 2.2 * dpr, 0, 7); ctx.fill();
      ctx.lineWidth = 3 * dpr; ctx.strokeStyle = dark ? 'rgba(0,0,0,.65)' : 'rgba(255,255,255,.8)';
      let text = label;
      if (values) { const v = values(c); if (v != null) text = `${label} ${v}`; }
      ctx.strokeText(text, x + 5 * dpr, y);
      ctx.fillText(text, x + 5 * dpr, y);
    }
  }
  ctx.restore();
}

// Marching squares on the sim grid; returns segments in view pixels.
export function contourSegments(sim, field, level) {
  const { NX, NY, cpx, cpy } = sim;
  const segs = [];
  const P = (i, j) => [(i + 0.5) * cpx, (j + 0.5) * cpy];
  for (let j = 0; j < NY - 1; j++) for (let i = 0; i < NX - 1; i++) {
    const k = j * NX + i;
    const a = field[k], b = field[k + 1], c = field[k + NX + 1], d = field[k + NX];
    const idx = (a > level ? 8 : 0) | (b > level ? 4 : 0) | (c > level ? 2 : 0) | (d > level ? 1 : 0);
    if (idx === 0 || idx === 15) continue;
    const lerp = (v1, v2) => (level - v1) / (v2 - v1);
    const [x0, y0] = P(i, j);
    const top = [x0 + lerp(a, b) * cpx, y0], right = [x0 + cpx, y0 + lerp(b, c) * cpy];
    const bottom = [x0 + lerp(d, c) * cpx, y0 + cpy], left = [x0, y0 + lerp(a, d) * cpy];
    const add = (p, q) => segs.push(p[0], p[1], q[0], q[1]);
    switch (idx) {
      case 1: case 14: add(left, bottom); break;
      case 2: case 13: add(bottom, right); break;
      case 3: case 12: add(left, right); break;
      case 4: case 11: add(top, right); break;
      case 5: add(left, top); add(bottom, right); break;
      case 6: case 9: add(top, bottom); break;
      case 7: case 8: add(left, top); break;
      case 10: add(left, bottom); add(top, right); break;
    }
  }
  return segs;
}

export function drawIsobars(ctx, view, sim, dpr, dark) {
  let mn = Infinity, mx = -Infinity;
  for (const v of sim.p) { if (v < mn) mn = v; if (v > mx) mx = v; }
  // Standard chart intervals: 4 hPa on continental maps, finer on small ones.
  const range = mx - mn, stepHpa = range > 26 ? 4 : range > 9 ? 2 : 1;
  ctx.save();
  ctx.translate(view.ox, view.oy);
  ctx.font = `600 ${Math.round(10 * dpr)}px Inter, system-ui, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  const placed = [];
  for (let lev = Math.ceil(mn / stepHpa) * stepHpa; lev <= mx; lev += stepHpa) {
    const s = contourSegments(sim, sim.p, lev);
    if (!s.length) continue;
    const strong = lev % 4 === 0;
    ctx.lineWidth = (strong ? 1.5 : 1) * dpr;
    ctx.strokeStyle = dark ? `rgba(255,255,255,${strong ? 0.85 : 0.5})` : `rgba(20,30,60,${strong ? 0.8 : 0.45})`;
    ctx.beginPath();
    for (let n = 0; n < s.length; n += 4) { ctx.moveTo(s[n], s[n + 1]); ctx.lineTo(s[n + 2], s[n + 3]); }
    ctx.stroke();
    if (!strong && stepHpa < 4 && range > 6) continue; // label the main isobars only
    // A few labels per level, kept apart from each other and from the edges.
    const every = Math.max(4, Math.floor(s.length / 4 / 3)) * 4;
    for (let n = Math.floor(every / 2) - (Math.floor(every / 2) % 4); n < s.length; n += every) {
      const x = s[n], y = s[n + 1];
      if (x < 24 * dpr || y < 24 * dpr || x > view.w - 24 * dpr || y > view.h - 24 * dpr) continue;
      if (placed.some(q => Math.abs(q[0] - x) < 60 * dpr && Math.abs(q[1] - y) < 22 * dpr)) continue;
      placed.push([x, y]);
      ctx.fillStyle = dark ? 'rgba(10,14,20,.7)' : 'rgba(255,255,255,.85)';
      ctx.fillRect(x - 15 * dpr, y - 6.5 * dpr, 30 * dpr, 13 * dpr);
      ctx.fillStyle = dark ? '#fff' : '#142040';
      ctx.fillText(String(lev), x, y);
    }
  }
  ctx.restore();
}

// Fronts of the lows, drawn like a synoptic chart: cold fronts blue with
// triangles, warm fronts red with semicircles, on the side they move to.
export function drawFronts(ctx, view, sim, dpr) {
  ctx.save();
  ctx.translate(view.ox, view.oy);
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  for (const S of sim.systems) {
    if (!(S.dp < -4) || S.fdir == null) continue;
    const alpha = Math.min(1, (-S.dp - 4) / 4);
    const mvx = S.mvx ?? Math.cos(S.fdir), mvy = S.mvy ?? Math.sin(S.fdir);
    for (let fi = 0; fi < 2; fi++) {
      const fr = frontGeom(S, fi);
      const pts = [];
      for (let t = 0; t <= 24; t++) {
        const a = (0.06 + 0.86 * t / 24) * fr.L * (fi === 0 ? 1 : 0.9), b = fr.kap * a * a;
        const east = a * fr.cd - b * fr.sd, north = a * fr.sd + b * fr.cd;
        const [la, lo] = offsetLL(S.lat, S.lon, east, north);
        pts.push([view.px(lo), view.py(la)]);
      }
      if (!pts.some(([x, y]) => x > -40 && y > -40 && x < view.w + 40 && y < view.h + 40)) continue;
      const col = fr.cold ? `rgba(30,80,220,${alpha})` : `rgba(215,35,35,${alpha})`;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.4 * dpr;
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
      // Symbols every ~34 px along the line, on the side facing the motion.
      const step = 34 * dpr, sz = 7 * dpr;
      let acc = step / 2;
      for (let i = 1; i < pts.length; i++) {
        const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
        const seg = Math.hypot(x1 - x0, y1 - y0);
        if (seg < 1e-3) continue;
        const ux = (x1 - x0) / seg, uy = (y1 - y0) / seg;
        let nx = -uy, ny = ux;
        if (nx * mvx - ny * mvy < 0) { nx = -nx; ny = -ny; } // screen y points south
        while (acc <= seg) {
          const cx = x0 + ux * acc, cy = y0 + uy * acc;
          ctx.beginPath();
          if (fr.cold) {
            ctx.moveTo(cx - ux * sz, cy - uy * sz); ctx.lineTo(cx + nx * sz * 1.3, cy + ny * sz * 1.3); ctx.lineTo(cx + ux * sz, cy + uy * sz);
          } else {
            const a0 = Math.atan2(uy, ux);
            ctx.arc(cx, cy, sz, a0, a0 + Math.PI, nx * uy - ny * ux > 0);
          }
          ctx.closePath(); ctx.fill();
          acc += step;
        }
        acc -= seg;
      }
    }
  }
  ctx.restore();
}

// Pressure centres as on a real synoptic chart: local minima/maxima of the
// pressure field that stand out from their surroundings (by >= 1 hPa).
export function pressureCentres(sim) {
  const { NX, NY, p } = sim;
  const kmCell = sim.dx[NY >> 1] / 1000;
  const W = Math.max(3, Math.min(40, Math.round(260 / kmCell)));
  const out = [];
  for (let j = 2; j < NY - 2; j++) for (let i = 2; i < NX - 2; i++) {
    const k = j * NX + i, v = p[k];
    let lo = true, hi = true;
    for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
      if (!di && !dj) continue;
      const q = p[k + dj * NX + di];
      if (q <= v) lo = false;
      if (q >= v) hi = false;
    }
    if (!lo && !hi) continue;
    let ok = true;
    for (let dj = -W; dj <= W && ok; dj++) {
      const jj = j + dj; if (jj < 0 || jj >= NY) continue;
      for (let di = -W; di <= W; di++) {
        const ii = i + di; if (ii < 0 || ii >= NX || (!di && !dj)) continue;
        const q = p[jj * NX + ii];
        if ((lo && q < v) || (hi && q > v)) { ok = false; break; }
      }
    }
    if (!ok) continue;
    let ring = 0, nr = 0;
    for (let a = 0; a < 16; a++) {
      const ii = Math.round(i + W * Math.cos(a * Math.PI / 8)), jj = Math.round(j + W * Math.sin(a * Math.PI / 8));
      if (ii >= 0 && jj >= 0 && ii < NX && jj < NY) { ring += p[jj * NX + ii]; nr++; }
    }
    if (nr < 6) continue;
    const prom = lo ? ring / nr - v : v - ring / nr;
    if (prom < 1) continue;
    out.push({ x: (i + 0.5) * sim.cpx, y: (j + 0.5) * sim.cpy, type: lo ? 'L' : 'H', p: v });
  }
  return out;
}

function drawCentre(ctx, x, y, low, pc, dpr) {
  ctx.font = `800 ${Math.round(26 * dpr)}px Inter, system-ui, sans-serif`;
  ctx.lineWidth = 4 * dpr; ctx.strokeStyle = 'rgba(255,255,255,.9)';
  ctx.strokeText(low ? 'L' : 'H', x, y);
  ctx.fillStyle = low ? '#d0202a' : '#1f4fd1';
  ctx.fillText(low ? 'L' : 'H', x, y);
  ctx.font = `700 ${Math.round(11 * dpr)}px Inter, system-ui, sans-serif`;
  ctx.lineWidth = 3 * dpr;
  ctx.strokeText(String(pc), x, y + 19 * dpr);
  ctx.fillText(String(pc), x, y + 19 * dpr);
}

// centres: true = label the pressure field's own centres (pressure charts);
// false = only the lows/highs you placed.
export function drawSystems(ctx, view, sim, dpr, centres = false) {
  ctx.save();
  ctx.translate(view.ox, view.oy);
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  if (centres) {
    for (const c of pressureCentres(sim)) drawCentre(ctx, c.x, c.y, c.type === 'L', Math.round(c.p), dpr);
  } else {
    for (const S of sim.systems) {
      if (S.bg) continue;
      const x = view.px(S.lon), y = view.py(S.lat);
      if (x < -20 || y < -20 || x > view.w + 20 || y > view.h + 20) continue;
      const pc = Math.round(sim.inside(S.lon, S.lat) ? sim.sampleLL(sim.p, S.lon, S.lat) : 1013 + S.dp);
      drawCentre(ctx, x, y, S.dp < 0, pc, dpr);
    }
  }
  ctx.restore();
}

// Wind barbs (knots) on a regular pixel grid.
export function drawBarbs(ctx, view, sim, u, v, dpr, color) {
  const step = 46 * dpr;
  ctx.save();
  ctx.translate(view.ox, view.oy);
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.3 * dpr; ctx.lineCap = 'round';
  const L = 15 * dpr;
  for (let y = step / 2; y < view.h; y += step) for (let x = step / 2; x < view.w; x += step) {
    const gx = x / sim.cpx - 0.5, gy = y / sim.cpy - 0.5;
    const uu = sim.sample(u, gx, gy), vv = sim.sample(v, gx, gy);
    const kt = Math.hypot(uu, vv) * 1.944;
    ctx.save();
    ctx.translate(x, y);
    if (kt < 2.5) { ctx.beginPath(); ctx.arc(0, 0, 3 * dpr, 0, 7); ctx.stroke(); ctx.restore(); continue; }
    // Staff points toward where the wind comes from.
    ctx.rotate(Math.atan2(-uu, -vv));
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, -L); ctx.stroke();
    let rem = Math.round(kt / 5) * 5, pos = -L;
    while (rem >= 50) { ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(7 * dpr, pos + 2.5 * dpr); ctx.lineTo(0, pos + 5 * dpr); ctx.fill(); pos += 6 * dpr; rem -= 50; }
    while (rem >= 10) { ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(8 * dpr, pos - 3 * dpr); ctx.stroke(); pos += 3.5 * dpr; rem -= 10; }
    if (rem >= 5) { if (pos === -L) pos += 3.5 * dpr; ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(4.5 * dpr, pos - 1.7 * dpr); ctx.stroke(); }
    ctx.restore();
  }
  ctx.restore();
}

export function drawStrikes(ctx, view, strikes, now, windowMin, dpr, size = 3.2) {
  ctx.save();
  ctx.translate(view.ox, view.oy);
  const s = size * dpr;
  ctx.lineWidth = 1.6 * dpr;
  for (const k of strikes) {
    const ageMin = (now - k.t) / 60000;
    if (ageMin < 0 || ageMin > windowMin) continue;
    const x = view.px(k.lon), y = view.py(k.lat);
    if (x < 0 || y < 0 || x > view.w || y > view.h) continue;
    const f = ageMin / windowMin;
    ctx.strokeStyle = f < 0.1 ? '#ffffff' : f < 0.33 ? '#fff23a' : f < 0.66 ? '#ff9a1f' : '#e0301e';
    ctx.beginPath(); ctx.moveTo(x - s, y); ctx.lineTo(x + s, y); ctx.moveTo(x, y - s); ctx.lineTo(x, y + s); ctx.stroke();
  }
  ctx.restore();
}

export { distKm };
