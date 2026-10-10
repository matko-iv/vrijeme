// Radar Sandbox app: regions, modes, tools, timeline and rendering.

import { View, REGIONS, RADAR_SITES, bboxAround, mercX, mercY, invMercX, invMercY } from './geo.js';
import { buildDEM } from './terrain.js';
import { Sim, DEFAULT_PARAMS, dewpoint, qsat, windFrom, dirFrom } from './sim.js';
import { Raster } from './radar.js';
import {
  SCALES, RADAR_PALETTES, makeLUT, lutColor, computeShade, renderBasemap, gridImage,
  drawBorders, drawCities, drawIsobars, drawSystems, drawBarbs, drawStrikes,
} from './render.js';
import { CITIES } from './cities.js';

const $ = id => document.getElementById(id);
const D2R = Math.PI / 180;
const TZ = 'Europe/Podgorica';
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

const MODES = [
  { key: 'radar', label: 'Radar', base: 'radar' },
  { key: 'rain', label: 'Rain rate', base: 'light' },
  { key: 'acc', label: 'Accumulation', base: 'light' },
  { key: 'temp', label: 'Temperature', base: 'light' },
  { key: 'dew', label: 'Dew point', base: 'light' },
  { key: 'rh', label: 'Humidity', base: 'light' },
  { key: 'wind', label: 'Wind & gusts', base: 'dark' },
  { key: 'mslp', label: 'Pressure', base: 'light' },
  { key: 'sat', label: 'Satellite', base: null },
  { key: 'cape', label: 'CAPE', base: 'light' },
  { key: 'top', label: 'Echo tops', base: 'dark' },
];

const TOOLS = [
  { key: 'move', icon: '✋', label: 'Drag', opts: ['brush'], hint: 'Grab a storm or an L/H centre and move it. Drag empty space to push rain, clouds and air masses around.' },
  { key: 'rain', icon: '🌧', label: 'Rain', opts: ['brush', 'intensity'], hint: 'Paint lift and moisture: a rain area that drifts with the steering wind and fades once its forcing decays (~3 h).' },
  { key: 'cell', icon: '⛈', label: 'Storm', opts: ['intensity', 'celltype'], hint: 'Click to drop a thunderstorm. Strength nudges its peak reflectivity; CAPE and shear decide how long it lives.' },
  { key: 'line', icon: '〰', label: 'Squall', opts: [], hint: 'Drag a line of storms. It propagates forward and rebuilds itself while there is CAPE ahead of it.' },
  { key: 'low', icon: 'L', label: 'Low', opts: ['intensity'], hint: 'Press at the centre and drag outward to size the low. Strength = depth (3–30 hPa). Lows drift with the steering wind.' },
  { key: 'high', icon: 'H', label: 'High', opts: ['intensity'], hint: 'Press at the centre and drag outward to size the high. Strength = intensity (3–30 hPa).' },
  { key: 'cold', icon: '▲', label: 'Cold front', opts: [], hint: 'Drag along the front. Cold air goes on the side the steering wind blows from; a squall line fires if there is CAPE.' },
  { key: 'warm', icon: '◐', label: 'Warm front', opts: [], hint: 'Drag along the front. Warm air goes behind it and a wide rain shield forms ahead of it.' },
  { key: 'moist', icon: '💧', label: 'Moisture', opts: ['brush', 'sign'], hint: 'Paint humid (＋) or dry (－) air.' },
  { key: 'heat', icon: '🌡', label: 'Heat', opts: ['brush', 'sign'], hint: 'Paint warmer (＋) or colder (－) surface air. Heat plus moisture builds CAPE.' },
  { key: 'erase', icon: '🧽', label: 'Erase', opts: ['brush'], hint: 'Wipe rain, storms, lightning and pressure centres under the brush.' },
  { key: 'radar', icon: '📡', label: 'Radar site', opts: [], hint: 'Click to place a radar: range ring, beam overshoot, terrain blocking, clutter and C-band attenuation.' },
];

const SLIDERS = [
  { key: 'shear', label: 'Shear 0–6 km', min: 0, max: 35, step: 1, fmt: v => `${v} m/s` },
  { key: 't850', label: 'T 850 hPa', min: -20, max: 26, step: 0.5, fmt: v => `${v}°` },
  { key: 't500', label: 'T 500 hPa', min: -42, max: -4, step: 0.5, fmt: v => `${v}°` },
  { key: 'rh', label: 'Humidity', min: 20, max: 98, step: 1, fmt: v => `${v}%` },
  { key: 'sst', label: 'Sea temp', min: 4, max: 31, step: 0.5, fmt: v => `${v}°` },
  { key: 'sun', label: 'Sun heating', min: 0, max: 1.6, step: 0.05, fmt: v => `${(+v).toFixed(2)}×` },
  { key: 'trigger', label: 'Storm trigger', min: 0, max: 3, step: 0.05, fmt: v => `${(+v).toFixed(2)}×` },
  { key: 'breeze', label: 'Sea breeze', min: 0, max: 2, step: 0.1, fmt: v => `${(+v).toFixed(1)}×` },
];

// Local wall time in Europe/Podgorica -> UTC ms.
function tzOffsetMin(utcMs) {
  const parts = new Intl.DateTimeFormat('en-GB', { timeZone: TZ, hourCycle: 'h23', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).formatToParts(new Date(utcMs));
  const g = t => +parts.find(p => p.type === t).value;
  return (Date.UTC(g('year'), g('month') - 1, g('day'), g('hour'), g('minute')) - utcMs) / 60000;
}
function localToUtc(y, mo, d, h, mi) {
  const guess = Date.UTC(y, mo, d, h, mi);
  return guess - tzOffsetMin(guess - tzOffsetMin(guess) * 60000) * 60000;
}
function localParts(utcMs) {
  const d = new Date(utcMs + tzOffsetMin(utcMs) * 60000);
  return { y: d.getUTCFullYear(), mo: d.getUTCMonth(), d: d.getUTCDate(), h: d.getUTCHours(), mi: d.getUTCMinutes() };
}
function todayAt(h, mi) { const p = localParts(Date.now()); return localToUtc(p.y, p.mo, p.d, h, mi); }
function dateAt(mo, d, h, mi) { const p = localParts(Date.now()); return localToUtc(p.y, mo, d, h, mi); }

const PRESETS = {
  autumn: { label: 'Adriatic autumn', sub: 'showers over a warm sea', P: { steerDir: 215, steerSpd: 13, shear: 15, t850: 8, t500: -20, rh: 72, sst: 22, sun: 1, trigger: 1, breeze: 1 },
    time: () => todayAt(10, 30), spin: 3, systems: [['L', 45.0, 11.2, 11, 650]] },
  summer: { label: 'Summer heat storms', sub: 'mountains fire up after noon', P: { steerDir: 265, steerSpd: 6, shear: 8, t850: 20, t500: -13, rh: 58, sst: 26, sun: 1.1, trigger: 1.6, breeze: 1.2 },
    time: () => dateAt(6, 18, 9, 0), spin: 3, systems: [] },
  genoa: { label: 'Genoa low · jugo', sub: 'orographic downpours', P: { steerDir: 205, steerSpd: 20, shear: 18, t850: 10, t500: -20, rh: 82, sst: 20, sun: 0.9, trigger: 0.9, breeze: 0.6 },
    time: () => dateAt(10, 20, 3, 0), spin: 4, systems: [['L', 43.6, 10.5, 20, 750]], fronts: [['warm', 41.5, 16.8, 44.5, 19.6]] },
  bora: { label: 'Bura', sub: 'cold NE downslope gale', P: { steerDir: 35, steerSpd: 14, shear: 10, t850: -3, t500: -27, rh: 50, sst: 15, sun: 0.7, trigger: 0.8, breeze: 0.4 },
    time: () => dateAt(0, 15, 6, 0), spin: 3, systems: [['H', 48.6, 16.5, 12, 950], ['L', 40.3, 19.5, 6, 450]] },
  supercell: { label: 'Supercell day', sub: 'strong shear, big CAPE', P: { steerDir: 240, steerSpd: 17, shear: 28, t850: 18, t500: -15, rh: 64, sst: 25, sun: 1.1, trigger: 0.4, breeze: 1 },
    time: () => dateAt(5, 22, 11, 0), spin: 2.5, systems: [['L', 46.5, 12.5, 7, 700]] },
  blank: { label: 'Blank canvas', sub: 'calm and stable: you draw', P: { steerDir: 270, steerSpd: 6, shear: 6, t850: 10, t500: -14, rh: 60, sst: 20, sun: 1, trigger: 0, breeze: 1 },
    time: () => todayAt(12, 0), spin: 0, systems: [] },
};

const state = {
  region: 'uljenje', mode: 'radar', base: 'auto', tool: 'move', brushKm: 30, intensity: 5, cellType: 'single', sign: 1,
  layers: { borders: true, cities: true, lightning: true, systems: true, isobars: false, particles: false, barbs: false, clutter: true, attenuation: true },
  palette: 'dhmz', siteKey: 'none', customSite: null, satMode: 'auto', stepMin: 15, fps: 4, playing: false,
  P: { ...DEFAULT_PARAMS }, preset: 'autumn',
};

const LAYER_LABELS = { borders: 'Borders', cities: 'Cities', lightning: 'Lightning', systems: 'L / H', isobars: 'Isobars', particles: 'Wind flow', barbs: 'Wind barbs', clutter: 'Radar clutter', attenuation: 'Attenuation' };

let cv, ctx, fx, fxc, ui, uic, dpr = 1;
let view, dem, shade, sim, raster, layout = {};
let borders = [], lakes = [];
const basemaps = new Map();
let history = [], hIdx = -1;
let dirty = true, renderQueued = false, building = null;
let rasterCanvas, rasterImg, palLUT;

// ---------- UI construction ----------
function buildUI() {
  const regions = $('regions');
  for (const [k, r] of Object.entries(REGIONS)) {
    const b = document.createElement('button');
    b.textContent = r.label; b.dataset.region = k;
    b.onclick = () => switchRegion(k);
    regions.appendChild(b);
  }
  const modes = $('modes');
  for (const m of MODES) {
    const b = document.createElement('button');
    b.textContent = m.label; b.dataset.mode = m.key;
    b.onclick = () => setMode(m.key);
    modes.appendChild(b);
  }
  const tools = $('tools');
  for (const t of TOOLS) {
    const b = document.createElement('button');
    b.innerHTML = `<i>${t.icon}</i>${t.label}`; b.dataset.tool = t.key; b.title = t.hint;
    b.onclick = () => setTool(t.key);
    tools.appendChild(b);
  }
  const sl = $('sliders');
  for (const s of SLIDERS) {
    const row = document.createElement('label');
    row.className = 'row';
    row.innerHTML = `<span>${s.label}</span><input type="range" min="${s.min}" max="${s.max}" step="${s.step}" data-key="${s.key}"><output></output>`;
    const inp = row.querySelector('input'), out = row.querySelector('output');
    inp.value = state.P[s.key]; out.textContent = s.fmt(state.P[s.key]);
    inp.oninput = () => { state.P[s.key] = +inp.value; out.textContent = s.fmt(inp.value); paramsChanged(); };
    sl.appendChild(row);
  }
  const ps = $('presets');
  for (const [k, p] of Object.entries(PRESETS)) {
    const b = document.createElement('button');
    b.innerHTML = `${p.label}<small>${p.sub}</small>`; b.dataset.preset = k;
    b.onclick = () => applyPreset(k);
    ps.appendChild(b);
  }
  const ly = $('layers');
  for (const [k, label] of Object.entries(LAYER_LABELS)) {
    const l = document.createElement('label');
    l.innerHTML = `<input type="checkbox" data-layer="${k}"> ${label}`;
    const c = l.querySelector('input');
    c.checked = state.layers[k];
    c.onchange = () => { state.layers[k] = c.checked; autoLayers.delete(k); if (k === 'particles') resetParticles(); if (k === 'clutter' || k === 'attenuation') dirty = true; requestRender(); };
    ly.appendChild(l);
  }
  const site = $('site');
  for (const [k, s] of Object.entries(RADAR_SITES)) site.insertAdjacentHTML('beforeend', `<option value="${k}">${s.name}</option>`);
  site.insertAdjacentHTML('beforeend', `<option value="custom" disabled>Custom (placed)</option>`);
  site.onchange = () => { state.siteKey = site.value; applySite(); };
  $('basemap').onchange = e => { state.base = e.target.value; requestRender(); };
  $('palette').onchange = e => { state.palette = e.target.value; palLUT = null; updateLegend(); requestRender(); };
  $('satMode').onchange = e => { state.satMode = e.target.value; dirty = true; updateLegend(); requestRender(); };

  const brush = $('brush'), inten = $('intensity');
  brush.oninput = () => { state.brushKm = +brush.value; $('brushOut').textContent = `${brush.value} km`; };
  inten.oninput = () => { state.intensity = +inten.value; $('intensityOut').textContent = inten.value; };
  brush.oninput(); inten.oninput();
  document.querySelectorAll('#optCellType button').forEach(b => b.onclick = () => {
    state.cellType = b.dataset.ct;
    document.querySelectorAll('#optCellType button').forEach(x => x.classList.toggle('on', x === b));
  });
  document.querySelectorAll('#optSign button').forEach(b => b.onclick = () => {
    state.sign = +b.dataset.sign;
    document.querySelectorAll('#optSign button').forEach(x => x.classList.toggle('on', x === b));
  });

  $('play').onclick = togglePlay;
  $('stepBtn').onclick = () => { stopPlay(); forward(); };
  $('back').onclick = () => { stopPlay(); if (hIdx > 0) goTo(hIdx - 1); };
  $('scrub').oninput = e => { stopPlay(); goTo(+e.target.value); };
  $('stepMin').value = String(state.stepMin);
  $('stepMin').onchange = e => { state.stepMin = +e.target.value; requestRender(); };
  $('fps').oninput = e => { state.fps = +e.target.value; };
  $('resetAcc').onclick = () => { sim.acc.fill(0); commitEdit(); };
  $('clearAll').onclick = () => {
    sim.cells = []; sim.systems = []; sim.strikes = [];
    for (const f of [sim.C, sim.Fz, sim.R, sim.Rc, sim.out, sim.anvil, sim.acc]) f.fill(0);
    commitEdit();
  };
  $('startTime').onchange = e => {
    const m = /^(\d+)-(\d+)-(\d+)T(\d+):(\d+)/.exec(e.target.value);
    if (!m) return;
    sim.time = localToUtc(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
    sim.diagnose(); resetHistory(); dirty = true; requestRender();
  };
  $('sideToggle').onclick = () => document.querySelector('.app').classList.toggle('sideopen');

  window.addEventListener('keydown', e => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    else if (e.code === 'ArrowRight') { stopPlay(); forward(); }
    else if (e.code === 'ArrowLeft') { stopPlay(); if (hIdx > 0) goTo(hIdx - 1); }
  });
  setupCompass();
  setTool('move');
  highlight();
}

function highlight() {
  document.querySelectorAll('#regions button').forEach(b => b.classList.toggle('on', b.dataset.region === state.region));
  document.querySelectorAll('#modes button').forEach(b => b.classList.toggle('on', b.dataset.mode === state.mode));
  document.querySelectorAll('#presets button').forEach(b => b.classList.toggle('on', b.dataset.preset === state.preset));
  const frame = REGIONS[state.region]?.radarFrame;
  $('site').disabled = !!frame;
  $('site').value = frame ? REGIONS[state.region].radar : state.siteKey;
}

function syncSliders() {
  document.querySelectorAll('#sliders input').forEach(inp => {
    const s = SLIDERS.find(x => x.key === inp.dataset.key);
    inp.value = state.P[s.key];
    inp.parentElement.querySelector('output').textContent = s.fmt(state.P[s.key]);
  });
  drawCompass();
}

function setTool(k) {
  state.tool = k;
  const t = TOOLS.find(x => x.key === k);
  document.querySelectorAll('#tools button').forEach(b => b.classList.toggle('on', b.dataset.tool === k));
  $('optBrush').style.display = t.opts.includes('brush') ? '' : 'none';
  $('optIntensity').style.display = t.opts.includes('intensity') ? '' : 'none';
  $('optCellType').style.display = t.opts.includes('celltype') ? '' : 'none';
  $('optSign').style.display = t.opts.includes('sign') ? '' : 'none';
  $('toolHint').textContent = t.hint;
  cv && (cv.style.cursor = k === 'move' ? 'grab' : 'crosshair');
}

const autoLayers = new Set();
function setMode(k) {
  // Layers a mode switched on by itself go off again when you leave it.
  for (const l of autoLayers) state.layers[l] = false;
  autoLayers.clear();
  state.mode = k;
  if (k === 'wind' && !state.layers.particles) { state.layers.particles = true; autoLayers.add('particles'); }
  if (k === 'mslp' && !state.layers.isobars) { state.layers.isobars = true; autoLayers.add('isobars'); }
  syncLayerBoxes(); resetParticles();
  dirty = true;
  highlight(); updateLegend(); requestRender();
}
function syncLayerBoxes() {
  document.querySelectorAll('#layers input').forEach(c => { c.checked = state.layers[c.dataset.layer]; });
}

// ---------- compass (steering wind) ----------
let compass, cctx;
function setupCompass() {
  compass = $('compass'); cctx = compass.getContext('2d');
  let dragging = false;
  const set = e => {
    const r = compass.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width * 2 - 1, y = (e.clientY - r.top) / r.height * 2 - 1;
    const len = Math.min(1, Math.hypot(x, y) / 0.76);
    // Arrow points where the wind goes; direction FROM is opposite.
    const toward = (Math.atan2(x, -y) / D2R + 360) % 360;
    state.P.steerDir = Math.round((toward + 180) % 360);
    state.P.steerSpd = Math.round(len * 40);
    paramsChanged();
  };
  compass.addEventListener('pointerdown', e => { dragging = true; compass.setPointerCapture(e.pointerId); set(e); });
  compass.addEventListener('pointermove', e => dragging && set(e));
  compass.addEventListener('pointerup', () => { dragging = false; });
  drawCompass();
}
const CARD = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
const cardinal = d => CARD[Math.round(((d % 360) + 360) % 360 / 22.5) % 16];
function drawCompass() {
  const g = cctx, W = compass.width, c = W / 2, R = W * 0.38;
  g.clearRect(0, 0, W, W);
  g.strokeStyle = '#2f3946'; g.lineWidth = 2;
  for (const f of [0.25, 0.5, 0.75, 1]) { g.beginPath(); g.arc(c, c, R * f, 0, 7); g.stroke(); }
  g.fillStyle = '#8d98a7'; g.font = '600 22px Inter, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText('N', c, c - R - 13); g.fillText('S', c, c + R + 14); g.fillText('E', c + R + 13, c); g.fillText('W', c - R - 13, c);
  const toward = (state.P.steerDir + 180) * D2R, len = state.P.steerSpd / 40 * R;
  const ex = c + Math.sin(toward) * len, ey = c - Math.cos(toward) * len;
  g.strokeStyle = '#3aa0ff'; g.fillStyle = '#3aa0ff'; g.lineWidth = 6; g.lineCap = 'round';
  g.beginPath(); g.moveTo(c, c); g.lineTo(ex, ey); g.stroke();
  const a = Math.atan2(ey - c, ex - c);
  g.beginPath(); g.moveTo(ex + Math.cos(a) * 10, ey + Math.sin(a) * 10);
  g.lineTo(ex + Math.cos(a + 2.5) * 18, ey + Math.sin(a + 2.5) * 18); g.lineTo(ex + Math.cos(a - 2.5) * 18, ey + Math.sin(a - 2.5) * 18); g.fill();
  g.beginPath(); g.arc(ex, ey, 9, 0, 7); g.fillStyle = '#e6e9ee'; g.fill();
  $('steerOut').textContent = `${cardinal(state.P.steerDir)} ${state.P.steerSpd} m/s`;
}

let paramTimer = null;
function paramsChanged() {
  drawCompass();
  clearTimeout(paramTimer);
  paramTimer = setTimeout(() => { if (sim) { sim.diagnose(); dirty = true; requestRender(); } }, 60);
}

// ---------- canvas & region ----------
function setupCanvas() {
  const stage = $('stage');
  const cw = stage.clientWidth, ch = stage.clientHeight;
  const devDpr = window.devicePixelRatio || 1;
  dpr = Math.max(1, Math.min(devDpr, 2200 / cw, 1500 / ch));
  for (const c of [cv, fx, ui]) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
}

function siteFor() {
  const R = REGIONS[state.region];
  if (R?.radarFrame) return RADAR_SITES[R.radar];
  if (state.siteKey === 'none') return null;
  if (state.siteKey === 'custom') return state.customSite;
  return RADAR_SITES[state.siteKey];
}

async function buildRegion() {
  const token = {};
  building = token;
  showLoading('Loading terrain…');
  setupCanvas();
  const W = cv.width, H = cv.height;
  const R = REGIONS[state.region];
  layout = {};
  if (R.radarFrame) {
    const s = RADAR_SITES[R.radar];
    const bbox = bboxAround(s.lat, s.lon, s.rangeKm + 10);
    const P = Math.round(Math.min(W, H) * 0.085);
    const S = Math.min(W - P, H - P);
    const ox = Math.round((W - S - P) / 2), oy = Math.round((H - S - P) / 2);
    view = new View(bbox, ox, oy + P, S, S);
    layout.frame = { P, S, ox, oy };
  } else {
    view = new View(R.bbox, 0, 0, W, H);
  }
  const d = await buildDEM(view, view.w, view.h, (n, total) => { if (building === token) showLoading(`Loading terrain ${n}/${total}`); }, lakes);
  if (building !== token) return;
  dem = d;
  shade = computeShade(dem, view);
  const old = sim;
  sim = new Sim(view, dem, state.P, old ? old.time : PRESETS.autumn.time());
  if (old) sim.adoptFrom(old);
  const oldR = raster;
  raster = new Raster(view, dem, sim);
  if (oldR) { raster.offX = oldR.offX; raster.offY = oldR.offY; raster.frame = oldR.frame; }
  rasterCanvas = document.createElement('canvas');
  rasterCanvas.width = raster.W; rasterCanvas.height = raster.H;
  rasterImg = rasterCanvas.getContext('2d').createImageData(raster.W, raster.H);
  satBase = null;
  basemaps.clear();
  applySite(false);
  resetHistory();
  resetParticles();
  dirty = true;
  render();
  hideLoading();
  if (dem.ok < 0.6) showLoading('Some terrain tiles failed to load (offline?). The sandbox still works.', 4000);
}

function switchRegion(k) {
  if (k === state.region && view) return;
  state.region = k;
  highlight();
  buildRegion();
}

function applySite(render_ = true) {
  if (!raster) return;
  const site = siteFor();
  raster.setSite(site);
  dirty = true;
  highlight();
  if (render_) requestRender();
}

// ---------- basemaps ----------
function baseStyle() {
  if (state.base !== 'auto') return state.base;
  return MODES.find(m => m.key === state.mode).base || 'terrain';
}
function basemapFor(style) {
  const site = style === 'radar' ? siteFor() : null;
  const key = style + (site ? `@${site.lat.toFixed(3)},${site.lon.toFixed(3)}` : '');
  if (basemaps.has(key)) return basemaps.get(key);
  let coverage = null;
  if (site) {
    const sx = view.px(site.lon), sy = view.py(site.lat), rows = new Float32Array(view.h);
    for (let y = 0; y < view.h; y++) rows[y] = view.metersPerPx(view.lat(y)) / 1000;
    coverage = (x, y) => { const m = rows[y]; return ((x - sx) * m) ** 2 + ((y - sy) * m) ** 2 <= site.rangeKm ** 2; };
  }
  const c = renderBasemap(dem, view, style, shade, coverage);
  basemaps.set(key, c);
  return c;
}
let satBase = null;
function satBasePixels() {
  if (satBase) return satBase;
  const c = document.createElement('canvas');
  c.width = raster.W; c.height = raster.H;
  const g = c.getContext('2d');
  g.drawImage(basemapFor('terrain'), 0, 0, raster.W, raster.H);
  satBase = g.getImageData(0, 0, raster.W, raster.H).data;
  return satBase;
}

// ---------- rendering ----------
function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; render(); });
}

function render() {
  if (!sim) return;
  const W = cv.width, H = cv.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#0a0d11';
  ctx.fillRect(0, 0, W, H);
  const mode = state.mode, style = mode === 'sat' ? 'sat' : baseStyle();
  if (mode !== 'sat') ctx.drawImage(basemapFor(style), view.ox, view.oy);
  ctx.save();
  ctx.beginPath(); ctx.rect(view.ox, view.oy, view.w, view.h); ctx.clip();

  if (mode === 'radar' || mode === 'top') {
    if (dirty) raster.compute(sim, { proj: !!layout.frame, clutter: state.layers.clutter, attenuation: state.layers.attenuation });
    paintRaster(mode, style);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(rasterCanvas, view.ox, view.oy, view.w, view.h);
  } else if (mode === 'sat') {
    if (dirty || !rasterImg._sat) {
      const sm = state.satMode === 'auto' ? (sunAtCentre() > 0.12 ? 'vis' : 'ir') : state.satMode;
      raster.satellite(sim, sm, satBasePixels(), rasterImg.data);
      rasterImg._sat = sm;
      rasterCanvas.getContext('2d').putImageData(rasterImg, 0, 0);
      updateLegend();
    }
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(rasterCanvas, view.ox, view.oy, view.w, view.h);
  } else {
    const img = fieldImage(mode);
    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = mode === 'mslp' ? 0.55 : 0.82;
    ctx.drawImage(img, view.ox, view.oy, view.w, view.h);
    ctx.globalAlpha = 1;
  }
  if (mode !== 'sat') rasterImg._sat = null;

  // Overlays.
  const dark = style === 'dark' || style === 'sat';
  if (state.layers.borders) {
    const col = style === 'radar' ? 'rgba(30,24,20,.85)' : style === 'sat' ? 'rgba(255,236,120,.6)' : dark ? 'rgba(210,220,235,.5)' : style === 'terrain' ? 'rgba(255,255,255,.75)' : 'rgba(55,55,65,.6)';
    drawBorders(ctx, view, borders, col, (style === 'radar' ? 1 : 1.1) * dpr);
  }
  if (state.layers.isobars || mode === 'mslp') drawIsobars(ctx, view, sim, dpr, dark);
  if (state.layers.barbs) drawBarbs(ctx, view, sim, sim.u, sim.v, dpr, dark ? 'rgba(255,255,255,.85)' : 'rgba(20,25,35,.8)');
  if (state.layers.cities) {
    const values = mode === 'temp' ? c => fmtAt(c, sim.T, v => `${Math.round(v)}°`) : mode === 'wind' ? c => fmtAt(c, sim.gust, v => `${Math.round(v * 3.6)}`) :
      mode === 'acc' ? c => fmtAt(c, sim.acc, v => (v >= 0.5 ? `${v.toFixed(v < 10 ? 1 : 0)}` : null)) : null;
    drawCities(ctx, view, CITIES, { maxRank: cityRank(), style: style === 'sat' ? 'satellite' : style === 'radar' && mode === 'radar' ? 'radar' : style, dpr, values });
  }
  if (state.layers.lightning) drawStrikes(ctx, view, sim.strikes, sim.time, Math.max(30, state.stepMin * 2), dpr, cityRank() <= 2 ? 2.2 : 3.2);
  if (state.layers.systems) drawSystems(ctx, view, sim, dpr);
  const site = siteFor();
  if (site && !layout.frame && (mode === 'radar' || mode === 'top')) drawSiteRing(site);
  ctx.restore();
  if (layout.frame && (mode === 'radar' || mode === 'top')) drawFrame(site);
  else drawStamp();
  dirty = false;
  updateClock();
}

function sunAtCentre() { return sim.cosz[(sim.NY >> 1) * sim.NX + (sim.NX >> 1)]; }

function fmtAt(c, field, fmt) {
  if (!sim.inside(c[3], c[2])) return null;
  return fmt(sim.sampleLL(field, c[3], c[2]));
}

function cityRank() {
  const span = view.bbox[2] - view.bbox[0];
  return span > 30 ? 1 : span > 12 ? 2 : span > 4.5 ? 3 : 4;
}

function paintRaster(mode, style) {
  const d = rasterImg.data, N = raster.W * raster.H;
  if (mode === 'top') {
    const L = SCALES.top.lut;
    for (let p = 0; p < N; p++) {
      const o = p * 4;
      if (raster.dbz[p] < 15) { d[o + 3] = 0; continue; }
      lutColor(L, raster.top[p], d, o);
    }
  } else {
    if (!palLUT) palLUT = makeLUT(RADAR_PALETTES[state.palette], -10, 80, true);
    const a = style === 'radar' ? 255 : 235;
    for (let p = 0; p < N; p++) {
      const o = p * 4;
      lutColor(palLUT, raster.dbz[p], d, o);
      if (d[o + 3]) d[o + 3] = Math.min(d[o + 3], a);
    }
  }
  rasterCanvas.getContext('2d').putImageData(rasterImg, 0, 0);
}

function fieldImage(mode) {
  const s = sim;
  let fn, scale = SCALES[mode];
  switch (mode) {
    case 'rain': fn = k => s.R[k] + s.Rc[k]; break;
    case 'acc': fn = k => s.acc[k]; break;
    case 'temp': fn = k => s.T[k]; break;
    case 'dew': fn = k => dewpoint(s.q[k], s.p[k] * Math.exp(-s.elev[k] / 8000)); break;
    case 'rh': fn = k => clamp(100 * s.q[k] / qsat(s.T[k], s.p[k] * Math.exp(-s.elev[k] / 8000)), 0, 100); break;
    case 'wind': fn = k => s.gust[k] * 3.6; break;
    case 'mslp': fn = k => s.p[k]; break;
    case 'cape': fn = k => s.cape[k]; break;
  }
  return gridImage(sim, fn, scale);
}

function drawSiteRing(site) {
  ctx.save();
  ctx.translate(view.ox, view.oy);
  const x = view.px(site.lon), y = view.py(site.lat), r = site.rangeKm * 1000 / view.metersPerPx(site.lat);
  ctx.strokeStyle = 'rgba(120,80,40,.9)'; ctx.lineWidth = 2 * dpr;
  ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
  ctx.setLineDash([4 * dpr, 6 * dpr]); ctx.lineWidth = 1 * dpr; ctx.strokeStyle = 'rgba(80,60,40,.5)';
  for (const f of [0.25, 0.5, 0.75]) { ctx.beginPath(); ctx.arc(x, y, r * f, 0, 7); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.fillStyle = '#111'; ctx.beginPath(); ctx.arc(x, y, 3.5 * dpr, 0, 7); ctx.fill();
  ctx.restore();
}

function fmtUTC(ms) {
  const d = new Date(ms), z = n => String(n).padStart(2, '0');
  return `${d.getUTCFullYear()}-${z(d.getUTCMonth() + 1)}-${z(d.getUTCDate())} ${z(d.getUTCHours())}:${z(d.getUTCMinutes())}`;
}

function drawStamp() {
  const label = state.mode === 'radar' || state.mode === 'top' ? (siteFor() ? siteFor().name : 'Radar composite') : MODES.find(m => m.key === state.mode).label;
  ctx.save();
  ctx.font = `${Math.round(12 * dpr)}px Verdana, sans-serif`;
  const t = `${label}  ${fmtUTC(sim.time)} UTC`;
  const w = ctx.measureText(t).width + 14 * dpr;
  ctx.fillStyle = 'rgba(255,255,255,.88)'; ctx.fillRect(view.ox + 8 * dpr, view.oy + 8 * dpr, w, 20 * dpr);
  ctx.fillStyle = '#222'; ctx.textBaseline = 'middle';
  ctx.fillText(t, view.ox + 15 * dpr, view.oy + 18 * dpr);
  ctx.restore();
}

// The Uljenje-style frame: max-reflectivity side views, header and ring.
function drawFrame(site) {
  const { P, S, ox, oy } = layout.frame;
  const isRadar = state.mode === 'radar' || state.mode === 'top';
  ctx.save();
  // Range ring.
  if (isRadar) {
    const x = view.ox + view.px(site.lon), y = view.oy + view.py(site.lat), r = site.rangeKm * 1000 / view.metersPerPx(site.lat);
    ctx.save(); ctx.beginPath(); ctx.rect(view.ox, view.oy, view.w, view.h); ctx.clip();
    ctx.strokeStyle = '#956c42'; ctx.lineWidth = 2 * dpr; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
    ctx.restore();
  }
  // Panels.
  ctx.fillStyle = '#d49b5f';
  ctx.fillRect(ox, oy, S, P); ctx.fillRect(ox + S, oy + P, P, S);
  ctx.fillStyle = '#000'; ctx.fillRect(ox + S, oy, P, P);
  if (isRadar && raster.projTop) {
    const NZ = raster.NZ, W = raster.W, H = raster.H;
    if (!palLUT) palLUT = makeLUT(RADAR_PALETTES[state.palette], -10, 80, true);
    const topC = document.createElement('canvas'); topC.width = W; topC.height = NZ;
    const ti = topC.getContext('2d').createImageData(W, NZ);
    for (let zi = 0; zi < NZ; zi++) for (let x = 0; x < W; x++) {
      const o = ((NZ - 1 - zi) * W + x) * 4;
      if (zi * 0.25 < raster.terrTop[x]) { ti.data[o] = 149; ti.data[o + 1] = 108; ti.data[o + 2] = 66; ti.data[o + 3] = 255; continue; }
      lutColor(palLUT, raster.projTop[zi * W + x], ti.data, o);
    }
    topC.getContext('2d').putImageData(ti, 0, 0);
    const sideC = document.createElement('canvas'); sideC.width = NZ; sideC.height = H;
    const si = sideC.getContext('2d').createImageData(NZ, H);
    for (let y = 0; y < H; y++) for (let zi = 0; zi < NZ; zi++) {
      const o = (y * NZ + zi) * 4;
      if (zi * 0.25 < raster.terrSide[y]) { si.data[o] = 149; si.data[o + 1] = 108; si.data[o + 2] = 66; si.data[o + 3] = 255; continue; }
      lutColor(palLUT, raster.projSide[zi * H + y], si.data, o);
    }
    sideC.getContext('2d').putImageData(si, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(topC, ox, oy, S, P);
    ctx.drawImage(sideC, ox + S, oy + P, P, S);
  }
  // Height reference lines (5 and 10 km of 14).
  ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.lineWidth = 1 * dpr;
  for (const km of [5, 10]) {
    const yy = oy + P - P * km / 14, xx = ox + S + P * km / 14;
    ctx.beginPath(); ctx.moveTo(ox, yy); ctx.lineTo(ox + S, yy); ctx.moveTo(xx, oy + P); ctx.lineTo(xx, oy + P + S); ctx.stroke();
  }
  ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5 * dpr;
  ctx.strokeRect(ox, oy, S, P); ctx.strokeRect(ox + S, oy + P, P, S); ctx.strokeRect(view.ox, view.oy, view.w, view.h);
  ctx.fillStyle = '#fff'; ctx.font = `${Math.round(10 * dpr)}px Verdana, sans-serif`; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText('10 km', ox + S + P - 4 * dpr, oy + P * 0.28);
  ctx.fillText('5 km', ox + S + P - 4 * dpr, oy + P * 0.62);
  // Header.
  const label = `${isRadar ? site.name : MODES.find(m => m.key === state.mode).label}  ${fmtUTC(sim.time)}`;
  ctx.font = `${Math.round(14 * dpr)}px Verdana, sans-serif`; ctx.textAlign = 'left';
  const w = ctx.measureText(label).width + 16 * dpr;
  ctx.fillStyle = '#fff'; ctx.fillRect(view.ox + 3 * dpr, view.oy + 3 * dpr, w, 24 * dpr);
  ctx.strokeStyle = '#888'; ctx.strokeRect(view.ox + 3 * dpr, view.oy + 3 * dpr, w, 24 * dpr);
  ctx.fillStyle = '#222'; ctx.fillText(label, view.ox + 11 * dpr, view.oy + 15.5 * dpr);
  ctx.restore();
}

// ---------- legend, clock, readout ----------
function updateLegend() {
  const el = $('legend');
  const mode = state.mode;
  let html = '';
  if (mode === 'radar') {
    const pal = RADAR_PALETTES[state.palette];
    const segs = pal.map(s => `<div style="flex:1;background:rgb(${s[1]},${s[2]},${s[3]})"></div>`).join('');
    const ticks = pal.map((s, i) => `<span style="left:${(i / pal.length) * 100}%">${s[0]}</span>`).join('') + `<span style="left:100%">dBZ</span>`;
    html = `<div><div class="bar">${segs}</div><div class="ticks">${ticks}</div></div>`;
  } else if (mode === 'sat') {
    const sm = rasterImg?._sat || state.satMode;
    if (sm === 'vis') html = `<span class="unit">Visible · reflected sunlight (dark at night)</span>`;
    else {
      const bar = sm === 'ir-enh'
        ? 'linear-gradient(90deg,#111,#888 55%,#ddd 70%,#50a0ff 74%,#28dc50 80%,#fadc1e 86%,#f03c1e 93%,#fff)'
        : 'linear-gradient(90deg,#111,#fff)';
      html = `<div><div class="bar" style="background:${bar}"></div><div class="ticks"><span style="left:0">+30°C</span><span style="left:50%">-15</span><span style="left:100%">-60°C</span></div></div><span class="unit">cloud-top temperature</span>`;
    }
  } else {
    const sc = SCALES[mode];
    const n = 48, segs = [];
    const lo = sc.log ? Math.log10(sc.stops[0][0] * 2) : sc.min, hi = sc.log ? Math.log10(sc.max) : sc.max;
    const tmp = [0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      const x = lo + (hi - lo) * (i + 0.5) / n;
      lutColor(sc.lut, sc.log ? 10 ** x : x, tmp, 0);
      segs.push(`<div style="flex:1;background:rgba(${tmp[0]},${tmp[1]},${tmp[2]},${Math.max(0.35, tmp[3] / 255)})"></div>`);
    }
    const pos = v => ((sc.log ? Math.log10(v) : v) - lo) / (hi - lo) * 100;
    const conv = mode === 'wind' ? 'km/h gusts' : sc.unit;
    const ticks = sc.ticks.map(t => `<span style="left:${pos(t)}%">${t}</span>`).join('');
    html = `<div><div class="bar">${segs.join('')}</div><div class="ticks">${ticks}</div></div><span class="unit">${conv}</span>`;
  }
  el.innerHTML = html;
}

function updateClock() {
  if (!sim) return;
  const fmt = new Intl.DateTimeFormat('en-GB', { timeZone: TZ, weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
  $('clockLocal').textContent = fmt.format(new Date(sim.time));
  const d = new Date(sim.time), z = n => String(n).padStart(2, '0');
  const cz = sunAtCentre();
  $('clockUtc').textContent = `${z(d.getUTCHours())}:${z(d.getUTCMinutes())} UTC · ${cz > 0.05 ? '☀ day' : cz > -0.1 ? '◐ twilight' : '☾ night'} · ${sim.cells.length} storms`;
  const p = localParts(sim.time);
  const v = `${p.y}-${z(p.mo + 1)}-${z(p.d)}T${z(p.h)}:${z(p.mi)}`;
  if (document.activeElement !== $('startTime')) $('startTime').value = v;
}

function readout(vx, vy) {
  const el = $('readout');
  if (vx < 0 || vy < 0 || vx > view.w || vy > view.h) { el.style.display = 'none'; return; }
  const lon = view.lon(vx), lat = view.lat(vy);
  const gx = vx / sim.cpx - 0.5, gy = vy / sim.cpy - 0.5;
  const S = f => sim.sample(f, gx, gy);
  const di = Math.min(dem.h - 1, vy | 0) * dem.w + Math.min(dem.w - 1, vx | 0);
  const elev = dem.water[di] ? 0 : Math.max(0, Math.round(dem.elev[di]));
  const T = S(sim.T), q = S(sim.q), ps = S(sim.p) * Math.exp(-elev / 8000);
  const rh = clamp(100 * q / qsat(T, ps), 0, 100), td = dewpoint(q, ps);
  const u = S(sim.u), v = S(sim.v), spd = Math.hypot(u, v);
  const rp = Math.min(raster.W * raster.H - 1, Math.floor(vy / raster.k) * raster.W + Math.floor(vx / raster.k));
  const dbz = raster.dbz[rp];
  const R = S(sim.R) + S(sim.Rc);
  el.innerHTML = `<b>${lat.toFixed(2)}°N ${lon.toFixed(2)}°E</b> · ${dem.water[di] ? 'sea' : `${elev} m`}
  <div class="g">
  <span>Radar</span><span>${dbz > -50 ? `${dbz.toFixed(0)} dBZ` : '—'}</span>
  <span>Rain</span><span>${R >= 0.05 ? R.toFixed(1) : '0'} mm/h · Σ ${S(sim.acc).toFixed(1)} mm</span>
  <span>Temp</span><span>${T.toFixed(1)}°C · dew ${td.toFixed(1)}°C</span>
  <span>Humidity</span><span>${rh.toFixed(0)}%</span>
  <span>Wind</span><span>${cardinal(dirFrom(u, v))} ${(spd * 3.6).toFixed(0)} km/h · gust ${(S(sim.gust) * 3.6).toFixed(0)}</span>
  <span>Pressure</span><span>${S(sim.p).toFixed(1)} hPa</span>
  <span>CAPE</span><span>${S(sim.cape).toFixed(0)} J/kg</span>
  <span>Cloud</span><span>${(S(sim.cf) * 100).toFixed(0)}%</span>
  </div>`;
  el.style.display = 'block';
}

// ---------- history & time ----------
function snap() { return { s: sim.snapshot(), offX: raster.offX, offY: raster.offY, frame: raster.frame }; }
function resetHistory() { history = [snap()]; hIdx = 0; updateScrub(); }
function updateScrub() { const s = $('scrub'); s.max = String(history.length - 1); s.value = String(hIdx); }
function goTo(i) {
  i = clamp(i, 0, history.length - 1);
  const h = history[i];
  sim.restore(h.s); raster.offX = h.offX; raster.offY = h.offY; raster.frame = h.frame;
  hIdx = i; updateScrub(); dirty = true; render();
}
function forward() {
  if (hIdx < history.length - 1) { goTo(hIdx + 1); return; }
  const dt = state.stepMin * 60;
  sim.step(dt);
  raster.advance(sim, dt);
  history.push(snap());
  if (history.length > 72) history.shift();
  hIdx = history.length - 1;
  updateScrub();
  dirty = true;
  render();
}
// After a user edit the current frame changes and the future is rewritten.
function commitEdit() {
  sim.diagnose();
  history.length = hIdx + 1;
  history[hIdx] = snap();
  updateScrub();
  dirty = true;
  requestRender();
}
let playTimer = null;
function togglePlay() { state.playing ? stopPlay() : startPlay(); }
function startPlay() {
  state.playing = true; $('play').textContent = '⏸';
  const tick = () => {
    if (!state.playing) return;
    const t = performance.now();
    forward();
    playTimer = setTimeout(tick, Math.max(15, 1000 / state.fps - (performance.now() - t)));
  };
  tick();
}
function stopPlay() { state.playing = false; $('play').textContent = '▶'; clearTimeout(playTimer); }

async function applyPreset(k) {
  const p = PRESETS[k];
  stopPlay();
  state.preset = k;
  Object.assign(state.P, DEFAULT_PARAMS, p.P);
  syncSliders();
  highlight();
  showLoading(p.spin ? 'Spinning up the weather…' : 'Setting up…');
  await new Promise(r => setTimeout(r, 30));
  sim = new Sim(view, dem, state.P, p.time());
  for (const [type, lat, lon, dp, R] of p.systems) sim.addSystem(type, lat, lon, dp, R);
  sim.diagnose();
  for (const f of p.fronts || []) sim.addFront(f[0], f[1], f[2], f[3], f[4]);
  raster.offX = 0; raster.offY = 0;
  const chunks = Math.round((p.spin || 0) * 4);
  for (let i = 0; i < chunks; i++) {
    sim.step(900); raster.advance(sim, 900);
    if (i % 2 === 1) { showLoading(`Spinning up the weather… ${Math.round((i + 1) / chunks * 100)}%`); await new Promise(r => setTimeout(r, 0)); }
  }
  sim.acc.fill(0);
  sim.strikes = sim.strikes.filter(s => s.t > sim.time - 30 * 60000);
  sim.diagnose();
  resetHistory();
  dirty = true;
  render();
  hideLoading();
}

// ---------- pointer interaction ----------
let drag = null, paintLoop = null, lastPaint = 0, lastDiag = 0, hover = null;
function evPos(e) {
  const r = cv.getBoundingClientRect();
  const x = (e.clientX - r.left) * cv.width / r.width, y = (e.clientY - r.top) * cv.height / r.height;
  return { cx: x, cy: y, vx: x - view.ox, vy: y - view.oy };
}
function ll(p) { return { lat: view.lat(p.vy), lon: view.lon(p.vx) }; }
function kmPerPx(lat) { return view.metersPerPx(lat) / 1000; }

function hitTest(p) {
  const r = 22 * dpr;
  for (const S of sim.systems) {
    if (Math.hypot(view.px(S.lon) - p.vx, view.py(S.lat) - p.vy) < r) return { kind: 'sys', obj: S };
  }
  let best = null, bd = Infinity;
  for (const c of sim.cells) {
    const d = Math.hypot(view.px(c.lon) - p.vx, view.py(c.lat) - p.vy);
    const rad = Math.max(12 * dpr, c.r * 2 / kmPerPx(c.lat));
    if (d < rad && d < bd) { bd = d; best = c; }
  }
  return best ? { kind: 'cell', obj: best } : null;
}

function setupPointer() {
  cv.addEventListener('pointerdown', e => {
    if (!sim || e.button === 2) return;
    const p = evPos(e);
    if (p.vx < 0 || p.vy < 0 || p.vx > view.w || p.vy > view.h) return;
    cv.setPointerCapture(e.pointerId);
    const { lat, lon } = ll(p);
    const t = state.tool;
    if (t === 'move') {
      const h = hitTest(p);
      drag = h ? { kind: h.kind, obj: h.obj, last: { lat, lon } } : { kind: 'smudge', last: { lat, lon } };
      cv.style.cursor = 'grabbing';
    } else if (t === 'rain' || t === 'moist' || t === 'heat' || t === 'erase') {
      drag = { kind: 'paint', at: { lat, lon } };
      lastPaint = performance.now();
      applyPaint(0.12);
      paintLoop = requestAnimationFrame(paintTick);
    } else if (t === 'cell') {
      const c = sim.addCell(lat, lon, { type: state.cellType, boost: (state.intensity - 5) * 1.8 });
      c.age = c.life * (c.type === 'super' ? 0.08 : 0.2); c.forced = true;
      c.dbz = sim.cellDbz(c);
      commitEdit();
    } else if (t === 'radar') {
      state.customSite = { name: 'Custom radar', short: 'Custom', lat, lon, rangeKm: 240 };
      state.siteKey = 'custom';
      const opt = $('site').querySelector('option[value=custom]'); opt.disabled = false;
      if (REGIONS[state.region].radarFrame) { state.region = 'adriatic'; buildRegion(); }
      else applySite();
      highlight();
    } else {
      drag = { kind: 'shape', a: { lat, lon }, b: { lat, lon }, pa: p, pb: p };
    }
    drawUI(p);
  });
  cv.addEventListener('pointermove', e => {
    if (!sim) return;
    const p = evPos(e);
    hover = p;
    readout(p.vx, p.vy);
    if (drag) {
      const { lat, lon } = ll(p);
      if (drag.kind === 'sys' || drag.kind === 'cell') {
        drag.obj.lat = lat; drag.obj.lon = lon;
        if (drag.kind === 'sys') sim.diagnose();
        dirty = true; requestRender();
      } else if (drag.kind === 'smudge') {
        const e_ = (lon - drag.last.lon) * 111320 * Math.cos(lat * D2R), n_ = (lat - drag.last.lat) * 111320;
        if (Math.hypot(e_, n_) > 200) {
          sim.smudge(drag.last.lat, drag.last.lon, state.brushKm, e_, n_);
          drag.last = { lat, lon };
          dirty = true; requestRender();
        }
      } else if (drag.kind === 'paint') drag.at = { lat, lon };
      else if (drag.kind === 'shape') { drag.b = { lat, lon }; drag.pb = p; }
    }
    drawUI(p);
  });
  const end = () => {
    if (!drag) return;
    const d = drag;
    drag = null;
    cancelAnimationFrame(paintLoop);
    cv.style.cursor = state.tool === 'move' ? 'grab' : 'crosshair';
    if (d.kind === 'shape') {
      const t = state.tool, { a, b } = d;
      const km = Math.hypot((b.lat - a.lat) * 111.32, (b.lon - a.lon) * 111.32 * Math.cos(a.lat * D2R));
      if (t === 'low' || t === 'high') sim.addSystem(t === 'low' ? 'L' : 'H', a.lat, a.lon, 3 * state.intensity, Math.max(150, km || 400));
      else if (km > 15) {
        if (t === 'line') sim.addLine(a.lat, a.lon, b.lat, b.lon, { forced: true });
        else if (t === 'cold' || t === 'warm') { sim.diagnose(); sim.addFront(t, a.lat, a.lon, b.lat, b.lon); }
      }
    }
    commitEdit();
    drawUI(hover);
  };
  cv.addEventListener('pointerup', end);
  cv.addEventListener('pointercancel', end);
  cv.addEventListener('pointerleave', () => { $('readout').style.display = 'none'; hover = null; drawUI(null); });
  cv.addEventListener('contextmenu', e => e.preventDefault());
  cv.addEventListener('wheel', onWheel, { passive: false });
}

function paintTick() {
  if (!drag || drag.kind !== 'paint') return;
  const now = performance.now(), dt = Math.min(0.1, (now - lastPaint) / 1000);
  lastPaint = now;
  applyPaint(dt);
  paintLoop = requestAnimationFrame(paintTick);
}
function applyPaint(dt) {
  const { lat, lon } = drag.at, km = state.brushKm, t = state.tool;
  if (t === 'rain') sim.paintRain(lat, lon, km, state.intensity / 5 * dt * 4);
  else if (t === 'moist') sim.paintMoisture(lat, lon, km, state.sign * dt * 3);
  else if (t === 'heat') sim.paintHeat(lat, lon, km, state.sign * dt * 3);
  else if (t === 'erase') sim.erase(lat, lon, km);
  const now = performance.now();
  if (now - lastDiag > 250) { sim.diagnose(); lastDiag = now; }
  dirty = true;
  requestRender();
}

function drawUI(p) {
  const g = uic;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, ui.width, ui.height);
  if (!p || !sim) return;
  g.save();
  g.translate(view.ox, view.oy);
  const t = state.tool;
  const lat = view.lat(p.vy);
  if (TOOLS.find(x => x.key === t).opts.includes('brush')) {
    const r = state.brushKm / kmPerPx(lat);
    g.strokeStyle = 'rgba(255,255,255,.9)'; g.lineWidth = 1.5 * dpr;
    g.setLineDash([5 * dpr, 4 * dpr]);
    g.beginPath(); g.arc(p.vx, p.vy, r, 0, 7); g.stroke();
    g.strokeStyle = 'rgba(0,0,0,.6)'; g.lineDashOffset = 4 * dpr; g.beginPath(); g.arc(p.vx, p.vy, r, 0, 7); g.stroke();
    g.setLineDash([]);
  }
  if (t === 'move' && !drag) {
    const h = hitTest(p);
    if (h) {
      const x = view.px(h.obj.lon), y = view.py(h.obj.lat);
      g.strokeStyle = '#3aa0ff'; g.lineWidth = 2.5 * dpr;
      g.beginPath(); g.arc(x, y, h.kind === 'sys' ? 24 * dpr : Math.max(12 * dpr, h.obj.r * 2 / kmPerPx(h.obj.lat)), 0, 7); g.stroke();
    }
  }
  if (drag && drag.kind === 'shape') {
    const a = drag.pa, b = drag.pb;
    g.lineWidth = 3 * dpr;
    if (t === 'low' || t === 'high') {
      const r = Math.max(Math.hypot(b.vx - a.vx, b.vy - a.vy), 150 / kmPerPx(lat));
      g.strokeStyle = t === 'low' ? '#ff4d4d' : '#4d7dff';
      g.beginPath(); g.arc(a.vx, a.vy, r, 0, 7); g.stroke();
      g.fillStyle = g.strokeStyle; g.font = `800 ${Math.round(22 * dpr)}px Inter, sans-serif`; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(t === 'low' ? 'L' : 'H', a.vx, a.vy);
      g.font = `600 ${Math.round(12 * dpr)}px Inter, sans-serif`;
      g.fillText(`${t === 'low' ? '−' : '+'}${3 * state.intensity} hPa · ${Math.round(r * kmPerPx(lat))} km`, a.vx, a.vy + 22 * dpr);
    } else {
      g.strokeStyle = t === 'cold' ? '#2f6bff' : t === 'warm' ? '#ff3b3b' : '#ffd24a';
      g.beginPath(); g.moveTo(a.vx, a.vy); g.lineTo(b.vx, b.vy); g.stroke();
      if (t === 'cold' || t === 'warm') {
        const L = Math.hypot(b.vx - a.vx, b.vy - a.vy), n = Math.floor(L / (26 * dpr));
        const ux = (b.vx - a.vx) / L, uy = (b.vy - a.vy) / L;
        const [su, sv] = windFrom(state.P.steerDir, state.P.steerSpd);
        // Symbols point downwind (direction of motion).
        let nx = -uy, ny = ux;
        if (nx * su + ny * -sv < 0) { nx = -nx; ny = -ny; }
        g.fillStyle = g.strokeStyle;
        for (let i = 0; i < n; i++) {
          const cx = a.vx + ux * (i + 0.5) * 26 * dpr, cy = a.vy + uy * (i + 0.5) * 26 * dpr, s = 8 * dpr;
          g.beginPath();
          if (t === 'cold') { g.moveTo(cx - ux * s, cy - uy * s); g.lineTo(cx + nx * s * 1.2, cy + ny * s * 1.2); g.lineTo(cx + ux * s, cy + uy * s); }
          else g.arc(cx, cy, s, Math.atan2(uy, ux), Math.atan2(uy, ux) + Math.PI, nx * uy - ny * ux > 0);
          g.fill();
        }
      }
    }
  }
  g.restore();
}

// ---------- wheel zoom ----------
let zoomTimer = null, zoomAcc = 1, zoomAt = null;
function onWheel(e) {
  e.preventDefault();
  if (!view) return;
  const p = evPos(e);
  zoomAcc *= Math.exp(e.deltaY * 0.0015);
  zoomAcc = clamp(zoomAcc, 0.15, 6);
  zoomAt = { mx: view.mx0 + p.vx / view.scale, my: view.my0 + p.vy / view.scale, fx: p.vx / view.w, fy: p.vy / view.h };
  const r = cv.getBoundingClientRect();
  cv.style.transformOrigin = `${e.clientX - r.left}px ${e.clientY - r.top}px`;
  cv.style.transform = `scale(${1 / zoomAcc})`;
  clearTimeout(zoomTimer);
  zoomTimer = setTimeout(applyZoom, 380);
}
function applyZoom() {
  const f = zoomAcc;
  zoomAcc = 1;
  cv.style.transform = '';
  const spanX = view.w / view.scale * f, spanY = view.h / view.scale * f;
  let mx0 = zoomAt.mx - zoomAt.fx * spanX, my0 = zoomAt.my - zoomAt.fy * spanY;
  const W = invMercX(mx0), E = invMercX(mx0 + spanX), N = invMercY(my0), S = invMercY(my0 + spanY);
  if (E - W < 0.7 || E - W > 75) { requestRender(); return; }
  REGIONS.custom = { label: 'Custom', bbox: [W, S, E, N] };
  if (!document.querySelector('#regions button[data-region=custom]')) {
    const b = document.createElement('button');
    b.textContent = 'Custom'; b.dataset.region = 'custom';
    b.onclick = () => switchRegion('custom');
    $('regions').appendChild(b);
  }
  state.region = 'custom';
  highlight();
  buildRegion();
}

// ---------- wind particles ----------
let parts = [];
function resetParticles() {
  fxc.setTransform(1, 0, 0, 1, 0, 0);
  fxc.clearRect(0, 0, fx.width, fx.height);
  parts = [];
  if (!state.layers.particles || !view) return;
  const n = Math.round(view.w * view.h / (1400 * dpr * dpr));
  for (let i = 0; i < n; i++) parts.push(newPart());
}
function newPart() { return { x: Math.random() * view.w, y: Math.random() * view.h, age: Math.floor(Math.random() * 90) }; }
function particleTick() {
  requestAnimationFrame(particleTick);
  if (!state.layers.particles || !sim || !parts.length) return;
  const g = fxc;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.globalCompositeOperation = 'destination-in';
  g.fillStyle = 'rgba(0,0,0,0.9)';
  g.fillRect(0, 0, fx.width, fx.height);
  g.globalCompositeOperation = 'source-over';
  const dark = state.mode === 'sat' || baseStyle() === 'dark';
  g.strokeStyle = dark ? 'rgba(255,255,255,.75)' : 'rgba(25,35,55,.7)';
  g.lineWidth = 1.1 * dpr;
  g.beginPath();
  const k = 0.16 * dpr;
  for (const q of parts) {
    const gx = q.x / sim.cpx - 0.5, gy = q.y / sim.cpy - 0.5;
    const u = sim.sample(sim.u, gx, gy), v = sim.sample(sim.v, gx, gy);
    const nx = q.x + u * k, ny = q.y - v * k;
    g.moveTo(view.ox + q.x, view.oy + q.y); g.lineTo(view.ox + nx, view.oy + ny);
    q.x = nx; q.y = ny; q.age++;
    if (q.age > 100 || nx < 0 || ny < 0 || nx > view.w || ny > view.h) Object.assign(q, newPart(), { age: 0 });
  }
  g.stroke();
}

// ---------- loading overlay ----------
let loadTimer = null;
function showLoading(text, autoHideMs) {
  $('loadingText').textContent = text;
  $('loading').classList.remove('hide');
  clearTimeout(loadTimer);
  if (autoHideMs) loadTimer = setTimeout(hideLoading, autoHideMs);
}
function hideLoading() { $('loading').classList.add('hide'); }

// ---------- boot ----------
async function boot() {
  cv = $('map'); ctx = cv.getContext('2d');
  fx = $('fx'); fxc = fx.getContext('2d');
  ui = $('ui'); uic = ui.getContext('2d');
  buildUI();
  setupPointer();
  updateLegend();
  requestAnimationFrame(particleTick);
  try { borders = await (await fetch('data/borders.json')).json(); } catch (e) { borders = []; }
  try { lakes = await (await fetch('data/lakes.json')).json(); } catch (e) { lakes = []; }
  await buildRegion();
  await applyPreset('autumn');
  let resizeTimer = null, lastSize = `${$('stage').clientWidth}x${$('stage').clientHeight}`;
  new ResizeObserver(() => {
    const size = `${$('stage').clientWidth}x${$('stage').clientHeight}`;
    if (size === lastSize) return;
    lastSize = size;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (sim) buildRegion(); }, 300);
  }).observe($('stage'));
  window.__sandbox = { get sim() { return sim; }, get raster() { return raster; }, state, forward, applyPreset, setMode, switchRegion };
}
boot();
