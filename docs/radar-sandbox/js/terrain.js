// Terrain: AWS Terrain Tiles (Mapzen terrarium encoding, SRTM/GMTED based,
// public with CORS) stitched into an elevation grid at canvas resolution.

const TILE_URL = (z, x, y) => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${z}/${x}/${y}.png`;
const tileCache = new Map(); // key -> Promise<Float32Array|null>
const MAX_CACHE = 600;

function decodeTile(img) {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, 256, 256).data;
  const out = new Float32Array(256 * 256);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = d[p] * 256 + d[p + 1] + d[p + 2] / 256 - 32768;
  }
  return out;
}

function loadTile(z, x, y) {
  const n = 1 << z;
  x = ((x % n) + n) % n;
  if (y < 0 || y >= n) return Promise.resolve(null);
  const key = `${z}/${x}/${y}`;
  if (tileCache.has(key)) return tileCache.get(key);
  const p = new Promise(resolve => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => { try { resolve(decodeTile(img)); } catch (e) { resolve(null); } };
    img.onerror = () => resolve(null);
    img.src = TILE_URL(z, x, y);
  });
  tileCache.set(key, p);
  if (tileCache.size > MAX_CACHE) tileCache.delete(tileCache.keys().next().value);
  return p;
}

async function pool(tasks, limit, onDone) {
  const results = new Array(tasks.length);
  let next = 0, done = 0;
  async function worker() {
    while (next < tasks.length) {
      const i = next++;
      results[i] = await tasks[i]();
      onDone && onDone(++done, tasks.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

// Returns {elev: Float32Array(w*h), water, w, h, ok: fraction of tiles loaded, z}.
export async function buildDEM(view, w, h, onProgress, lakes) {
  let z = Math.ceil(Math.log2(view.scale / 256) + 0.15);
  z = Math.max(2, Math.min(12, z));
  let tx0, tx1, ty0, ty1;
  for (;;) {
    const n = 1 << z;
    tx0 = Math.floor(view.mx0 * n); tx1 = Math.floor((view.mx0 + w / view.scale) * n);
    ty0 = Math.floor(view.my0 * n); ty1 = Math.floor((view.my0 + h / view.scale) * n);
    if ((tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= 140 || z <= 2) break;
    z--;
  }
  const cols = tx1 - tx0 + 1, rows = ty1 - ty0 + 1;
  const tasks = [];
  for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) tasks.push(() => loadTile(z, tx, ty));
  const tiles = await pool(tasks, 12, onProgress);
  const ok = tiles.filter(Boolean).length / tiles.length;

  const n = 1 << z, S = 256 * n, MW = cols * 256;
  const elev = new Float32Array(w * h);
  const sample = (gx, gy) => {
    const c = Math.floor(gx / 256), r = Math.floor(gy / 256);
    if (c < 0 || r < 0 || c >= cols || r >= rows) return 0;
    const t = tiles[r * cols + c];
    if (!t) return 0;
    return t[(gy - r * 256) * 256 + (gx - c * 256)];
  };
  const MH = rows * 256;
  for (let y = 0; y < h; y++) {
    const fy = (view.my0 + (y + 0.5) / view.scale) * S - ty0 * 256 - 0.5;
    const iy = Math.max(0, Math.min(MH - 2, Math.floor(fy))), wy = Math.min(1, Math.max(0, fy - iy));
    for (let x = 0; x < w; x++) {
      const fx = (view.mx0 + (x + 0.5) / view.scale) * S - tx0 * 256 - 0.5;
      const ix = Math.max(0, Math.min(MW - 2, Math.floor(fx))), wx = Math.min(1, Math.max(0, fx - ix));
      const a = sample(ix, iy), b = sample(ix + 1, iy), c = sample(ix, iy + 1), d = sample(ix + 1, iy + 1);
      elev[y * w + x] = (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy;
    }
  }
  return { elev, w, h, ok, z, water: detectWater(elev, w, h, view, lakes) };
}

// Water = below sea level (cleaned of the speckle that interpolation leaves
// along coasts) plus lakes rasterised from Natural Earth polygons.
function detectWater(elev, w, h, view, lakes) {
  const raw = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) raw[i] = elev[i] <= 0 ? 1 : 0;
  const water = new Uint8Array(raw);
  const R = 2;
  for (let y = R; y < h - R; y++) {
    for (let x = R; x < w - R; x++) {
      const i = y * w + x;
      if (Math.abs(elev[i]) > 12) continue;
      let n = 0;
      for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) n += raw[i + dy * w + dx];
      water[i] = n * 2 > (2 * R + 1) ** 2 ? 1 : 0;
    }
  }
  if (lakes && lakes.length) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.fillStyle = '#fff';
    g.beginPath();
    for (const ring of lakes) {
      for (let k = 0; k < ring.length; k++) {
        const x = view.px(ring[k][0]), y = view.py(ring[k][1]);
        if (k === 0) g.moveTo(x, y); else g.lineTo(x, y);
      }
      g.closePath();
    }
    g.fill();
    const d = g.getImageData(0, 0, w, h).data;
    for (let i = 0; i < w * h; i++) if (d[i * 4 + 3] > 127) water[i] = 1;
  }
  return water;
}
