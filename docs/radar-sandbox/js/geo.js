// Web-Mercator helpers, region presets and the View that maps lon/lat to
// canvas pixels.

export const EARTH_CIRC = 40075016.686;
export const R_EARTH = 6371000;
const D2R = Math.PI / 180;

export const mercX = lon => (lon + 180) / 360;
export function mercY(lat) {
  const s = Math.sin(Math.max(-85, Math.min(85, lat)) * D2R);
  return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
}
export const invMercX = x => x * 360 - 180;
export const invMercY = y => Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) / D2R;

export const RADAR_SITES = {
  uljenje: { name: 'MRC Uljenje', short: 'Uljenje', lat: 42.8944, lon: 17.4783, rangeKm: 248 },
  bilogora: { name: 'Bilogora', short: 'Bilogora', lat: 45.8838, lon: 17.2006, rangeKm: 240 },
  osijek: { name: 'Osijek', short: 'Osijek', lat: 45.5027, lon: 18.5613, rangeKm: 240 },
  samanli: { name: 'Samanli', short: 'Samanli', lat: 42.98, lon: 21.31, rangeKm: 240 },
};

export const REGIONS = {
  europe: { label: 'Europe', bbox: [-12, 34.5, 42, 62] },
  balkans: { label: 'SE Europe', bbox: [12.3, 36.6, 30.2, 47.9] },
  adriatic: { label: 'Adriatic', bbox: [12.0, 39.6, 21.4, 46.1] },
  montenegro: { label: 'Montenegro', bbox: [18.2, 41.72, 20.6, 43.62] },
  uljenje: { label: 'Radar Uljenje', radar: 'uljenje', radarFrame: true },
};

export function bboxAround(lat, lon, km) {
  const dLat = km / 111.32, dLon = km / (111.32 * Math.cos(lat * D2R));
  return [lon - dLon, lat - dLat, lon + dLon, lat + dLat];
}

export class View {
  // Covers the rectangle [ox, oy, w, h] of the canvas with the bbox,
  // expanding the bbox to the rectangle's aspect ratio.
  constructor(bbox, ox, oy, w, h) {
    const [W, S, E, N] = bbox;
    const x0 = mercX(W), x1 = mercX(E), y0 = mercY(N), y1 = mercY(S);
    this.scale = Math.min(w / (x1 - x0), h / (y1 - y0));
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    this.mx0 = cx - w / 2 / this.scale;
    this.my0 = cy - h / 2 / this.scale;
    this.ox = ox; this.oy = oy; this.w = w; this.h = h;
    this.bbox = [invMercX(this.mx0), invMercY(this.my0 + h / this.scale),
      invMercX(this.mx0 + w / this.scale), invMercY(this.my0)];
  }
  // Pixel coordinates relative to the view rectangle (not the canvas).
  px(lon) { return (mercX(lon) - this.mx0) * this.scale; }
  py(lat) { return (mercY(lat) - this.my0) * this.scale; }
  lon(px) { return invMercX(this.mx0 + px / this.scale); }
  lat(py) { return invMercY(this.my0 + py / this.scale); }
  metersPerPx(lat) { return EARTH_CIRC * Math.cos(lat * D2R) / this.scale; }
  contains(lon, lat, pad = 0) {
    const x = this.px(lon), y = this.py(lat);
    return x >= -pad && y >= -pad && x <= this.w + pad && y <= this.h + pad;
  }
}

export function distKm(lat1, lon1, lat2, lon2) {
  const dLat = (lat2 - lat1) * D2R, dLon = (lon2 - lon1) * D2R;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * D2R) * Math.cos(lat2 * D2R) * Math.sin(dLon / 2) ** 2;
  return 2 * R_EARTH * Math.asin(Math.min(1, Math.sqrt(a))) / 1000;
}

// Moves a lon/lat point by (east, north) metres.
export function offsetLL(lat, lon, east, north) {
  return [lat + north / 111320, lon + east / (111320 * Math.cos(lat * D2R))];
}
