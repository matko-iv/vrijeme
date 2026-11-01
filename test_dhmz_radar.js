// Tests the in-browser DHMZ radar scan in docs/forecast.html.
// Run:  node test_dhmz_radar.js      (exit 0 = pass)
//
// The scan is inline in forecast.html, so we slice the real source out of the
// file and evaluate it on synthetic 720x751 RGBA frames. The slice runs from the
// DHMZ marker comment to just before fetchDhmzState(); if you move or rename
// either anchor, update MARK_START / MARK_END below.

const fs = require('fs');
const path = require('path');

const MARK_START = '// ---- DHMZ radar read in the browser, no pipeline needed ----';
const MARK_END = '// The current DHMZ frame as a radarState';

const html = fs.readFileSync(path.join(__dirname, 'docs', 'forecast.html'), 'utf8');
const start = html.indexOf(MARK_START);
const end = html.indexOf(MARK_END, start);
if (start < 0 || end < 0) {
    console.error('FAIL: could not locate the DHMZ region in docs/forecast.html');
    process.exit(1);
}
const api = new Function(html.slice(start, end) + `
    return { scanDhmzPixels, dhmzClassify, DHMZ_GEOM };
`)();

let failures = 0;
function eq(actual, expected, what) {
    const ok = Object.is(actual, expected);
    if (!ok) { failures++; console.error(`FAIL: ${what}\n  expected: ${expected}\n  actual:   ${actual}`); }
    else console.log(`ok: ${what}`);
}

const { w, h, budva, pxPerKm } = api.DHMZ_GEOM;
const SEA = [0, 60, 112];
const GREEN_32 = [0, 192, 39];      // 32.5 dBZ
const YELLOW_42 = [255, 255, 0];    // 42.5 dBZ
const RED_57 = [255, 0, 0];         // 57.5 dBZ
function frame() {
    const px = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) px.set([...SEA, 255], i * 4);
    return px;
}
function disc(px, cx, cy, r, rgb) {
    for (let y = cy - r; y <= cy + r; y++) {
        for (let x = cx - r; x <= cx + r; x++) {
            if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) px.set(rgb, (y * w + x) * 4);
        }
    }
}

eq(api.dhmzClassify(...GREEN_32), 32.5, 'legend green reads 32.5 dBZ');
eq(Number.isNaN(api.dhmzClassify(...SEA)), true, 'sea tone is not echo');
eq(Number.isNaN(api.dhmzClassify(212, 155, 95)), true, 'brown relief tone is not mistaken for orange echo');

let s = api.scanDhmzPixels(frame());
eq(s.rainAtLocation, false, 'empty frame: no rain at Budva');
eq(s.bestRain, null, 'empty frame: no closest rain');

let px = frame();
disc(px, budva.x, budva.y, 6, GREEN_32);
s = api.scanDhmzPixels(px);
eq(s.rainAtLocation, true, 'rain disc over Budva: rain at Budva');
eq(s.bestRain.dbz, 32.5, 'rain disc over Budva: its intensity');
eq(s.severe, false, '32.5 dBZ is not severe');

px = frame();
px.set(GREEN_32, (budva.y * w + budva.x) * 4);
s = api.scanDhmzPixels(px);
eq(s.rainAtLocation, false, 'a lone wet pixel on Budva is speckle, not rain');

// 12 km south-west of Budva, radius 5 px (~3.7 km): the near edge is ~8.8 km away.
const off = Math.round(12 * pxPerKm / Math.SQRT2);
px = frame();
disc(px, budva.x - off, budva.y + off, 5, YELLOW_42);
s = api.scanDhmzPixels(px);
eq(s.rainAtLocation, false, 'cell 12 km SW: not at Budva');
eq(s.bestRain.cardinal, 'SW', 'cell 12 km SW: direction');
eq(Math.round(s.bestRain.km), 9, 'cell 12 km SW: closest edge ~9 km');
eq(s.bestRain.dbz, 42.5, 'cell 12 km SW: intensity');

px = frame();
disc(px, budva.x, budva.y, 7, GREEN_32);
disc(px, budva.x, budva.y, 3, RED_57);
s = api.scanDhmzPixels(px);
eq(s.severe, true, '57.5 dBZ core over Budva is severe');

// Rain 200 km away is outside the 150 km scan.
px = frame();
disc(px, budva.x, budva.y - Math.round(200 * pxPerKm), 6, GREEN_32);
s = api.scanDhmzPixels(px);
eq(s.bestRain, null, 'rain beyond 150 km is ignored');

if (failures) { console.error(`\n${failures} test(s) failed`); process.exit(1); }
console.log('\nall tests passed');
