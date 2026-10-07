// Tests the SKALA NOWCAST hourly gating in docs/forecast.html.
// Run:  node test_nowcast_hourly.js      (exit 0 = pass)
//
// The gating functions are inline in forecast.html, so we slice the real source
// out of the file and evaluate it — nothing is copied into this test. The slice
// runs from the marker comment to just before dailyIconKey(); if you move or
// rename either anchor, update MARK_START / MARK_END below.

if (process.env.TZ !== 'Europe/Podgorica') {
    // The UTC-bucket -> local-cell join is timezone-sensitive, and TZ must be set
    // before the first Date use, so re-exec with the page's timezone.
    const r = require('child_process').spawnSync(process.execPath, [__filename], {
        stdio: 'inherit', env: Object.assign({}, process.env, { TZ: 'Europe/Podgorica' })
    });
    process.exit(r.status === null ? 1 : r.status);
}

const fs = require('fs');
const path = require('path');

const MARK_START = '// ---- SKALA gating of near-term rain icons ----';
const MARK_END = 'function dailyIconKey(';

const html = fs.readFileSync(path.join(__dirname, 'docs', 'forecast.html'), 'utf8');
const start = html.indexOf(MARK_START);
const end = html.indexOf(MARK_END, start);
if (start < 0 || end < 0) {
    console.error('FAIL: could not locate the gating region in docs/forecast.html');
    process.exit(1);
}
const src = html.slice(start, end);

const api = new Function(src + `
    return { nowcastHour, nowcastCellMm, gateWeatherCode, nowcastFresh, nowcastIntensityCode,
             nowcastImminent, nowcastImminentCode, nowcastNowRate, escalateCloud, cloudToCode,
             cellIsCurrentHour, HOUR_COVER_MIN, rainCellParts, POSSIBLE_RAIN_POP,
             setState: (ns, rs) => { nowcastState = ns; radarState = rs; } };
`)();

let failures = 0;
function eq(actual, expected, what) {
    const ok = Object.is(actual, expected);
    if (!ok) { failures++; console.error(`FAIL: ${what}\n  expected: ${expected}\n  actual:   ${actual}`); }
    else console.log(`ok: ${what}`);
}

const realNow = Date.now;
function at(epochMs, fn) {
    Date.now = () => epochMs;
    try { fn(); } finally { Date.now = realNow; }
}

// Sanity: the whole join rests on the process running in Budva's timezone.
eq(new Date('2026-07-09T13:00:00Z').getHours(), 15, 'TZ check: 13:00Z is 15:00 local (CEST)');

// Base 13:40Z = 15:40 local. Rain starts at 14:05Z, i.e. entirely inside the
// 14:00Z clock hour (= the 16:00 local cell). The old lead-offset math integrated
// leads 0-60 (13:40Z-14:40Z) into the CURRENT hour's cell, crediting 4.0 mm of the
// 16:00 cell's rain to the 15:00 cell — and its icon with it.
const BASE = Date.parse('2026-07-09T13:40:00Z');
const series = [];
for (let lead = 5; lead <= 80; lead += 5) {
    const wet = lead >= 25;
    series.push({ lead_min: lead, point_mmh: wet ? 6.0 : 0.0, disc_max_mmh: wet ? 8.0 : 0.0 });
}
const ns = {
    ok: true, base_epoch_ms: BASE, horizon_min: 80, timestep_min: 5,
    now: { point_mmh: 0.0, disc_max_mmh: 0.0 },
    series,
    hourly_mm: [
        { hour: '2026-07-09T13:00:00Z', mm: 0.0, covered_min: 20, peak_point_mmh: 0.0, peak_disc_mmh: 0.0 },
        { hour: '2026-07-09T14:00:00Z', mm: 6.0, covered_min: 60, peak_point_mmh: 6.0, peak_disc_mmh: 8.0 },
    ],
};

at(BASE + 5 * 60000, () => {
    api.setState(ns, null);

    // 1. Mid-hour bug: the buckets must land on the right clock-hour cells.
    const cur = api.nowcastHour(ns, '2026-07-09T15:00:00');
    const nxt = api.nowcastHour(ns, '2026-07-09T16:00:00');
    eq(cur && cur.hour, '2026-07-09T13:00:00Z', '15:00 local cell -> 13:00Z bucket');
    eq(nxt && nxt.hour, '2026-07-09T14:00:00Z', '16:00 local cell -> 14:00Z bucket');
    eq(cur && cur.mm, 0.0, '15:00 cell mm = 0.0 (rain is all in the next clock hour)');
    eq(nxt && nxt.mm, 6.0, '16:00 cell mm = 6.0');
    eq(api.nowcastCellMm(cur, true), 0.0, 'current hour overrides mm even at 20 min coverage');
    eq(api.nowcastCellMm(nxt, false), 6.0, 'fully covered hour overrides mm');

    // Same bug, on the icon: the rain icon belongs to the 16:00 cell, not the 15:00 one.
    eq(api.gateWeatherCode(0, 5, cur, true, null), 0, '15:00 cell stays clear (no rain in its hour)');
    eq(api.gateWeatherCode(0, 5, nxt, false, null), 63, '16:00 cell gets the rain icon (6 mm/h -> 63)');

    // 2. Icon upgrade on rain at Budva.
    const b10 = { hour: '2026-07-09T14:00:00Z', mm: 5, covered_min: 60, peak_point_mmh: 10, peak_disc_mmh: 12 };
    eq(api.gateWeatherCode(0, 10, b10, false, null), 82, 'peak_point 10 mm/h upgrades clear -> 82');

    // 3. Never downgrade a heavier model code.
    const bLight = { hour: '2026-07-09T14:00:00Z', mm: 0.3, covered_min: 60, peak_point_mmh: 0.5, peak_disc_mmh: 0.5 };
    eq(api.gateWeatherCode(95, 90, bLight, false, null), 95, 'model 95 + light nowcast rain stays 95');

    // 4. Nearby cell, dry at Budva.
    const bStrong = { hour: '2026-07-09T14:00:00Z', mm: 0, covered_min: 60, peak_point_mmh: 0, peak_disc_mmh: 30 };
    const bWeak = { hour: '2026-07-09T14:00:00Z', mm: 0, covered_min: 60, peak_point_mmh: 0, peak_disc_mmh: 5 };
    eq(api.gateWeatherCode(0, 10, bStrong, false, null), 95, 'strong nearby cell -> 95');
    eq(api.gateWeatherCode(0, 10, bWeak, false, null), api.escalateCloud(0), 'weak nearby cell -> cloud nudge, no rain code');
    eq(api.gateWeatherCode(0, 10, bWeak, false, null), 2, 'weak nearby cell: clear -> partly cloudy');

    // 5. Strip model rain only when the hour is actually covered.
    const bDryCovered = { hour: '2026-07-09T14:00:00Z', mm: 0, covered_min: 60, peak_point_mmh: 0, peak_disc_mmh: 0 };
    const bDryThin = { hour: '2026-07-09T14:00:00Z', mm: 0, covered_min: 10, peak_point_mmh: 0, peak_disc_mmh: 0 };
    eq(api.gateWeatherCode(61, 80, bDryCovered, false, null), api.cloudToCode(80), 'dry + covered 60 min strips model rain');
    eq(api.gateWeatherCode(61, 80, bDryCovered, false, null), 3, 'stripped code is cloudToCode(80) = 3');
    eq(api.gateWeatherCode(61, 80, bDryThin, false, null), 61, 'dry but only 10 min covered keeps the model rain code');
    eq(api.nowcastCellMm(bDryThin, false), null, 'thinly covered non-current hour keeps the NWP mm');

    // 5b. The HOUR_COVER_MIN boundary — the real JSON routinely sits right around it.
    const bDry29 = { hour: '2026-07-09T14:00:00Z', mm: 0, covered_min: 29, peak_point_mmh: 0, peak_disc_mmh: 0 };
    const bDry30 = { hour: '2026-07-09T14:00:00Z', mm: 0, covered_min: 30, peak_point_mmh: 0, peak_disc_mmh: 0 };
    eq(api.gateWeatherCode(61, 80, bDry29, false, null), 61, 'covered 29 min keeps the model rain code');
    eq(api.gateWeatherCode(61, 80, bDry30, false, null), 3, 'covered 30 min strips it (>= HOUR_COVER_MIN)');
    eq(api.nowcastCellMm(bDry29, false), null, 'covered 29 min does not override mm');
    eq(api.nowcastCellMm(bDry30, false), 0, 'covered 30 min overrides mm');

    // 5c. Old-schema bucket (fresh JSON, generator not yet redeployed): mm/covered_min but
    // no peaks. Absent peaks are NOT 0 — such a bucket may neither strip the icon nor
    // restate the mm, or the cell would show a cloudy icon next to a blue 6.0mm.
    const bOld = { hour: '2026-07-09T14:00:00Z', mm: 6.0, covered_min: 60 };
    eq(api.gateWeatherCode(63, 80, bOld, false, null), 63, 'bucket without peaks does not strip the model rain code');
    eq(api.gateWeatherCode(0, 80, bOld, false, null), 0, 'bucket without peaks does not upgrade either');
    eq(api.nowcastCellMm(bOld, false), null, 'bucket without peaks leaves the NWP mm -> icon and mm agree');
    eq(api.nowcastCellMm(bOld, true), null, 'same for the current hour');

    // 6. Beyond the horizon: no bucket, nothing changes.
    eq(api.nowcastHour(ns, '2026-07-09T18:00:00'), null, '18:00 local cell is beyond the horizon');
    eq(api.gateWeatherCode(61, 80, null, false, null), 61, 'no bucket -> code unchanged');
    eq(api.nowcastCellMm(null, false), null, 'no bucket -> mm unchanged');

    // 6b. A status file with no hourly_mm at all (or an empty one) must not throw.
    const nsNoHourly = { ok: true, base_epoch_ms: BASE, series };
    eq(api.nowcastHour(nsNoHourly, '2026-07-09T15:00:00'), null, 'missing hourly_mm -> no bucket');
    eq(api.nowcastHour({ ok: true, base_epoch_ms: BASE, hourly_mm: [] }, '2026-07-09T15:00:00'), null, 'empty hourly_mm -> no bucket');
    eq(api.nowcastHour(ns, ''), null, 'cell with no datetime -> no bucket');
    api.setState(nsNoHourly, null);
    eq(api.gateWeatherCode(61, 80, api.nowcastHour(nsNoHourly, '2026-07-09T15:00:00'), true, null), 61,
       'missing hourly_mm -> clean fall-through, code unchanged');
    api.setState(ns, null);

    // 7. Imminent = DGMR says rain is falling at Budva RIGHT NOW (the series sample at
    // the forecast's current age) — NEVER before the onset actually arrives.
    const nsImm = {
        ok: true, base_epoch_ms: BASE, timestep_min: 5,
        now: { point_mmh: 0.0, disc_max_mmh: 0.0 },
        series: [
            { lead_min: 5, point_mmh: 6.0, disc_max_mmh: 6.0 },    // current lead (age 5) -> 63
            { lead_min: 30, point_mmh: 40.0, disc_max_mmh: 40.0 }, // later peak must NOT leak in
        ],
    };
    api.setState(nsImm, null);
    eq(api.nowcastImminent(), true, 'raining at the current lead -> imminent');
    eq(api.nowcastImminentCode(), 63, 'imminent intensity is the CURRENT rate, not a later peak (6 mm/h -> 63)');
    api.setState(nsImm, { ok: true, ageMin: 5, rainAtLocation: true });
    eq(api.nowcastImminent(), false, 'SKALA RAIN already shows the rain -> no force needed');
    const nsPre = {
        ok: true, base_epoch_ms: BASE, timestep_min: 5,
        now: { point_mmh: 0.0, disc_max_mmh: 0.0 },
        series: [
            { lead_min: 5, point_mmh: 0.0, disc_max_mmh: 0.0 },
            { lead_min: 10, point_mmh: 8.0, disc_max_mmh: 8.0 },   // onset 5 min from now
        ],
    };
    api.setState(nsPre, null);
    eq(api.nowcastImminent(), false, 'onset 5 min away is NOT imminent — the icon waits for the rain');
    const nsBaseFrame = {
        ok: true, base_epoch_ms: BASE, timestep_min: 5,
        now: { point_mmh: 2.5, disc_max_mmh: 3.0 },
        series: [{ lead_min: 10, point_mmh: 0.0, disc_max_mmh: 0.0 }],  // first lead still ahead
    };
    api.setState(nsBaseFrame, null);
    eq(api.nowcastImminent(), true, 'age below the first lead reads the base (now) frame');
    eq(api.nowcastImminentCode(), 63, 'now-frame 2.5 mm/h -> 63');
    api.setState(ns, null);

    // SKALA RAIN fallback stays current-hour only, and only when the nowcast is absent.
    api.setState(null, null);
    const rs = { ok: true, ageMin: 5, rainAtLocation: true, bestRain: { dbz: 40 } };
    eq(api.gateWeatherCode(0, 10, null, true, rs), 63, 'no nowcast: SKALA RAIN gates the current hour');
    eq(api.gateWeatherCode(0, 10, null, false, rs), 0, 'no nowcast: SKALA RAIN does not gate later hours');
});

// 7b. CURRENT hour: the icon means NOW. Rain later in the same clock hour must not
// paint the cell (nor the big 'now' icon) before the onset actually arrives — it may
// still be sunny outside. Onset is 14:05Z (lead 25); the 14:00Z bucket peaks 6 mm/h.
at(BASE + 22 * 60000, () => {   // 14:02Z = 16:02 local — 3 min BEFORE onset, still dry
    api.setState(ns, null);
    const b = api.nowcastHour(ns, '2026-07-09T16:00:00');
    eq(b && b.peak_point_mmh, 6.0, 'sanity: the current-hour bucket does peak 6 mm/h');
    eq(api.gateWeatherCode(0, 5, b, true, null), 0, 'current hour stays clear until rain actually starts');
    eq(api.gateWeatherCode(61, 80, b, true, null), 3, 'current hour: DGMR dry at this moment strips the model rain code');
    eq(api.nowcastCellMm(b, true), 6.0, 'current hour mm still shows the DGMR hour total');
});
at(BASE + 30 * 60000, () => {   // 14:10Z — rain started at 14:05Z
    api.setState(ns, null);
    const b = api.nowcastHour(ns, '2026-07-09T16:00:00');
    eq(api.gateWeatherCode(0, 5, b, true, null), 63, 'current hour flips to rain once it is falling (6 mm/h -> 63)');
});
// Same rule for the nearby-cell (disc) escalation: a strong cell due later in the
// hour must not raise the thunderstorm icon while the sky is still quiet.
const seriesD = [];
for (let lead = 5; lead <= 80; lead += 5) {
    seriesD.push({ lead_min: lead, point_mmh: 0.0, disc_max_mmh: lead >= 25 ? 30.0 : 0.0 });
}
const nsD = {
    ok: true, base_epoch_ms: BASE, horizon_min: 80, timestep_min: 5,
    now: { point_mmh: 0.0, disc_max_mmh: 0.0 }, series: seriesD,
    hourly_mm: [
        { hour: '2026-07-09T13:00:00Z', mm: 0.0, covered_min: 20, peak_point_mmh: 0.0, peak_disc_mmh: 0.0 },
        { hour: '2026-07-09T14:00:00Z', mm: 0.0, covered_min: 60, peak_point_mmh: 0.0, peak_disc_mmh: 30.0 },
    ],
};
at(BASE + 22 * 60000, () => {
    api.setState(nsD, null);
    const b = api.nowcastHour(nsD, '2026-07-09T16:00:00');
    eq(api.gateWeatherCode(0, 5, b, true, null), 0, 'strong cell due later this hour does not paint 95 early');
});
at(BASE + 30 * 60000, () => {
    api.setState(nsD, null);
    const b = api.nowcastHour(nsD, '2026-07-09T16:00:00');
    eq(api.gateWeatherCode(0, 5, b, true, null), 95, 'strong cell NEARBY RIGHT NOW -> thunderstorm icon');
});

// 7c. Hour rollover: render() runs once and stamps data-now, so applyRadarGating()
// must take "is this the current hour" from the wall clock, not the stamp — otherwise
// after midnight the new hour's cell is gated as a FUTURE hour and its bucket peak
// keeps the model's rain icon while it is still dry outside.
at(BASE + 25 * 60000, () => {   // 14:05Z = 16:05 local
    eq(api.cellIsCurrentHour('2026-07-09T16:00:00'), true, 'wall clock 16:05 -> the 16:00 cell IS the current hour');
    eq(api.cellIsCurrentHour('2026-07-09T15:00:00'), false, 'the 15:00 cell no longer is');
    eq(api.cellIsCurrentHour('2026-07-09T17:00:00'), false, 'the 17:00 cell not yet');
    eq(api.cellIsCurrentHour(''), false, 'no datetime -> not the current hour');
});

// 7d. CURRENT hour, dry at Budva but a WEAK cell nearby (disc): the cloud nudge must
// not keep the model's rain code — it is not raining at Budva at this moment.
const seriesN = [];
for (let lead = 5; lead <= 80; lead += 5) {
    seriesN.push({ lead_min: lead, point_mmh: 0.0, disc_max_mmh: 3.0 });
}
const nsN = {
    ok: true, base_epoch_ms: BASE, horizon_min: 80, timestep_min: 5,
    now: { point_mmh: 0.0, disc_max_mmh: 3.0 }, series: seriesN,
    hourly_mm: [
        { hour: '2026-07-09T13:00:00Z', mm: 0.0, covered_min: 20, peak_point_mmh: 0.0, peak_disc_mmh: 3.0 },
    ],
};
at(BASE + 10 * 60000, () => {
    api.setState(nsN, null);
    const b = api.nowcastHour(nsN, '2026-07-09T15:00:00');
    eq(api.gateWeatherCode(61, 80, b, true, null), 3, 'current hour: weak nearby cell strips model rain, keeps the cloud nudge');
    eq(api.gateWeatherCode(61, 20, b, true, null), 2, 'stripped code still gets the one-step nudge (cloud 20 -> 1 -> 2)');
    const bFut = { hour: '2026-07-09T14:00:00Z', mm: 0.5, covered_min: 60, peak_point_mmh: 0.0, peak_disc_mmh: 3.0 };
    eq(api.gateWeatherCode(61, 80, bFut, false, null), 61, 'future hour: weak nearby cell keeps the model rain code');
});

// 8. Stale nowcast (base older than 90 min) -> no nowcast gating at all.
at(BASE + 120 * 60000, () => {
    api.setState(ns, null);
    eq(api.nowcastFresh(ns), false, 'base 120 min old is stale');
    const b = api.nowcastHour(ns, '2026-07-09T16:00:00');
    eq(api.gateWeatherCode(0, 10, b, false, null), 0, 'stale nowcast does not upgrade the icon');
    eq(api.nowcastCellMm(b, false), null, 'stale nowcast does not override mm');
});

// 9. Winter (CET = UTC+1). Everything above is CEST (UTC+2), so a join that just added
// a fixed +2 would pass it — this is the case that pins the epoch join as DST-safe.
const W_BASE = Date.parse('2026-01-15T13:40:00Z');
const nsW = {
    ok: true, base_epoch_ms: W_BASE, horizon_min: 80, timestep_min: 5,
    now: { point_mmh: 0.0, disc_max_mmh: 0.0 }, series: [],
    hourly_mm: [
        { hour: '2026-01-15T13:00:00Z', mm: 0.0, covered_min: 20, peak_point_mmh: 0.0, peak_disc_mmh: 0.0 },
        { hour: '2026-01-15T14:00:00Z', mm: 4.0, covered_min: 60, peak_point_mmh: 5.0, peak_disc_mmh: 6.0 },
    ],
};
at(W_BASE + 5 * 60000, () => {
    api.setState(nsW, null);
    eq(new Date('2026-01-15T14:00:00Z').getHours(), 15, 'TZ check: 14:00Z is 15:00 local (CET, +1)');

    const w15 = api.nowcastHour(nsW, '2026-01-15T15:00:00');
    eq(w15 && w15.hour, '2026-01-15T14:00:00Z', 'winter: 15:00 local cell -> 14:00Z bucket');
    eq(w15 && w15.mm, 4.0, 'winter: 15:00 cell mm = 4.0');
    eq(api.nowcastCellMm(w15, false), 4.0, 'winter: 15:00 cell mm override = 4.0');
    eq(api.gateWeatherCode(0, 5, w15, false, null), 63, 'winter: 15:00 cell gets the rain icon (5 mm/h -> 63)');

    // With a +2 offset the 14:00Z bucket would land here instead.
    eq(api.nowcastHour(nsW, '2026-01-15T16:00:00'), null, 'winter: 16:00 local cell has no bucket');
    eq(api.gateWeatherCode(0, 5, api.nowcastHour(nsW, '2026-01-15T16:00:00'), false, null), 0,
       'winter: 16:00 cell keeps the model code');

    const w14 = api.nowcastHour(nsW, '2026-01-15T14:00:00');
    eq(w14 && w14.hour, '2026-01-15T13:00:00Z', 'winter: 14:00 local cell -> 13:00Z bucket');
    eq(w14 && w14.mm, 0.0, 'winter: 14:00 cell mm = 0.0');
});

// 9. Rain-cell text: rain shown below POSSIBLE_RAIN_POP is "possible" (dimmed icon,
// "do X mm" = what falls if it rains, "moguća NN%"); at or above it, "X mm" + "NN%".
eq(api.POSSIBLE_RAIN_POP, 0.6, 'possible-rain PoP threshold is 60%');
let rp = api.rainCellParts(1.84, 0.47);
eq(rp.rain, 'do 1.8mm', 'possible rain: amount reads "do 1.8mm"');
eq(rp.pop, 'moguća 47%', 'possible rain: PoP line reads "moguća 47%"');
eq(rp.possible, true, 'possible rain: flagged for the dimmed icon');
rp = api.rainCellParts(4.1, 0.6);
eq(rp.rain, '4.1mm', 'likely rain (PoP = threshold): plain amount');
eq(rp.pop, '60%', 'likely rain: PoP shown without "moguća"');
eq(rp.possible, false, 'likely rain: icon not dimmed');
rp = api.rainCellParts(1.2, 0.598);
eq(rp.pop + '|' + rp.possible, '60%|false', 'PoP 59.8% shows as 60%, so it is not labelled "moguća"');
eq(api.rainCellParts(1.2, 0.594).pop, 'moguća 59%', 'PoP 59.4% shows as "moguća 59%"');
rp = api.rainCellParts(0.0, 0.25);
eq(rp.rain + rp.pop, '', 'dry hour: no rain or PoP text even with a PoP');
eq(api.rainCellParts(0.04, 0.9).rain, '', 'amounts <= 0.05 mm are not shown');
rp = api.rainCellParts(2.2, null);
eq(rp.rain, '2.2mm', 'JSON without PoP: old amount text');
eq(rp.pop + String(rp.possible), 'false', 'JSON without PoP: no PoP line, not dimmed');
eq(api.rainCellParts(NaN, 0.5).rain, '', 'missing amount: nothing shown');

if (failures) { console.error(`\n${failures} test(s) failed`); process.exit(1); }
console.log('\nall tests passed');
