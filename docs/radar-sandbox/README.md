# Radar Sandbox

A weather sandbox, not a forecast. Paint rain, drop thunderstorms, place
lows, highs and fronts on real terrain, then step time and watch radar,
satellite and surface maps evolve with sensible 5–60 minute steps.

Open `index.html` through any static server (GitHub Pages serves it from
`docs/radar-sandbox/`). There is no build step. ES modules only. Terrain
tiles load from AWS Terrain Tiles at runtime.

## Regions

Europe, SE Europe, Adriatic, Montenegro and **Radar Uljenje**. The
Uljenje view copies the DHMZ product: 248 km range ring, E-W and N-S
max-reflectivity side views, and the DHMZ colour scale. Scroll the mouse
wheel to zoom anywhere; that makes a *Custom* region. The weather carries
over between regions.

## Map modes

Radar (composite reflectivity), rain rate, accumulation, 2 m temperature,
dew point, humidity, wind and gusts (with flow particles), pressure
(isobars, L/H), satellite (visible by day, IR or enhanced IR), CAPE and
echo tops.

## Tools

| Tool | What it does |
|---|---|
| Drag | Move storms and L/H centres. Drag empty space to push rain, clouds and air masses. |
| Rain | Paints lift and moisture. The rain area drifts with the steering wind and fades once its forcing decays (~3 h). |
| Storm | Click drops one storm; hold and drag to paint storms across the brush, spaced like real cells (~9 km single, ~12 km multicell, ~30 km supercell). Choose single, multicell or supercell (right-mover with hook echo). Storms you place live their full life cycle even in stable air, and placed multicells keep regenerating for ~3 h. |
| Squall | A line of storms that propagates forward and rebuilds itself. |
| Low / High | Press at the centre, drag outward for size. Strength sets the depth. |
| Cold / warm front | Temperature step plus a lift band. A cold front fires a squall line when there is CAPE. |
| Moisture / Heat | Paint humid or dry, warm or cold air. |
| Erase | Wipes everything under the brush. |
| Radar site | Puts a radar anywhere, with its ring, beam overshoot, terrain blocking, clutter and C-band attenuation. |

Atmosphere controls: the steering wind (drag the compass arrow), 0–6 km
shear, T850, T500, humidity, sea temperature, sun heating, storm trigger and
sea breeze. Six scenarios start you off: Adriatic autumn showers, summer
heat storms, a Genoa low with jugo, bura, a supercell day and a blank
canvas.

**Auto weather** (on by default) does two gentle things while you play:

- **Wind:** the steering wind turns toward a new direction every 5–11
  simulated hours, by at most ~12° per hour, and its speed drifts slowly.
- **Your rain:** whatever you paint with the Rain tool keeps evolving.
  Each painted area becomes a parent rain pocket. Every mature pocket buds
  children at its edges, on average ~1.3 per generation, slowly weakening.
  So your rain spreads, splits and drifts as a cluster for many hours and
  then dies out. In a test, one stroke grew for ~7 h and was gone after
  ~14 h.

It never creates rain in dry areas, nor lows, highs, fronts or air-mass
changes. The **Natural rain** slider decides whether the sandbox also
makes its own rain (terrain, convergence, spontaneous storms). The blank
canvas uses 0, "only mine". The scenarios use 100%. Pocket shapes and
lifecycles are modelled on 4,266 rain objects (≥15 dBZ) tracked in the
budva-radar Uljenje tiles:

| | Real | Pockets (blank canvas) |
|---|---|---|
| Area, median / 90% | 41 / 465 km² | 66 / 323 km² |
| Length/width, median | 1.8 | 1.6 |
| Growth / decay to peak | 15 / 20 min | 20 / 15 min |
| Peak dBZ, median / 90% | 27 / 43 | 24 / 35 |
| Drift speed | 18 km/h | 15 km/h (follows the wind) |

Lifetime rises with size (√area, 30 min to 6 h). This dynamic scaling of
precipitation lifetime is the one known from radar nowcasting research
(Germann & Zawadzki). Pockets only add lift and moisture, so the rain
itself grows and fades through the normal rain physics.
**Flow variety** scales the free atmosphere described below.

Keys: `Space` play/pause, `→` step, `←` back. The timeline keeps the last
72 frames. Editing an earlier frame rewrites the future.

## How the weather works (`js/sim.js`)

The engine runs on a ~200-cell grid over the visible map.

- **Pressure and wind.** A background gradient comes from the steering wind,
  plus Gaussian lows and highs with a gradient-wind correction. Surface
  wind is turned toward low pressure by friction (more over land), blocked
  by slopes, accelerated downslope (bura) and modulated by a diurnal sea
  breeze.
- **Free atmosphere.** Even with no lows or highs placed, one broad, slowly
  morphing long-wave trough/ridge pattern (~1200 km) bends the wind gently
  across the map. It adds matching pressure and ascent in its troughs.
  Mesoscale organisation (~60 km) scales condensation inside areas that are
  already rising. It gives bands and gaps without creating rain out of
  nothing.
- **Temperature and moisture.** Both are advected semi-Lagrangian and
  relaxed toward an equilibrium. That equilibrium comes from the air mass
  (T850), the lapse rate over the terrain, sea temperature and the sun:
  solar zenith angle for the real date and place, damped by cloud.
- **Stratiform rain.** Lift from convergence, upslope flow, warm advection
  and lows condenses moisture into a cloud-water reservoir. It rains out
  over ~25 min and drifts with the steering wind. It needs a deep moist
  layer (surface RH plus a mid-level humidity field), so dry summer days
  stay convective.
- **Convection.** CAPE comes from parcel theory: the theta-e of a lightly
  mixed surface parcel against the 500 hPa temperature. Storms are
  Lagrangian cells with growth, mature and decay stages. They start where
  CAPE meets a trigger: lift, daytime heating of high ground, gust fronts,
  or cold air over a warm sea. Each storm rains, builds a cold pool and a
  gust front, and spawns daughters on its right-forward flank (or upwind
  when backbuilding against terrain). Supercells move right of the mean
  wind. Lightning rate rises with reflectivity and echo top.
- **Radar** (`js/radar.js`) renders at ~1 km. Stratiform texture rides on
  two flow-following coordinate layers, so it moves and deforms with the
  local wind. The layers are advected and reset alternately every 3 h. The
  texture adds bands, holes and embedded heavier cores. Storm cells have irregular shapes, internal cores and hooks.
  In single-site mode the beam follows 4/3-earth geometry with a terrain
  horizon per azimuth, a range-dependent detection threshold and C-band
  path-integrated attenuation.

## Data

- Terrain: [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/)
  (SRTM, GMTED; terrarium encoding).
- Borders and lakes: [Natural Earth](https://www.naturalearthdata.com/) 1:10m,
  clipped to Europe in `data/`.
- Radar site and colour scale: DHMZ Uljenje (`budva-radar` config).

## Calibration against real radar

`budva-radar` holds real Uljenje data: ODIM volumes (24 Jun, 6 and 8 Jul
2026) and 1,692 five-minute 1 km rain tiles around Budva (23–28 May 2023).
The sandbox was tuned to match their statistics at Budva, with the same 1 km
grid and clutter off:

| | Real | Sandbox (autumn / Genoa / summer) |
|---|---|---|
| Wet area ≥10 dBZ, 256 km tile | 18% rainy day, 0.5–6% shower days | 21% / 47% / 10% |
| Share of wet pixels at 10–15 / 15–20 / 20–25 / 25–30 / 30–35 dBZ | .29 / .23 / .22 / .14 / .07 | .29 / .27 / .22 / .13 / .05 |
| Spectral slope of the dBZ field | −2.6 … −3.0 | −2.8 … −3.3 |
| Overlap of ≥15 dBZ area after 5 / 15 / 30 min | .79 / .61 / .48 | .73 / .48–.59 / .35–.52 |

## Real-data textures (`data/rain_atlas.*`, `data/cloud_atlas.*`)

Rain and clouds are not drawn from noise. They are cut from real imagery:

- **Rain:** 36 rain scenes from the budva-radar Uljenje 1 km tiles, rotated
  and blended into a seamless 960 km pattern. The simulation's areal rain
  rate decides how much of an area is wet: drizzle-level rain gives a few
  sparse, weak pockets; 1 mm/h gives ~40%; heavy rain gives a near-solid
  shield. Each pixel's dBZ comes from its place in the real pattern and the
  real wet-pixel dBZ distribution. So light rain is light, cores sit inside
  shields, and edges are ragged like real echoes. The pattern rides on
  flow-following coordinates, so it moves and slowly deforms with the wind.
- **Large-scale rain organisation:** frontal bands and rain systems decoded
  from the EUMETNET OPERA European composite (12 Oct 2026, FMI CDN),
  4.5 km/px. This layer modulates rain intensity over hundreds of km, so
  rain areas organise into bands and lumps. The Uljenje pattern supplies
  the 1 km detail inside them.
- **Clouds:** six real scenes from EUMETSAT MTG-I1 FCI, fetched through
  EUMETView WMS: Nov 2025 cirrus, a Mar 2026 cold-front band, May and Jul
  2026 convection, a Sep 2026 cyclone and an Oct 2026 comma cloud. For
  each scene the Meteosat cloud mask removes the clear land and sea
  (coastlines, snow). The rest is stored as a 3-channel atlas:
  - visible 0.6 µm reflectance: cloud brightness and thickness;
  - 10.5 µm IR coldness: cloud-top height;
  - a combined field that the simulation's cloud fraction thresholds.

  The six scenes are cross-faded into a 3×2 mosaic and mirror-wrapped.
- **Satellite modes:** **Visible is always daylit.** At any hour it shows
  the real reflectance texture: thin cloud grey and translucent, thick
  cloud bright. Infrared and enhanced IR use the cloud-top channel. Anvils
  are oval downwind canopies with a sharp upwind edge.

Measured with the same tracker as the real data (Budva 256 km tile, 1 km,
clutter off):

| | Real | Sandbox autumn / Genoa |
|---|---|---|
| Wet ≥10 dBZ | 6–18% | 13% / 22–26% |
| 10–15 / 15–20 / 20–25 / 25–30 dBZ share | .29 / .23 / .22 / .14 | .27–.31 / .24–.32 / .16–.19 / .09–.13 |
| ≥15 dBZ overlap after 5 / 15 / 30 min (at the real data's 18 km/h drift) | .79 / .61 / .48 | .76 / .50 / .32 |
