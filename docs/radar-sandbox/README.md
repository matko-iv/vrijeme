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
| Storm | Single cell, multicell or supercell (right-mover with hook echo). |
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

Keys: `Space` play/pause, `→` step, `←` back. The timeline keeps the last
72 frames. Editing an earlier frame rewrites the future.

## How the weather works (`js/sim.js`)

The engine runs on a ~200-cell grid over the visible map.

- **Pressure and wind.** A background gradient comes from the steering wind,
  plus Gaussian lows and highs with a gradient-wind correction. Surface
  wind is turned toward low pressure by friction (more over land), blocked
  by slopes, accelerated downslope (bura) and modulated by a diurnal sea
  breeze.
- **Temperature and moisture.** Both are advected semi-Lagrangian and
  relaxed toward an equilibrium. That equilibrium comes from the air mass
  (T850), the lapse rate over the terrain, sea temperature and the sun:
  solar zenith angle for the real date and place, damped by cloud.
- **Stratiform rain.** Lift from convergence, upslope flow, warm advection
  and lows condenses moisture into a cloud-water reservoir. It rains out
  over ~15 min and drifts with the steering wind. It needs a deep moist
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
- **Radar** (`js/radar.js`) renders at ~1 km. Stratiform texture is stretched
  along the flow. Storm cells have irregular shapes, internal cores and hooks.
  In single-site mode the beam follows 4/3-earth geometry with a terrain
  horizon per azimuth, a range-dependent detection threshold and C-band
  path-integrated attenuation.

## Data

- Terrain: [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/)
  (SRTM, GMTED; terrarium encoding).
- Borders and lakes: [Natural Earth](https://www.naturalearthdata.com/) 1:10m,
  clipped to Europe in `data/`.
- Radar site and colour scale: DHMZ Uljenje (`budva-radar` config).
