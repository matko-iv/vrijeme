# Organic map imagery

Imagery: **NASA Earth Observatory, Blue Marble Next Generation**, served by NASA GIBS.
The unmodified downloaded crops and their exact request URLs are in `sources/`.
`atlas.json` records the crop and atlas slot for each terrain. Water uses procedural
surface colours; volcanic terrain uses mountain imagery and procedural crater relief.
No AI-generated imagery is used by this renderer.

Rebuild from the included original crops (Python + Pillow):

```sh
python scripts/prepare-organic-terrain.py
```

The game samples these textures continuously in world coordinates, blends neighbouring
terrain classes, and applies lakes and seasonal snow from actual world data. Imagery
depicts material and terrain character, not the geographic coordinates of the fictional world.

Cities use deterministic geometry with cached Canvas architectural surfaces, sampled
at 8, 24 or 64 pixels per world unit. Hipped roofs, material grain, eaves, chimneys,
gardens and tree crowns appear at closer zoom. Installations and vehicles use vector
geometry. Layouts and architectural surfaces keep
the same world dimensions at every zoom. The shared scale is eight display metres
per map unit: a 7 m two-lane road, a 4.6 × 1.85 m sedan, and 6.5–10 m house widths.
This architectural scale does not change strategic travel distances or game rules.

The Organic layer uses WebGL, selected explicitly by the map component. If the atlas
cannot load, flat ground and procedural settlements remain usable. No legacy AI settlement
images are loaded as a fallback. Layout caches are bounded across visited areas and
reused on unchanged data refreshes; roads, traffic, ownership overlays and input remain
separate layers.

Sources:

- [NASA Blue Marble Next Generation](https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-map/)
- [NASA GIBS](https://www.earthdata.nasa.gov/data/tools/gibs)

Local validation (with the development server running):

```sh
node scripts/test-organic-map.cjs http://127.0.0.1:5175 docs/previews/organic
node scripts/test-map-regressions.cjs http://127.0.0.1:5175
node scripts/test-map-traffic.cjs http://127.0.0.1:5175
```
