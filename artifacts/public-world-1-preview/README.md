# Public World 1: organic cities preview

Open `public-world-1-preview.html` directly in a browser. The five views have pan and magnification controls, plus links to the original 3200 × 2000 PNG captures.

The map was read from the public production endpoints on 15 September 2026. `snapshot.json` records the precise fetch timestamp, season, tick, and camera positions. `tiles.json` and `rivers.json` retain that public snapshot: 22,500 tiles and 108 cities. No private nation data or authenticated endpoints were used.

These are captures of the revised local `rpn-web` Pixi renderer. The full-world view allows the camera to fit the entire world rather than using the game's crop-to-fill zoom limit; an assertion verifies that every live tile is included. The other views use regular map zoom levels. Icons, labels, borders, and weather effects are switched off to inspect the artwork.

City positions, city names, building counts, inter-hex roads, railways, terrain, lakes, and rivers come from the world snapshot. Individual houses, local lanes, gardens and trees are deterministic visual representations, not actual individual structures from the database. Traffic is the renderer's visual simulation. The city artwork is procedural, with no generated AI images or photographic city cutouts. NASA imagery supplies the terrain; it does not supply roof-level detail.

The revision uses irregular street blocks, hipped roofs with material grain, eaves, visible walls, chimneys, gardens, denser foliage, and soft shadows. Architectural surfaces retain the same world size and placement across zoom levels. Detail textures reach 64 samples per world unit for 24× retina viewing; offscreen surfaces are culled and the inactive cache is bounded by texture memory. Rail corridors are excluded from roof placement. Power lines use architectural dimensions in the organic view, while resource badges stay small and respect the icons toggle.

This remains procedural game art and does not reproduce the photographic appearance of the earlier AI concept exactly. The preview is a dated snapshot, not an automatically refreshing live map. Changes are local; nothing was deployed.

Reproduce while the local game Vite server runs at port 5175:

```powershell
node artifacts/public-world-1-preview/render.cjs
python artifacts/public-world-1-preview/build-preview.py
```
