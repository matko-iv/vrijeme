<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { MapRenderer, type MapMode, type TileData, type BuildMode, type BuildPathCallback } from '$lib/game/map-renderer';
  import { mapApi, economyApi, borderApi, warApi, militaryApi } from '$lib/api';
  import { nation } from '$lib/stores/nation';
  import { adjacencyPreview } from '$lib/stores/map';
  import { appearance } from '$lib/stores/settings';
  import { createRefreshQueue } from '$lib/game/refresh-queue';
  import type { Season } from '$lib/game/weather-layer';

  const SEASONS = ['spring', 'summer', 'autumn', 'winter'] as const;
  function isSeason(value: unknown): value is Season {
    return typeof value === 'string' && (SEASONS as readonly string[]).includes(value);
  }

  export let mode: MapMode = 'political';
  export let onHexSelect: (q: number, r: number, tile: TileData | null) => void = () => {};
  export let onZoomChange: (zoom: number) => void = () => {};
  export let onNotPlaced: () => void = () => {};
  export let onBuildPath: BuildPathCallback = () => {};

  let canvasDiv: HTMLDivElement;
  let renderer: MapRenderer | null = null;
  let loading = true;
  let error = '';
  let initialLoad = true;
  let pendingWorldInfo: [number, number, number, number] | null = null;
  let weatherSound = true;
  const overlayVisibility = new Map<Parameters<MapRenderer['setOverlayVisibility']>[0], boolean>();
  let pendingBuildMode: BuildMode = 'pan';
  let riversLoaded = false;
  let riverWorldKey = '';
  let riversInFlight: Promise<void> | null = null;
  let seasonRequest = 0;
  let warRequest = 0;
  let borderRequest = 0;
  let powerRequest = 0;
  let pixiApp: import('pixi.js').Application | null = null;
  let componentDestroyed = false;
  let resizeObserver: ResizeObserver | null = null;

  $: renderer?.setAdjacencyPreview($adjacencyPreview);

  // Canvas labels are rasterised with a resolved font string, not a CSS var,
  // so a typeface change has to reach the renderer explicitly. The bundled
  // faces load lazily: wait for the fetch before re-cutting, or Pixi measures
  // the fallback and the labels come out the wrong width.
  $: applyFont($appearance.font);

  function applyFont(_font: string) {
    if (!renderer) return;
    const r = renderer;
    const done = typeof document !== 'undefined' && document.fonts
      ? document.fonts.ready
      : Promise.resolve();
    void done.then(() => { if (renderer === r) r.refreshFont(); });
  }

  onMount(async () => {
    try {
      const { Application } = await import('pixi.js');
      if (componentDestroyed) return;
      const app = new Application();
      await app.init({
        resizeTo: canvasDiv,
        backgroundColor: 0x0a0a1a,
        antialias: true,
        preserveDrawingBuffer: true,
        // Pixi defaults resolution to 1, so on any HiDPI screen the whole
        // canvas was drawn at half or a third of the device's real pixels
        // and stretched back up: every label and icon soft, the hex art
        // included. Capped at 2 because a 3x display would triple the
        // fragment work for a difference nobody can see, and this canvas
        // fills the window.
        resolution: Math.min(window.devicePixelRatio || 1, 2),
        autoDensity: true,
      });
      if (componentDestroyed) {
        app.destroy(true, true);
        return;
      }
      pixiApp = app;
      canvasDiv.appendChild(app.canvas);

      renderer = new MapRenderer(app);
      renderer.setOnZoomChange((zoom) => onZoomChange(zoom));
      renderer.setWeatherSound(weatherSound);
      renderer.setBuildMode(pendingBuildMode);
      for (const [key, value] of overlayVisibility) renderer.setOverlayVisibility(key, value);
      if (pendingWorldInfo) renderer.setWorldInfo(...pendingWorldInfo);
      applyFont($appearance.font);
      // resizeTo only reacts to window resizes; layout shifts (the
      // toolbar wrapping when a mode adds tools) resize the container
      // without one, leaving hit tests a hex off until this fires.
      resizeObserver = new ResizeObserver(() => {
        app.resize();
        renderer?.handleResize();
      });
      resizeObserver.observe(canvasDiv);
      renderer.setOnHexSelect((q, r, tile) => {
        onHexSelect(q, r, tile);
      });
      renderer.setOnBuildPath((path, buildingKey) => {
        onBuildPath(path, buildingKey);
      });

      await queueReload(false);
    } catch (e: any) {
      error = e.message ?? 'Failed to load map';
    }
    loading = false;
  });

  let legacyChecked = false;

  /** The sky follows the world's season, so a glance at the map says
   *  which one it is. A failure here costs the weather and nothing
   *  else, so it is caught and dropped. */
  async function syncSeason() {
    const target = renderer;
    if (!target) return;
    const request = ++seasonRequest;
    // One key, read by both the toolbar and the renderer, so a player who
    // turned the weather off last week does not get a frame of rain.
    try {
      if (localStorage.getItem('rpn_map_weather') === '0') {
        target.setWeatherEnabled(false);
      }
    } catch { /* private mode */ }
    try {
      const info = await mapApi.tickInfo();
      // The server is the authority on the season, but it is still a
      // string off the wire, so a value the sky has no artwork for is
      // dropped rather than passed through.
      if (renderer !== target || request !== seasonRequest) return;
      if (isSeason(info?.season)) target.setSeason(info.season);
      if (pendingWorldInfo && Number.isFinite(info?.tick_count)) {
        const [seed, width, height] = pendingWorldInfo;
        setWorldInfo(seed, width, height, info.tick_count);
      }
    } catch {
      // The map is more important than the clouds over it.
    }
  }

  // Centring is a first-load courtesy, not something a refresh may do.
  // Somebody watching a war on the far side of the map should stay there
  // when a stranger lays a road.
  // The shared map is redacted for everyone, so a nation's own buildings
  // and stats arrive separately and are laid over it. loadTiles keys by
  // hex, so these overwrite the redacted rows for the same hexes.
  // Signed out, or before a nation exists, there is nothing to overlay.
  async function myTiles(): Promise<any[]> {
    if (!$nation?.userid) return [];
    try {
      const mine = await mapApi.myTiles(true);
      return mine.tiles ?? [];
    } catch {
      // A redacted map still draws; it just shows none of your own detail.
      return [];
    }
  }

  async function loadMapData(fresh = false, recenter = false) {
    const target = renderer;
    if (!target) return;
    const [data, mine] = await Promise.all([
      mapApi.tiles({ includeStats: true, fresh }), myTiles(),
    ]);
    if (renderer !== target) return;
    if (data.tiles && data.tiles.length > 0) {
      error = '';
      if (data.meta) {
        const { seed, width, height } = data.meta;
        if ([seed, width, height].every(Number.isFinite)) {
          const key = `${seed}:${width}:${height}`;
          if (key !== riverWorldKey) {
            riverWorldKey = key;
            riversLoaded = false;
            riversInFlight = null;
          }
          setWorldInfo(seed, width, height, pendingWorldInfo?.[3] ?? 0);
        }
      }
      target.loadTiles(
        [...data.tiles, ...mine],
        data.nations, data.edges, data.hv_hexes, data.border_postures, data.customs_hexes);
      target.setMode(mode);
      void syncSeason();
      void loadRivers();
      if (mode === 'electricity') void loadPowerGrid();
      if (mode === 'borders') void loadBorderOverlay();

      const nationData = $nation;
      if (nationData?.userid) {
        const found = renderer.hasNation(nationData.userid);
        if (recenter && found) renderer.centerOnNation(nationData.userid);
        if (!found) {
          onNotPlaced();
        } else if (!legacyChecked) {
          // Auto-place legacy buildings on first load
          legacyChecked = true;
          try {
            const result = await mapApi.autoPlaceLegacy();
            if (result.placed > 0) {
              console.log(`Auto-placed ${result.placed} legacy buildings (${result.skipped} skipped)`);
              // Reload to show newly placed buildings
              const [refreshed, ownTiles] = await Promise.all([
                mapApi.tiles({ includeStats: true, fresh: true }), myTiles(),
              ]);
              if (renderer !== target) return;
              target.loadTiles(
                [...refreshed.tiles, ...ownTiles],
                refreshed.nations, refreshed.edges, refreshed.hv_hexes, refreshed.border_postures, refreshed.customs_hexes);
              target.setMode(mode);
            }
          } catch (e) {
            // Not critical: legacy placement is best-effort
            console.warn('Legacy auto-placement failed:', e);
          }
        }
      }
    } else {
      error = 'No map data. World needs to be generated.';
    }
  }

  $: if (renderer) {
    renderer.setMode(mode);
  }

  // Fetch an overlay each time it is opened, including after a failed request.
  $: if (renderer) {
    if (mode === 'electricity') void loadPowerGrid();
    if (mode === 'borders') void loadBorderOverlay();
    if (mode === 'military') void loadWarOverlay();
  }

  async function loadWarOverlay() {
    const target = renderer;
    if (!target) return;
    const request = ++warRequest;
    try {
      const [orders, deployments] = await Promise.all([
        warApi.orders(),
        militaryApi.v2Deployments(),
      ]);
      if (renderer === target && request === warRequest) target.setWarOverlay(orders.orders ?? [], deployments.map ?? []);
    } catch {
      // No session or no military yet: the overlay simply stays empty
    }
  }

  export function refreshWarOverlay(): Promise<void> {
    return loadWarOverlay();
  }

  // Rivers are terrain, not an overlay, so they load once with the map and
  // stay drawn in every mode.
  function loadRivers(): Promise<void> {
    const target = renderer;
    if (!target || riversLoaded) return Promise.resolve();
    if (riversInFlight) return riversInFlight;
    const worldKey = riverWorldKey;
    riversInFlight = mapApi.rivers().then((data) => {
      if (renderer !== target || worldKey !== riverWorldKey) return;
      target.setRivers(data.edges, data.lakes, data.major_flow);
      riversLoaded = true;
    }).catch((e) => {
      console.warn('Rivers failed to load:', e);
    }).finally(() => { if (worldKey === riverWorldKey) riversInFlight = null; });
    return riversInFlight;
  }

  async function loadBorderOverlay() {
    const target = renderer;
    if (!target) return;
    const request = ++borderRequest;
    try {
      const data = await borderApi.overlay();
      if (renderer === target && request === borderRequest) target.setBorderOverlay(data.hexes);
    } catch (e) {
      console.warn('Border overlay failed:', e);
    }
  }

  async function loadPowerGrid() {
    const target = renderer;
    if (!target) return;
    const request = ++powerRequest;
    try {
      const grid = await economyApi.powerGrid();
      if (renderer === target && request === powerRequest) target.setPowerGrid(grid.hexes ?? []);
    } catch {
      // Not placed yet or no grid: static nameplate labels still render
    }
  }

  export function zoomIn() {
    if (renderer) {
      renderer.setZoom(renderer.zoom * 1.3);
      onZoomChange(renderer.zoom);
    }
  }

  export function zoomOut() {
    if (renderer) {
      renderer.setZoom(renderer.zoom * 0.77);
      onZoomChange(renderer.zoom);
    }
  }

  export function getZoom(): number {
    return renderer?.zoom ?? 0.4;
  }

  /** Hand the sky the world it is over, so it is the same for everyone. */
  export function setWorldInfo(seed: number, width: number, height: number, tick = 0) {
    pendingWorldInfo = [seed, width, height, tick];
    renderer?.setWorldInfo(seed, width, height, tick);
  }

  /** Centre on a hex and open its panel. Used by deep links. */
  export function focusHex(q: number, r: number): boolean {
    return renderer ? renderer.focusHex(q, r) : false;
  }

  export function centerOnMyNation(): boolean {
    const nationData = $nation;
    if (renderer && nationData?.userid) {
      return renderer.centerOnNation(nationData.userid);
    }
    return false;
  }

  const queueReload = createRefreshQueue(async (fresh) => {
    try {
      await loadMapData(fresh, initialLoad);
      initialLoad = false;
    } catch (e: any) {
      if (!componentDestroyed) error = e.message ?? 'Could not load the map.';
    }
  }, () => !componentDestroyed && renderer !== null);

  /** Refresh after a local edit, bypassing the shared server cache. */
  export function reloadTiles(): Promise<void> {
    return queueReload(true);
  }

  /** Coalesce other players' edits while using the shared server cache. */
  export function reloadTilesShared(): Promise<void> {
    return queueReload(false);
  }

  export function setBuildMode(mode: BuildMode) {
    pendingBuildMode = mode;
    renderer?.setBuildMode(mode);
  }

  export async function exportImage() {
    await renderer?.exportImage();
  }

  export function setWeatherSound(enabled: boolean) {
    weatherSound = enabled;
    renderer?.setWeatherSound(enabled);
  }

  export function setOverlayVisibility(key: 'icons' | 'roads' | 'traffic' | 'power' | 'borders' | 'labels' | 'textures' | 'weather', value: boolean) {
    overlayVisibility.set(key, value);
    renderer?.setOverlayVisibility(key, value);
  }

  onDestroy(() => {
    componentDestroyed = true;
    resizeObserver?.disconnect();
    resizeObserver = null;
    renderer?.destroy();
    renderer = null;
    pixiApp?.destroy(true, true);
    pixiApp = null;
  });
</script>

<div class="hex-map" bind:this={canvasDiv}>
  {#if loading}
    <div class="map-loading">
      <p>Loading world map...</p>
    </div>
  {/if}
  {#if error}
    <div class="map-error">
      <p>{error}</p>
    </div>
  {/if}
</div>

<style>
  .hex-map {
    flex: 1;
    overflow: hidden;
    position: relative;
  }
  .hex-map :global(canvas) {
    display: block;
    width: 100% !important;
    height: 100% !important;
  }
  .map-loading, .map-error {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
  }
  .map-loading p { color: var(--color-text-muted); font-size: calc(0.9rem * var(--font-scale)); }
  .map-error p { color: var(--color-danger); font-size: calc(0.85rem * var(--font-scale)); }
</style>
