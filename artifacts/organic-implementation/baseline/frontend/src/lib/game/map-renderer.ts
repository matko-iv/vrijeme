/**
 * Core Pixi.js hex map renderer.
 *
 * Draws a 200x200 hex grid with terrain coloring, pan/zoom camera,
 * hex hover/selection, city markers, building icons, road/rail lines,
 * and multiple map modes.
 */
import { Container, Graphics, Text, Sprite, Assets, type Application, type TextStyle, type Texture } from 'pixi.js';
import { WeatherLayer, type Season } from './weather-layer';
import { GameAudio, type Shot } from './game-audio';
import { axialToPixel, pixelToAxial, hexCorners, hexKey, axialNeighbors, HEX_SIZE } from './hex-math';
import {
  TERRAIN_TYPES, RESOURCE_COLORS, richnessAlpha,
  resourceDotShape, type ResourceDotShape,
} from './terrain-colors';
import { getBuildingIcon, getUnitIcon } from './icons';
import { preloadTerrainTextures, getTileTexture, getSettlementTexture, getInstallationTexture, areTerrainTexturesLoaded, installationTypeFor, installationFrameMs, installationFrames, getTextureByUrl } from './tile-textures';
import { rainAt, soilMoisture } from './rainfall';
import { TransportNetwork, samplePath, type TransportPath } from './transport';
import { TrafficLayer } from './traffic-layer';
import { CheckpointLayer } from './checkpoint-layer';
import { findCrossings, type Crossing, type BorderPosture } from './border-crossings';
// Accessibility settings and the shared deposit-marker lookups. The map
// legend reads the same two tables, so codes and palettes can't drift.
import {
  a11y, RESOURCE_CODES, resourceMarkerColors,
  type A11ySettings, type ColorblindMode,
} from '../stores/settings';

/** What the renderer actually consumes from the a11y store. */
type A11yView = { colorblind: ColorblindMode; resourceLabels: boolean };
const A11Y_FALLBACK: A11yView = { colorblind: 'off', resourceLabels: false };

// Power plant output in MW (mirrors backend constants)
const POWER_PLANT_OUTPUT: Record<string, number> = {
  coal_power_p: 1.0,
  oil_power_p: 1.0,
  nuclear_power_p: 75.0,
  wind_power_p: 0.75,
  hydro_power_p: 5.0,
  solar_power_p: 0.5,
  geothermal_power_p: 2.0,
};

// Power line capacity in MW
const POWER_LINE_CAPACITY: Record<string, number> = {
  power_line: 15,
  high_voltage_line: 150,
};

// Buildings whose icons stay visible in electricity mode
const POWER_BUILDINGS = new Set([
  'coal_power_p', 'oil_power_p', 'nuclear_power_p',
  'wind_power_p', 'hydro_power_p', 'solar_power_p',
  'geothermal_power_p', 'power_line', 'high_voltage_line',
]);

// Network buildings drawn as lines, not icons
const LINE_BUILDINGS = new Set(['road_network', 'dirt_road', 'railway', 'power_line', 'high_voltage_line']);

// Axial direction for each flat-top hex edge d (corners d -> d+1).
// Edge midpoints sit at angle d*60+30 degrees, matching these neighbours.
const BORDER_DIRS: [number, number][] = [
  [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
];

// Unit-circle corners of a flat-top hex, so per-tile outlines are 12
// multiply-adds instead of 12 trig calls. At 15k visible tiles that is the
// difference between ~2ms and ~0.1ms per pass.
const HEX_UNIT_CORNERS: number[] = (() => {
  const pts: number[] = [];
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 3) * i;
    pts.push(Math.cos(a), Math.sin(a));
  }
  return pts;
})();

/** Corners of one hex as [x, y, ...]; a fresh array because Pixi keeps the
 *  reference in the path instruction until the frame is tessellated. */
function hexPts(cx: number, cy: number, sz: number): number[] {
  const p = new Array<number>(12);
  for (let i = 0; i < 6; i++) {
    p[i * 2] = cx + HEX_UNIT_CORNERS[i * 2] * sz;
    p[i * 2 + 1] = cy + HEX_UNIT_CORNERS[i * 2 + 1] * sz;
  }
  return p;
}

// Hex outlines stop being drawn past this many visible tiles. Borders are
// hairlines at that density — they read as a grey wash, not as borders — and
// they are what made the 0.35-0.5 zoom band crawl. Tuned by measurement:
// see the frame-time notes in redraw().
const BORDER_MAX_TILES = 9000;

// Resource labels only appear when hexes are big enough to hold four
// characters, and are capped so a wide screen can't spawn unbounded Text.
const RESOURCE_LABEL_MIN_ZOOM = 0.95;
// Country names carry the whole middle of the zoom range, not just the
// far end: you should be able to read who owns what while still seeing
// buildings. They overlap the city names (which start at 0.5) and fade
// out as those take over, rather than switching off at a hard edge.
const NATION_LABEL_MAX_ZOOM = 1.1;
// Below this they are at full strength; above it they thin out.
const NATION_LABEL_FULL_ZOOM = 0.75;
const NATION_LABEL_MIN_HEXES = 3;
const NATION_LABEL_MIN_SCALE = 0.3;
const NATION_LABEL_MAX_SCALE = 2.2;
// The size the label's glyphs are rasterised at before fitting. A label is
// re-rasterised at its fitted size rather than scaled up from this one:
// magnifying a 16px bitmap to 2.2x is what made big nations look soft.
const NATION_LABEL_BASE_FONT = 16;
// Re-rasterising costs a texture upload, so sizes snap to this step
// instead of changing on every frame of a pan.
const NATION_LABEL_FONT_STEP = 2;
const RESOURCE_LABEL_MAX = 700;
// Text is rasterised once at this size and scaled per zoom: changing the
// style would re-rasterise every label on every zoom step.
const RESOURCE_LABEL_BASE_FONT = 16;

// Canvas text can't read a CSS variable, so the map resolves --font-sans
// itself and caches the string. Invalidated by refreshFont() when the player
// picks a different typeface.
const MAP_FONT_FALLBACK = 'system-ui, Segoe UI, Arial, sans-serif';
let mapFontStack: string | null = null;

/** The UI's current sans stack, in the form a Pixi TextStyle wants. */
export function mapFont(): string {
  if (mapFontStack) return mapFontStack;
  if (typeof document === 'undefined') return MAP_FONT_FALLBACK;
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue('--font-sans').trim();
  mapFontStack = v || MAP_FONT_FALLBACK;
  return mapFontStack;
}

/** Drop the cached stack so the next label picks up a new typeface. */
export function clearMapFont(): void {
  mapFontStack = null;
}

// Military mode highlights hexes carrying these (mirrors MILITARY_BUILDINGS)
export const MILITARY_BUILDING_KEYS = new Set([
  'tank_factory', 'plane_factory', 'ship_factory', 'missile_factory',
  'military_logistics_centre', 'tank_storage', 'plane_storage',
  'ship_storage', 'missile_storage', 'espionage_agency',
]);

/** Tundra and arctic. Snow falls here whatever the calendar says. */
const FROZEN_TERRAIN = new Set([5, 12]);

export type MapMode = 'political' | 'military' | 'economic' | 'terrain'
  | 'pollution' | 'crime' | 'disease' | 'fire_coverage' | 'fire_hazard'
  | 'police_coverage' | 'hospital_coverage'
  | 'electricity' | 'water' | 'hydrology'
  | 'happiness' | 'education' | 'safety' | 'health'
  | 'borders';



export interface TileData {
  q: number;
  r: number;
  terrain: number;
  resource: string | null;
  richness: number;
  // Nation ids are Discord snowflakes, too large for a JS number to hold
  // exactly, so they travel as strings end to end.
  owner: string | null;
  // Owner of an Economic Zone. Water is claimed through its own column, so
  // an ocean hex has a seaOwner and never an owner.
  seaOwner?: string | null;
  cityId?: number | null;
  cityName?: string | null;
  isCapital?: boolean;
  buildingCount?: number;
  buildingKeys?: string[];
  // Hex stats (included when include_stats=true)
  pollution?: number;
  crime?: number;
  disease?: number;
  electricity?: boolean;
  roadConnected?: boolean;
  railConnected?: boolean;
  fireCoverage?: number;
  fireHazard?: number;
  waterSupply?: number;
  policeCoverage?: number;
  hospitalCoverage?: number;
  safety?: number;
  health?: number;
  education?: number;
  happiness?: number;
  inhabited?: boolean;
  // Anarchy info
  anarchyUntil?: string | null;
  occupier?: string | null;
}

export interface NationMeta {
  flag?: string | null;
  country: string;
  primary_color: string;
  secondary_color: string;
}

export type HexSelectCallback = (q: number, r: number, tile: TileData | null) => void;
export type BuildPathCallback = (path: { q: number; r: number }[], buildingKey: string) => void;
export type BuildMode =
  | 'pan' | 'road' | 'dirt' | 'rail' | 'power' | 'hv_power'
  | 'move' | 'attack_line' | 'defense_line';

/** Drag modes that draw war orders instead of buildings. */
export const WAR_ORDER_MODES: ReadonlySet<BuildMode> = new Set(
  ['move', 'attack_line', 'defense_line'] as BuildMode[],
);

export interface WarOrder {
  id: number;
  kind: 'move' | 'attack_line' | 'defense_line';
  path: [number, number][];
  progress: number;
}

export interface WarGarrison {
  q: number;
  r: number;
  total: number;
  units?: { key: string; count: number }[];
  in_supply: boolean;
  entrenchment: number;
}

export class MapRenderer {
  private app: Application;
  private worldContainer: Container;
  private hexGfx: Graphics;
  private overlayGfx: Graphics;
  private labelContainer: Container;
  private nationLabelContainer!: Container;
  private powerLabelContainer: Container;

  private tiles: Map<string, TileData> = new Map();
  // Keyed by nation id as a string: snowflake ids exceed JS number precision.
  private nations: Map<string, NationMeta> = new Map();
  private nationColorCache: Map<string, number> = new Map();

  private mode: MapMode = 'political';
  private borderOverlay: Map<string, { tension: number; crossing: string }> = new Map();

  // Camera
  private camX = 0;
  private camY = 0;
  private _zoom = 0.4;
  // The floor when the map has not been measured yet. Once it has, the
  // real floor is the zoom that still fills the screen with land.
  private minZoom = 0.15;
  private maxZoom = 4.0;
  // Pixel extent of the hex field, half a hex past the outermost centres.
  private worldBox: { minX: number; minY: number; maxX: number; maxY: number } | null = null;

  // Interaction
  private dragging = false;
  private dragStart = { x: 0, y: 0 };
  private camStart = { x: 0, y: 0 };
  private hoveredHex: string | null = null;
  private selectedHex: string | null = null;
  private adjacencyPreview: { q: number; r: number; radius: number; positive: boolean } | null = null;

  private onHexSelect: HexSelectCallback = () => {};
  private onZoomChange: (zoom: number) => void = () => {};
  private reportedZoom = NaN;
  private dirty = true;
  private overlayDirty = false;
  private readonly interactionController = new AbortController();
  private destroyed = false;
  private readonly renderTick = () => {
    if (this.destroyed) return;
    if (this.dirty) {
      this.redraw();
      this.dirty = false;
      this.overlayDirty = false;
    } else if (this.overlayDirty) {
      const z = this._zoom;
      this.drawOverlay(
        this.app.screen.width / 2 - this.camX * z,
        this.app.screen.height / 2 - this.camY * z, z,
      );
      this.overlayDirty = false;
    }
    if (this.reportedZoom !== this._zoom) {
      this.reportedZoom = this._zoom;
      this.onZoomChange(this._zoom);
    }
  };

  /** Redraw after the canvas container changed size. The PIXI app only
   *  tracks window resizes on its own, so layout shifts (a toolbar
   *  wrapping to a second row, a banner appearing) leave app.screen
   *  stale — and every screenToWorld hit test lands a hex off. */
  handleResize() {
    this.dirty = true;
  }

  // Build mode (drag-to-build roads/rails)
  private buildMode: BuildMode = 'pan';
  private buildPath: { q: number; r: number }[] = [];
  private buildPathKeys: Set<string> = new Set();
  private building = false; // true while dragging in build mode
  private onBuildPath: BuildPathCallback = () => {};

  // City label cache
  private cityLabels: Map<string, Text> = new Map();
  private lastLabelZoom = 0;
  // One entry per nation holding land: where its name sits, how far it
  // may stretch, and the angle its territory leans at. Measured when the
  // tiles change, not per frame.
  private nationLabelGeometry: Map<string, {
    cx: number; cy: number; angle: number; span: number; hexes: number;
  }> = new Map();
  private nationLabels: Map<string, Text> = new Map();
  private lastNationLabelZoom = 0;

  // Power MW label cache (electricity mode)
  private powerLabels: Map<string, Text> = new Map();
  private lastPowerLabelZoom = 0;

  // War overlay (military mode): own garrisons and active map orders
  private warOrders: WarOrder[] = [];
  private warGarrisons: WarGarrison[] = [];
  private warLabelContainer!: Container;
  private warLabels: Map<string, Container> = new Map();
  private lastWarLabelZoom = 0;
  // Live grid figures per hex from /economy/power-grid: what the lines
  // deliver here, the capacity carrying it, and what the hex draws.
  private powerGrid: Map<string, {
    delivered_mw: number; capacity_mw: number | null;
    consumption_mw: number; electrified: boolean;
  }> = new Map();

  // Building icon texture cache & sprite pool
  private iconContainer!: Container;
  private iconTextures: Map<string, Texture> = new Map();
  private spritePool: Sprite[] = [];

  // Network edges from backend (canonical edge keys)
  private networkEdges: Map<string, string> = new Map(); // edgeKey -> network_type
  private transport = new TransportNetwork(new Map(), new Map());
  private traffic!: TrafficLayer;
  private checkpoints!: CheckpointLayer;
  private crossings: Crossing[] = [];
  private trafficTick: ((ticker: { deltaMS: number }) => void) | null = null;
  private _showTraffic = true;

  // Overlay visibility toggles
  private _showIcons = true;
  private _showRoads = true;
  private _showPower = true;
  private _showBorders = true;
  private _showLabels = true;
  private _showTextures = true;

  // Flat terrain colour painted under the tile art
  private baseGfx!: Graphics;

  // Tile sprites whose installation is an animation (the windmill hex).
  // Rebuilt by redraw(); advanced by a ticker that swaps textures in place
  // rather than redrawing, because a full redraw ~8 times a second would
  // cost far more than the handful of sprites actually changing.
  private animatedTiles: { sprite: Sprite; type: string }[] = [];
  private animAccumMs = 0;
  private animFrame = 0;
  private animTick: ((ticker: { deltaMS: number }) => void) | null = null;

  // Hex tile texture sprites (terrain mode)
  private tileContainer!: Container;
  private tileSpritePool: Sprite[] = [];
  private terrainTexturesLoaded = false;
  private flowGfx!: Graphics;
  private roadGfx!: Graphics;
  private flowMs = 0;
  private flowTick: ((ticker: { deltaMS: number }) => void) | null = null;
  private hvHexes = new Set<string>();
  private riverEdges: [number, number, number, number, number, number][] = [];
  private riverGeometry = new Map<string, { x: number; y: number }[] | null>();
  private lakeHexes = new Set<string>();
  private hexFlow = new Map<string, number>();
  private season: Season | null = null;
  private snowHexes = new Set<string>();
  private riverMajorFlow = 500;

  // Ocean hexes adjacent to land (rendered as lighter shallow water)
  private shallowWater: Set<string> = new Set();
  // Land hexes touching an ocean tile. The settlement art is picked from
  // this: only a hex in here may be drawn with water in it.
  private coastalLand: Set<string> = new Set();

  // Resource dot labels (accessibility). Each code is rasterised once into a
  // shared texture; on-screen labels are pooled Sprites pointing at it. A
  // Text per label costs ~0.1ms/frame each in the text pipeline — at 500
  // deposits that is a 50ms frame, which defeats the point of task 1.
  private resourceLabelContainer!: Container;
  private resourceLabelPool: Sprite[] = [];
  private resourceCodeTextures: Map<string, Texture | null> = new Map();

  // Accessibility settings, mirrored from the a11y store
  private a11y: A11yView = A11Y_FALLBACK;
  private a11yUnsub: (() => void) | null = null;

  // ── Profiling ──
  /** Wall time of the last redraw() in ms. Read it from the console
   *  (`renderer.lastRedrawMs`) or from a benchmark harness. */
  lastRedrawMs = 0;
  /** Tiles that survived the viewport cull on the last redraw(). */
  lastVisibleTiles = 0;
  /** Reused cull output: rebuilt in place each frame so panning does not
   *  allocate ~15k objects per frame. */
  private visibleBuf: { tile: TileData; sx: number; sy: number }[] = [];
  /** Clouds and rain. Scenery, and the season made visible. */
  private weather!: WeatherLayer;
  /** The world's declared size, once the client has been told it. */
  private worldSize: { width: number; height: number } | null = null;
  private readonly audio = new GameAudio();
  private weatherTick: ((ticker: { deltaMS: number }) => void) | null = null;

  constructor(app: Application) {
    this.app = app;

    this.worldContainer = new Container();
    app.stage.addChild(this.worldContainer);

    // Flat terrain colour under the artwork. Tile art is not obliged to
    // fill its hex: the pixel mountains and forests taper to a peak, so
    // the upper corners of the hex are transparent and used to read as
    // black gaps in the grid. This paints the ground they stand on.
    this.baseGfx = new Graphics();
    this.worldContainer.addChild(this.baseGfx);

    // Tile texture sprites sit below hexGfx so color overlays draw on top
    this.tileContainer = new Container();
    this.worldContainer.addChild(this.tileContainer);

    this.hexGfx = new Graphics();
    this.worldContainer.addChild(this.hexGfx);

    // The rivers' moving highlights. Their own layer because they redraw
    // every frame while the map underneath redraws only when it changes.
    this.flowGfx = new Graphics();
    this.worldContainer.addChild(this.flowGfx);

    // Roads and the markers that ride on them sit above the flow highlights.
    // Inside hexGfx they were under it, so the dashes slid across every
    // bridge deck and junction the water passes beneath.
    this.roadGfx = new Graphics();
    this.worldContainer.addChild(this.roadGfx);

    this.traffic = new TrafficLayer();
    this.worldContainer.addChild(this.traffic.container);
    this.checkpoints = new CheckpointLayer();
    this.worldContainer.addChild(this.checkpoints.container);
    try { this._showTraffic = localStorage.getItem('rpn_map_traffic') !== '0'; } catch { /* private mode */ }

    // Weather sits at two depths inside the world, because its two halves
    // belong at two depths. A cloud shadow falls on the ground, so it goes
    // under the buildings.
    this.weather = new WeatherLayer(app.renderer);
    this.worldContainer.addChild(this.weather.shadowLayer);

    this.iconContainer = new Container();
    this.worldContainer.addChild(this.iconContainer);

    this.overlayGfx = new Graphics();
    this.worldContainer.addChild(this.overlayGfx);

    // And the clouds themselves pass above the buildings, but below the
    // text: a cloud drifting over a city should not hide its name.
    this.worldContainer.addChild(this.weather.cloudLayer);
    this.weather.setSeason('summer');

    // Country names sit under the city names: at the zoom where one is
    // readable the other is hidden, and when they do overlap the city is
    // the more specific thing to read.
    this.nationLabelContainer = new Container();
    this.worldContainer.addChild(this.nationLabelContainer);

    this.labelContainer = new Container();
    this.worldContainer.addChild(this.labelContainer);

    // Deposit codes sit above the fills but below city names
    this.resourceLabelContainer = new Container();
    this.worldContainer.addChild(this.resourceLabelContainer);

    this.powerLabelContainer = new Container();
    this.worldContainer.addChild(this.powerLabelContainer);

    this.warLabelContainer = new Container();
    this.worldContainer.addChild(this.warLabelContainer);

    this.setupInteraction();

    // Accessibility settings drive deposit dot shape and labels; subscribe
    // fires immediately, so this also seeds the initial value.
    this.a11yUnsub = a11y.subscribe((s: A11ySettings) => {
      this.setA11y(s);
    });

    // Center camera on map middle
    const [cx, cy] = axialToPixel(100, 100, HEX_SIZE);
    this.camX = cx;
    this.camY = cy;

    // Render loop
    app.ticker.add(this.renderTick);

    // Installation animations. Every animated type shares one clock, so the
    // windmills across a nation turn together instead of drifting apart.
    this.animTick = (ticker: { deltaMS: number }) => {
      if (this.animatedTiles.length === 0) return;
      this.animAccumMs += ticker.deltaMS;
      const step = installationFrameMs(this.animatedTiles[0].type) ?? 130;
      if (this.animAccumMs < step) return;
      this.animAccumMs %= step;
      this.animFrame++;
      for (const { sprite, type } of this.animatedTiles) {
        const frames = installationFrames(type);
        if (!frames.length) continue;
        const tex = getTextureByUrl(frames[this.animFrame % frames.length]);
        if (tex) sprite.texture = tex;
      }
    };
    app.ticker.add(this.animTick);

    // Water moves whether or not the map needs redrawing, the same way the
    // sky does. Cheap: a handful of short dashes on the visible edges.
    this.flowTick = (ticker: { deltaMS: number }) => {
      this.flowMs += ticker.deltaMS;
      this.drawFlow();
    };
    app.ticker.add(this.flowTick);
    this.trafficTick = ticker => { this.traffic.update(ticker.deltaMS); this.checkpoints.update(ticker.deltaMS, this.traffic.simulation.time); };
    app.ticker.add(this.trafficTick);

    // Weather moves every frame whether or not the map needs redrawing,
    // which is the point of it: the map is still and the sky is not.
    this.weatherTick = (ticker: { deltaMS: number }) => {
      this.weather.update(ticker.deltaMS, this.zoom);
      this.audio.setBed('rain', this.weather.rainAmount);
      this.audio.setBed('wind', this.weather.windAmount);
      const clap = this.weather.takeThunder();
      if (clap > 0) this.audio.thunder(clap);
    };
    app.ticker.add(this.weatherTick);
  }

  // ── Public API ───────────────────────────────────────────────────

  setOnHexSelect(cb: HexSelectCallback) {
    this.onHexSelect = cb;
  }

  setOnZoomChange(cb: (zoom: number) => void) {
    this.onZoomChange = cb;
    this.reportedZoom = NaN;
  }

  setOnBuildPath(cb: BuildPathCallback) {
    this.onBuildPath = cb;
  }

  /** Turn map sound on or off. Separate from the weather overlay:
   *  plenty of people want the sky and not the noise. */
  setWeatherSound(enabled: boolean) {
    this.audio.setMuted(!enabled);
  }

  /** Fire a one-off, for whatever just happened on the map. */
  playSound(shot: Shot, volume = 1) {
    this.audio.play(shot, volume);
  }

  /** Called from a real user gesture on the canvas. Browsers will not
   *  start audio outside one, so without this the rain stays silent. */
  resumeAudio() {
    this.audio.resume();
  }

  /** Tell the sky which world it is over.
   *
   * The seed makes the cloud field identical for every player, and the
   * fixed world size stops the extent moving when somebody claims a hex:
   * bounds used to be measured from the tiles actually loaded, so another
   * nation expanding rebuilt your weather.
   */
  setWorldInfo(seed: number, width: number, height: number, tick = 0) {
    if (this.worldSeed !== (seed >>> 0) || this.worldTick !== tick) this.dirty = true;
    this.weather?.setSeed(seed);
    // Hydrology reads the same rainfall field the server simulates, so the
    // map shows where it has actually been raining rather than a static
    // per-terrain guess.
    this.worldSeed = seed >>> 0;
    this.worldTick = tick;
    if (width > 0 && height > 0) {
      this.worldSize = { width, height };
      this.publishWorldBounds();
    }
  }

  /** The tick the rainfall field is evaluated at. */
  private worldSeed = 0;
  private worldTick = 0;

  setOverlayVisibility(key: 'icons' | 'roads' | 'traffic' | 'power' | 'borders' | 'labels' | 'textures' | 'weather', value: boolean) {
    if (key === 'icons') this._showIcons = value;
    else if (key === 'roads') this._showRoads = value;
    else if (key === 'traffic') this._showTraffic = value;
    else if (key === 'power') this._showPower = value;
    else if (key === 'borders') this._showBorders = value;
    else if (key === 'labels') this._showLabels = value;
    else if (key === 'textures') this._showTextures = value;
    else if (key === 'weather') {
      // Weather draws itself on the ticker, so it needs no redraw. The
      // others set a flag that redraw() reads.
      this.weather?.setEnabled(value);
      return;
    }
    this.dirty = true;
  }

  /** Apply accessibility settings. Wired to the a11y store in the
   *  constructor; also public so callers without the store (tests,
   *  benchmarks) can drive it directly. */
  setA11y(s: Partial<A11yView> | null | undefined) {
    const next: A11yView = {
      colorblind: s?.colorblind ?? 'off',
      resourceLabels: s?.resourceLabels === true,
    };
    if (next.colorblind === this.a11y.colorblind
        && next.resourceLabels === this.a11y.resourceLabels) return;
    this.a11y = next;
    this.dirty = true;
  }

  /** The world's season: the sky, and in winter the ground as well. */
  setSeason(season: Season) {
    const changed = this.season !== season;
    this.season = season;
    this.weather?.setSeason(season);
    if (changed) this.dirty = true;
  }

  /** How far the snow line creeps out of the cold band in winter.
   *
   *  The terrain does not change: a grassland hex under snow is still
   *  grassland and still grows what grassland grows. Only the art does, so
   *  winter looks like winter without moving a single placement rule.
   *  The depth wobbles per hex so the edge is ragged rather than a ring.
   */
  private computeSnowLine() {
    this.snowHexes.clear();
    if (!this.tiles.size) return;

    const MAX_DEPTH = 5;
    let frontier: TileData[] = [];
    for (const tile of this.tiles.values()) {
      if (FROZEN_TERRAIN.has(tile.terrain)) frontier.push(tile);
    }

    const depth = new Map<string, number>();
    for (const t of frontier) depth.set(hexKey(t.q, t.r), 0);
    for (let step = 1; step <= MAX_DEPTH; step++) {
      const next: TileData[] = [];
      for (const t of frontier) {
        for (const [nq, nr] of axialNeighbors(t.q, t.r)) {
          const k = hexKey(nq, nr);
          if (depth.has(k)) continue;
          const n = this.tiles.get(k);
          if (!n || n.terrain === 0) continue;
          // A deterministic wobble per hex, so the same map always freezes
          // the same way and the line is not a clean arc.
          const jitter = Math.abs(Math.sin(nq * 12.9898 + nr * 78.233) * 43758.5453) % 1;
          if (step > MAX_DEPTH * (0.45 + jitter * 0.55)) continue;
          depth.set(k, step);
          next.push(n);
        }
      }
      frontier = next;
      if (!frontier.length) break;
    }
    for (const k of depth.keys()) this.snowHexes.add(k);
  }

  setWeatherEnabled(on: boolean) {
    this.weather?.setEnabled(on);
  }

  isWeatherEnabled(): boolean {
    return this.weather?.isEnabled() ?? false;
  }

  setBuildMode(mode: BuildMode) {
    this.buildMode = mode;
    this.buildPath = [];
    this.buildPathKeys.clear();
    this.building = false;
    this.dirty = true;
    const canvas = this.app.canvas as HTMLCanvasElement;
    canvas.style.cursor = mode === 'pan' ? 'grab' : 'crosshair';
  }

  getBuildMode(): BuildMode {
    return this.buildMode;
  }

  loadTiles(
    rawTiles: any[], nationsMeta: Record<string, any>, rawEdges?: any[],
    hvHexes?: [number, number][],
    borderPostures: BorderPosture[] = [], customsHexes: [number, number][] = [],
  ) {
    // Pylons are visible from the ground, so they travel on the shared
    // map. The building list they used to be read from does not.
    this.hvHexes.clear();
    for (const [q, r] of hvHexes ?? []) this.hvHexes.add(hexKey(q, r));
    this.tiles.clear();
    this.nations.clear();
    this.nationColorCache.clear();
    this.clearLabels();
    this.clearNationLabels();
    this.networkEdges.clear();

    for (const t of rawTiles) {
      // Two payload shapes: 14 fields (no stats) or 30 fields (with stats,
      // inhabited at [26]). Anarchy/occupier/seaOwner close the tuple, so
      // read them from the end - that also tolerates a cached payload from
      // before the inhabited flag (28) or the sea owner (13/29) existed.
      const hasStats = t.length >= 20;
      const hasSeaOwner = t.length === 14 || t.length >= 30;
      const tail = hasSeaOwner ? 1 : 0;
      const tile: TileData = {
        q: t[0], r: t[1], terrain: t[2],
        resource: t[3], richness: t[4],
        owner: t[5] == null ? null : String(t[5]),
        cityId: t[6] ?? null,
        cityName: t[7] ?? null,
        isCapital: t[8] ?? false,
        buildingCount: t[9] ?? 0,
        buildingKeys: t[10] ?? [],
        pollution: hasStats ? (t[11] ?? 0) : 0,
        crime: hasStats ? (t[12] ?? 0) : 0,
        disease: hasStats ? (t[13] ?? 0) : 0,
        electricity: hasStats ? (t[14] ?? false) : false,
        roadConnected: hasStats ? (t[15] ?? false) : false,
        railConnected: hasStats ? (t[16] ?? false) : false,
        fireCoverage: hasStats ? (t[17] ?? 0) : 0,
        fireHazard: hasStats ? (t[18] ?? 0) : 0,
        waterSupply: hasStats ? (t[19] ?? 0) : 0,
        policeCoverage: hasStats ? (t[20] ?? 0) : 0,
        hospitalCoverage: hasStats ? (t[21] ?? 0) : 0,
        safety: hasStats ? (t[22] ?? 0) : 0,
        health: hasStats ? (t[23] ?? 0) : 0,
        education: hasStats ? (t[24] ?? 0) : 0,
        happiness: hasStats ? (t[25] ?? 0) : 0,
        inhabited: hasStats ? (t.length >= 29 ? (t[26] ?? false) : true) : true,
        anarchyUntil: t[t.length - 2 - tail] ?? null,
        occupier: (() => {
          const o = t[t.length - 1 - tail];
          return o == null ? null : String(o);
        })(),
        // Water is owned through its own column, so an Economic Zone claim
        // never shows up in `owner`. id stays a string, like owner.
        seaOwner: (() => {
          if (!hasSeaOwner) return null;
          const o = t[t.length - 1];
          return o == null ? null : String(o);
        })(),
      };
      this.tiles.set(hexKey(tile.q, tile.r), tile);
    }

    // Shallow-water cache: ocean hexes that touch land render lighter,
    // giving coastlines visible depth in flat-color modes.
    this.shallowWater.clear();
    this.coastalLand.clear();
    for (const tile of this.tiles.values()) {
      const key = hexKey(tile.q, tile.r);
      for (const [nq, nr] of axialNeighbors(tile.q, tile.r)) {
        const n = this.tiles.get(hexKey(nq, nr));
        if (!n) continue;
        if (tile.terrain === 0 && n.terrain !== 0) {
          this.shallowWater.add(key);
          break;
        }
        // The other half of the same coastline, read from the land side.
        if (tile.terrain !== 0 && n.terrain === 0) {
          this.coastalLand.add(key);
          break;
        }
      }
    }

    // Where the snow reaches once winter comes round.
    this.computeSnowLine();

    for (const [id, meta] of Object.entries(nationsMeta)) {
      // id stays a string to match tile.owner; never parseInt a snowflake
      this.nations.set(id, meta as NationMeta);
      this.nationColorCache.set(id, parseInt((meta as NationMeta).primary_color.slice(1), 16));
    }

    // Load network edges (from_q, from_r, to_q, to_r, network_type)
    if (rawEdges) {
      for (const e of rawEdges) {
        const fq = e[0], fr = e[1], tq = e[2], tr = e[3], ntype: string = e[4];
        // Canonical order: smaller key first
        const kA = hexKey(fq, fr), kB = hexKey(tq, tr);
        const edgeKey = kA < kB ? `${kA}_${kB}` : `${kB}_${kA}`;
        // Store with network type (may have multiple types per edge, store as comma-separated)
        const existing = this.networkEdges.get(edgeKey);
        if (existing) {
          if (!existing.includes(ntype)) {
            this.networkEdges.set(edgeKey, `${existing},${ntype}`);
          }
        } else {
          this.networkEdges.set(edgeKey, ntype);
        }
      }
    }

    this.transport = new TransportNetwork(this.tiles, this.networkEdges);
    this.crossings = findCrossings(this.transport, this.tiles, borderPostures, customsHexes);
    this.traffic.setNetwork(this.transport);
    this.publishWorldBounds();

    this.dirty = true;
    this.preloadBuildingTextures();
    this.preloadTerrainTileTextures();
  }

  /**
   * Measure the tiles and hand the extent to the weather.
   *
   * The sky used to carry its own idea of how big the map is, written
   * against a hex size the game does not use, so the clouds were spread
   * over a box three times too wide and most of them sat off the east
   * edge with nothing under them.
   */
  private publishWorldBounds() {
    if (this.tiles.size === 0) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const tile of this.tiles.values()) {
      const [x, y] = axialToPixel(tile.q, tile.r, HEX_SIZE);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    if (!Number.isFinite(minX) || !Number.isFinite(minY)) return;
    // A hex is drawn from its centre, so the outermost ones hang half a
    // hex past the extreme centres.
    this.worldBox = {
      minX: minX - HEX_SIZE, minY: minY - HEX_SIZE,
      maxX: maxX + HEX_SIZE, maxY: maxY + HEX_SIZE,
    };
    // The camera was free to drift off the edge before the map had been
    // measured. Now that it has, pull it back.
    this.clampView();
    this.measureNationLabels();

    if (!this.weather) return;
    // Prefer the world's declared size. Measuring the loaded tiles makes
    // the sky depend on who owns what, which is how another player's
    // claim ended up rearranging everybody's clouds.
    let skyMinX = minX - HEX_SIZE, skyMinY = minY - HEX_SIZE;
    let skyMaxX = maxX + HEX_SIZE, skyMaxY = maxY + HEX_SIZE;
    if (this.worldSize) {
      const [ax, ay] = axialToPixel(0, 0, HEX_SIZE);
      const [bx, by] = axialToPixel(
        this.worldSize.width, this.worldSize.height, HEX_SIZE);
      skyMinX = Math.min(ax, bx) - HEX_SIZE;
      skyMinY = Math.min(ay, by) - HEX_SIZE;
      skyMaxX = Math.max(ax, bx) + HEX_SIZE;
      skyMaxY = Math.max(ay, by) + HEX_SIZE;
    }
    this.weather.setWorldBounds(skyMinX, skyMinY, skyMaxX, skyMaxY);
    // The bounding box is not the map: axialToPixel skews y by q, so the
    // hex field is a parallelogram inside it. Let the sky ask which
    // points are actually over a tile.
    this.weather.setCoverage((x, y) => {
      const [q, r] = pixelToAxial(x, y, HEX_SIZE);
      return this.tiles.has(hexKey(q, r));
    });
    // And which of them are frozen, so what falls on the ice cap is
    // snow in every season rather than in one of them.
    this.weather.setFrozenGround((x, y) => {
      const [q, r] = pixelToAxial(x, y, HEX_SIZE);
      const tile = this.tiles.get(hexKey(q, r));
      return !!tile && FROZEN_TERRAIN.has(tile.terrain);
    });

    // And the outline to clip them to. Four corners in axial space, which
    // are four corners of a parallelogram in pixel space, pushed out from
    // the middle by a hex so the rim tiles are covered rather than
    // shaved.
    let q0 = Infinity, q1 = -Infinity, r0 = Infinity, r1 = -Infinity;
    for (const tile of this.tiles.values()) {
      if (tile.q < q0) q0 = tile.q;
      if (tile.q > q1) q1 = tile.q;
      if (tile.r < r0) r0 = tile.r;
      if (tile.r > r1) r1 = tile.r;
    }
    const corners = [[q0, r0], [q1, r0], [q1, r1], [q0, r1]]
      .map(([q, r]) => axialToPixel(q, r, HEX_SIZE));
    const cx = corners.reduce((t, c) => t + c[0], 0) / corners.length;
    const cy = corners.reduce((t, c) => t + c[1], 0) / corners.length;
    const points: number[] = [];
    for (const [x, y] of corners) {
      const dx = x - cx, dy = y - cy;
      const len = Math.hypot(dx, dy) || 1;
      points.push(x + (dx / len) * HEX_SIZE * 2, y + (dy / len) * HEX_SIZE * 2);
    }
    this.weather.setClip(points);
  }

  private async preloadTerrainTileTextures() {
    try {
      const terrainIds = new Set<number>();
      for (const tile of this.tiles.values()) {
        terrainIds.add(tile.terrain);
      }
      await preloadTerrainTextures(terrainIds);
      this.terrainTexturesLoaded = areTerrainTexturesLoaded();
      console.log(`[map-renderer] Terrain textures loaded: ${this.terrainTexturesLoaded}`);
      this.dirty = true;
    } catch (e) {
      console.error('[map-renderer] Failed to preload terrain textures:', e);
    }
  }

  /** Fetch SVG data from the Iconify API and create Pixi textures for building icons. */
  private async preloadBuildingTextures() {
    // Collect unique Iconify icon names from loaded tiles
    const needed = new Set<string>();
    for (const tile of this.tiles.values()) {
      for (const key of tile.buildingKeys ?? []) {
        if (LINE_BUILDINGS.has(key)) continue;
        const def = getBuildingIcon(key);
        if (!this.iconTextures.has(def.icon)) needed.add(def.icon);
      }
    }
    await this.loadIconTextures(needed);
  }

  /** Load any not-yet-cached Iconify icons into Pixi textures. */
  private async loadIconTextures(needed: Set<string>) {
    for (const icon of [...needed]) {
      if (this.iconTextures.has(icon)) needed.delete(icon);
    }
    if (needed.size === 0) return;

    // Group icons by prefix for batch fetch
    const byPrefix = new Map<string, string[]>();
    for (const icon of needed) {
      const [prefix, name] = icon.split(':');
      if (!byPrefix.has(prefix)) byPrefix.set(prefix, []);
      byPrefix.get(prefix)!.push(name);
    }

    for (const [prefix, names] of byPrefix) {
      try {
        const url = `https://api.iconify.design/${prefix}.json?icons=${names.join(',')}`;
        const resp = await fetch(url);
        if (!resp.ok) continue;
        const data = await resp.json();
        const defaultW = data.width ?? 512;
        const defaultH = data.height ?? 512;

        const loadPromises: Promise<void>[] = [];
        for (const [name, info] of Object.entries(data.icons as Record<string, any>)) {
          const body: string = info.body.replace(/currentColor/g, 'white');
          const w = info.width ?? defaultW;
          const h = info.height ?? defaultH;
          const iconKey = `${prefix}:${name}`;
          // Rasterise near display size: icons draw at <=28px, so a 512px
          // bitmap (256 logical at resolution 2) keeps minification inside
          // the mipmap chain instead of shimmering down from 2048px, and
          // cuts GPU memory per icon from 16MB to 256KB.
          const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 ${w} ${h}" shape-rendering="geometricPrecision">${body}</svg>`;
          // Use encodeURIComponent for proper UTF-8 handling (some SVG paths contain special chars)
          const dataUrl = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;

          loadPromises.push(
            Assets.load<Texture>({
              src: dataUrl,
              alias: `icon-${iconKey}-${Date.now()}`,
              data: {
                resolution: 2,
                autoGenerateMipmaps: true,
                scaleMode: 'linear',
              },
            })
              .then(tex => { this.iconTextures.set(iconKey, tex); })
              .catch(() => { /* skip missing icon */ })
          );
        }
        await Promise.all(loadPromises);
      } catch {
        /* API fetch failed – icons will fall back to colored circles */
      }
    }

    this.dirty = true;
  }

  /** Feed live power-grid figures for the electricity overlay. */
  setPowerGrid(hexes: {
    q: number; r: number; delivered_mw: number; capacity_mw: number | null;
    consumption_mw: number; electrified: boolean;
  }[]) {
    this.powerGrid.clear();
    for (const h of hexes ?? []) {
      this.powerGrid.set(hexKey(h.q, h.r), {
        delivered_mw: h.delivered_mw,
        capacity_mw: h.capacity_mw,
        consumption_mw: h.consumption_mw,
        electrified: h.electrified,
      });
    }
    this.clearPowerLabels();
    this.dirty = true;
  }

  /** Re-cut every canvas label after the player changes typeface. Text is
   *  rasterised once and cached, so the caches have to go with the font. */
  refreshFont() {
    clearMapFont();
    this.clearLabels();
    this.clearNationLabels();
    this.clearWarLabels();
    this.clearResourceLabels();
    this.dirty = true;
  }

  setMode(mode: MapMode) {
    if (this.mode !== mode) {
      this.mode = mode;
      this.clearPowerLabels();
      this.dirty = true;
    }
  }

  /** Rivers run on the edges between hexes and are terrain, not an
   *  overlay, so they are drawn in every map mode. */
  setRivers(
    edges: [number, number, number, number, number, number][],
    lakes: [number, number, number][],
    majorFlow: number,
  ) {
    this.riverEdges = edges;
    this.riverGeometry.clear();
    this.lakeHexes.clear();
    for (const [q, r] of lakes) this.lakeHexes.add(hexKey(q, r));
    // Both hexes flanking an edge stand on that bank, so per-hex flow is
    // the most water any of its six edges carries.
    this.hexFlow.clear();
    for (const [q1, r1, q2, r2, flow] of edges) {
      for (const k of [hexKey(q1, r1), hexKey(q2, r2)]) {
        this.hexFlow.set(k, Math.max(this.hexFlow.get(k) ?? 0, flow));
      }
    }
    this.riverMajorFlow = majorFlow || 500;
    this.dirty = true;
  }

  /** Border overlay for 'borders' mode: my frontier hexes with the
   * hottest neighbor's tension and the crossing state. */
  setBorderOverlay(hexes: { q: number; r: number; tension: number; crossing: string }[]) {
    this.borderOverlay.clear();
    for (const h of hexes) {
      this.borderOverlay.set(hexKey(h.q, h.r), { tension: h.tension, crossing: h.crossing });
    }
    this.dirty = true;
  }

  /** Highlight the adjacency radius around a hex while picking a building. */
  setAdjacencyPreview(preview: { q: number; r: number; radius: number; positive: boolean } | null) {
    this.adjacencyPreview = preview;
    this.overlayDirty = true;
  }

  /** Own garrisons and active map orders, rendered in military mode. */
  setWarOverlay(orders: WarOrder[], garrisons: WarGarrison[]) {
    this.warOrders = orders;
    this.warGarrisons = garrisons;
    this.clearWarLabels();
    // Badge icons: fetch any unit textures we don't have yet, then rebuild
    // the cached badges so late-arriving icons actually appear.
    const needed = new Set<string>();
    for (const g of garrisons) {
      // A garrison with no per-unit breakdown still draws one chip, on the
      // generic icon, so that has to be fetched too.
      if (!g.units?.length) needed.add(getUnitIcon(''));
      for (const u of g.units ?? []) needed.add(getUnitIcon(u.key));
    }
    void this.loadIconTextures(needed).then(() => {
      this.clearWarLabels();
      this.dirty = true;
    });
    this.dirty = true;
  }

  getMode(): MapMode {
    return this.mode;
  }

  get zoom(): number {
    return this._zoom;
  }

  /** The zoom at which the map still covers the viewport.
   *
   * Zoom out past this and the void around the world comes into frame,
   * which is the one view the game has nothing to draw in. Recomputed
   * from the live screen size, so it follows a window resize.
   */
  private minZoomForBounds(): number {
    if (!this.worldBox) return this.minZoom;
    const bw = this.worldBox.maxX - this.worldBox.minX;
    const bh = this.worldBox.maxY - this.worldBox.minY;
    if (bw <= 0 || bh <= 0) return this.minZoom;
    const fit = Math.max(this.app.screen.width / bw, this.app.screen.height / bh);
    return Math.min(this.maxZoom, fit);
  }

  private clampZoom(z: number): number {
    return Math.max(this.minZoomForBounds(), Math.min(this.maxZoom, z));
  }

  /** Keep the viewport inside the map, on both axes.
   *
   * Clamping the zoom alone is half the job: at the fitting zoom you can
   * still drag the world off screen and end up looking at the abyss from
   * the side. An axis shorter than the viewport centres instead.
   */
  private clampView() {
    this._zoom = this.clampZoom(this._zoom);
    if (!this.worldBox) return;
    const { minX, minY, maxX, maxY } = this.worldBox;
    const halfW = this.app.screen.width / (2 * this._zoom);
    const halfH = this.app.screen.height / (2 * this._zoom);
    this.camX = minX + halfW > maxX - halfW
      ? (minX + maxX) / 2
      : Math.min(Math.max(this.camX, minX + halfW), maxX - halfW);
    this.camY = minY + halfH > maxY - halfH
      ? (minY + maxY) / 2
      : Math.min(Math.max(this.camY, minY + halfH), maxY - halfH);
  }

  setZoom(z: number) {
    this._zoom = this.clampZoom(z);
    this.clampView();
    this.dirty = true;
  }

  centerOn(q: number, r: number) {
    const [px, py] = axialToPixel(q, r, HEX_SIZE);
    this.camX = px;
    this.camY = py;
    this.clampView();
    this.dirty = true;
  }

  /** Centre on a hex and open it, as though the player had clicked it.
   *
   * The Problems tab links to a specific hex, so "take me there" has to
   * do what a click does rather than only move the camera: a centred map
   * with no panel open still leaves the player hunting.
   */
  focusHex(q: number, r: number): boolean {
    const tile = this.tiles.get(hexKey(q, r));
    this.centerOn(q, r);
    this.onHexSelect(q, r, tile ?? null);
    return tile != null;
  }

  /** Whether this nation holds any hex, without touching the camera.
   *
   * centerOnNation used to answer this as a side effect of moving, which
   * meant every refresh that wanted the answer also threw the view back
   * to the player's own territory.
   */
  hasNation(nationId: number | string): boolean {
    const target = String(nationId);
    for (const tile of this.tiles.values()) {
      if (tile.owner === target) return true;
    }
    return false;
  }

  centerOnNation(nationId: number | string): boolean {
    const target = String(nationId);
    let sumQ = 0, sumR = 0, count = 0;
    for (const tile of this.tiles.values()) {
      if (tile.owner === target) {
        sumQ += tile.q;
        sumR += tile.r;
        count++;
      }
    }
    if (count === 0) return false;
    const avgQ = Math.round(sumQ / count);
    const avgR = Math.round(sumR / count);
    this.centerOn(avgQ, avgR);
    if (this._zoom < 0.8) {
      this._zoom = this.clampZoom(0.8);
    }
    this.clampView();
    this.dirty = true;
    return true;
  }

  getSelectedTile(): TileData | null {
    if (!this.selectedHex) return null;
    return this.tiles.get(this.selectedHex) ?? null;
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.interactionController.abort();
    this.app.ticker.remove(this.renderTick);
    if (this.animTick) this.app.ticker.remove(this.animTick);
    this.animTick = null;
    if (this.flowTick) this.app.ticker.remove(this.flowTick);
    this.flowTick = null;
    if (this.trafficTick) this.app.ticker.remove(this.trafficTick);
    this.trafficTick = null;
    this.traffic.destroy();
    this.checkpoints.destroy();
    if (this.weatherTick) this.app.ticker.remove(this.weatherTick);
    this.weatherTick = null;
    this.weather?.destroy();
    this.animatedTiles = [];
    this.a11yUnsub?.();
    this.a11yUnsub = null;
    this.clearLabels();
    this.clearNationLabels();
    this.clearResourceLabels();
    this.visibleBuf = [];
    // Pools retain sprites that were detached when zooming or changing modes.
    for (const sprite of [...this.spritePool, ...this.tileSpritePool]) sprite.destroy();
    this.spritePool = [];
    this.tileSpritePool = [];
    this.iconTextures.clear();
    this.riverGeometry.clear();
    this.audio.destroy();
    this.app.stage.removeChild(this.worldContainer);
    this.worldContainer.destroy({ children: true });
  }

  /** Export the current map view as a high-quality PNG with branding (landscape). */
  async exportImage(): Promise<void> {
    // Force a synchronous redraw + render so the WebGL buffer has content
    this.redraw();
    this.app.renderer.render(this.app.stage);

    const canvas = this.app.canvas as HTMLCanvasElement;
    const srcW = canvas.width;
    const srcH = canvas.height;

    // Determine landscape export dimensions
    const scale = 2;
    let exportW: number;
    let exportH: number;
    if (srcW >= srcH) {
      // Already landscape
      exportW = srcW * scale;
      exportH = srcH * scale;
    } else {
      // Portrait source → force 16:9 landscape by using width as the long side
      exportW = Math.max(srcW, srcH) * scale;
      exportH = Math.round(exportW * 9 / 16);
    }

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = exportW;
    exportCanvas.height = exportH;
    const ctx = exportCanvas.getContext('2d')!;

    // Fill background
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, exportW, exportH);

    // Center the source canvas content in the export canvas
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    const drawW = srcW * scale;
    const drawH = srcH * scale;
    const dx = (exportW - drawW) / 2;
    const dy = (exportH - drawH) / 2;
    ctx.drawImage(canvas, dx, dy, drawW, drawH);

    // Add branding in bottom-right corner
    const brandPadding = 20 * scale;
    const brandFontSize = 16 * scale;
    const brandText = 'Roleplay Nations';

    ctx.save();

    const brandImg = new Image();
    brandImg.crossOrigin = 'anonymous';
    const imgLoaded = new Promise<boolean>((resolve) => {
      brandImg.onload = () => resolve(true);
      brandImg.onerror = () => resolve(false);
      brandImg.src = '/brandwhite.png';
    });
    const loaded = await imgLoaded;

    const imgH = 28 * scale;
    const imgW = loaded ? (brandImg.naturalWidth / brandImg.naturalHeight) * imgH : 0;

    ctx.font = `bold ${brandFontSize}px ${mapFont()}`;
    const textWidth = ctx.measureText(brandText).width;
    const gap = 8 * scale;
    const totalW = (loaded ? imgW + gap : 0) + textWidth;

    const bx = exportW - totalW - brandPadding;
    const by = exportH - imgH - brandPadding;

    ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
    ctx.shadowBlur = 6 * scale;
    ctx.shadowOffsetX = 1 * scale;
    ctx.shadowOffsetY = 1 * scale;

    if (loaded) {
      ctx.drawImage(brandImg, bx, by, imgW, imgH);
    }

    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(brandText, bx + (loaded ? imgW + gap : 0), by + imgH / 2);

    ctx.restore();

    // Download
    const link = document.createElement('a');
    link.download = `rpn-map-${Date.now()}.png`;
    link.href = exportCanvas.toDataURL('image/png');
    link.click();
  }

  // ── Rendering ────────────────────────────────────────────────────

  private redraw() {
    const perfStart = performance.now();
    // A window resize changes what fits without touching the camera.
    this.clampView();
    const gfx = this.hexGfx;
    gfx.clear();
    const roadGfx = this.roadGfx;
    roadGfx.clear();

    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const z = this._zoom;
    const sz = HEX_SIZE * z;

    // Camera offset: center of screen maps to (camX, camY) in world space
    const offsetX = w / 2 - this.camX * z;
    const offsetY = h / 2 - this.camY * z;

    // The weather draws its own children at world coordinates, so it needs
    // the same camera this function bakes into everything else.
    this.weather?.setCamera(offsetX, offsetY, z, w, h);

    // Screen margin for culling
    const margin = sz * 3;

    // ── Pass 0: cull once ───────────────────────────────────────────
    // The survivor count decides the border LOD, so the cull has to run
    // before anything is drawn. Results land in a reused buffer.
    const vis = this.visibleBuf;
    let visCount = 0;
    for (const tile of this.tiles.values()) {
      const [wx, wy] = axialToPixel(tile.q, tile.r, HEX_SIZE);
      const sx = wx * z + offsetX;
      const sy = wy * z + offsetY;
      if (sx < -margin || sx > w + margin || sy < -margin || sy > h + margin) continue;
      const slot = vis[visCount];
      if (slot) { slot.tile = tile; slot.sx = sx; slot.sy = sy; }
      else vis[visCount] = { tile, sx, sy };
      visCount++;
    }
    this.lastVisibleTiles = visCount;

    // Hex outlines are gated on how many tiles are actually on screen, not
    // on zoom: a zoom threshold alone left the 0.35-0.5 band drawing borders
    // for ~15k tiles, which is exactly where panning stalled. Below the
    // threshold every outline goes into ONE accumulated path stroked once —
    // in Pixi v8 each .stroke() is its own instruction, path clone and
    // batch, so per-tile stroking is what actually costs.
    const drawBorders = z > 0.35 && visCount <= BORDER_MAX_TILES;
    const drawResources = z > 0.6 && (this.mode === 'terrain' || this.mode === 'economic');
    const drawCityMarkers = z > 0.3;
    const drawBuildingDots = z > 1.0;

    // Collect city tiles so markers can be drawn AFTER roads
    const cityMarkers: { sx: number; sy: number; tile: TileData }[] = [];

    // Hex outlines, accumulated and stroked in one instruction after the fills
    const borderPolys: number[][] = [];
    // Deposit dots, bucketed by colour+alpha so each bucket is one fill
    const resourceDots: { sx: number; sy: number; resource: string; richness: number }[] = [];

    // Tile texture sprites: terrain mode shows the raw artwork; political
    // mode reuses it as a base layer under a translucent nation tint.
    const texturesReady = this.terrainTexturesLoaded || areTerrainTexturesLoaded();
    const useTextures = this._showTextures && texturesReady
      && (this.mode === 'terrain' || this.mode === 'political');
    this.tileContainer.removeChildren();
    this.tileContainer.visible = useTextures;
    this.animatedTiles.length = 0;
    this.baseGfx.clear();
    this.baseGfx.visible = useTextures;
    let tileSpriteIdx = 0;

    // National border segments and anarchy hexes, drawn after all fills
    const showNationBorders = this._showBorders
      && (this.mode === 'political' || this.mode === 'military') && z > 0.25;
    const borderSegs: { x1: number; y1: number; x2: number; y2: number; color: number }[] = [];
    const anarchyHexes: { sx: number; sy: number }[] = [];
    const occupiedHexes: { sx: number; sy: number; color: number }[] = [];

    for (let vi = 0; vi < visCount; vi++) {
      const { tile, sx, sy } = vis[vi];

      // One corner array per tile, shared by the fill and the outline: Pixi
      // only reads it when the frame is tessellated, so sharing is safe as
      // long as nothing mutates it afterwards.
      let pts: number[] | null = null;
      const corners = () => (pts ??= hexPts(sx, sy, sz));

      if (useTextures) {
        // Priority: city settlement > installation > terrain
        let tex: Texture | null = null;
        let animatedHere: string | null = null;
        if (tile.cityId && tile.buildingCount && tile.buildingCount > 0) {
          tex = getSettlementTexture(
            tile.q, tile.r, tile.buildingCount,
            this.coastalLand.has(hexKey(tile.q, tile.r)),
          );
        } else if (tile.buildingKeys && tile.buildingKeys.length > 0) {
          tex = getInstallationTexture(tile.buildingKeys, tile.q, tile.r);
          const instType = installationTypeFor(tile.buildingKeys);
          if (instType && installationFrameMs(instType) !== null) {
            animatedHere = instType;
            // Redraws are driven by hover and pan, not by the animation
            // clock. Handing back the base texture here would rewind every
            // windmill to frame 0 on each pointer move, so pick up the frame
            // the shared clock is actually on.
            const frames = installationFrames(instType);
            if (frames.length) {
              const cur = getTextureByUrl(frames[this.animFrame % frames.length]);
              if (cur) tex = cur;
            }
          }
        }
        if (!tex) {
          const shallow = tile.terrain !== 0
            || this.shallowWater.has(hexKey(tile.q, tile.r));
          // A lake is water, so it takes water art. Without this the flat
          // colour said lake and the texture on top of it still said
          // grassland, which is why no lake was visible on the map.
          const isLake = this.lakeHexes.has(hexKey(tile.q, tile.r));
          // In winter the snow reaches past the cold band. Terrain 5 is
          // tundra art; the tile's own terrain is untouched.
          const underSnow = !isLake && this.season === 'winter'
            && this.snowHexes.has(hexKey(tile.q, tile.r));
          const artTerrain = isLake ? 0 : (underSnow ? 5 : tile.terrain);
          tex = getTileTexture(tile.q, tile.r, artTerrain, shallow);
        }
        if (tex) {
          // Ground first, art on top: whatever the art leaves transparent
          // reads as terrain rather than as a hole in the map.
          this.baseGfx.poly(corners()).fill({ color: this.getHexColor(tile) });
          let sprite: Sprite;
          if (tileSpriteIdx < this.tileSpritePool.length) {
            sprite = this.tileSpritePool[tileSpriteIdx];
          } else {
            sprite = new Sprite();
            sprite.anchor.set(0.5, 0.5);
            this.tileSpritePool.push(sprite);
          }
          sprite.texture = tex;
          sprite.x = sx;
          sprite.y = sy;
          // Flat-top hex: width = 2*sz, height = sqrt(3)*sz. Every tile
          // spans the full canvas width and sits centred on it, so the
          // canvas aspect is the scale factor. Reading it off the texture
          // rather than hardcoding one lets sets differ: the painted hexes
          // were 509x443 (1.149, the hex exactly), while the pixel terrain
          // is a 128x128 square whose extra height is peaks and treetops
          // deliberately overhanging the hex above and below.
          const aspect = tex.height / tex.width;
          sprite.width = sz * 2.04;
          sprite.height = sz * 2.04 * aspect;
          sprite.alpha = 1.0;
          this.tileContainer.addChild(sprite);
          if (animatedHere) this.animatedTiles.push({ sprite, type: animatedHere });
          tileSpriteIdx++;
        } else {
          // Texture not loaded yet; fall back to color fill
          gfx.poly(corners()).fill({ color: this.getHexColor(tile) });
        }
        // Political tint over the artwork so ownership reads clearly
        if (this.mode === 'political' && tile.terrain !== 0) {
          if (tile.anarchyUntil) {
            gfx.poly(corners()).fill({ color: 0x7f1d1d, alpha: 0.5 });
          } else if (tile.owner) {
            const c = this.nationColorCache.get(tile.owner) ?? 0x555555;
            gfx.poly(corners()).fill({ color: c, alpha: tile.buildingCount ? 0.25 : 0.38 });
          }
        } else if (this.mode === 'political' && tile.seaOwner) {
          // An Economic Zone is a claim on water, not sovereign ground, so
          // it tints lighter than territory and the sea still reads as sea.
          const c = this.nationColorCache.get(tile.seaOwner) ?? 0x555555;
          gfx.poly(corners()).fill({ color: c, alpha: 0.22 });
        }
        // Outlines are accumulated and stroked once, after every fill
        if (drawBorders) borderPolys.push(corners());
      } else {
        // Non-texture mode: flat color fill with subtle per-hex variation
        // in terrain/political so large areas don't look like plastic.
        let fillColor = this.getHexColor(tile);
        if (this.mode === 'terrain' || this.mode === 'political') {
          fillColor = this.varyColor(fillColor, tile.q, tile.r);
        }
        gfx.poly(corners()).fill({ color: fillColor });

        if (drawBorders) borderPolys.push(corners());
      }

      // Anarchy hexes get a hatched overlay in political/military modes
      if (tile.anarchyUntil && z > 0.3 && (this.mode === 'political' || this.mode === 'military')) {
        anarchyHexes.push({ sx, sy });
      }

      // Occupied hexes hatch in the occupier's color: ownership only moves
      // at the peace table, so the base fill stays the owner's.
      if (tile.occupier && tile.occupier !== tile.owner && z > 0.3
          && (this.mode === 'political' || this.mode === 'military')) {
        occupiedHexes.push({
          sx, sy,
          color: this.nationColorCache.get(tile.occupier) ?? 0xff3333,
        });
      }

      // Collect national border edges: any edge whose neighbour has a
      // different owner (or none) gets a stroke in this nation's color.
      if (showNationBorders && tile.owner) {
        const color = this.nationColorCache.get(tile.owner) ?? 0xffffff;
        for (let d = 0; d < 6; d++) {
          const nb = this.tiles.get(hexKey(tile.q + BORDER_DIRS[d][0], tile.r + BORDER_DIRS[d][1]));
          if (nb && nb.owner === tile.owner) continue;
          const e = ((d + 1) % 6) * 2;
          // Inset slightly so two nations' borders sit side by side
          const inset = sz * 0.9;
          borderSegs.push({
            x1: sx + HEX_UNIT_CORNERS[d * 2] * inset,
            y1: sy + HEX_UNIT_CORNERS[d * 2 + 1] * inset,
            x2: sx + HEX_UNIT_CORNERS[e] * inset,
            y2: sy + HEX_UNIT_CORNERS[e + 1] * inset,
            color,
          });
        }
      }

      // Resource dots are bucketed and drawn after the loop
      if (drawResources && tile.resource && tile.richness > 0) {
        resourceDots.push({ sx, sy, resource: tile.resource, richness: tile.richness });
      }

      // Collect city tiles for drawing AFTER roads
      if (drawCityMarkers && tile.cityId) {
        cityMarkers.push({ sx, sy, tile });
      }

    }

    // Hex outlines: every visible hex goes into one path, stroked once.
    // One .stroke() means one instruction, one path clone and one batch for
    // the whole grid instead of one of each per tile.
    if (borderPolys.length > 0) {
      for (let i = 0; i < borderPolys.length; i++) gfx.poly(borderPolys[i]);
      gfx.stroke({
        color: 0x000000,
        width: useTextures ? (z > 0.6 ? 0.5 : 0.2) : (z > 0.6 ? 0.7 : 0.3),
        alpha: useTextures ? (z > 0.5 ? 0.15 : 0.05) : (z > 0.5 ? 0.25 : 0.08),
      });
    }

    // Deposit dots + accessibility labels
    this.drawResourceDots(gfx, resourceDots, sz, z);

    // National borders: dark underlay then colored line, over the fills
    if (borderSegs.length > 0) {
      const bw = Math.max(1.2, sz * 0.14);
      for (const s of borderSegs) {
        gfx.moveTo(s.x1, s.y1).lineTo(s.x2, s.y2)
          .stroke({ color: 0x000000, width: bw + Math.max(0.8, bw * 0.5), alpha: 0.35, cap: 'round' });
      }
      for (const s of borderSegs) {
        gfx.moveTo(s.x1, s.y1).lineTo(s.x2, s.y2)
          .stroke({ color: s.color, width: bw, alpha: 0.95, cap: 'round' });
      }
    }

    // Anarchy hatching (over fills and borders, under roads)
    for (const a of anarchyHexes) {
      this.drawAnarchyHatch(gfx, a.sx, a.sy, sz);
    }
    for (const o of occupiedHexes) {
      this.drawAnarchyHatch(gfx, o.sx, o.sy, sz, o.color, 0.85);
    }

    // Rivers, under the roads so a bridge reads as crossing over the water
    this.drawRivers(gfx, offsetX, offsetY, z);

    // Draw road/rail network lines (respects overlay toggles), on the layer
    // above the flow highlights.
    this.drawNetworkLines(roadGfx, offsetX, offsetY, z);

    // Bridges wherever a route crosses a river, over the deck it carries.
    this.drawBridges(roadGfx, offsetX, offsetY, z);

    // City markers (drawn OVER roads so they remain visible)
    for (const { sx, sy, tile: ct } of cityMarkers) {
      const paintedSettlement = useTextures && !!ct.buildingCount && z >= 0.65;
      // The settlement itself marks an inhabited hex at readable zoom.
      if (paintedSettlement && !ct.isCapital) continue;
      const markerY = paintedSettlement ? sy - sz * 0.58 : sy;
      const markerSize = paintedSettlement ? Math.max(2, Math.min(5, sz * 0.12)) : Math.max(2, sz * 0.25);
      const markerColor = ct.isCapital ? 0xffd700 : 0xffffff;
      if (ct.isCapital) {
        this.drawStar(roadGfx, sx, markerY, markerSize, markerColor);
      } else {
        const half = markerSize * 0.6;
        roadGfx.rect(sx - half, sy - half, half * 2, half * 2)
          .fill({ color: markerColor, alpha: 0.9 })
          .stroke({ color: 0x000000, width: 0.5, alpha: 0.5 });
      }
    }

    // Building SVG icons (drawn OVER roads & city markers via iconContainer)
    this.iconContainer.removeChildren();
    let spriteIdx = 0;

    // At overview zoom the artwork communicates settlements. Badges return
    // progressively as there is room for them, along the upper hex edge.
    if (drawBuildingDots && this._showIcons && (!useTextures || z >= 2.4)) {
      for (const tile of this.tiles.values()) {
        if (!tile.buildingKeys || tile.buildingKeys.length === 0) continue;
        const [wx, wy] = axialToPixel(tile.q, tile.r, HEX_SIZE);
        const sx = wx * z + offsetX;
        const sy = wy * z + offsetY;
        if (sx < -margin || sx > w + margin || sy < -margin || sy > h + margin) continue;

        const uniqueKeys = [...new Set(tile.buildingKeys)];
        const iconsToDraw = uniqueKeys
          .filter(k => {
            if (LINE_BUILDINGS.has(k)) return false;
            if (this.mode === 'electricity' && !POWER_BUILDINGS.has(k)) return false;
            return true;
          })
          .slice(0, useTextures ? (z < 3 ? 1 : z < 3.8 ? 2 : 3) : 4);
        const count = iconsToDraw.length;
        if (count === 0) continue;

        // Icon sizing: shrinks with count so nothing overlaps
        const baseSize = useTextures ? Math.max(10, Math.min(14, sz * 0.24)) : Math.max(12, Math.min(28, sz * 0.6));
        const iconSize = count === 1 ? baseSize
          : count === 2 ? baseSize * 0.85
          : baseSize * 0.7;
        const r = iconSize * 0.55;

        // Layout positions: each icon gets its own non-overlapping slot
        const positions: { x: number; y: number }[] = [];
        if (useTextures) {
          const gap = iconSize * 1.25 + 3;
          const badgeY = sy - sz * (tile.isCapital ? 0.28 : 0.58);
          for (let i = 0; i < count; i++) {
            positions.push({ x: sx + (i - (count - 1) / 2) * gap, y: badgeY });
          }
        } else if (count === 1) {
          positions.push({ x: sx, y: sy });
        } else if (count === 2) {
          const gap = iconSize * 0.75;
          positions.push({ x: sx - gap, y: sy }, { x: sx + gap, y: sy });
        } else {
          // 2×2 grid centred on the hex
          const gap = iconSize * 0.72;
          positions.push(
            { x: sx - gap, y: sy - gap },
            { x: sx + gap, y: sy - gap },
            { x: sx - gap, y: sy + gap },
            { x: sx + gap, y: sy + gap },
          );
        }

        for (let i = 0; i < count; i++) {
          const def = getBuildingIcon(iconsToDraw[i]);
          const tex = this.iconTextures.get(def.icon);
          const ix = positions[i].x;
          const iy = positions[i].y;
          const tintColor = parseInt(def.color.slice(1), 16);

          // Rounded-rect backing for contrast (dark pill with border)
          roadGfx.roundRect(ix - r - 1.5, iy - r - 1.5, (r + 1.5) * 2, (r + 1.5) * 2, 4)
            .fill({ color: 0x000000, alpha: 0.7 });
          roadGfx.roundRect(ix - r, iy - r, r * 2, r * 2, 3)
            .fill({ color: 0x111827, alpha: 0.95 });

          if (tex) {
            let sprite: Sprite;
            if (spriteIdx < this.spritePool.length) {
              sprite = this.spritePool[spriteIdx];
            } else {
              sprite = new Sprite();
              sprite.anchor.set(0.5, 0.5);
              // Snap to whole pixels: half-pixel placement is the main
              // source of blur on small tinted icons
              sprite.roundPixels = true;
              this.spritePool.push(sprite);
            }
            sprite.texture = tex;
            sprite.x = ix;
            sprite.y = iy;
            sprite.width = iconSize * 0.85;
            sprite.height = iconSize * 0.85;
            sprite.tint = tintColor;
            sprite.alpha = 1.0;
            this.iconContainer.addChild(sprite);
            spriteIdx++;
          } else {
            roadGfx.circle(ix, iy, r * 0.6).fill({ color: tintColor, alpha: 0.9 });
          }
        }
      }
    }

    // Draw city name labels at higher zoom levels
    this.drawCityLabels(offsetX, offsetY, z);
    this.drawNationLabels(offsetX, offsetY, z);

    // Draw MW labels on power hexes in electricity mode
    this.drawPowerLabels(offsetX, offsetY, z);
    this.drawWarLabels(offsetX, offsetY, z);

    // Draw overlay (hover + selection)
    this.drawOverlay(offsetX, offsetY, z);

    this.lastRedrawMs = performance.now() - perfStart;
  }

  /**
   * Deposit dots, plus the accessibility label under each one.
   *
   * Dots are bucketed by resource and richness so a screen full of deposits
   * costs one fill instruction per bucket (at most 12 resources x richness
   * steps) instead of one per hex. In colourblind mode each resource also
   * gets its own shape, so hue is never the only channel.
   */
  private drawResourceDots(
    gfx: Graphics,
    dots: { sx: number; sy: number; resource: string; richness: number }[],
    sz: number,
    z: number,
  ) {
    const cbMode = this.a11y.colorblind;
    const cb = cbMode !== 'off';
    const showLabels = (this.a11y.resourceLabels || cb)
      && dots.length > 0 && z >= RESOURCE_LABEL_MIN_ZOOM;

    // Labels are pooled Text; hide the pool up front and reveal what we use
    this.resourceLabelContainer.visible = showLabels;

    if (dots.length === 0) {
      this.hideResourceLabelsFrom(0);
      return;
    }

    const palette = resourceMarkerColors(cbMode);
    const dotR = Math.max(1.5, sz * 0.15);

    // Bucket key: resource + richness (alpha follows richness)
    const buckets = new Map<string, { resource: string; richness: number; items: number[] }>();
    for (const d of dots) {
      const key = `${d.resource}|${d.richness}`;
      let b = buckets.get(key);
      if (!b) { b = { resource: d.resource, richness: d.richness, items: [] }; buckets.set(key, b); }
      b.items.push(d.sx, d.sy);
    }

    for (const b of buckets.values()) {
      const shape = resourceDotShape(b.resource, cb);
      const color = palette?.[b.resource] ?? RESOURCE_COLORS[b.resource] ?? 0xffffff;
      for (let i = 0; i < b.items.length; i += 2) {
        this.addDotShape(gfx, b.items[i], b.items[i + 1], dotR, shape);
      }
      gfx.fill({ color, alpha: richnessAlpha(b.richness) });
      // A thin dark keyline keeps light dots (limestone, arctic) readable
      if (cb) {
        for (let i = 0; i < b.items.length; i += 2) {
          this.addDotShape(gfx, b.items[i], b.items[i + 1], dotR, shape);
        }
        gfx.stroke({ color: 0x000000, width: Math.max(0.4, dotR * 0.22), alpha: 0.55 });
      }
    }

    if (!showLabels) {
      this.hideResourceLabelsFrom(0);
      return;
    }

    // Text codes under each dot, capped so a wide screen cannot spawn an
    // unbounded number of sprites. Same LOD gate as the dots themselves.
    const scale = (sz * 0.42) / RESOURCE_LABEL_BASE_FONT;
    let used = 0;
    for (const d of dots) {
      if (used >= RESOURCE_LABEL_MAX) break;
      const code = RESOURCE_CODES[d.resource];
      if (!code) continue;   // unknown deposit: never invent a code
      const tex = this.getCodeTexture(code);
      if (!tex) continue;
      const label = this.acquireResourceLabel(used);
      if (label.texture !== tex) label.texture = tex;
      label.scale.set(scale);
      label.x = d.sx;
      label.y = d.sy + dotR + Math.max(1, sz * 0.06);
      label.visible = true;
      used++;
    }
    this.hideResourceLabelsFrom(used);
  }

  /** One shared texture per deposit code, rasterised on first use. */
  private getCodeTexture(code: string): Texture | null {
    const cached = this.resourceCodeTextures.get(code);
    if (cached !== undefined) return cached;

    let tex: Texture | null = null;
    try {
      const text = new Text({
        text: code,
        style: {
          fontSize: RESOURCE_LABEL_BASE_FONT,
          fontFamily: mapFont(),
          fontWeight: 'bold',
          fill: 0xffffff,
          stroke: { color: 0x000000, width: 3, join: 'round' },
        } as Partial<TextStyle>,
      });
      // Cut at 4x, not 2x. This one texture is magnified rather than
      // re-cut: the sprite is scaled by (sz * 0.42) / 16, which reaches
      // about 3x at full zoom, so a 2x cut was being blown up past its
      // own resolution. There are only ~22 codes and each is a few
      // characters, so the extra memory is negligible.
      tex = this.app.renderer.generateTexture({ target: text, resolution: 4 });
      text.destroy();
    } catch {
      tex = null;   // renderer not ready: skip labels rather than break the map
    }
    this.resourceCodeTextures.set(code, tex);
    return tex;
  }

  /** One dot outline added to the current path (filled/stroked by caller). */
  private addDotShape(
    gfx: Graphics, cx: number, cy: number, r: number, shape: ResourceDotShape,
  ) {
    switch (shape) {
      case 'square':
        gfx.rect(cx - r * 0.88, cy - r * 0.88, r * 1.76, r * 1.76);
        break;
      case 'triangle': {
        const h = r * 1.15;
        gfx.poly([cx, cy - h, cx + h * 0.95, cy + h * 0.7, cx - h * 0.95, cy + h * 0.7]);
        break;
      }
      case 'diamond':
        gfx.poly([cx, cy - r * 1.25, cx + r * 1.05, cy, cx, cy + r * 1.25, cx - r * 1.05, cy]);
        break;
      case 'hexagon':
        gfx.poly(hexPts(cx, cy, r * 1.1));
        break;
      default:
        gfx.circle(cx, cy, r);
    }
  }

  private acquireResourceLabel(index: number): Sprite {
    let label = this.resourceLabelPool[index];
    if (!label) {
      label = new Sprite();
      label.anchor.set(0.5, 0);
      label.roundPixels = true;
      this.resourceLabelContainer.addChild(label);
      this.resourceLabelPool[index] = label;
    }
    return label;
  }

  private hideResourceLabelsFrom(index: number) {
    for (let i = index; i < this.resourceLabelPool.length; i++) {
      this.resourceLabelPool[i].visible = false;
    }
  }

  private clearResourceLabels() {
    for (const label of this.resourceLabelPool) label.destroy();
    this.resourceLabelPool = [];
    this.resourceLabelContainer.removeChildren();
    for (const tex of this.resourceCodeTextures.values()) tex?.destroy(true);
    this.resourceCodeTextures.clear();
  }

  /** Diagonal hatch lines marking a hex in anarchy. */
  private drawAnarchyHatch(
    gfx: Graphics, cx: number, cy: number, sz: number,
    color = 0x000000, alpha = 0.4,
  ) {
    const dirx = 0.707, diry = -0.707;   // 45° stroke direction
    const perpx = 0.707, perpy = 0.707;
    const len = sz * 0.52;
    const width = Math.max(0.8, sz * 0.08);
    for (const o of [-sz * 0.45, 0, sz * 0.45]) {
      const mx = cx + perpx * o;
      const my = cy + perpy * o;
      const l = len * (1 - Math.abs(o) / (sz * 1.4));
      gfx.moveTo(mx - dirx * l, my - diry * l)
        .lineTo(mx + dirx * l, my + diry * l)
        .stroke({ color, width, alpha, cap: 'round' });
    }
  }

  /** Deterministic per-hex brightness jitter (±6%) to break up flat color. */
  private varyColor(color: number, q: number, r: number): number {
    let h = (q * 374761393 + r * 668265263) | 0;
    h = (h ^ (h >> 13)) * 1274126177;
    const t = ((h >>> 0) % 1000) / 1000;
    const f = 0.94 + t * 0.12;
    const rr = Math.min(255, Math.round(((color >> 16) & 0xff) * f));
    const gg = Math.min(255, Math.round(((color >> 8) & 0xff) * f));
    const bb = Math.min(255, Math.round((color & 0xff) * f));
    return (rr << 16) | (gg << 8) | bb;
  }

  private drawStar(gfx: Graphics, cx: number, cy: number, size: number, color: number) {
    const pts: number[] = [];
    for (let i = 0; i < 5; i++) {
      const outerAngle = (i * 72 - 90) * Math.PI / 180;
      const innerAngle = ((i * 72 + 36) - 90) * Math.PI / 180;
      pts.push(cx + Math.cos(outerAngle) * size, cy + Math.sin(outerAngle) * size);
      pts.push(cx + Math.cos(innerAngle) * size * 0.45, cy + Math.sin(innerAngle) * size * 0.45);
    }
    gfx.poly(pts).fill({ color, alpha: 0.95 }).stroke({ color: 0x000000, width: 0.5, alpha: 0.5 });
  }

  /** The shared edge of two adjacent hexes: the segment between the two
   *  corners they have in common. */
  private sharedEdge(aq: number, ar: number, bq: number, br: number) {
    // River geometry stays fixed while the camera and animation move.
    const key = `${aq},${ar},${bq},${br}`;
    if (this.riverGeometry.has(key)) return this.riverGeometry.get(key)!;
    const [ax, ay] = axialToPixel(aq, ar, HEX_SIZE);
    const [bx, by] = axialToPixel(bq, br, HEX_SIZE);
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i;
      const cx = ax + HEX_SIZE * Math.cos(angle);
      const cy = ay + HEX_SIZE * Math.sin(angle);
      for (let j = 0; j < 6; j++) {
        const bangle = (Math.PI / 3) * j;
        const dx = bx + HEX_SIZE * Math.cos(bangle) - cx;
        const dy = by + HEX_SIZE * Math.sin(bangle) - cy;
        if (dx * dx + dy * dy < 0.01) pts.push({ x: cx, y: cy });
      }
    }
    const edge = pts.length === 2 ? pts : null;
    this.riverGeometry.set(key, edge);
    return edge;
  }

  /** Water running downstream: short highlights sliding along each river,
   *  redrawn every frame on their own layer. Pixi has no dashed stroke, so
   *  the dashes are drawn as segments, which is also what lets them move.
   *  Every hex edge is the same length, so one phase keeps the whole
   *  network in step without tracking distance per edge. */
  private drawFlow() {
    const g = this.flowGfx;
    g.clear();
    if (!this.riverEdges.length) return;
    const z = this.zoom;
    if (z < 0.45) return;               // invisible at this size anyway

    const sz = HEX_SIZE * z;
    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const margin = sz * 3;
    // The same camera every other pass bakes in. worldContainer is never
    // translated, so reading its x/y gave zero and the water sat still in
    // screen space while the map moved under it.
    const offsetX = w / 2 - this.camX * z;
    const offsetY = h / 2 - this.camY * z;

    const DASH = 0.30;                  // fraction of an edge
    const GAP = 0.70;
    const phase = (this.flowMs / 2600) % 1;

    for (const e of this.riverEdges) {
      const raw = this.sharedEdge(e[0], e[1], e[2], e[3]);
      if (!raw) continue;
      // Downstream end first, so every segment of one river runs the same
      // way. The generator says whether the downstream corner is the
      // lexicographically smaller of the two; sorting here reproduces the
      // ordering it used, because the two coordinate spaces differ only by
      // a positive scale.
      const sorted = raw[0].x < raw[1].x
        || (raw[0].x === raw[1].x && raw[0].y <= raw[1].y)
        ? raw : [raw[1], raw[0]];
      const downIsMin = e[5] === 1;
      const from = downIsMin ? sorted[1] : sorted[0];
      const to = downIsMin ? sorted[0] : sorted[1];
      const x0 = from.x * z + offsetX, y0 = from.y * z + offsetY;
      const x1 = to.x * z + offsetX, y1 = to.y * z + offsetY;
      if ((x0 < -margin || x0 > w + margin || y0 < -margin || y0 > h + margin)
        && (x1 < -margin || x1 > w + margin || y1 < -margin || y1 > h + margin)) {
        continue;
      }
      const flow = e[4];
      const width = Math.max(0.4, sz * (0.030 + 0.048 * Math.log2(flow / 80 + 1)) * 0.5);

      // Two dashes per edge, a period apart, so the run never breaks at a
      // junction however the two edges happen to be oriented.
      for (const base of [0, DASH + GAP]) {
        let t0 = base + phase * (DASH + GAP) - (DASH + GAP);
        while (t0 < 1) {
          const a = Math.max(0, t0);
          const b2 = Math.min(1, t0 + DASH);
          if (b2 > a) {
            g.moveTo(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)
              .lineTo(x0 + (x1 - x0) * b2, y0 + (y1 - y0) * b2)
              .stroke({
                color: flow >= this.riverMajorFlow ? 0xbfe6ff : 0xa8d8f0,
                width, alpha: 0.75, cap: 'round',
              });
          }
          t0 += DASH + GAP;
        }
      }
    }
  }

  /** Rivers, drawn along the edge between the two hexes they separate.
   *  Width comes from the flow the generator measured, so the trunk is
   *  wide and the headwaters are hairlines. */
  private drawRivers(gfx: Graphics, offsetX: number, offsetY: number, z: number) {
    if (!this.riverEdges.length) return;
    const sz = HEX_SIZE * z;
    // Culled the way every other pass is: a 150x150 map carries about
    // 1,500 river edges and only a handful are ever on screen.
    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const margin = sz * 3;
    const onScreen = (x: number, y: number) =>
      x >= -margin && x <= w + margin && y >= -margin && y <= h + margin;

    // The shared edge of two adjacent hexes is the segment between the two
    // corners they have in common, which is what makes a river a border
    // rather than a stripe down the middle of a tile.
    const shared = (aq: number, ar: number, bq: number, br: number) =>
      this.sharedEdge(aq, ar, bq, br);

    const line = (
      e: [number, number, number, number, number, number],
      color: number, widthMult: number, alpha: number,
    ) => {
      const pts = shared(e[0], e[1], e[2], e[3]);
      if (!pts) return;
      const x0 = pts[0].x * z + offsetX, y0 = pts[0].y * z + offsetY;
      const x1 = pts[1].x * z + offsetX, y1 = pts[1].y * z + offsetY;
      if (!onScreen(x0, y0) && !onScreen(x1, y1)) return;
      const flow = e[4];
      const width = Math.max(
        0.5,
        sz * (0.030 + 0.048 * Math.log2(flow / 80 + 1)) * widthMult,
      );
      gfx.moveTo(x0, y0).lineTo(x1, y1)
        .stroke({ color: color, width: width, alpha, cap: 'round', join: 'round' });
    };

    // Casing first so the water sits in a cut channel, then the water.
    for (const e of this.riverEdges) line(e, 0x0b2534, 1.7, 0.75);
    for (const e of this.riverEdges) {
      line(e, e[4] >= this.riverMajorFlow ? 0x3f86b6 : 0x3d7ea8, 1, 0.98);
    }
  }

  /** A deck wherever a route crosses water.
   *
   *  Rivers run along hex edges and routes run between hex centres, so a
   *  route crosses exactly the edge shared by the two hexes it joins. That
   *  turns the crossing test into a key lookup instead of a segment sweep
   *  over every river on the map. */
  private drawBridges(gfx: Graphics, offsetX: number, offsetY: number, z: number) {
    if (!this.riverEdges.length || z < 0.5) return;
    const sz = HEX_SIZE * z;
    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const margin = sz * 3;

    for (const e of this.riverEdges) {
      const [aq, ar, bq, br, flow] = e;
      const [ax, ay] = axialToPixel(aq, ar, HEX_SIZE);
      const [bx, by] = axialToPixel(bq, br, HEX_SIZE);
      const mx = (ax + bx) / 2 * z + offsetX, my = (ay + by) / 2 * z + offsetY;
      if (mx < -margin || mx > w + margin || my < -margin || my > h + margin) continue;

      const forward = `${hexKey(aq, ar)}_${hexKey(bq, br)}`;
      const back = `${hexKey(bq, br)}_${hexKey(aq, ar)}`;
      for (const kind of ['road', 'rail'] as const) {
        const path = this.transport.get(forward, kind) ?? this.transport.get(back, kind);
        if (!path) continue;

        // Where the fitted route actually meets the shared boundary, which
        // on a bend is not the midpoint between the two hex centres.
        const side = (p: { x: number; y: number }) =>
          (p.x - (ax + bx) / 2) * (bx - ax) + (p.y - (ay + by) / 2) * (by - ay);
        let cx = (ax + bx) / 2, cy = (ay + by) / 2;
        let angle = Math.atan2(by - ay, bx - ax);
        for (let i = 1; i < path.points.length; i++) {
          const p = path.points[i - 1], q = path.points[i];
          const u = side(p), v = side(q);
          if (u <= 0 && v >= 0) {
            const t = -u / (v - u || 1);
            cx = p.x + (q.x - p.x) * t;
            cy = p.y + (q.y - p.y) * t;
            angle = Math.atan2(q.y - p.y, q.x - p.x);
            break;
          }
        }
        const x = cx * z + offsetX, y = cy * z + offsetY;

        // Long enough to land on both banks: the river's own width plus an
        // abutment either side.
        const water = Math.max(0.5, sz * (0.030 + 0.048 * Math.log2(flow / 80 + 1)) * 1.7);
        const half = water * 0.5 + Math.max(1.2, sz * 0.05);
        const rail = kind === 'rail';
        const deck = rail
          ? Math.max(1.4, z * 1.7)
          : Math.max(2.2, z * 2.0) + Math.max(0.6, z * 0.65);

        const c = Math.cos(angle), s = Math.sin(angle);
        const at = (u: number, v: number): [number, number] =>
          [x + u * c - v * s, y + u * s + v * c];

        // Shadow on the water, then the deck, then a parapet down each side.
        gfx.poly([...at(-half, -deck * 0.5 + 0.8), ...at(half, -deck * 0.5 + 0.8),
                  ...at(half, deck * 0.5 + 1.4), ...at(-half, deck * 0.5 + 1.4)])
          .fill({ color: 0x05131c, alpha: 0.45 });
        gfx.poly([...at(-half, -deck * 0.5), ...at(half, -deck * 0.5),
                  ...at(half, deck * 0.5), ...at(-half, deck * 0.5)])
          .fill({ color: rail ? 0x5a5148 : 0x8b8d86 });
        for (const v of [-deck * 0.5, deck * 0.5]) {
          gfx.moveTo(...at(-half, v)).lineTo(...at(half, v))
            .stroke({ color: 0xc8c5ba, width: Math.max(0.35, z * 0.2), alpha: 0.7 });
        }
        // Piers read as a bridge rather than a paved ford once close in.
        if (z > 1.1) for (const u of [-half * 0.55, half * 0.55]) {
          gfx.moveTo(...at(u, -deck * 0.5)).lineTo(...at(u, deck * 0.5))
            .stroke({ color: 0x3f3a34, width: Math.max(0.3, z * 0.22), alpha: 0.55 });
        }
      }
    }
  }

  /** Draw connected road curves, railway tracks and overhead power. */
  private drawNetworkLines(gfx: Graphics, offsetX: number, offsetY: number, z: number) {
    if (z < 0.3) {
      this.checkpoints.setView([], this.tiles, this.nations, offsetX, offsetY, z, this.app.screen.width, this.app.screen.height, false);
      this.traffic.setView([], offsetX, offsetY, z, this.app.screen.width, this.app.screen.height, false, false);
      return;
    }

    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const margin = HEX_SIZE * z * 3;

    const roadWidth = Math.max(1.5, z * 2.0);
    const railWidth = Math.max(0.6, z * 0.7);
    const tieLength = Math.max(1.2, z * 2.0);

    // ── Pass 1: collect edges from stored network_edges ────────────
    type Edge = { sx: number; sy: number; nx: number; ny: number; sq: number; sr: number; nq: number; nr: number };
    type PwEdge = Edge & { hv: boolean };
    const roadEdges: Edge[] = [];
    const railEdges: Edge[] = [];
    const pwEdges: PwEdge[] = [];

    // Per-edge tracking of which infrastructure types are present
    const edgeTypes = new Map<string, { road: boolean; rail: boolean; power: boolean }>();

    const roadHexScreenPos = new Map<string, { x: number; y: number; count: number }>();
    const railHexScreenPos = new Map<string, { x: number; y: number; count: number }>();
    const pwHexPos = new Map<string, { x: number; y: number; count: number; hv: boolean }>();

    // Screen position cache
    const screenPosCache = new Map<string, { x: number; y: number } | null>();
    const getScreenPos = (key: string): { x: number; y: number } | null => {
      if (screenPosCache.has(key)) return screenPosCache.get(key)!;
      const tile = this.tiles.get(key);
      if (!tile) { screenPosCache.set(key, null); return null; }
      const [wx, wy] = axialToPixel(tile.q, tile.r, HEX_SIZE);
      const pos = { x: wx * z + offsetX, y: wy * z + offsetY };
      screenPosCache.set(key, pos);
      return pos;
    };

    for (const [edgeKey, types] of this.networkEdges) {
      const [kA, kB] = edgeKey.split('_');
      const posA = getScreenPos(kA);
      const posB = getScreenPos(kB);
      if (!posA || !posB) continue;

      // Viewport cull
      if (posA.x < -margin && posB.x < -margin) continue;
      if (posA.x > w + margin && posB.x > w + margin) continue;
      if (posA.y < -margin && posB.y < -margin) continue;
      if (posA.y > h + margin && posB.y > h + margin) continue;

      const tileA = this.tiles.get(kA)!;
      const tileB = this.tiles.get(kB)!;

      // Ensure edgeTypes entry exists
      if (!edgeTypes.has(edgeKey)) {
        edgeTypes.set(edgeKey, { road: false, rail: false, power: false });
      }
      const et = edgeTypes.get(edgeKey)!;

      const typeArr = types.split(',');
      for (const ntype of typeArr) {
        if ((ntype === 'road' || ntype === 'dirt') && this._showRoads) {
          roadEdges.push({ sx: posA.x, sy: posA.y, nx: posB.x, ny: posB.y, sq: tileA.q, sr: tileA.r, nq: tileB.q, nr: tileB.r });
          et.road = true;
          const e1 = roadHexScreenPos.get(kA);
          if (e1) e1.count++; else roadHexScreenPos.set(kA, { x: posA.x, y: posA.y, count: 1 });
          const e2 = roadHexScreenPos.get(kB);
          if (e2) e2.count++; else roadHexScreenPos.set(kB, { x: posB.x, y: posB.y, count: 1 });
        } else if (ntype === 'rail' && this._showRoads) {
          railEdges.push({ sx: posA.x, sy: posA.y, nx: posB.x, ny: posB.y, sq: tileA.q, sr: tileA.r, nq: tileB.q, nr: tileB.r });
          et.rail = true;
          const e1 = railHexScreenPos.get(kA);
          if (e1) e1.count++; else railHexScreenPos.set(kA, { x: posA.x, y: posA.y, count: 1 });
          const e2 = railHexScreenPos.get(kB);
          if (e2) e2.count++; else railHexScreenPos.set(kB, { x: posB.x, y: posB.y, count: 1 });
        } else if (ntype === 'power' && this._showPower) {
          const aHasHV = this.hvHexes.has(kA);
          const bHasHV = this.hvHexes.has(kB);
          const edgeHV = aHasHV && bHasHV;
          pwEdges.push({ sx: posA.x, sy: posA.y, nx: posB.x, ny: posB.y, sq: tileA.q, sr: tileA.r, nq: tileB.q, nr: tileB.r, hv: edgeHV });
          et.power = true;
          const e1 = pwHexPos.get(kA);
          if (e1) { e1.count++; e1.hv = e1.hv || aHasHV; }
          else pwHexPos.set(kA, { x: posA.x, y: posA.y, count: 1, hv: aHasHV });
          const e2 = pwHexPos.get(kB);
          if (e2) { e2.count++; e2.hv = e2.hv || bHasHV; }
          else pwHexPos.set(kB, { x: posB.x, y: posB.y, count: 1, hv: bHasHV });
        }
      }
    }

    const visibleTransport: TransportPath[] = [];
    const screenPoints = (path: TransportPath) => path.points
      .filter((_, i) => z >= .85 || i % 8 === 0)
      .map(p => ({ x: p.x * z + offsetX, y: p.y * z + offsetY }));
    const trace = (points: { x: number; y: number }[]) => {
      gfx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) gfx.lineTo(points[i].x, points[i].y);
    };
    // Paint every shoulder first, then every surface. Branches merge into a
    // single intersection instead of one segment's dark outline cutting it.
    for (const pass of [0, 1]) for (const e of roadEdges) {
      const path = this.transport.get(`${hexKey(e.sq, e.sr)}_${hexKey(e.nq, e.nr)}`, 'road');
      if (!path) continue;
      if (pass === 0) visibleTransport.push(path);
      const pts = screenPoints(path);
      const dirt = path.surface === 'dirt', busy = !dirt && path.activity > .55;
      const width = roadWidth * (dirt ? .72 : busy ? 1.2 : 1);
      if (pass === 0) {
        trace(pts);
        gfx.stroke({ color: dirt ? 0x8d7955 : 0x353b3d, width: width + Math.max(.6, z * .65), alpha: dirt ? .6 : .95, cap: 'round', join: 'round' });
        continue;
      }
      trace(pts);
      gfx.stroke({ color: dirt ? 0xb99a6a : busy ? 0x69717a : 0x73787b, width, alpha: .98, cap: 'round', join: 'round' });
      if (dirt && z > 1.1) for (const lane of [-.4, .4]) {
        trace(path.lengths.map(d => {
          const p = samplePath(path, d, false, lane);
          return { x: p.x * z + offsetX, y: p.y * z + offsetY };
        }));
        gfx.stroke({ color: 0x887148, width: z * .22, alpha: .65 });
      }
      // Distance-based markings stay even through the bends.
      if (z > 1.1 && busy) {
        const start = (this.transport.adjacent.get(`${path.a}:road`)?.length ?? 0) > 2 ? 4 : 0;
        const end = (this.transport.adjacent.get(`${path.b}:road`)?.length ?? 0) > 2 ? path.length - 4 : path.length;
        for (let d = start + 1; d < end - 1; d += 4) {
          const a = samplePath(path, d), b = samplePath(path, Math.min(d + 1.8, path.length - 1));
          gfx.moveTo(a.x * z + offsetX, a.y * z + offsetY).lineTo(b.x * z + offsetX, b.y * z + offsetY);
        }
        gfx.stroke({ color: 0xefdcaa, width: Math.max(.3, z * .25), alpha: .8 });
      }
    }

    // Junction mouths use the fitted arms, including their actual lane offset.
    for (const [key, pos] of roadHexScreenPos) {
      if (pos.count >= 3) {
        const paths = this.transport.adjacent.get(`${key}:road`) ?? [];
        if (!paths.length) continue;
        const anchor = paths[0].a === key ? paths[0].points[0] : paths[0].points.at(-1)!;
        const dirt = paths.every(p => p.surface === 'dirt'), busy = paths.some(p => p.activity > .55);
        const arms = paths.map(path => ({ path, p: samplePath(path, 3, path.b === key) }));
        const outline = arms.flatMap(({ path, p }) => [-1, 1].map(side => {
          const half = path.surface === 'dirt' ? .72 : path.activity > .55 ? 1.2 : 1;
          return { x: p.x - Math.sin(p.angle) * half * side, y: p.y + Math.cos(p.angle) * half * side };
        })).sort((a, b) => Math.atan2(a.y - anchor.y, a.x - anchor.x) - Math.atan2(b.y - anchor.y, b.x - anchor.x));
        gfx.poly(outline.flatMap(p => [p.x * z + offsetX, p.y * z + offsetY])).fill(dirt ? 0xb99a6a : busy ? 0x69717a : 0x73787b);
        if (!dirt && z > 1.2) {
          let through: number[] = [], best = -.75;
          for (let i = 0; i < arms.length; i++) for (let j = i + 1; j < arms.length; j++) {
            const score = Math.cos(arms[i].p.angle - arms[j].p.angle);
            if (score < best) { best = score; through = [i, j]; }
          }
          arms.forEach(({ path }, i) => {
            if (through.includes(i) || path.surface === 'dirt') return;
            const p = samplePath(path, 4, path.b === key), nx = -Math.sin(p.angle), ny = Math.cos(p.angle);
            gfx.moveTo((p.x - nx) * z + offsetX, (p.y - ny) * z + offsetY)
              .lineTo((p.x + nx) * z + offsetX, (p.y + ny) * z + offsetY)
              .stroke({ color: 0xeee5ce, width: z * .3 });
          });
        }
      }
    }

    // ── Pass 3: draw rails (tapered lane polylines) ──────────────
    for (const e of railEdges) {
      const path = this.transport.get(`${hexKey(e.sq, e.sr)}_${hexKey(e.nq, e.nr)}`, 'rail');
      if (!path) continue;
      visibleTransport.push(path);
      trace(screenPoints(path));
      gfx.stroke({ color: 0x8c8170, width: tieLength * 2.4, alpha: .5, cap: 'round', join: 'round' });
      if (z < .85) {
        trace(screenPoints(path));
        gfx.stroke({ color: 0xc3c7ca, width: railWidth, alpha: .9 });
        continue;
      }
      for (let d = 1.5; d < path.length; d += 3.5) {
        const p = samplePath(path, d);
        const px = -Math.sin(p.angle) * tieLength, py = Math.cos(p.angle) * tieLength;
        const x = p.x * z + offsetX, y = p.y * z + offsetY;
        gfx.moveTo(x - px, y - py).lineTo(x + px, y + py);
      }
      gfx.stroke({ color: 0x4c3928, width: Math.max(.4, z * .55), alpha: .95 });
      for (const side of [-1, 1]) {
        const points = path.lengths.map(d => {
          const p = samplePath(path, d, false, side * .8, false);
          return { x: p.x * z + offsetX, y: p.y * z + offsetY };
        });
        trace(points);
        gfx.stroke({ color: 0xbec8cc, width: railWidth * .55, alpha: .9, join: 'round' });
      }
    }
    this.traffic.setView(visibleTransport, offsetX, offsetY, z, w, h,
      this._showRoads && this._showTraffic, this.season === 'winter');
    this.checkpoints.setView(this.crossings, this.tiles, this.nations, offsetX, offsetY, z, w, h, this._showRoads);

    // Rail junction dots
    for (const [, pos] of railHexScreenPos) {
      if (pos.count >= 2) {
        gfx.circle(pos.x, pos.y, railWidth * 0.9)
          .fill({ color: 0x6b7280, alpha: 0.75 });
      }
    }

    // ── Pass 4: draw power lines (cables + node dots, no pylons) ──
    // Regular power line: thin grey wire with a gentle droop, anchored by a dot
    // High voltage line: three bold amber cables anchored by a larger dot

    // Helper: draw a subtle catenary (sagging curve) between two points
    const drawCatenary = (x1: number, y1: number, x2: number, y2: number, sag: number, color: number, width: number, alpha: number) => {
      const segments = 8;
      gfx.moveTo(x1, y1);
      for (let i = 1; i <= segments; i++) {
        const t = i / segments;
        const mx = x1 + (x2 - x1) * t;
        const my = y1 + (y2 - y1) * t;
        // Parabolic sag: max at center (t=0.5)
        const sagAmount = sag * 4 * t * (1 - t);
        gfx.lineTo(mx, my + sagAmount);
      }
      gfx.stroke({ color, width, alpha });
    };

    for (const e of pwEdges) {
      // Power runs center-to-center: it is aerial, so it crosses roads and
      // rails overhead rather than claiming a ground lane. Slightly lower
      // alpha where the edge also carries a road keeps the asphalt readable.
      const x1 = e.sx, y1 = e.sy, x2 = e.nx, y2 = e.ny;
      const edgeId = `${hexKey(e.sq, e.sr)}_${hexKey(e.nq, e.nr)}`;
      const et = edgeTypes.get(edgeId);
      const overGround = !!(et && (et.road || et.rail));
      const dim = overGround ? 0.75 : 1.0;

      const dx = x2 - x1, dy = y2 - y1;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len < 1) continue;
      const ux = dx / len, uy = dy / len;
      const ppx = -uy, ppy = ux;

      if (e.hv) {
        // ── High-voltage line: bold amber cables with catenary droop ──
        const hvGap = Math.max(1.5, z * 2.5);
        const hvColor = 0xf59e0b;
        const hvLineW = Math.max(0.6, z * 0.8);
        const sagAmt = Math.max(1.5, z * 3);

        // Three catenary wires (left, center, right)
        drawCatenary(x1 - ppx * hvGap, y1 - ppy * hvGap, x2 - ppx * hvGap, y2 - ppy * hvGap, sagAmt, hvColor, hvLineW, 0.85 * dim);
        drawCatenary(x1, y1, x2, y2, sagAmt * 0.6, hvColor, hvLineW * 0.7, 0.6 * dim);
        drawCatenary(x1 + ppx * hvGap, y1 + ppy * hvGap, x2 + ppx * hvGap, y2 + ppy * hvGap, sagAmt, hvColor, hvLineW, 0.85 * dim);

      } else {
        // ── Regular power line: thin grey single wire with gentle droop ──
        const pwLineW = Math.max(0.4, z * 0.5);
        const sagAmt = Math.max(1, z * 2);

        drawCatenary(x1, y1, x2, y2, sagAmt, 0x9ca3af, pwLineW, 0.75 * dim);
      }
    }

    // Node dots at hex centers, replacing the old pylons. The cables meet
    // center-to-center, so the dot sits exactly where they anchor — the same
    // way road/rail junction dots sit on their line endpoints, which lets the
    // grid read as one network layer. Tinted to the line colour (amber for HV,
    // grey for distribution) and cased with the road casing tone so it blends
    // with the asphalt rather than floating over it.
    const pwCaseW = Math.max(0.4, z * 0.5);
    for (const [hexId, pos] of pwHexPos) {
      const overGround = roadHexScreenPos.has(hexId) || railHexScreenPos.has(hexId);
      const dim = overGround ? 0.8 : 1.0;
      const dotColor = pos.hv ? 0xf59e0b : 0x9ca3af;
      const r = pos.hv ? Math.max(1.0, z * 1.4) : Math.max(0.8, z * 1.1);
      gfx.circle(pos.x, pos.y, r)
        .fill({ color: dotColor, alpha: 0.95 * dim })
        .stroke({ color: 0x1a1d23, width: pwCaseW, alpha: 0.7 * dim });
    }
  }

  private clearLabels() {
    for (const label of this.cityLabels.values()) {
      label.destroy();
    }
    this.cityLabels.clear();
    this.labelContainer.removeChildren();
    this.clearPowerLabels();
  }

  private drawCityLabels(offsetX: number, offsetY: number, z: number) {
    // Only show labels above a certain zoom (and when toggled on)
    const showLabels = this._showLabels && z > 0.5;
    this.labelContainer.visible = showLabels;
    if (!showLabels) return;

    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const margin = HEX_SIZE * z * 3;
    const fontSize = Math.max(8, Math.min(14, 10 * z));

    // Rebuild labels if zoom changed significantly
    const zoomChanged = Math.abs(z - this.lastLabelZoom) > 0.05;
    if (zoomChanged) {
      this.clearLabels();
      this.lastLabelZoom = z;
    }

    for (const tile of this.tiles.values()) {
      if (!tile.cityName) continue;

      const [wx, wy] = axialToPixel(tile.q, tile.r, HEX_SIZE);
      const sx = wx * z + offsetX;
      const sy = wy * z + offsetY;

      if (sx < -margin || sx > w + margin || sy < -margin || sy > h + margin) continue;

      const key = hexKey(tile.q, tile.r);
      let label = this.cityLabels.get(key);

      if (!label) {
        const displayName = tile.isCapital ? `★ ${tile.cityName}` : tile.cityName;
        label = new Text({
          text: displayName,
          style: {
            fontSize,
            fontFamily: mapFont(),
            fill: tile.isCapital ? 0xffd700 : 0xffffff,
            fontWeight: tile.isCapital ? 'bold' : 'normal',
            dropShadow: { color: 0x000000, distance: 1, blur: 2, alpha: 1, angle: Math.PI / 4 },
          } as Partial<TextStyle>,
        });
        label.anchor.set(0.5, 0);
        this.labelContainer.addChild(label);
        this.cityLabels.set(key, label);
      }

      label.x = sx;
      const paintedMap = this._showTextures && (this.terrainTexturesLoaded || areTerrainTexturesLoaded())
        && (this.mode === 'terrain' || this.mode === 'political');
      label.y = sy + HEX_SIZE * z * (paintedMap ? 0.82 : 0.5);
    }
  }

  /** Where each nation's name goes, and which way it leans.
   *
   * The centre is the mean of the nation's hexes. The angle is the long
   * axis of that cloud of points, found the cheap way: the covariance of
   * the offsets has a closed-form principal axis, which for a territory
   * shaped like a country is the direction it sprawls in. The tilt is
   * then clamped, because a name standing on its head is not a nice
   * touch, it is a bug.
   */
  private measureNationLabels() {
    this.nationLabelGeometry.clear();
    const points = new Map<string, number[]>();
    for (const tile of this.tiles.values()) {
      if (!tile.owner) continue;
      const [wx, wy] = axialToPixel(tile.q, tile.r, HEX_SIZE);
      let list = points.get(tile.owner);
      if (!list) { list = []; points.set(tile.owner, list); }
      list.push(wx, wy);
    }

    for (const [owner, coords] of points) {
      const n = coords.length / 2;
      let cx = 0, cy = 0;
      for (let i = 0; i < coords.length; i += 2) { cx += coords[i]; cy += coords[i + 1]; }
      cx /= n; cy /= n;

      // Covariance of the hex positions about that centre.
      let sxx = 0, syy = 0, sxy = 0;
      for (let i = 0; i < coords.length; i += 2) {
        const dx = coords[i] - cx;
        const dy = coords[i + 1] - cy;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
      }
      sxx /= n; syy /= n; sxy /= n;

      // Principal axis. A perfectly round territory has no long axis and
      // the angle is meaningless, so leave those level.
      let angle = 0;
      if (Math.abs(sxy) > 1e-6 || Math.abs(sxx - syy) > 1e-6) {
        angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
      }

      // How elongated the territory actually is, from the eigenvalues of
      // that covariance. The tilt is then a fraction of the maximum rather
      // than the maximum itself: a hex field sprawls diagonally, so the raw
      // angle is steep even for a blob, and sixteen countries all leaning
      // at exactly the clamp reads as a rendering fault, not a flourish.
      const half = (sxx + syy) / 2;
      const spread = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy);
      const major = half + spread;
      const minor = Math.max(1e-6, half - spread);
      const elongation = Math.min(1, (major - minor) / (major + minor));

      // A name leaning past this reads as broken rather than as styled.
      const MAX_TILT = 0.28; // ~16 degrees
      angle = Math.max(-MAX_TILT, Math.min(MAX_TILT, angle)) * elongation;

      // How far the territory reaches along that axis, which is how much
      // room the name has before it hangs over someone else's land.
      const ux = Math.cos(angle), uy = Math.sin(angle);
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < coords.length; i += 2) {
        const t = (coords[i] - cx) * ux + (coords[i + 1] - cy) * uy;
        if (t < min) min = t;
        if (t > max) max = t;
      }
      const span = Math.max(HEX_SIZE * 2, (max - min) + HEX_SIZE * 2);
      this.nationLabelGeometry.set(owner, { cx, cy, angle, span, hexes: n });
    }
  }

  /** Country names across their territory, the way a strategy map does it.
   *
   * These are the zoomed-out layer: they appear as the city names go, so
   * the map always says something about who owns what without ever
   * carrying both sets of text at once.
   */
  private drawNationLabels(offsetX: number, offsetY: number, z: number) {
    const show = this._showLabels && z <= NATION_LABEL_MAX_ZOOM;
    this.nationLabelContainer.visible = show;
    if (!show) return;

    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const margin = HEX_SIZE * z * 6;

    // Text is rasterised at a size, so it is rebuilt in steps rather than
    // on every wheel notch.
    if (Math.abs(z - this.lastNationLabelZoom) > 0.02) {
      this.clearNationLabels();
      this.lastNationLabelZoom = z;
    }

    const seen = new Set<string>();
    for (const [owner, geo] of this.nationLabelGeometry) {
      const meta = this.nations.get(owner);
      const name = meta?.country;
      if (!name) continue;
      // One or two hexes cannot carry a name without covering themselves.
      if (geo.hexes < NATION_LABEL_MIN_HEXES) continue;

      const sx = geo.cx * z + offsetX;
      const sy = geo.cy * z + offsetY;
      if (sx < -margin || sx > w + margin || sy < -margin || sy > h + margin) continue;

      seen.add(owner);
      let label = this.nationLabels.get(owner);
      if (!label) {
        label = new Text({
          text: name.toUpperCase(),
          style: {
            fontSize: NATION_LABEL_BASE_FONT,
            fontFamily: mapFont(),
            fill: 0xf5f0e6,
            fontWeight: 'bold',
            letterSpacing: 2,
            stroke: { color: 0x000000, width: 3 },
            dropShadow: { color: 0x000000, distance: 1, blur: 3, alpha: 0.8, angle: Math.PI / 4 },
          } as Partial<TextStyle>,
          resolution: Math.min(2, window.devicePixelRatio || 1),
        });
        label.anchor.set(0.5);
        this.nationLabelContainer.addChild(label);
        this.nationLabels.set(owner, label);
      }

      label.x = sx;
      label.y = sy;
      label.rotation = geo.angle;

      // Fit the name to the land it names: as wide as the territory
      // allows, never so small it cannot be read, never so large it
      // swamps a neighbour.
      const target = geo.span * z * 0.82;
      // Measure against the base size whatever the label is currently set
      // to, or the fit would chase its own last result.
      const currentSize = (label.style.fontSize as number) || NATION_LABEL_BASE_FONT;
      const natural = label.width * (NATION_LABEL_BASE_FONT / currentSize);
      const fit = natural > 0 ? target / natural : 1;
      const scale = Math.max(NATION_LABEL_MIN_SCALE, Math.min(NATION_LABEL_MAX_SCALE, fit));

      // Draw at 1:1 and rebuild the glyphs at the size they are shown, so
      // a wide nation's name is sharp instead of an upscaled 16px bitmap.
      const wanted = Math.max(
        NATION_LABEL_FONT_STEP,
        Math.round(NATION_LABEL_BASE_FONT * scale / NATION_LABEL_FONT_STEP)
          * NATION_LABEL_FONT_STEP,
      );
      if (currentSize !== wanted) {
        const ratio = wanted / NATION_LABEL_BASE_FONT;
        label.style.fontSize = wanted;
        // Outline and tracking were tuned at the base size; hold them
        // proportional or a large label grows a hairline stroke.
        label.style.letterSpacing = 2 * ratio;
        label.style.stroke = { color: 0x000000, width: 3 * ratio };
      }
      label.scale.set(1);
      // Full strength across the zoomed-out half, then thinning as the
      // city names take over, so the handover is a crossfade rather than
      // a switch and neither layer ever shouts over the other.
      label.alpha = z <= NATION_LABEL_FULL_ZOOM
        ? 1
        : Math.max(0.3, (NATION_LABEL_MAX_ZOOM - z)
            / (NATION_LABEL_MAX_ZOOM - NATION_LABEL_FULL_ZOOM));
      label.visible = true;
    }

    // Nations that scrolled off screen keep their Text for the next frame
    // but must not be drawn where they are not.
    for (const [owner, label] of this.nationLabels) {
      if (!seen.has(owner)) label.visible = false;
    }
  }

  private clearNationLabels() {
    for (const label of this.nationLabels.values()) label.destroy();
    this.nationLabels.clear();
    this.nationLabelContainer.removeChildren();
  }

  private clearPowerLabels() {
    for (const label of this.powerLabels.values()) {
      label.destroy();
    }
    this.powerLabels.clear();
    this.powerLabelContainer.removeChildren();
  }

  private clearWarLabels() {
    for (const badge of this.warLabels.values()) {
      badge.destroy({ children: true });
    }
    this.warLabels.clear();
    this.warLabelContainer.removeChildren();
  }

  /** One garrison badge: a plate carrying a chip per unit type.
   *
   *  Each chip is icon + count on its own recessed pill, so four unit types
   *  read as four things rather than one run of digits. A supply rail down
   *  the left edge carries the status colour, and a tail points the plate at
   *  its hex. */
  private buildGarrisonBadge(g: WarGarrison, z: number): Container {
    const s = Math.max(0.55, Math.min(1.35, z));
    const iconSize = 13 * s;
    const fontSize = Math.max(8, 10 * s);
    const chipH = iconSize + 5 * s;
    const chipPad = 4 * s;
    const chipGap = 3 * s;
    const railW = 3 * s;
    const padX = 5 * s;
    const padY = 4 * s;
    const radius = 5 * s;

    const MAX_ENTRIES = 4;
    const units = (g.units && g.units.length > 0)
      ? g.units
      : [{ key: '', count: g.total }];
    const shown = units.slice(0, MAX_ENTRIES);
    const overflow = units.length - shown.length;

    const supplyColor = g.in_supply ? 0x4ade80 : 0xef4444;

    const badge = new Container();
    const bg = new Graphics();
    badge.addChild(bg);
    const chipGfx = new Graphics();
    badge.addChild(chipGfx);

    let x = railW + padX;
    const centerY = padY + chipH / 2;

    for (const u of shown) {
      const chipStart = x;
      let inner = x + chipPad;

      const tex = this.iconTextures.get(getUnitIcon(u.key));
      if (tex) {
        const sprite = new Sprite(tex);
        sprite.anchor.set(0, 0.5);
        sprite.x = inner;
        sprite.y = centerY;
        sprite.width = iconSize;
        sprite.height = iconSize;
        sprite.tint = 0xe5e7eb;
        badge.addChild(sprite);
      } else {
        // The icons come from a remote sprite API. When it is unreachable the
        // chip still has to say which unit it is, so fall back to the first
        // two letters of the key rather than an empty slot.
        const glyph = new Text({
          text: (u.key || '?').slice(0, 2).toUpperCase(),
          style: {
            fontSize: Math.max(7, fontSize - 1),
            fontFamily: mapFont(),
            fill: 0x93a4bd,
            fontWeight: 'bold',
          } as Partial<TextStyle>,
        });
        glyph.anchor.set(0.5, 0.5);
        glyph.x = inner + iconSize / 2;
        glyph.y = centerY;
        badge.addChild(glyph);
      }
      inner += iconSize + 3 * s;

      const count = new Text({
        text: `${u.count}`,
        style: {
          fontSize,
          fontFamily: mapFont(),
          fill: 0xf9fafb,
          fontWeight: 'bold',
        } as Partial<TextStyle>,
      });
      count.anchor.set(0, 0.5);
      count.x = inner;
      count.y = centerY;
      badge.addChild(count);
      inner += count.width;

      const chipW = inner + chipPad - chipStart;
      chipGfx.roundRect(chipStart, padY, chipW, chipH, 3 * s)
        .fill({ color: 0x1b2230, alpha: 0.95 })
        .stroke({ color: 0x2f3a4d, width: Math.max(1, 0.8 * s), alpha: 0.9 });

      x = chipStart + chipW + chipGap;
    }
    x -= chipGap;

    // Overflow and entrenchment are annotations on the whole stack, not on
    // any one unit, so they sit outside the chips.
    const extras: string[] = [];
    if (overflow > 0) extras.push(`+${overflow}`);
    if (g.entrenchment > 0) extras.push(`▲${Math.round(g.entrenchment)}%`);
    if (extras.length > 0) {
      x += 4 * s;
      const tail = new Text({
        text: extras.join(' '),
        style: {
          fontSize: Math.max(7, fontSize - 1),
          fontFamily: mapFont(),
          fill: g.entrenchment > 0 ? 0xfbbf24 : 0x9ca3af,
          fontWeight: 'bold',
        } as Partial<TextStyle>,
      });
      tail.anchor.set(0, 0.5);
      tail.x = x;
      tail.y = centerY;
      badge.addChild(tail);
      x += tail.width;
    }

    const boxW = x + padX;
    const boxH = chipH + padY * 2;
    const tailW = 5 * s;
    const tailH = 4 * s;

    bg.roundRect(0, 0, boxW, boxH, radius)
      .fill({ color: 0x0b0e14, alpha: 0.92 })
      .stroke({ color: 0x39445a, width: Math.max(1, 1 * s), alpha: 0.95 });
    // Supply rail: a solid bar is legible at zooms where a 1px border colour
    // is a single dim pixel.
    bg.roundRect(0, 0, railW + radius, boxH, radius)
      .fill({ color: supplyColor, alpha: g.in_supply ? 0.85 : 1 });
    bg.rect(railW, 0, radius, boxH)
      .fill({ color: 0x0b0e14, alpha: 0.92 });
    // Tail, pointing down at the hex the plate belongs to.
    bg.poly([
      boxW / 2 - tailW, boxH,
      boxW / 2 + tailW, boxH,
      boxW / 2, boxH + tailH,
    ]).fill({ color: 0x0b0e14, alpha: 0.92 });

    // Bottom-center pivot: the badge hangs above its anchor point, clear
    // of the city labels that live below the hex.
    badge.pivot.set(boxW / 2, boxH + tailH);
    return badge;
  }

  /** Garrison badges in military mode: per-unit icon + count on a plate. */
  private drawWarLabels(offsetX: number, offsetY: number, z: number) {
    const show = this.mode === 'military' && z > 0.3 && this.warGarrisons.length > 0;
    this.warLabelContainer.visible = show;
    if (!show) { return; }

    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const margin = HEX_SIZE * z * 3;

    if (Math.abs(z - this.lastWarLabelZoom) > 0.08) {
      this.clearWarLabels();
      this.lastWarLabelZoom = z;
    }

    for (const g of this.warGarrisons) {
      const [wx, wy] = axialToPixel(g.q, g.r, HEX_SIZE);
      const sx = wx * z + offsetX;
      const sy = wy * z + offsetY;
      if (sx < -margin || sx > w + margin || sy < -margin || sy > h + margin) continue;

      const key = hexKey(g.q, g.r);
      let badge = this.warLabels.get(key);
      if (!badge) {
        badge = this.buildGarrisonBadge(g, z);
        this.warLabelContainer.addChild(badge);
        this.warLabels.set(key, badge);
      }
      badge.x = sx;
      badge.y = sy - HEX_SIZE * z * 0.35;
    }
  }

  /** Show MW labels on power plant & power line hexes in electricity mode. */
  private drawPowerLabels(offsetX: number, offsetY: number, z: number) {
    const show = this.mode === 'electricity' && z > 0.6;
    this.powerLabelContainer.visible = show;
    if (!show) { return; }

    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const margin = HEX_SIZE * z * 3;
    const fontSize = Math.max(7, Math.min(12, 9 * z));

    const zoomChanged = Math.abs(z - this.lastPowerLabelZoom) > 0.05;
    if (zoomChanged) {
      this.clearPowerLabels();
      this.lastPowerLabelZoom = z;
    }

    for (const tile of this.tiles.values()) {
      if (!tile.buildingKeys || tile.buildingKeys.length === 0) continue;

      // Determine MW value to display
      let mwText: string | null = null;
      let labelColor = 0xfbbf24; // amber for power

      // Live grid figures when we have them: what the lines actually carry
      // to this hex, its capacity, and what the hex draws.
      const live = this.powerGrid.get(hexKey(tile.q, tile.r));

      // A hex's own draw is load, not a grid deficit: "↓0.59 MW" reads
      // as consumption where "−0.59 MW" read as being short by that much.
      // Check for power plant
      for (const key of tile.buildingKeys) {
        const output = POWER_PLANT_OUTPUT[key];
        if (output !== undefined) {
          const gen = live?.delivered_mw ?? output;
          mwText = `⚡${gen} MW`;
          if (live && live.consumption_mw > 0) mwText += `\n↓${live.consumption_mw} MW`;
          labelColor = 0x4ade80; // green for generators
          break;
        }
      }

      // If no power plant, check for power line
      if (!mwText) {
        for (const key of tile.buildingKeys) {
          const cap = POWER_LINE_CAPACITY[key];
          if (cap !== undefined) {
            if (live) {
              // carried / capacity, then this hex's own draw
              mwText = `${live.delivered_mw}/${live.capacity_mw ?? cap} MW`;
              if (live.consumption_mw > 0) mwText += `\n↓${live.consumption_mw} MW`;
            } else {
              mwText = `${cap} MW`;
            }
            labelColor = key === 'high_voltage_line' ? 0xf97316 : 0x94a3b8; // orange for HV, slate for normal
            break;
          }
        }
      }

      // Consumer hex with no grid infrastructure of its own
      if (!mwText && live && live.consumption_mw > 0) {
        mwText = `↓${live.consumption_mw} MW`;
        labelColor = live.electrified ? 0xfbbf24 : 0xef4444;
      }

      if (!mwText) continue;

      const [wx, wy] = axialToPixel(tile.q, tile.r, HEX_SIZE);
      const sx = wx * z + offsetX;
      const sy = wy * z + offsetY;
      if (sx < -margin || sx > w + margin || sy < -margin || sy > h + margin) continue;

      const key = hexKey(tile.q, tile.r);
      let label = this.powerLabels.get(key);

      if (!label) {
        label = new Text({
          text: mwText,
          style: {
            fontSize,
            fontFamily: mapFont(),
            fill: labelColor,
            fontWeight: 'bold',
            dropShadow: { color: 0x000000, distance: 1, blur: 3, alpha: 1, angle: Math.PI / 4 },
          } as Partial<TextStyle>,
        });
        label.anchor.set(0.5, 0.5);
        this.powerLabelContainer.addChild(label);
        this.powerLabels.set(key, label);
      } else if (label.text !== mwText) {
        // Live grid figures change between ticks; a cached label must not
        // keep showing the old megawatts.
        label.text = mwText;
      }

      label.x = sx;
      label.y = sy - HEX_SIZE * z * 0.35;
    }
  }

  private drawOverlay(offsetX: number, offsetY: number, z: number) {
    const gfx = this.overlayGfx;
    gfx.clear();
    const sz = HEX_SIZE * z;

    // Build mode preview: highlight path hexes and draw connecting line
    if (this.buildPath.length > 0) {
      const previewColor =
        this.buildMode === 'dirt' ? 0xb99a6a : this.buildMode === 'road' ? 0x69717a
        : (this.buildMode === 'power' || this.buildMode === 'hv_power') ? 0xfbbf24
        : this.buildMode === 'move' ? 0x22c55e
        : this.buildMode === 'attack_line' ? 0xef4444
        : this.buildMode === 'defense_line' ? 0x3b82f6
        : 0x6b7280;
      const previewAlpha = 0.35;

      // Highlight each hex in the path
      for (const h of this.buildPath) {
        const [wx, wy] = axialToPixel(h.q, h.r, HEX_SIZE);
        const sx = wx * z + offsetX;
        const sy = wy * z + offsetY;
        gfx.poly(hexCorners(sx, sy, sz))
          .fill({ color: previewColor, alpha: previewAlpha })
          .stroke({ color: 0xffffff, width: 1.5, alpha: 0.6 });
      }

      // Draw a connecting line through the path centers
      if (this.buildPath.length > 1) {
        const lineColor =
          this.buildMode === 'dirt' ? 0xb99a6a : this.buildMode === 'road' ? 0x69717a
          : (this.buildMode === 'power' || this.buildMode === 'hv_power') ? 0xfbbf24
          : this.buildMode === 'move' ? 0x22c55e
          : this.buildMode === 'attack_line' ? 0xef4444
          : this.buildMode === 'defense_line' ? 0x3b82f6
          : 0x9ca3af;
        const [wx0, wy0] = axialToPixel(this.buildPath[0].q, this.buildPath[0].r, HEX_SIZE);
        gfx.moveTo(wx0 * z + offsetX, wy0 * z + offsetY);
        for (let i = 1; i < this.buildPath.length; i++) {
          const [wx, wy] = axialToPixel(this.buildPath[i].q, this.buildPath[i].r, HEX_SIZE);
          gfx.lineTo(wx * z + offsetX, wy * z + offsetY);
        }
        gfx.stroke({ color: lineColor, width: Math.max(2, z * 3), alpha: 0.7 });
      }
    }

    // War orders overlay: arrows and lines, military mode only
    if (this.mode === 'military' && this.warOrders.length > 0) {
      for (const order of this.warOrders) {
        if (!order.path || order.path.length === 0) continue;
        const pts = order.path.map(([q, r]) => {
          const [wx, wy] = axialToPixel(q, r, HEX_SIZE);
          return { x: wx * z + offsetX, y: wy * z + offsetY };
        });
        const color = order.kind === 'move' ? 0x22c55e
          : order.kind === 'attack_line' ? 0xef4444 : 0x3b82f6;
        const width = Math.max(2, z * 4);

        if (pts.length > 1) {
          gfx.moveTo(pts[0].x, pts[0].y);
          for (let i = 1; i < pts.length; i++) gfx.lineTo(pts[i].x, pts[i].y);
          gfx.stroke({ color, width, alpha: order.kind === 'defense_line' ? 0.55 : 0.8 });
        }

        if (order.kind === 'defense_line') {
          // Anchor markers along the line read as fortification pins
          for (const p of pts) {
            gfx.rect(p.x - width, p.y - width, width * 2, width * 2)
              .fill({ color, alpha: 0.9 });
          }
          continue;
        }

        // Arrowhead on the last segment (or a lone objective hex)
        const tip = pts[pts.length - 1];
        const prev = pts.length > 1 ? pts[pts.length - 2] : { x: tip.x - 1, y: tip.y };
        const angle = Math.atan2(tip.y - prev.y, tip.x - prev.x);
        const ah = Math.max(6, z * 14);
        gfx.poly([
          tip.x, tip.y,
          tip.x - ah * Math.cos(angle - 0.5), tip.y - ah * Math.sin(angle - 0.5),
          tip.x - ah * Math.cos(angle + 0.5), tip.y - ah * Math.sin(angle + 0.5),
        ]).fill({ color, alpha: 0.9 });

        // Progress dot: where the column currently stands
        if (order.kind === 'move') {
          const at = pts[Math.min(order.progress, pts.length - 1)];
          gfx.circle(at.x, at.y, Math.max(3, z * 7)).fill({ color, alpha: 1 });
        }
      }
    }

    // Adjacency radius preview while choosing a building
    if (this.adjacencyPreview) {
      const { q, r, radius, positive } = this.adjacencyPreview;
      const color = positive ? 0x22c55e : 0xef4444;
      for (let dq = -radius; dq <= radius; dq++) {
        const lo = Math.max(-radius, -dq - radius);
        const hi = Math.min(radius, -dq + radius);
        for (let dr = lo; dr <= hi; dr++) {
          if (!this.tiles.has(hexKey(q + dq, r + dr))) continue;
          const [wx, wy] = axialToPixel(q + dq, r + dr, HEX_SIZE);
          const sx = wx * z + offsetX;
          const sy = wy * z + offsetY;
          const isCenter = dq === 0 && dr === 0;
          gfx.poly(hexCorners(sx, sy, sz))
            .fill({ color, alpha: isCenter ? 0.4 : 0.22 })
            .stroke({ color, width: 1, alpha: 0.6 });
        }
      }
    }

    if (this.hoveredHex) {
      const tile = this.tiles.get(this.hoveredHex);
      if (tile) {
        const [wx, wy] = axialToPixel(tile.q, tile.r, HEX_SIZE);
        const sx = wx * z + offsetX;
        const sy = wy * z + offsetY;
        gfx.poly(hexCorners(sx, sy, sz))
          .fill({ color: 0xffffff, alpha: 0.08 })
          .stroke({ color: 0xffffff, width: 1.5, alpha: 0.55 });
      }
    }

    if (this.selectedHex) {
      const tile = this.tiles.get(this.selectedHex);
      if (tile) {
        const [wx, wy] = axialToPixel(tile.q, tile.r, HEX_SIZE);
        const sx = wx * z + offsetX;
        const sy = wy * z + offsetY;
        gfx.poly(hexCorners(sx, sy, sz)).stroke({ color: 0xffd700, width: 2, alpha: 0.9 });
      }
    }
  }

  private getHexColor(tile: TileData): number {
    let terrainColor = TERRAIN_TYPES[tile.terrain]?.color ?? 0x333333;
    // A lake is where a river ended and the basin filled. It is water, so
    // it paints as water whatever the terrain underneath used to be.
    if (this.lakeHexes.has(hexKey(tile.q, tile.r))) {
      return this.lerpColor(TERRAIN_TYPES[0].color, 0x4f8cae, 0.30);
    }
    // Shallow water: ocean touching land renders lighter so coastlines pop
    if (tile.terrain === 0 && this.shallowWater.has(hexKey(tile.q, tile.r))) {
      terrainColor = this.lerpColor(terrainColor, 0x4f8cae, 0.45);
    }

    switch (this.mode) {
      case 'terrain':
        return terrainColor;

      case 'political':
        if (tile.terrain === 0) {
          // Claimed water blends toward its owner without becoming land.
          if (!tile.seaOwner) return terrainColor;
          return this.lerpColor(
            terrainColor,
            this.nationColorCache.get(tile.seaOwner) ?? 0x555555,
            0.32,
          );
        }
        if (tile.anarchyUntil) return 0x8b0000; // dark red for anarchy hexes
        if (!tile.owner) return this.desaturate(terrainColor, 0.4);
        return this.nationColorCache.get(tile.owner) ?? 0x555555;

      case 'economic':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.resource || tile.richness === 0) return this.desaturate(terrainColor, 0.3);
        const intensity = 0.3 + tile.richness * 0.14;
        return this.lerpColor(0x1a1a2e, 0x4ade80, intensity);

      case 'military': {
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner) return this.desaturate(terrainColor, 0.4);
        const base = this.nationColorCache.get(tile.owner) ?? 0x555555;
        const mil = (tile.buildingKeys ?? []).filter((k) => MILITARY_BUILDING_KEYS.has(k)).length;
        if (mil === 0) return this.desaturate(base, 0.55);
        // Installations burn toward red as they stack up
        return this.lerpColor(base, 0xdc2626, 0.35 + Math.min(1, mil / 4) * 0.5);
      }

      case 'borders': {
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner) return this.desaturate(terrainColor, 0.4);
        const ov = this.borderOverlay.get(hexKey(tile.q, tile.r));
        if (ov) {
          // Frontier hexes glow green (calm) to red (inflamed). Every
          // nation's frontier is in the overlay, not only your own, so a
          // border between two neighbours you are not part of still reads
          // as a border.
          const t = Math.min(1, Math.max(0, ov.tension) / 100);
          return this.lerpColor(0x22c55e, 0xdc2626, t);
        }
        // Interior land: the owner's colour, pushed well down so it never
        // competes with the frontier scale. It used to be one flat slate
        // for everybody, which made a map with no frontier on it look
        // broken rather than empty.
        const owner = this.nationColorCache.get(tile.owner) ?? 0x64748b;
        return this.lerpColor(this.desaturate(owner, 0.72), 0x1f2937, 0.42);
      }

      case 'pollution':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, (tile.pollution ?? 0) / 100);
          return this.lerpColor(0x2d6a4f, 0x854d0e, t); }

      case 'crime':
        if (tile.terrain === 0) return terrainColor;
        // People-stats only exist where people live: uninhabited hexes
        // render like wilderness instead of pretending a stat of zero.
        if (!tile.owner || !tile.inhabited) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, (tile.crime ?? 0) / 100);
          return this.lerpColor(0x2d5a27, 0xdc2626, t); }

      case 'disease':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner || !tile.inhabited) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, (tile.disease ?? 0) / 100);
          return this.lerpColor(0x2d5a27, 0x9333ea, t); }

      case 'fire_coverage':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, (tile.fireCoverage ?? 0) / 10);
          return this.lerpColor(0x7f1d1d, 0x22c55e, t); }

      case 'police_coverage':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, (tile.policeCoverage ?? 0) / 10);
          return this.lerpColor(0x1e1b4b, 0x3b82f6, t); }

      case 'hospital_coverage':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, (tile.hospitalCoverage ?? 0) / 10);
          return this.lerpColor(0x4a1515, 0xef4444, t); }

      case 'electricity':
        if (tile.terrain === 0) return 0x0a0a1a;  // very dark ocean
        if (!tile.owner) return 0x111122;  // dark unowned
        return tile.electricity ? 0xfbbf24 : 0x111122;  // amber glow for powered, dark for unpowered

      case 'water':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner) return this.desaturate(terrainColor, 0.3);
        { const ws = tile.waterSupply ?? 0;
          if (ws <= 0) return 0x78350f;  // brown/dry
          const t = Math.min(1, ws / 150000);
          return this.lerpColor(0x7dd3fc, 0x1d4ed8, t); }  // light blue → deep blue

      case 'hydrology': {
        // Where the fresh water is, rather than where the pipes are. Reads
        // the river network the generator carved, so it answers the two
        // questions rivers introduced: what can farm, and what can dock.
        if (tile.terrain === 0) return terrainColor;
        const flow = this.hexFlow.get(hexKey(tile.q, tile.r)) ?? 0;
        if (flow >= this.riverMajorFlow) return 0x1d4ed8;   // barges reach here
        if (flow > 0) return 0x38bdf8;                       // a stream
        if (tile.terrain === 7) return 0x7dd3fc;             // salt water only
        // Dry land, shaded by how wet the ground actually is. Terrain still
        // sets what a hex can hold - desert reads parched, swamp damp - but
        // recent rain is what moves it, so the layer changes as fronts pass.
        const moisture = soilMoisture(
          this.worldSeed, this.worldTick, tile.q, tile.r, tile.terrain);
        const ground = this.lerpColor(0x78350f, 0x4d7c0f, moisture);
        // Under active rain, pull the hex toward a wet slate so the storm
        // itself is legible, not just the ground it leaves behind.
        const rain = rainAt(this.worldSeed, this.worldTick, tile.q, tile.r);
        return rain > 0 ? this.lerpColor(ground, 0x2563eb, rain * 0.45) : ground;
      }

      case 'happiness':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner || !tile.inhabited) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, Math.max(0, (tile.happiness ?? 0)) / 100);
          return this.lerpColor(0xdc2626, 0x22c55e, t); }

      case 'education':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner || !tile.inhabited) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, (tile.education ?? 0) / 50);
          return this.lerpColor(0x4b5563, 0x3b82f6, t); }

      case 'safety':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner || !tile.inhabited) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, (tile.safety ?? 0) / 50);
          return this.lerpColor(0xdc2626, 0x22c55e, t); }

      case 'health':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner || !tile.inhabited) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, (tile.health ?? 0) / 100);
          return this.lerpColor(0xdc2626, 0x22c55e, t); }

      case 'fire_hazard':
        if (tile.terrain === 0) return terrainColor;
        if (!tile.owner) return this.desaturate(terrainColor, 0.3);
        { const t = Math.min(1, (tile.fireHazard ?? 0) / 50);
          return this.lerpColor(0x22c55e, 0xdc2626, t); }

      default:
        return terrainColor;
    }
  }

  // ── Interaction ──────────────────────────────────────────────────

  private setupInteraction() {
    const canvas = this.app.canvas as HTMLCanvasElement;
    const signal = this.interactionController.signal;

    canvas.addEventListener('pointerdown', (e: PointerEvent) => {
      if (!e.isPrimary || e.button !== 0) return;
      // The gesture the browser wants before it will let audio play.
      this.audio.resume();
      if (this.buildMode !== 'pan') {
        // Build mode: start path collection
        this.building = true;
        this.buildPath = [];
        this.buildPathKeys.clear();
        const world = this.screenToWorld(e.offsetX, e.offsetY);
        const [q, r] = pixelToAxial(world[0], world[1], HEX_SIZE);
        const key = hexKey(q, r);
        if (this.tiles.has(key)) {
          this.buildPath.push({ q, r });
          this.buildPathKeys.add(key);
          this.overlayDirty = true;
        }
        canvas.style.cursor = 'crosshair';
      } else {
        this.dragging = true;
        this.dragStart = { x: e.clientX, y: e.clientY };
        this.camStart = { x: this.camX, y: this.camY };
        canvas.style.cursor = 'grabbing';
      }
    }, { signal });

    canvas.addEventListener('pointermove', (e: PointerEvent) => {
      if (!e.isPrimary || active.size > 1) return;
      if (this.building && this.buildMode !== 'pan') {
        // Build mode drag: extend the path, or rewind it when the pointer
        // comes back over a hex already on the path. Rewinding truncates to
        // that hex rather than popping one entry, so a fast drag that skips
        // several hexes on the way back still lands on the right prefix.
        const world = this.screenToWorld(e.offsetX, e.offsetY);
        const [q, r] = pixelToAxial(world[0], world[1], HEX_SIZE);
        const key = hexKey(q, r);
        if (this.tiles.has(key)) {
          if (this.buildPathKeys.has(key)) {
            const idx = this.buildPath.findIndex(p => p.q === q && p.r === r);
            if (idx >= 0 && idx < this.buildPath.length - 1) {
              for (let i = idx + 1; i < this.buildPath.length; i++) {
                this.buildPathKeys.delete(hexKey(this.buildPath[i].q, this.buildPath[i].r));
              }
              this.buildPath.length = idx + 1;
              this.overlayDirty = true;
            }
          } else {
            this.buildPath.push({ q, r });
            this.buildPathKeys.add(key);
            this.overlayDirty = true;
          }
        }
      } else if (this.dragging) {
        const dx = (e.clientX - this.dragStart.x) / this._zoom;
        const dy = (e.clientY - this.dragStart.y) / this._zoom;
        this.camX = this.camStart.x - dx;
        this.camY = this.camStart.y - dy;
        this.clampView();
        this.dirty = true;
      } else {
        const world = this.screenToWorld(e.offsetX, e.offsetY);
        const [q, r] = pixelToAxial(world[0], world[1], HEX_SIZE);
        const key = hexKey(q, r);
        if (key !== this.hoveredHex) {
          this.hoveredHex = this.tiles.has(key) ? key : null;
          this.overlayDirty = true;
        }
      }
    }, { signal });

    canvas.addEventListener('pointerup', (e: PointerEvent) => {
      if (!e.isPrimary || (!this.dragging && !this.building)) return;
      if (this.building && this.buildMode !== 'pan') {
        // Build mode release: fire callback with path
        this.building = false;
        if (this.buildPath.length > 0) {
          // War order modes report the mode itself; build modes their building
          const buildingKey = WAR_ORDER_MODES.has(this.buildMode)
            ? this.buildMode
            : this.buildMode === 'road' ? 'road_network'
            : this.buildMode === 'dirt' ? 'dirt_road'
            : this.buildMode === 'rail' ? 'railway'
            : this.buildMode === 'hv_power' ? 'high_voltage_line'
            : 'power_line';
          this.onBuildPath([...this.buildPath], buildingKey);
        }
        this.buildPath = [];
        this.buildPathKeys.clear();
        this.overlayDirty = true;
        canvas.style.cursor = 'crosshair';
        return;
      }

      const dx = Math.abs(e.clientX - this.dragStart.x);
      const dy = Math.abs(e.clientY - this.dragStart.y);

      if (dx < 5 && dy < 5) {
        const world = this.screenToWorld(e.offsetX, e.offsetY);
        const [q, r] = pixelToAxial(world[0], world[1], HEX_SIZE);
        const key = hexKey(q, r);
        const tile = this.tiles.get(key) ?? null;

        if (tile) {
          this.selectedHex = key;
          this.onHexSelect(q, r, tile);
        } else {
          this.selectedHex = null;
          this.onHexSelect(q, r, null);
        }
        this.overlayDirty = true;
      }

      this.dragging = false;
      canvas.style.cursor = this.buildMode === 'pan' ? 'grab' : 'crosshair';
    }, { signal });

    const cancelGesture = () => {
      if (this.building) {
        // Cancel build on leave
        this.building = false;
        this.buildPath = [];
        this.buildPathKeys.clear();
      }
      this.dragging = false;
      this.hoveredHex = null;
      this.overlayDirty = true;
      canvas.style.cursor = this.buildMode === 'pan' ? 'grab' : 'crosshair';
    };

    canvas.addEventListener('pointerleave', cancelGesture, { signal });
    canvas.addEventListener('pointercancel', cancelGesture, { signal });

    canvas.addEventListener('wheel', (e: WheelEvent) => {
      e.preventDefault();
      this.zoomAbout(e.offsetX, e.offsetY, e.deltaY < 0 ? 1.12 : 0.89);
    }, { passive: false, signal });

    // ── Touch ──────────────────────────────────────────────────────
    // Without this the browser claims the gesture for page scroll and
    // pinch, and pointermove never fires: on a phone the map simply did
    // not move. It has to be set on the element, not a stylesheet, because
    // Pixi creates the canvas itself.
    canvas.style.touchAction = 'none';

    // Two fingers pinch to zoom. Tracked here rather than through the
    // pointer handlers above so a second finger landing mid-drag ends the
    // pan cleanly instead of yanking the camera to the midpoint.
    const active = new Map<number, { x: number; y: number }>();
    let pinchDistance = 0;

    const midpoint = () => {
      const pts = [...active.values()];
      return {
        x: (pts[0].x + pts[1].x) / 2,
        y: (pts[0].y + pts[1].y) / 2,
        d: Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y),
      };
    };

    canvas.addEventListener('pointerdown', (e: PointerEvent) => {
      if (e.pointerType !== 'touch') return;
      active.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (active.size === 2) {
        // A pinch is starting, so whatever the first finger began is over.
        this.dragging = false;
        this.building = false;
        this.buildPath = [];
        this.buildPathKeys.clear();
        this.overlayDirty = true;
        pinchDistance = midpoint().d;
      }
    }, { signal });

    canvas.addEventListener('pointermove', (e: PointerEvent) => {
      if (e.pointerType !== 'touch' || !active.has(e.pointerId)) return;
      active.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (active.size !== 2) return;
      const { x, y, d } = midpoint();
      if (pinchDistance > 0 && d > 0) {
        const rect = canvas.getBoundingClientRect();
        this.zoomAbout(x - rect.left, y - rect.top, d / pinchDistance);
      }
      pinchDistance = d;
    }, { signal });

    const endTouch = (e: PointerEvent) => {
      if (e.pointerType !== 'touch') return;
      active.delete(e.pointerId);
      if (active.size < 2) pinchDistance = 0;
    };
    canvas.addEventListener('pointerup', endTouch, { signal });
    canvas.addEventListener('pointercancel', endTouch, { signal });
    canvas.addEventListener('pointerleave', endTouch, { signal });

    canvas.style.cursor = 'grab';
  }

  /** Scale about a point on the canvas, keeping that point still. */
  private zoomAbout(sx: number, sy: number, factor: number) {
    const before = this.screenToWorld(sx, sy);
    this._zoom = this.clampZoom(this._zoom * factor);
    const after = this.screenToWorld(sx, sy);
    this.camX -= after[0] - before[0];
    this.camY -= after[1] - before[1];
    this.clampView();
    this.dirty = true;
  }

  // ── Coordinate helpers ───────────────────────────────────────────

  private screenToWorld(sx: number, sy: number): [number, number] {
    const w = this.app.screen.width;
    const h = this.app.screen.height;
    const offsetX = w / 2 - this.camX * this._zoom;
    const offsetY = h / 2 - this.camY * this._zoom;
    return [(sx - offsetX) / this._zoom, (sy - offsetY) / this._zoom];
  }

  // ── Color helpers ────────────────────────────────────────────────

  private desaturate(color: number, amount: number): number {
    const r = (color >> 16) & 0xff;
    const g = (color >> 8) & 0xff;
    const b = color & 0xff;
    const gray = Math.round(r * 0.299 + g * 0.587 + b * 0.114);
    const nr = Math.round(r + (gray - r) * amount);
    const ng = Math.round(g + (gray - g) * amount);
    const nb = Math.round(b + (gray - b) * amount);
    return (nr << 16) | (ng << 8) | nb;
  }

  private lerpColor(a: number, b: number, t: number): number {
    const ar = (a >> 16) & 0xff, ag = (a >> 8) & 0xff, ab = a & 0xff;
    const br = (b >> 16) & 0xff, bg = (b >> 8) & 0xff, bb = b & 0xff;
    const nr = Math.round(ar + (br - ar) * t);
    const ng = Math.round(ag + (bg - ag) * t);
    const nb = Math.round(ab + (bb - ab) * t);
    return (nr << 16) | (ng << 8) | nb;
  }
}
