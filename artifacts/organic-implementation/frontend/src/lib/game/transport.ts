/** Deterministic, public-map-only road geometry and ambient traffic. */
import { axialToPixel, HEX_SIZE } from './hex-math';
import { fitRoutes } from './route-geometry';
import { VEHICLE_DIMENSIONS, LANE_OFFSET, DIRT_WIDTH, metres } from './map-scale';

export type TransportKind = 'road' | 'rail';
export interface TransportTile {
  q: number; r: number; terrain: number; owner: string | null;
  cityId?: number | null; isCapital?: boolean; anarchyUntil?: string | null;
}
export interface Point { x: number; y: number }
export interface TransportPath {
  id: string; a: string; b: string; kind: TransportKind;
  points: Point[]; lengths: number[]; length: number;
  activity: number; speed: number; seed: number;
  surface: 'paved' | 'dirt' | 'rail';
  blocked?: boolean;
  checkpoint?: { distance: number; sides: [boolean, boolean] };
}

/** A short inspection window followed by enough green time for a small platoon. */
export function inspectionOpen(time: number, path: TransportPath, reverse: boolean) {
  return (time + path.seed * 12 + (reverse ? 6 : 0)) % 12 >= 6;
}
export function hash(value: string): number {
  let n = 2166136261;
  for (let i = 0; i < value.length; i++) n = Math.imul(n ^ value.charCodeAt(i), 16777619);
  return (n >>> 0) / 4294967296;
}
const ROUGH = new Set([2, 3, 6, 8, 9, 11]);

export function makePath(a: TransportTile, b: TransportTile, kind: TransportKind, shared: boolean): TransportPath {
  // Canonical orientation makes curves and shared lanes independent of payload order.
  if (`${a.q},${a.r}` > `${b.q},${b.r}`) [a, b] = [b, a];
  const ka = `${a.q},${a.r}`, kb = `${b.q},${b.r}`;
  const id = `${ka}_${kb}:${kind}`, seed = hash(id);
  const [ax, ay] = axialToPixel(a.q, a.r, HEX_SIZE);
  const [bx, by] = axialToPixel(b.q, b.r, HEX_SIZE);
  const dx = bx - ax, dy = by - ay, distance = Math.hypot(dx, dy);
  const nx = -dy / (distance || 1), ny = dx / (distance || 1);
  const urban = a.cityId != null || b.cityId != null;
  const rough = ROUGH.has(a.terrain) || ROUGH.has(b.terrain);
  const points: Point[] = [], lengths = [0];
  // Smooth drift with zero displacement and derivative at the junctions.
  for (let i = 0; i <= 12; i++) {
    const t = i / 12;
    const drift = 0; // Whole-network fitting supplies the final alignment.
    const point = { x: ax + dx * t + nx * drift, y: ay + dy * t + ny * drift };
    if (i) lengths.push(lengths[i - 1] + Math.hypot(point.x - points[i - 1].x, point.y - points[i - 1].y));
    points.push(point);
  }
  return { id, a: ka, b: kb, kind, seed, points, lengths, length: lengths[12], activity: 0, surface: kind === 'rail' ? 'rail' : 'paved',
    speed: metres((kind === 'rail' ? 90 : urban ? 30 : 60) / 3.6) * (rough ? .72 : 1) };
}

/** Arc-length sampling keeps cars and sleepers evenly spaced around a bend. */
export function samplePath(path: TransportPath, distance: number, reverse = false, lane = 0, mergeLane = true) {
  let d = Math.max(0, Math.min(path.length, distance));
  if (reverse) d = path.length - d;
  let i = 1;
  while (i < path.lengths.length - 1 && path.lengths[i] < d) i++;
  const a = path.points[i - 1], b = path.points[i];
  const length = path.lengths[i] - path.lengths[i - 1];
  const t = length ? (d - path.lengths[i - 1]) / length : 0;
  const direction = reverse ? -1 : 1;
  const ux = (b.x - a.x) / (length || 1) * direction;
  const uy = (b.y - a.y) / (length || 1) * direction;
  // Lanes merge through the node so traffic cannot jump sideways at turns.
  const offset = lane * (mergeLane ? Math.min(1, d / 3, (path.length - d) / 3) : 1);
  return { x: a.x + (b.x - a.x) * t - uy * offset,
    y: a.y + (b.y - a.y) * t + ux * offset, angle: Math.atan2(uy, ux) };
}

export class TransportNetwork {
  paths = new Map<string, TransportPath>();
  adjacent = new Map<string, TransportPath[]>();

  constructor(tiles: Map<string, TransportTile>, edges: Map<string, string>) {
    for (const [key, raw] of edges) {
      const [ka, kb] = key.split('_');
      const a = tiles.get(ka), b = tiles.get(kb);
      if (!a || !b || a.terrain === 0 || b.terrain === 0) continue;
      const dq = a.q - b.q, dr = a.r - b.r;
      if (Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr)) !== 1) continue;
      const types = new Set(raw.split(','));
      if (types.has('dirt')) types.add('road');
      for (const kind of ['road', 'rail'] as const) {
        if (!types.has(kind)) continue;
        const path = makePath(a, b, kind, types.has('road') && types.has('rail'));
        if (kind === 'road' && types.has('dirt')) { path.surface = 'dirt'; path.speed *= .5; }
        if (this.paths.has(path.id)) continue;
        this.paths.set(path.id, path);
        for (const node of [path.a, path.b]) {
          const key = `${node}:${kind}`;
          if (!this.adjacent.has(key)) this.adjacent.set(key, []);
          this.adjacent.get(key)!.push(path);
        }
      }
    }
    fitRoutes(this.paths, this.adjacent, tiles);
    // City influence travels along built connections, not across empty land.
    const influence = new Map<string, number>();
    let frontier: Array<[string, number]> = [];
    for (const [node, tile] of tiles) if (tile.cityId != null) {
      for (const kind of ['road', 'rail']) {
        const key = `${node}:${kind}`, weight = tile.isCapital ? 1 : .75;
        influence.set(key, weight); frontier.push([key, weight]);
      }
    }
    for (let hop = 0; hop < 4; hop++) {
      const next: typeof frontier = [];
      for (const [key, value] of frontier) for (const path of this.adjacent.get(key) ?? []) {
        for (const node of [path.a, path.b]) {
          const target = `${node}:${path.kind}`, weight = value * .66;
          if (weight > (influence.get(target) ?? 0)) { influence.set(target, weight); next.push([target, weight]); }
        }
      }
      frontier = next;
    }
    for (const path of this.paths.values()) {
      const a = tiles.get(path.a)!, b = tiles.get(path.b)!;
      const degree = Math.max(this.adjacent.get(`${path.a}:${path.kind}`)!.length, this.adjacent.get(`${path.b}:${path.kind}`)!.length);
      const city = Math.max(influence.get(`${path.a}:${path.kind}`) ?? 0, influence.get(`${path.b}:${path.kind}`) ?? 0);
      const settled = a.owner != null && b.owner != null;
      const disrupted = a.anarchyUntil || b.anarchyUntil;
      path.activity = Math.min(1, .08 + (settled ? .2 : 0) + city * .6 + Math.min(3, degree - 1) * .045) * (disrupted ? .2 : 1);
      if (path.surface === 'dirt') path.activity *= .15;
    }
  }

  get(edge: string, kind: TransportKind) { return this.paths.get(`${edge}:${kind}`); }
}

export interface Vehicle {
  id: string; path: TransportPath; reverse: boolean; distance: number;
  history: Array<{ path: TransportPath; reverse: boolean }>;
  turns: number; color: number; carriages: number;
  model?: 'compact' | 'sedan' | 'van' | 'bus' | 'truck';
  waiting?: boolean;
}
export function vehicleLength(v: Vehicle) { return VEHICLE_DIMENSIONS[v.model ?? 'sedan'].length; }

/** Drivers strongly prefer paving, while dirt-only connections remain usable. */
export function chooseRoute(choices: TransportPath[], current: TransportPath, seed: number) {
  if (!choices.length) return current;
  const weight = (p: TransportPath) => p.surface === 'dirt' ? .03 : 1;
  let remaining = seed * choices.reduce((sum, p) => sum + weight(p), 0);
  for (const path of choices) { remaining -= weight(path); if (remaining < 0) return path; }
  return choices[choices.length - 1];
}

export class TrafficSimulation {
  time = 0;
  vehicles = new Map<string, Vehicle>();
  network = new TransportNetwork(new Map(), new Map());
  static readonly CAR_LIMIT = 160;
  static readonly TRAIN_LIMIT = 18;

  setNetwork(network: TransportNetwork) {
    this.network = network;
    for (const [id, vehicle] of this.vehicles) {
      const path = network.paths.get(vehicle.path.id);
      if (!path) this.vehicles.delete(id);
      else {
        vehicle.path = path;
        const history: Vehicle['history'] = [];
        for (const previous of vehicle.history) {
          const path = network.paths.get(previous.path.id);
          if (!path) break; // Carriages cannot skip a removed section of track.
          history.push({ path, reverse: previous.reverse });
        }
        vehicle.history = history;
      }
    }
  }

  populate(visible: TransportPath[], winter: boolean) {
    const ids = new Set(visible.filter(p => !p.blocked).map(p => p.id));
    for (const [id, v] of this.vehicles) if (!ids.has(v.path.id)) this.vehicles.delete(id);
    for (const kind of ['road', 'rail'] as const) {
      const paths = visible.filter(p => p.kind === kind && !p.blocked);
      const limit = kind === 'road' ? TrafficSimulation.CAR_LIMIT : TrafficSimulation.TRAIN_LIMIT;
      const demandFor = (p: TransportPath) => (p.activity + (p.checkpoint && kind === 'road' ? .6 * (p.surface === 'dirt' ? .15 : 1) : 0)) * (kind === 'road' ? 2.7 : .12) * (winter ? .72 : 1);
      const demand = paths.reduce((n, p) => n + demandFor(p), 0);
      const target = Math.min(limit, Math.round(demand));
      const dirtTarget = Math.round(paths.filter(p => p.surface === 'dirt').reduce((n,p) => n + demandFor(p), 0));
      let dirtCount = [...this.vehicles.values()].filter(v => v.path.kind === kind && v.path.surface === 'dirt').length;
      for (const [id,v] of this.vehicles) if (v.path.kind === kind && v.path.surface === 'dirt' && dirtCount > dirtTarget) { this.vehicles.delete(id); dirtCount--; }
      let count = [...this.vehicles.values()].filter(v => v.path.kind === kind).length;
      for (const [id, v] of this.vehicles) if (v.path.kind === kind && count > target) { this.vehicles.delete(id); count--; }
      const ranked = paths.sort((a, b) => b.activity * (.5 + b.seed) - a.activity * (.5 + a.seed));
      for (let slot = 0; slot < (kind === 'road' ? 6 : 2) && count < target; slot++) for (const path of ranked) {
        if (count >= target) break;
        if (path.surface === 'dirt' && dirtCount >= dirtTarget) continue;
        const id = `${path.id}:${slot}`;
        if (this.vehicles.has(id)) continue;
        // Avoid stacking new vehicles on top of traffic arriving from another edge.
        const seed = hash(id), reverse = slot % 2 === 0;
        const model = seed < .25 ? 'compact' : seed < .75 ? 'sedan' : seed < .88 ? 'van' : seed < .93 ? 'bus' : 'truck';
        const distance = path.length * (.18 + Math.floor(slot / 2) * .29);
        const length = vehicleLength({model} as Vehicle);
        if ([...this.vehicles.values()].some(v => v.path.id === path.id &&
          (kind === 'rail' || (v.reverse === reverse && Math.abs(v.distance-distance) < (vehicleLength(v)+length)/2+1)))) continue;
        const entrance = reverse ? path.b : path.a;
        if (kind === 'road' && [...this.vehicles.values()].some(v => v.path.kind === 'road' && v.path !== path &&
          (v.reverse ? v.path.a : v.path.b) === entrance && v.path.length-v.distance+distance < (vehicleLength(v)+length)/2+1)) continue;
        const palette = [0xd9d9d2, 0xa7adae, 0x753f36, 0x3c5361, 0x454b4b];
        this.vehicles.set(id, { id, path, reverse,
          distance, history: [], turns: 0, model,
          color: kind === 'rail' ? 0xc27646 : palette[Math.floor(seed * palette.length)],
          carriages: kind === 'rail' ? 2 + Math.floor(path.seed * 3) : 1 });
        count++;
        if (path.surface === 'dirt') dirtCount++;
      }
    }
  }

  step(deltaMs: number, winter: boolean) {
    const seconds = Math.min(100, Math.max(0, deltaMs)) / 1000;
    this.time += seconds;
    const lanes = new Map<string, Vehicle[]>();
    const laneKey = (path: TransportPath, reverse: boolean) => `${path.id}:${reverse}`;
    for (const v of this.vehicles.values()) {
      const key = laneKey(v.path,v.reverse);
      if (!lanes.has(key)) lanes.set(key,[]);
      lanes.get(key)!.push(v);
    }
    for (const list of lanes.values()) list.sort((a,b) => b.distance-a.distance);
    // Front to back within each lane: a following driver cannot overtake a queue.
    for (const vehicle of [...lanes.values()].flat()) {
      if (vehicle.path.blocked) continue;
      const node = vehicle.reverse ? vehicle.path.a : vehicle.path.b;
      const available = this.network.adjacent.get(`${node}:${vehicle.path.kind}`) ?? [];
      const choices = available.filter(p => p.id !== vehicle.path.id && !p.blocked);
      const next = chooseRoute(choices, vehicle.path, hash(`${vehicle.id}:${vehicle.turns}`));
      const reverse = next.b === node;
      const previous = vehicle.distance, road = vehicle.path.kind === 'road';
      const factor = vehicle.model === 'truck' ? .72 : vehicle.model === 'bus' ? .82 : 1;
      let distance = previous + vehicle.path.speed * seconds * (road ? factor : 1) * (winter ? vehicle.path.surface === 'dirt' ? .5 : .78 : 1);
      if (road) {
        const length = vehicleLength(vehicle);
        const checkpoint = vehicle.path.checkpoint;
        if (checkpoint?.sides[vehicle.reverse ? 1 : 0] && !inspectionOpen(this.time,vehicle.path,vehicle.reverse)) {
          const gate = vehicle.reverse ? vehicle.path.length-checkpoint.distance-3 : checkpoint.distance-3;
          const stop = Math.max(0,gate-length/2-.35);
          if (previous <= stop) distance = Math.min(distance,stop);
        }
        // Busy junctions release alternating approaches; rural through roads flow freely.
        const stop = vehicle.path.length-length/2-.6;
        if (available.length >= 3 && vehicle.path.activity > .45 &&
          (this.time + hash(node)*8 + vehicle.path.seed*4) % 8 < 4 && previous <= stop) distance = Math.min(distance,stop);
        for (const ahead of lanes.get(laneKey(vehicle.path,vehicle.reverse)) ?? []) {
          if (ahead !== vehicle && ahead.path === vehicle.path && ahead.reverse === vehicle.reverse && ahead.distance > previous) distance = Math.min(distance,ahead.distance-(length+vehicleLength(ahead))/2-1);
        }
        // A tailback on the next segment propagates across the hex boundary.
        for (const ahead of lanes.get(laneKey(next,reverse)) ?? []) {
          if (ahead !== vehicle && ahead.path === next && ahead.reverse === reverse) distance = Math.min(distance,vehicle.path.length+ahead.distance-(length+vehicleLength(ahead))/2-1);
        }
      }
      vehicle.distance = Math.max(previous,distance);
      vehicle.waiting = vehicle.distance-previous < seconds*.5;
      if (vehicle.distance < vehicle.path.length) continue;
      vehicle.turns++;
      vehicle.distance -= vehicle.path.length;
      vehicle.history.unshift({ path: vehicle.path, reverse: vehicle.reverse });
      vehicle.history.length = Math.min(4, vehicle.history.length);
      vehicle.path = next;
      vehicle.reverse = reverse;
      const key = laneKey(next,reverse);
      if (!lanes.has(key)) lanes.set(key,[]);
      lanes.get(key)!.push(vehicle);
    }
  }

  position(vehicle: Vehicle, behind = 0) {
    let distance = vehicle.distance - behind;
    let path = vehicle.path, reverse = vehicle.reverse;
    for (const previous of vehicle.history) {
      if (distance >= 0) break;
      path = previous.path; reverse = previous.reverse; distance += path.length;
    }
    if (distance < 0) return null;
    return samplePath(path, distance, reverse, path.kind === 'road' ? path.surface === 'dirt' ? DIRT_WIDTH / 4 : LANE_OFFSET : 0);
  }
}
