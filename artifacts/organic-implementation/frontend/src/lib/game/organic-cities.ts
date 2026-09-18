import { Container, Graphics } from 'pixi.js';
import { axialToPixel, HEX_SIZE } from './hex-math';
import { hash, type Point, type TransportNetwork } from './transport';
import { metres, ROAD_WIDTH } from './map-scale';
import { installationTypeFor } from './tile-textures';
import { citySurface } from './city-surface';
import { SettlementLayout } from './settlement-layout';
export interface CityTile {
    q: number;
    r: number;
    terrain: number;
    owner?: string | null;
    cityId?: number | null;
    buildingCount?: number;
    buildingKeys?: string[];
}
export interface Roof {
    x: number;
    y: number;
    width: number;
    height: number;
    angle: number;
    tone: number;
    kind?: 'hip' | 'gable' | 'villa' | 'terrace' | 'apartment' | 'warehouse' | 'barn';
    storeys?: number;
}
export interface CityPlan {
    streets: Point[][];
    roofs: Roof[];
    trees: Point[];
    gardens: Point[];
    installation?: string;
}
const inside = (p: Point) => Math.abs(p.x) < HEX_SIZE * .93 && Math.abs(p.y) < HEX_SIZE * .805 && Math.abs(p.x) * .8660254 + Math.abs(p.y) * .5 < HEX_SIZE * .805;
export function nearestStreet(p: Point, streets: Point[][]) {
    let distance = Infinity, angle = 0;
    for (const line of streets)
        for (let i = 1; i < line.length; i++) {
            const a = line[i - 1], b = line[i], dx = b.x - a.x, dy = b.y - a.y;
            const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy || 1)));
            const d = Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
            if (d < distance) {
                distance = d;
                angle = Math.atan2(dy, dx);
            }
        }
    return { distance, angle };
}
/** Stable world geometry. Zoom changes detail, never placement or object size. */
export function planCity(tile: CityTile, network: TransportNetwork): CityPlan {
    const key = `${tile.q},${tile.r}`, [cx, cy] = axialToPixel(tile.q, tile.r, HEX_SIZE);
    const n = tile.buildingCount ?? 0, tier = n >= 15 ? 3 : n >= 8 ? 2 : n >= 3 ? 1 : 0;
    const radius = [6.8, 8.8, 11.8, 13][tier], limit = [18, 38, 80, 135][tier];
    const rotation = hash(`city:${tile.cityId ?? key}`) * .7 - .35, streets: Point[][] = [];
    const rotate = (x: number, y: number) => ({ x: x * Math.cos(rotation) - y * Math.sin(rotation), y: x * Math.sin(rotation) + y * Math.cos(rotation) });
    for (const path of network.adjacent.get(`${key}:road`) ?? []) {
        const points = path.a === key ? path.points : [...path.points].reverse();
        streets.push(points.map(p => ({ x: p.x - cx, y: p.y - cy })).filter(p => Math.hypot(p.x, p.y) < HEX_SIZE * 1.15));
    }
    // Local lanes share junction coordinates. Small bends break up the blocks,
    // while the town's primary streets still join the actual transport network.
    const offsets = tier === 3 ? [-8.5, -4.6, 0, 4.1, 8.6] : tier === 2 ? [-5.1, 0, 4.3] : tier === 1 ? [-3.5, 0, 3.5] : [0];
    const node = (x: number, y: number) => rotate(x + Math.sin(y * .32 + rotation) * 1.15 + Math.sin(x * .29) * .35, y + Math.sin(x * .31) * 1.3 + Math.sin(y * .41) * .45);
    for (const y of offsets) {
        const end = Math.sqrt(Math.max(0, radius * radius - y * y)) * .92;
        streets.push(Array.from({ length: 25 }, (_, i) => node((i / 24 * 2 - 1) * end, y)));
    }
    for (const [column, x] of offsets.entries()) {
        const end = Math.sqrt(Math.max(0, radius * radius - x * x)) * .92;
        const start = column % 3 === 1 ? (offsets[1] ?? -end) : -end;
        const finish = column % 3 === 2 ? (offsets[offsets.length - 2] ?? end) : end;
        streets.push(Array.from({ length: 25 }, (_, i) => node(x, start + (finish - start) * i / 24)));
    }
    const roads = streets.filter(s => s.length > 1), roofs: Roof[] = [], trees: Point[] = [], gardens: Point[] = [];
    const exclusionRoads = [...roads];
    for (const path of network.adjacent.get(`${key}:rail`) ?? [])
        exclusionRoads.push(path.points.map(p => ({ x: p.x - cx, y: p.y - cy })));
    const candidates: Point[] = [];
    for (let y = -radius; y <= radius; y += .65)
        for (let x = -radius; x <= radius; x += .65) {
            const seed = hash(`${key}:${x.toFixed(2)}:${y.toFixed(2)}`);
            candidates.push({ x: x + (seed - .5) * .4, y: y + (hash(`${seed}`) - .5) * .4 });
        }
    candidates.sort((a, b) => Math.hypot(a.x, a.y) - Math.hypot(b.x, b.y));
    for (const p of candidates) {
        if (roofs.length >= limit)
            break;
        const seed = hash(`${key}:${p.x}:${p.y}`), edge = Math.hypot(p.x, p.y) / radius;
        if (edge > 1 || seed < Math.max(0, edge - .65) * .7)
            continue;
        const near = nearestStreet(p, roads), width = metres(6.5 + seed * 3.5), height = metres(8 + hash(`${seed}:h`) * 4);
        if (near.distance < ROAD_WIDTH / 2 + Math.min(width, height) / 2 + .12 || near.distance > 3.5)
            continue;
        const angle = near.angle, co = Math.cos(angle), si = Math.sin(angle);
        const corners = [[-1, -1], [-1, 1], [1, 1], [1, -1]].map(([a, b]) => ({ x: p.x + a * width / 2 * co - b * height / 2 * si, y: p.y + a * width / 2 * si + b * height / 2 * co }));
        if (!corners.every(inside))
            continue;
        if (roofs.some(r => Math.hypot(r.x - p.x, r.y - p.y) < (Math.hypot(r.width, r.height) + Math.hypot(width, height)) * .43 + .06))
            continue;
        if (corners.some(c => nearestStreet(c, exclusionRoads).distance < metres(2.6)))
            continue;
        roofs.push({ ...p, width, height, angle, tone: Math.floor(seed * 4) });
        const garden = { x: p.x - Math.sin(angle) * (height / 2 + .5), y: p.y + Math.cos(angle) * (height / 2 + .5) };
        if (inside(garden) && nearestStreet(garden, roads).distance > .85)
            gardens.push(garden);
    }
    for (let i = 0; i < 1100; i++) {
        const a = hash(`${key}:tree:${i}`) * Math.PI * 2, r = Math.sqrt(hash(`${key}:radius:${i}`)) * radius * 1.05;
        const p = { x: Math.cos(a) * r, y: Math.sin(a) * r };
        if (inside(p) && nearestStreet(p, exclusionRoads).distance > .52 && !roofs.some(b => Math.hypot(b.x - p.x, b.y - p.y) < Math.hypot(b.width, b.height) / 2 + .12))
            trees.push(p);
    }
    return { streets: roads, roofs, trees, gardens };
}
function roofPoints(b: Roof, x0: number, y0: number, x1: number, y1: number, ox = 0, oy = 0) {
    const co = Math.cos(b.angle), si = Math.sin(b.angle);
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].flatMap(([x, y]) => [b.x + x * co - y * si + ox, b.y + x * si + y * co + oy]);
}
function planInstallation(tile: CityTile, type: string): CityPlan {
    const seed = hash(`${tile.q},${tile.r}`), roofs: Roof[] = [];
    for (let i = 0; i < (type === 'farm' ? 2 : type === 'windmill' ? 0 : 5); i++)
        roofs.push({ x: -7 + (i % 3) * 5, y: -4 + Math.floor(i / 3) * 7, width: metres(type === 'farm' ? 12 : 22), height: metres(type === 'farm' ? 16 : 30), angle: .12, tone: type === 'farm' ? 0 : 2, kind: type === 'farm' ? 'barn' : 'warehouse', storeys: 1 });
    return { installation: type, streets: [[{ x: -11, y: 1 }, { x: 0, y: 2 }, { x: 11, y: 1 }]], roofs, trees: [{ x: -10, y: -6 }, { x: 9, y: -7 }, { x: 8, y: 7 }], gardens: [] };
}
export function drawCity(plan: CityPlan, detail: number): Container {
    if ((!plan.installation && detail > 0) || plan.installation === 'farm') return citySurface(plan, detail);
    const g = new Graphics(), tones = [0x9a7055, 0x846957, 0x868780, 0xb39473];
    for (const b of plan.roofs)
        g.poly(roofPoints(b, -b.width * .64, -b.height * .59, b.width * .64, b.height * .59));
    g.fill({ color: 0x9d9983, alpha: .65 });
    for (const p of plan.gardens)
        g.roundRect(p.x - .48, p.y - .38, .96, .76, .08);
    g.fill({ color: 0x687a4c, alpha: .72 });
    for (const line of plan.streets) {
        g.moveTo(line[0].x, line[0].y);
        for (const p of line.slice(1))
            g.lineTo(p.x, p.y);
    }
    g.stroke({ color: 0x9a9485, width: metres(4.9), alpha: .8, cap: 'round', join: 'round' });
    for (const line of plan.streets) {
        g.moveTo(line[0].x, line[0].y);
        for (const p of line.slice(1))
            g.lineTo(p.x, p.y);
    }
    g.stroke({ color: 0x575b58, width: metres(3.8), cap: 'round', join: 'round' });
    for (const b of plan.roofs)
        g.poly(roofPoints(b, -b.width / 2, -b.height / 2, b.width / 2, b.height / 2, .15, .20));
    g.fill({ color: 0x16251c, alpha: .48 });
    for (let tone = 0; tone < 4; tone++) {
        for (const b of plan.roofs)
            if (b.tone === tone)
                g.poly(roofPoints(b, -b.width / 2, -b.height / 2, b.width / 2, b.height / 2));
        g.fill(tones[tone]);
    }
    if (detail > 0) {
        for (const b of plan.roofs)
            g.poly(roofPoints(b, 0, -b.height / 2, b.width / 2, b.height / 2));
        g.fill({ color: 0x302d26, alpha: .21 });
        for (const b of plan.roofs) {
            const co = Math.cos(b.angle), si = Math.sin(b.angle);
            g.moveTo(b.x + si * b.height / 2, b.y - co * b.height / 2).lineTo(b.x - si * b.height / 2, b.y + co * b.height / 2);
        }
        g.stroke({ color: 0xd6c0a0, width: .035, alpha: .6 });
    }
    if (detail > 1) {
        for (const b of plan.roofs) {
            g.poly(roofPoints(b, -b.width * .34, -b.height * .24, -b.width * .13, b.height * .05));
        }
        g.fill({ color: 0x35454a, alpha: .8 });
        for (const b of plan.roofs)
            g.poly(roofPoints(b, b.width * .19, -b.height * .31, b.width * .31, -b.height * .17, .025, .025));
        g.fill({ color: 0x302f29, alpha: .7 });
        for (const b of plan.roofs)
            g.poly(roofPoints(b, b.width * .19, -b.height * .31, b.width * .29, -b.height * .19));
        g.fill(0xae9b82);
        for (const b of plan.roofs) {
            const co = Math.cos(b.angle), si = Math.sin(b.angle);
            for (let y = -b.height / 2 + .13; y < b.height / 2; y += .16) {
                g.moveTo(b.x - b.width / 2 * co - y * si, b.y - b.width / 2 * si + y * co).lineTo(b.x + b.width / 2 * co - y * si, b.y + b.width / 2 * si + y * co);
            }
        }
        g.stroke({ color: 0x342f28, width: .018, alpha: .27 });
    }
    for (const p of plan.trees)
        g.circle(p.x + .12, p.y + .16, .4);
    g.fill({ color: 0x182b20, alpha: .45 });
    for (const p of plan.trees)
        g.circle(p.x, p.y, .34).circle(p.x + .19, p.y - .10, .25).circle(p.x - .13, p.y + .14, .26);
    g.fill(0x3b5231);
    if (detail > 0) {
        for (const p of plan.trees)
            g.circle(p.x - .09, p.y - .10, .20).circle(p.x + .12, p.y - .16, .16);
        g.fill({ color: 0x72805a, alpha: .65 });
    }
    if (plan.installation === 'windmill')
        for (const [x, y] of [[-7, -5], [5, -6], [-3, 6], [8, 5]]) {
            g.moveTo(x, y).lineTo(x + .8, y + 1.5).stroke({ color: 0x23332a, width: .16, alpha: .45 });
            g.circle(x, y, .14).fill(0xe2e3d8);
            for (let i = 0; i < 3; i++) {
                const a = i * Math.PI * 2 / 3 + .3;
                g.moveTo(x, y).lineTo(x + Math.cos(a) * 2.1, y + Math.sin(a) * 2.1);
            }
            g.stroke({ color: 0xd4d9d2, width: .10, cap: 'round' });
        }
    return g;
}
export class OrganicCities {
    readonly container = new Container();
    private cache = new Map<string, {
        g: Container;
        plan: CityPlan;
        detail: number;
        used: number;
        signature: string;
    }>();
    private frame = 0;
    readonly layout = new SettlementLayout();
    setWorld(tiles: Map<string, CityTile>, network: TransportNetwork, lakes: Set<string>) { this.layout.setWorld(tiles, network, lakes); }
    clear() { for (const e of this.cache.values())
        e.g.destroy({ children: true, texture: true, textureSource: true }); this.cache.clear(); }
    draw(visible: Array<{
        tile: CityTile;
        sx: number;
        sy: number;
    }>, count: number, network: TransportNetwork, z: number, enabled: boolean, lakes: Set<string>, width = Infinity, height = Infinity) {
        this.container.visible = enabled;
        this.container.removeChildren();
        this.frame++;
        if (!enabled)
            return;
        const detail = z >= 12 ? 2 : z >= 1.5 ? 1 : 0;
        for (let i = 0; i < count; i++) {
            const { tile, sx, sy } = visible[i], key = `${tile.q},${tile.r}`;
            if (sx + 18 * z < 0 || sy + 18 * z < 0 || sx - 18 * z > width || sy - 18 * z > height) continue;
            if (tile.terrain === 0 || lakes.has(key))
                continue;
            const installation = tile.cityId == null ? installationTypeFor(tile.buildingKeys ?? []) : null;
            if (!installation && !this.layout.hasSettlement(tile))
                continue;
            const arms = [...network.adjacent.get(`${key}:road`) ?? [], ...network.adjacent.get(`${key}:rail`) ?? []];
            const signature = `${this.layout.revision}:${tile.cityId}:${tile.buildingCount}:${tile.terrain}:${installation}:` + arms.map(p => `${p.id}:${p.length}:${p.points[0].x}:${p.points[0].y}`).join('|');
            let entry = this.cache.get(key);
            if (entry && entry.signature !== signature) {
                entry.g.destroy({ children: true, texture: true, textureSource: true });
                this.cache.delete(key);
                entry = undefined;
            }
            if (!entry) {
                const plan = installation ? planInstallation(tile, installation) : this.layout.plan(tile);
                entry = { g: drawCity(plan, detail), plan, detail, used: this.frame, signature };
                this.cache.set(key, entry);
            }
            else if (entry.detail !== detail) {
                entry.g.destroy({ children: true, texture: true, textureSource: true });
                entry.g = drawCity(entry.plan, detail);
                entry.detail = detail;
            }
            entry.used = this.frame;
            entry.g.position.set(sx, sy);
            entry.g.scale.set(z);
            this.container.addChild(entry.g);
        }
        let pixels = [...this.cache.values()].reduce((sum, e) => sum + (e.plan.installation ? 0 : (36 * [8, 24, 64][e.detail]) ** 2), 0);
        if (this.cache.size > 96 || pixels > 24 * 1024 * 1024)
            for (const [k, e] of [...this.cache].sort((a, b) => a[1].used - b[1].used)) {
                if (e.used !== this.frame) {
                    e.g.destroy({ children: true, texture: true, textureSource: true });
                    this.cache.delete(k);
                    pixels -= e.plan.installation ? 0 : (36 * [8, 24, 64][e.detail]) ** 2;
                    if (this.cache.size <= 96 && pixels <= 24 * 1024 * 1024)
                        break;
                }
            }
    }
    destroy() { this.clear(); this.layout.clear(); this.container.destroy({ children: true }); }
}
