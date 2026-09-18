import { Assets, Container, Graphics, Rectangle, Sprite, Texture } from 'pixi.js';
import { frontierPoint, type Crossing } from './border-crossings';
import { samplePath, inspectionOpen, type TransportPath, type TransportTile } from './transport';
import { worldApiUrl } from '../api/world';
import { ROAD_WIDTH, metres } from './map-scale';

interface Nation { flag?: string | null; primary_color: string; secondary_color?: string }
interface Flag { container: Container; strips: Sprite[]; url: string; phase: number; step: number }

/** Concrete apron, low metal roof, glazed frontage and a shallow cast shadow. */
function drawCheckpoint(gfx: Graphics, x: number, y: number, zoom: number, angle: number) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const point = (u: number, v: number, elevation = 0): [number, number] => [x + (u * c - v * s) * zoom, y + (u * s + v * c - elevation) * zoom];
  const panel = (u: number, v: number, w: number, h: number, color: number, elevation = 0, alpha = 1) => {
    gfx.poly([...point(u, v, elevation), ...point(u + w, v, elevation), ...point(u + w, v + h, elevation), ...point(u, v + h, elevation)])
      .fill({ color, alpha });
  };
  panel(-4.2, -2.8, 8.4, 5.6, 0x999c96);
  panel(-2.7, -1.4, 6.3, 3.7, 0x293237, 0, .22);
  panel(-3.2, -1.9, 6.4, 3.8, 0xb4b9b5);
  panel(-3.2, -1.9, 6.4, 3.8, 0x717c7b, .65);
  panel(-3, -1.7, 6, 3.4, 0x919c9b, .65);
  for (let u = -2.4; u < 3; u += 1.1) {
    gfx.moveTo(...point(u, -1.7, .65)).lineTo(...point(u, 1.7, .65)).stroke({ color: 0x717e7d, width: Math.max(.3, zoom * .12) });
  }
  // Glazing along the road-facing wall, with mullions instead of a bright icon window.
  for (let u = -2.7; u < 2; u += 1.15) {
    panel(u, -1.95, .9, .24, 0x465d65);
  }
  panel(1.5, .4, .8, 1, 0xc2c6c0, .7);
}

/** Static checkpoint furniture plus a small, independently animated flag layer. */
export class CheckpointLayer {
  readonly container = new Container();
  private furniture = new Graphics();
  private barriers = new Graphics();
  private gates: {path: TransportPath; reverse: boolean; closed: boolean; x: number; y: number; angle: number; zoom: number}[] = [];
  private flags = new Map<string, Flag>();
  private textures = new Map<string, Promise<Texture>>();
  private motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  private time = 0; private elapsed = 0;
  private trafficTime = 0;
  private destroyed = false;
  constructor() { this.container.eventMode = 'none'; this.container.addChild(this.furniture, this.barriers); }

  setView(crossings: Crossing[], tiles: Map<string, TransportTile>, nations: Map<string, Nation>, ox: number, oy: number, zoom: number, width: number, height: number, enabled: boolean) {
    const gfx = this.furniture.clear(), used = new Set<string>();
    this.gates = [];
    this.container.visible = enabled && zoom >= .85;
    if (this.container.visible) for (const crossing of crossings) {
      const { x, y } = crossing.point;
      const sx = x * zoom + ox, sy = y * zoom + oy;
      if (sx < -30 || sy < -30 || sx > width + 30 || sy > height + 30 || used.size > 46) continue;
      for (let side = 0; side < 2; side++) {
        if (used.size >= 48) continue;
        const sign = side ? 1 : -1, id = `${crossing.id}:${side}`;
        // Opposite verges keep both flags legible even on a north–south road.
        const site = crossing.sites?.[side];
        const pole = crossing.poles?.[side];
        if (crossing.showBuildings?.[side] && site) drawCheckpoint(gfx, site.x * zoom + ox, site.y * zoom + oy, zoom, site.angle);
        if (crossing.checkpoints[side]) for (const path of crossing.paths) {
          const at = frontierPoint(path, tiles.get(path.a)!, tiles.get(path.b)!).distance;
          const p = samplePath(path, at + sign * 3);
          this.gates.push({path, reverse: !!side, closed: crossing.closed, x:p.x*zoom+ox, y:p.y*zoom+oy, angle:p.angle, zoom});
        }
        // One flag per country per border stretch, not one per route.
        if (!pole || !crossing.showFlags?.[side]) continue;
        const px = pole.x * zoom + ox, py = pole.y * zoom + oy;
        const nation = nations.get(crossing.nations[side]);
        if (!nation) continue;
        // The pole and full cloth envelope were checked against owned territory.
        gfx.circle(px, py + zoom, .65 * zoom).fill(0x929992);
        gfx.moveTo(px + zoom * .5, py + zoom).lineTo(px + zoom * 3, py + zoom * 3)
          .stroke({ color: 0x26312b, alpha: .18, width: zoom * .3 });
        gfx.moveTo(px, py + zoom).lineTo(px, py - 10 * zoom).stroke({ color: 0xadb5b4, width: Math.max(.6, zoom * .22) });
        used.add(id);
        let flag = this.flags.get(id);
        const ref = nation.flag || '';
        // Include the world in the cache key so another lobby cannot reuse this flag.
        const url = ref.startsWith('/map/flags/') ? worldApiUrl(ref) : ref;
        if (flag && flag.url !== url) { this.release(flag); this.flags.delete(id); flag = undefined; }
        if (!flag) {
          const container = new Container();
          this.container.addChild(container);
          flag = { container, strips: [], url, phase: side * 1.3, step: 1 };
          this.flags.set(id, flag);
          if (url && /^(https?:\/\/|\/[^/]|data:image\/)/i.test(url)) this.loadFlag(flag, url);
        }
        flag.container.position.set(px, py - 9.5 * zoom); flag.container.scale.set(zoom);
      }
    }
    for (const [id, flag] of this.flags) if (!used.has(id)) { this.release(flag); this.flags.delete(id); }
    this.pose();
    this.drawBarriers();
  }

  private drawBarriers() {
    const gfx = this.barriers.clear();
    for (const gate of this.gates) {
      const {path,reverse,zoom,angle,x,y} = gate;
      const direction = reverse ? -1 : 1, nx = -Math.sin(angle)*direction, ny = Math.cos(angle)*direction;
      const bx = x+nx*(ROAD_WIDTH/2+metres(.25))*zoom, by = y+ny*(ROAD_WIDTH/2+metres(.25))*zoom;
      const closed = gate.closed || !inspectionOpen(this.trafficTime,path,reverse);
      // From above a raised boom is foreshortened along the lane, never a
      // screen-space diagonal. A lowered boom crosses its incoming lane.
      const ex = closed ? x+nx*.05*zoom : bx+Math.cos(angle)*direction*.3*zoom;
      const ey = closed ? y+ny*.05*zoom : by+Math.sin(angle)*direction*.3*zoom;
      gfx.circle(bx,by,zoom*metres(.3)).fill(0x51575c);
      gfx.moveTo(bx,by).lineTo(ex,ey).stroke({color:0xe7e4da,width:zoom*metres(.12)});
      if (closed) for (let i=0;i<3;i++) {
        const t=i/3, t2=t+.16;
        gfx.moveTo(bx+(ex-bx)*t,by+(ey-by)*t).lineTo(bx+(ex-bx)*t2,by+(ey-by)*t2).stroke({color:0xaf5148,width:zoom*metres(.12)});
      }
    }
  }

  private loadFlag(flag: Flag, url: string) {
    if (!this.textures.has(url)) this.textures.set(url, Assets.load<Texture>({ src: url, loadParser: 'loadTextures' }));
    void this.textures.get(url)!.then(texture => {
      if (this.destroyed || flag.container.destroyed) return;
      // Preserve each uploaded flag's aspect ratio.
      const width = Math.min(8, 4.5 * texture.width / texture.height), height = width * texture.height / texture.width;
      flag.step = width / 8;
      for (let i = 0; i < 8; i++) {
        const slice = new Texture({ source: texture.source, frame: new Rectangle(texture.frame.x + i * texture.width / 8, texture.frame.y, texture.width / 8, texture.height) });
        const sprite = new Sprite(slice); sprite.width = flag.step + .015; sprite.height = height;
        flag.strips.push(sprite); flag.container.addChild(sprite);
      }
      this.pose();
    }).catch(() => { /* Leave the pole bare when the real flag is unavailable. */ });
  }
  private pose() {
    for (const flag of this.flags.values()) {
      for (let i = 0; i < flag.strips.length; i++) {
        const waveAt = (n: number) => this.motion.matches ? 0 : Math.sin(this.time * 3 + flag.phase - n * .6) * .45 * n / 8;
        const wave = waveAt(i), tilt = Math.atan2(waveAt(i + 1) - wave, flag.step);
        const strip = flag.strips[i];
        strip.position.set(i * flag.step, wave);
        // Shear neighbouring strips to meet at the same height, avoiding a
        // staircase silhouette when the player zooms in on a waving flag.
        strip.skew.y = tilt;
        strip.scale.x = (flag.step + .015) / strip.texture.width / Math.cos(tilt);
      }
    }
  }
  update(deltaMs: number, trafficTime?: number) {
    if (!this.container.visible || document.hidden) return;
    this.elapsed += Math.min(100, Math.max(0, deltaMs));
    if (this.elapsed < 1000 / 30) return;
    if (!this.motion.matches) this.time += this.elapsed / 1000;
    this.trafficTime = trafficTime ?? this.time;
    this.elapsed = 0; this.pose(); this.drawBarriers();
  }
  private release(flag: Flag) {
    for (const sprite of flag.strips) sprite.texture.destroy(false);
    flag.container.destroy({ children: true });
  }
  destroy() {
    this.destroyed = true;
    for (const flag of this.flags.values()) this.release(flag);
    this.flags.clear(); this.textures.clear(); this.container.destroy({ children: true });
  }
}
