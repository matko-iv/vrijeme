import { Container, Graphics } from 'pixi.js';
import { TrafficSimulation, vehicleLength, type TransportNetwork, type TransportPath } from './transport';

/** Small pooled vehicle shapes; moving traffic never redraws the terrain. */
export class TrafficLayer {
  readonly container = new Container();
  readonly simulation = new TrafficSimulation();
  private pool: Graphics[] = [];
  private styles: string[] = [];
  private elapsed = 0;
  private populationMs = 0;
  private visiblePaths: TransportPath[] = [];
  private ox = 0; private oy = 0; private zoom = 1;
  private width = 0; private height = 0;
  private winter = false;
  private enabled = true;
  private reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  private onMotion = () => { if (this.reducedMotion.matches) this.hide(); };

  constructor() {
    this.container.eventMode = 'none';
    this.reducedMotion.addEventListener('change', this.onMotion);
  }
  setNetwork(network: TransportNetwork) { this.simulation.setNetwork(network); }
  setView(paths: TransportPath[], ox: number, oy: number, zoom: number, width: number, height: number, enabled: boolean, winter: boolean) {
    Object.assign(this, { ox, oy, zoom, width, height, enabled, winter });
    this.visiblePaths = paths;
    if (enabled && zoom >= .65) this.simulation.populate(paths, winter);
    else this.simulation.vehicles.clear();
    this.draw();
  }
  update(deltaMs: number) {
    if (!this.enabled || this.zoom < .65 || document.hidden || this.reducedMotion.matches) { this.hide(); return; }
    this.elapsed += Math.min(100, Math.max(0, deltaMs));
    this.populationMs += Math.min(100, Math.max(0, deltaMs));
    if (this.populationMs >= 2000) {
      this.simulation.populate(this.visiblePaths, this.winter);
      this.populationMs = 0;
    }
    if (this.elapsed < 1000 / 30) return;
    this.simulation.step(this.elapsed, this.winter);
    this.elapsed = 0;
    this.draw();
  }
  private hide() { this.container.visible = false; this.elapsed = 0; }
  private draw() {
    if (!this.enabled || this.zoom < .65 || document.hidden || this.reducedMotion.matches) { this.hide(); return; }
    this.container.visible = true;
    let used = 0;
    for (const vehicle of this.simulation.vehicles.values()) for (let carriage = 0; carriage < vehicle.carriages; carriage++) {
      const point = this.simulation.position(vehicle, carriage * 4.3);
      if (!point) continue;
      const x = point.x * this.zoom + this.ox, y = point.y * this.zoom + this.oy;
      if (x < -12 || y < -12 || x > this.width + 12 || y > this.height + 12) continue;
      const index = used++;
      let sprite = this.pool[index];
      if (!sprite) {
        sprite = new Graphics();
        this.pool.push(sprite); this.container.addChild(sprite);
      }
      const rail = vehicle.path.kind === 'rail', model = rail ? 'rail' : vehicle.model ?? 'sedan';
      const style = `${model}:${!!vehicle.waiting}`;
      if (this.styles[index] !== style) {
        this.styles[index] = style;
        const length = rail ? 3.2 : vehicleLength(vehicle), half = length/2;
        const width = model === 'compact' ? 1 : model === 'sedan' ? 1.1 : 1.3;
        sprite.clear().roundRect(-half,-width/2,length,width,model === 'truck' ? .08 : .23).fill(0xffffff);
        sprite.rect(half-.95,-width*.4,.45,width*.8).fill(0x283f4b);
        if (model === 'bus') for (let x=-half+.4;x<half-1;x+=.7) {
          sprite.rect(x,-.6,.4,.2).rect(x,.4,.4,.2).fill(0x344c59);
        }
        if (model === 'truck') sprite.rect(-half+.15,-.55,length-1.55,1.1).fill(0xc4c8bf)
          .rect(half-1.4,-.6,.15,1.2).fill(0x344c59);
        if (model === 'van') sprite.rect(-half+.5,-.35,length-1.8,.7).fill(0xdce0dc);
        if (!rail) sprite.rect(-half,-width*.4,.2,.25).rect(-half,width*.4-.25,.2,.25).fill(vehicle.waiting ? 0xff4834 : 0x7c3029);
      }
      sprite.visible = true;
      sprite.position.set(x, y); sprite.rotation = point.angle;
      sprite.scale.set(this.zoom * (vehicle.path.kind === 'rail' ? 1.18 : 1));
      sprite.tint = carriage ? 0xaebbc1 : vehicle.color;
    }
    for (let i = used; i < this.pool.length; i++) this.pool[i].visible = false;
  }
  destroy() {
    this.reducedMotion.removeEventListener('change', this.onMotion);
    this.simulation.vehicles.clear(); this.pool = [];
    this.visiblePaths = [];
    this.container.destroy({ children: true });
  }
}
