import { Container, Graphics } from 'pixi.js';
import { TrafficSimulation, vehicleLength, type TransportNetwork, type TransportPath } from './transport';
import { VEHICLE_DIMENSIONS, TRAIN_LENGTH, TRAIN_SPACING, metres } from './map-scale';

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
      const point = this.simulation.position(vehicle, carriage * TRAIN_SPACING);
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
        const length = rail ? TRAIN_LENGTH : vehicleLength(vehicle), half = length/2;
        const width = rail ? metres(2.85) : VEHICLE_DIMENSIONS[vehicle.model ?? 'sedan'].width;
        sprite.clear();
        // Every part is in the same world units as the road and city roofs.
        sprite.roundRect(-half+.03,-width/2+.04,length,width,width*.14).fill({color:0x101c21,alpha:.42});
        if (!rail) for(const axle of [-.29,.28]) for(const side of [-1,1]) {
          sprite.rect(length*axle-length*.06,side*width*.49-width*.06,length*.12,width*.12).fill(0x20282a);
        }
        sprite.roundRect(-half,-width/2,length,width,width*(model==='truck'?.07:.18)).fill(0xffffff);
        sprite.roundRect(-length*.22,-width*.35,length*.52,width*.70,width*.12).fill(0xc9d1d2);
        sprite.poly([length*.14,-width*.36,length*.28,-width*.30,length*.28,width*.30,length*.14,width*.36]).fill(0x283f49);
        if(model==='sedan'||model==='compact')sprite.poly([-length*.30,-width*.28,-length*.18,-width*.35,-length*.18,width*.35,-length*.30,width*.28]).fill(0x344954);
        if(model==='truck')sprite.rect(-half+.03,-width*.46,length*.68,width*.92).fill(0xc3c6bf);
        if(model==='bus'||rail)for(let x=-half+length*.10;x<half-length*.22;x+=length*.12){
          sprite.rect(x,-width*.45,length*.065,width*.13).rect(x,width*.32,length*.065,width*.13).fill(0x344951);
        }
        // Narrow highlights describe the roof; lamps never glow beyond the body.
        sprite.moveTo(-length*.12,-width*.27).lineTo(length*.10,-width*.27).stroke({color:0xffffff,width:width*.035,alpha:.7});
        if(!rail){
          sprite.rect(half-length*.045,-width*.37,length*.045,width*.18).rect(half-length*.045,width*.19,length*.045,width*.18).fill(0xeee9d5);
          sprite.rect(-half,-width*.38,length*.04,width*.19).rect(-half,width*.19,length*.04,width*.19).fill(vehicle.waiting?0xe54b36:0x6f302a);
        }
      }
      sprite.visible = true;
      sprite.position.set(x, y); sprite.rotation = point.angle;
      sprite.scale.set(this.zoom);
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
