import { Container, Graphics } from 'pixi.js';
import { axialToPixel, HEX_SIZE } from './hex-math';
import { hash, type Point, type TransportNetwork } from './transport';
import { metres, ROAD_WIDTH } from './map-scale';

export interface CityTile {q:number;r:number;terrain:number;cityId?:number|null;buildingCount?:number;buildingKeys?:string[]}
export interface Roof {x:number;y:number;width:number;height:number;angle:number;tone:number}
export interface CityPlan {streets:Point[][];roofs:Roof[];trees:Point[];gardens:Point[]}
const inside=(p:Point)=>Math.abs(p.x)<HEX_SIZE*.93 && Math.abs(p.y)<HEX_SIZE*.805 && Math.abs(p.x)*.8660254+Math.abs(p.y)*.5<HEX_SIZE*.805;

export function nearestStreet(p:Point,streets:Point[][]){
  let distance=Infinity,angle=0;
  for(const line of streets)for(let i=1;i<line.length;i++){
    const a=line[i-1],b=line[i],dx=b.x-a.x,dy=b.y-a.y;
    const t=Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/(dx*dx+dy*dy||1)));
    const d=Math.hypot(p.x-a.x-t*dx,p.y-a.y-t*dy);
    if(d<distance){distance=d;angle=Math.atan2(dy,dx);}
  }
  return {distance,angle};
}

/** Stable world geometry. Zoom changes detail, never placement or object size. */
export function planCity(tile:CityTile,network:TransportNetwork):CityPlan{
  const key=`${tile.q},${tile.r}`,[cx,cy]=axialToPixel(tile.q,tile.r,HEX_SIZE);
  const n=tile.buildingCount??0,tier=n>=15?3:n>=8?2:n>=3?1:0;
  const radius=[4.8,7.5,10.5,13][tier],limit=[9,26,55,100][tier];
  const rotation=hash(key)*Math.PI,streets:Point[][]=[];
  const rotate=(x:number,y:number)=>({x:x*Math.cos(rotation)-y*Math.sin(rotation),y:x*Math.sin(rotation)+y*Math.cos(rotation)});
  for(const path of network.adjacent.get(`${key}:road`)??[]){
    const points=path.a===key?path.points:[...path.points].reverse();
    streets.push(points.map(p=>({x:p.x-cx,y:p.y-cy})).filter(p=>Math.hypot(p.x,p.y)<HEX_SIZE*1.15));
  }
  // A centre lane and ring streets give every plot a connected access route.
  streets.push(Array.from({length:13},(_,i)=>rotate((i/12*2-1)*radius,Math.sin(i/12*Math.PI*2)*.6)));
  if(tier>0)streets.push(Array.from({length:13},(_,i)=>rotate(Math.sin(i/12*Math.PI)*.65,(i/12*2-1)*radius)));
  for(const r of tier===3?[5,9]:tier===2?[6.5]:tier===1?[4.5]:[]){
    streets.push(Array.from({length:33},(_,i)=>{const a=i/32*Math.PI*2;return rotate(Math.cos(a)*(r+.35*Math.sin(a*3)),Math.sin(a)*r*.8);}));
  }
  const roads=streets.filter(s=>s.length>1),roofs:Roof[]=[],trees:Point[]=[],gardens:Point[]=[];
  const candidates:Point[]=[];
  for(let y=-radius;y<=radius;y+=1.9)for(let x=-radius;x<=radius;x+=1.9){
    const seed=hash(`${key}:${x.toFixed(2)}:${y.toFixed(2)}`);
    candidates.push({x:x+(seed-.5)*.4,y:y+(hash(`${seed}`)-.5)*.4});
  }
  candidates.sort((a,b)=>Math.hypot(a.x,a.y)-Math.hypot(b.x,b.y));
  for(const p of candidates){
    if(roofs.length>=limit)break;
    const seed=hash(`${key}:${p.x}:${p.y}`),edge=Math.hypot(p.x,p.y)/radius;
    if(edge>1 || seed<Math.max(0,edge-.65)*.7)continue;
    const near=nearestStreet(p,roads),width=metres(7+seed*4),height=metres(8+hash(`${seed}:h`)*5);
    if(near.distance<ROAD_WIDTH/2+Math.hypot(width,height)/2+.18 || near.distance>3.5)continue;
    const angle=near.angle,co=Math.cos(angle),si=Math.sin(angle);
    const corners=[[-1,-1],[-1,1],[1,1],[1,-1]].map(([a,b])=>({x:p.x+a*width/2*co-b*height/2*si,y:p.y+a*width/2*si+b*height/2*co}));
    if(!corners.every(inside))continue;
    if(roofs.some(r=>Math.hypot(r.x-p.x,r.y-p.y)<(Math.hypot(r.width,r.height)+Math.hypot(width,height))*.5+.12))continue;
    roofs.push({...p,width,height,angle,tone:Math.floor(seed*4)});
    const garden={x:p.x-Math.sin(angle)*(height/2+.5),y:p.y+Math.cos(angle)*(height/2+.5)};
    if(inside(garden)&&nearestStreet(garden,roads).distance>.85)gardens.push(garden);
  }
  for(let i=0;i<70;i++){
    const a=hash(`${key}:tree:${i}`)*Math.PI*2,r=Math.sqrt(hash(`${key}:radius:${i}`))*radius*1.05;
    const p={x:Math.cos(a)*r,y:Math.sin(a)*r};
    if(inside(p)&&nearestStreet(p,roads).distance>.9&&!roofs.some(b=>Math.hypot(b.x-p.x,b.y-p.y)<Math.hypot(b.width,b.height)/2+.5))trees.push(p);
  }
  return {streets:roads,roofs,trees,gardens};
}

function roofPoints(b:Roof,x0:number,y0:number,x1:number,y1:number,ox=0,oy=0){
  const co=Math.cos(b.angle),si=Math.sin(b.angle);
  return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]].flatMap(([x,y])=>[b.x+x*co-y*si+ox,b.y+x*si+y*co+oy]);
}
export function drawCity(plan:CityPlan,detail:number):Graphics{
  const g=new Graphics(),tones=[0x9a7055,0x846957,0x868780,0xb39473];
  for(const p of plan.gardens)g.ellipse(p.x,p.y,.62,.48);
  g.fill({color:0x66744e,alpha:.6});
  for(const line of plan.streets){g.moveTo(line[0].x,line[0].y);for(const p of line.slice(1))g.lineTo(p.x,p.y);}
  g.stroke({color:0x9a9485,width:metres(5.7),alpha:.8,cap:'round',join:'round'});
  for(const line of plan.streets){g.moveTo(line[0].x,line[0].y);for(const p of line.slice(1))g.lineTo(p.x,p.y);}
  g.stroke({color:0x575b58,width:metres(4.4),cap:'round',join:'round'});
  for(const b of plan.roofs)g.poly(roofPoints(b,-b.width/2,-b.height/2,b.width/2,b.height/2,.15,.20));
  g.fill({color:0x16251c,alpha:.48});
  for(let tone=0;tone<4;tone++){
    for(const b of plan.roofs)if(b.tone===tone)g.poly(roofPoints(b,-b.width/2,-b.height/2,b.width/2,b.height/2));
    g.fill(tones[tone]);
  }
  if(detail>0){
    for(const b of plan.roofs)g.poly(roofPoints(b,0,-b.height/2,b.width/2,b.height/2));
    g.fill({color:0x302d26,alpha:.21});
    for(const b of plan.roofs){const co=Math.cos(b.angle),si=Math.sin(b.angle);g.moveTo(b.x+si*b.height/2,b.y-co*b.height/2).lineTo(b.x-si*b.height/2,b.y+co*b.height/2);}
    g.stroke({color:0xd6c0a0,width:.035,alpha:.6});
  }
  if(detail>1){
    for(const b of plan.roofs){
      g.poly(roofPoints(b,-b.width*.34,-b.height*.24,-b.width*.13,b.height*.05));
    }g.fill({color:0x35454a,alpha:.8});
    for(const b of plan.roofs){const co=Math.cos(b.angle),si=Math.sin(b.angle);for(let y=-b.height/2+.13;y<b.height/2;y+=.16){g.moveTo(b.x-b.width/2*co-y*si,b.y-b.width/2*si+y*co).lineTo(b.x+b.width/2*co-y*si,b.y+b.width/2*si+y*co);}}
    g.stroke({color:0x342f28,width:.018,alpha:.27});
  }
  for(const p of plan.trees)g.circle(p.x+.12,p.y+.16,.4);g.fill({color:0x182b20,alpha:.45});
  for(const p of plan.trees)g.circle(p.x,p.y,.34);g.fill(0x3b5231);
  if(detail>0){for(const p of plan.trees)g.circle(p.x-.09,p.y-.10,.23);g.fill({color:0x72805a,alpha:.65});}
  return g;
}

export class OrganicCities {
  readonly container=new Container();
  private cache=new Map<string,{g:Graphics;plan:CityPlan;detail:number;used:number}>();
  private frame=0;
  clear(){for(const e of this.cache.values())e.g.destroy();this.cache.clear();}
  draw(visible:Array<{tile:CityTile;sx:number;sy:number}>,count:number,network:TransportNetwork,z:number,enabled:boolean,lakes:Set<string>){
    this.container.visible=enabled;this.container.removeChildren();this.frame++;
    if(!enabled)return;
    const detail=z>=5?2:z>=1.5?1:0;
    for(let i=0;i<count;i++){
      const {tile,sx,sy}=visible[i],key=`${tile.q},${tile.r}`;
      if(tile.cityId==null || tile.terrain===0 || lakes.has(key))continue;
      let entry=this.cache.get(key);
      if(!entry){const plan=planCity(tile,network);entry={g:drawCity(plan,detail),plan,detail,used:this.frame};this.cache.set(key,entry);}
      else if(entry.detail!==detail){entry.g.destroy();entry.g=drawCity(entry.plan,detail);entry.detail=detail;}
      entry.used=this.frame;entry.g.position.set(sx,sy);entry.g.scale.set(z);this.container.addChild(entry.g);
    }
    if(this.cache.size>256)for(const [k,e]of this.cache){if(e.used!==this.frame){e.g.destroy();this.cache.delete(k);if(this.cache.size<=256)break;}}
  }
  destroy(){this.clear();this.container.destroy({children:true});}
}
