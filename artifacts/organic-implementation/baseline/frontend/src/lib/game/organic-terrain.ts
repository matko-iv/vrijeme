import { Assets, BufferImageSource, Container, Filter, GlProgram, Rectangle, Sprite, Texture } from 'pixi.js';
import { axialNeighbors, HEX_SIZE } from './hex-math';

export interface GroundTile { q: number; r: number; terrain: number }

const vertex = `
in vec2 aPosition;
out vec2 vScreen;
uniform vec4 uOutputFrame;
uniform vec4 uOutputTexture;
void main() {
  vec2 p = aPosition * uOutputFrame.zw + uOutputFrame.xy;
  vScreen = p;
  p.x = p.x * (2.0 / uOutputTexture.x) - 1.0;
  p.y = p.y * (2.0 * uOutputTexture.z / uOutputTexture.y) - uOutputTexture.z;
  gl_Position = vec4(p, 0.0, 1.0);
}`;

const fragment = `
precision highp float;
in vec2 vScreen;
out vec4 finalColor;
uniform sampler2D uGround;
uniform sampler2D uAtlas;
uniform vec4 uMap;
uniform vec4 uCamera;
float noise(vec2 p) {
  vec2 i=floor(p),f=fract(p); f=f*f*(3.0-2.0*f);
  float a=fract(sin(dot(i,vec2(127.1,311.7)))*43758.5453);
  float b=fract(sin(dot(i+vec2(1,0),vec2(127.1,311.7)))*43758.5453);
  float c=fract(sin(dot(i+vec2(0,1),vec2(127.1,311.7)))*43758.5453);
  float d=fract(sin(dot(i+vec2(1,1),vec2(127.1,311.7)))*43758.5453);
  return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}
vec2 axial(vec2 p) { return vec2(2.0*p.x/3.0, -p.x/3.0+p.y/1.7320508)/16.0; }
vec2 centre(vec2 a) { return vec2(24.0*a.x,16.0*(.8660254*a.x+1.7320508*a.y)); }
vec2 roundHex(vec2 p) {
  vec3 a=vec3(p.x,-p.x-p.y,p.y),r=floor(a+.5),d=abs(r-a);
  if(d.x>d.y && d.x>d.z) r.x=-r.y-r.z;
  else if(d.y>d.z) r.y=-r.x-r.z;
  else r.z=-r.x-r.y;
  return r.xz;
}
vec4 cell(vec2 qr) {
  vec2 uv=(qr-uMap.xy+.5)/uMap.zw;
  if(any(lessThan(uv,vec2(0))) || any(greaterThan(uv,vec2(1)))) return vec4(0);
  return texture(uGround,uv);
}
vec3 terrain(float t,vec2 p,vec2 c) {
  float slot=t;
  if(t==11.0) slot=3.0;
  vec2 coord=p/210.0;
  if(slot==3.0) coord=vec2(p.x*.78+p.y*.17,p.y*.61-p.x*.13)/250.0;
  coord += vec2(noise(p/90.0),noise(p/90.0+71.0))*.055;
  // Mirrored addressing preserves source ridges without hard repeat seams.
  vec2 uv=1.0-abs(mod(coord,2.0)-1.0);
  vec2 tile=vec2(mod(slot,4.0),floor(slot/4.0));
  vec3 color=texture(uAtlas,(tile+(uv*510.0+1.0)/512.0)/4.0).rgb;
  if(slot==3.0) color=mix(color,vec3(dot(color,vec3(.213,.715,.072))),.18)*.94+vec3(.025);
  if(t==11.0) {
    vec2 d=(p-c)/16.0;
    float rad=length(d),angle=atan(d.y,d.x);
    float rim=exp(-pow((rad-.20)*18.0,2.0));
    float cone=exp(-rad*rad*3.7);
    float gullies=(sin(angle*17.0+rad*17.0)+sin(angle*29.0-rad*10.0))*.025*cone;
    float light=dot(normalize(d+vec2(.001)),normalize(vec2(-1.0,-1.0)));
    color=mix(color,vec3(.29,.28,.235),cone*.7);
    color*=1.0+cone*light*.34+gullies;
    color+=rim*(.04+light*.04);
    color*=1.0-.46*(1.0-smoothstep(.06,.19,rad));
  }
  return color;
}
void main() {
  vec2 world=(vScreen-uCamera.xy)/uCamera.z;
  vec2 warp=(vec2(noise(world/33.0),noise(world/33.0+47.0))-.5)*4.2;
  vec2 p=world+warp,qr=roundHex(axial(p));
  vec2 dirs[7]; dirs[0]=vec2(0);dirs[1]=vec2(1,0);dirs[2]=vec2(-1,0);
  dirs[3]=vec2(0,1);dirs[4]=vec2(0,-1);dirs[5]=vec2(1,-1);dirs[6]=vec2(-1,1);
  vec3 land=vec3(0); float landW=0.0,waterW=0.0,total=0.0,depth=0.0,valid=0.0;
  for(int i=0;i<7;i++) {
    vec2 a=qr+dirs[i],c=centre(a);vec4 info=cell(a);
    float d=dot(p-c,p-c),weight=exp(-d/76.8);
    total+=weight;valid+=weight*info.a;
    if(info.a<.5) continue;
    float t=floor(info.r*255.0+.5);
    if(t==0.0) {waterW+=weight;depth+=info.g*weight;}
    else {land+=terrain(t,world,c)*weight;landW+=weight;}
  }
  float wet=waterW/max(waterW+landW,.00001);
  float shelf=exp(-depth/max(waterW,.00001)*255.0*.8);
  shelf=max(shelf,pow(1.0-wet,2.0));
  vec3 water=mix(vec3(.086,.20,.271),vec3(.225,.404,.424),shelf);
  water+=(noise(world/140.0)-.5)*.012;
  float coast=smoothstep(.35,.65,wet);
  vec3 color=mix(land/max(landW,.00001),water,coast);
  float alpha=smoothstep(.3,.65,valid/max(total,.00001));
  finalColor=vec4(color*alpha,alpha);
}`;

/** One screen-sized shader: continuous world-space imagery, independent of tile count. */
export class OrganicTerrain {
  readonly container = new Container();
  ready = false;
  dirty = true;
  private surface = new Sprite(Texture.WHITE);
  private ground = new BufferImageSource({ resource: new Uint8Array(4), width: 1, height: 1, scaleMode: 'nearest', alphaMode: 'no-premultiply-alpha' });
  private filter: Filter | null = null;
  private disposed = false;
  private bounds = new Float32Array([0, 0, 1, 1]);
  constructor() { this.container.addChild(this.surface); this.container.visible=false; }
  async load() {
    const atlas = await Assets.load<Texture>('/tiles/organic/blue-marble-atlas.jpg');
    if(this.disposed) return;
    this.filter = new Filter({ glProgram: GlProgram.from({ vertex, fragment, name:'organic-ground' }),
      resources: { uGround:this.ground,uAtlas:atlas.source,
        groundUniforms:{uMap:{value:this.bounds,type:'vec4<f32>'},uCamera:{value:new Float32Array([0,0,1,0]),type:'vec4<f32>'}} },
      resolution: 1, antialias:'off', padding:0 });
    this.surface.filters=[this.filter];this.ready=true;
  }
  setWorld(tiles: Map<string, GroundTile>, lakes: Set<string>, snow: Set<string>, winter: boolean) {
    if(!this.dirty || !tiles.size) return;
    this.dirty=false;
    let minQ=Infinity,minR=Infinity,maxQ=-Infinity,maxR=-Infinity;
    for(const t of tiles.values()){minQ=Math.min(minQ,t.q);maxQ=Math.max(maxQ,t.q);minR=Math.min(minR,t.r);maxR=Math.max(maxR,t.r);}
    const w=maxQ-minQ+1,h=maxR-minR+1,data=new Uint8Array(w*h*4);
    const water=new Set<string>(),distance=new Map<string,number>();let frontier:GroundTile[]=[];
    for(const [k,t] of tiles) if(t.terrain===0 || lakes.has(k)) water.add(k);
    for(const [k,t] of tiles) if(water.has(k) && axialNeighbors(t.q,t.r).some(([q,r])=>tiles.has(`${q},${r}`)&&!water.has(`${q},${r}`)) {distance.set(k,0);frontier.push(t);}
    for(let d=1;d<=8;d++){const next:GroundTile[]=[];for(const t of frontier)for(const [q,r] of axialNeighbors(t.q,t.r)){const k=`${q},${r}`;if(water.has(k)&&!distance.has(k)){distance.set(k,d);next.push(tiles.get(k)!);}}frontier=next;}
    for(const [k,t] of tiles){const i=((t.r-minR)*w+t.q-minQ)*4;data[i]=water.has(k)?0:winter&&snow.has(k)?5:t.terrain;data[i+1]=distance.get(k)??9;data[i+3]=255;}
    const old=this.ground;
    this.ground=new BufferImageSource({resource:data,width:w,height:h,scaleMode:'nearest',alphaMode:'no-premultiply-alpha'});
    this.bounds.set([minQ,minR,w,h]);
    if(this.filter){this.filter.resources.uGround=this.ground;this.filter.resources.groundUniforms.uniforms.uMap=this.bounds;}
    old.destroy();
  }
  setView(x:number,y:number,z:number,w:number,h:number,enabled:boolean) {
    this.container.visible=enabled&&this.ready;
    if(!this.filter || !enabled)return;
    this.surface.width=w;this.surface.height=h;
    this.surface.filterArea=new Rectangle(0,0,w,h);
    this.filter.resources.groundUniforms.uniforms.uCamera=new Float32Array([x,y,z,0]);
  }
  destroy(){this.disposed=true;this.ready=false;this.surface.filters=[];this.filter?.destroy();this.ground.destroy();this.container.destroy({children:true});}
}
