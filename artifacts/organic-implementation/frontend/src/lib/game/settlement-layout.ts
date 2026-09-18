import { axialToPixel, pixelToAxial, HEX_SIZE } from './hex-math';
import { hash, type Point, type TransportNetwork } from './transport';
import { installationTypeFor } from './tile-textures';
import { type CityTile, type CityPlan, type Roof } from './organic-cities';

const empty = (): CityPlan => ({streets: [], roofs: [], trees: [], gardens: []});
const keyOf = (tile: CityTile) => `${tile.q},${tile.r}`;
const centre = (tile: CityTile): Point => { const [x,y] = axialToPixel(tile.q,tile.r,HEX_SIZE); return {x,y}; };
const radiusOf = (tile: CityTile) => Math.min(43, 15 + Math.sqrt(tile.buildingCount ?? 0) * 4);
const terrainForSuburbs = new Set([1, 2, 7, 8, 10]);
function streetIndex(lines: Point[][], reach=5) {
    const buckets=new Map<string, Array<[Point,Point]>>(), size=5;
    for(const line of lines)for(let i=1;i<line.length;i++) {
        const a=line[i-1],b=line[i];
        for(let x=Math.floor(Math.min(a.x,b.x)/size);x<=Math.floor(Math.max(a.x,b.x)/size);x++)
            for(let y=Math.floor(Math.min(a.y,b.y)/size);y<=Math.floor(Math.max(a.y,b.y)/size);y++) {
                const key=`${x},${y}`; if(!buckets.has(key))buckets.set(key,[]);buckets.get(key)!.push([a,b]);
            }
    }
    return (p:Point)=>{
        let squared=Infinity, angle=0;
        for(let x=Math.floor((p.x-reach)/size);x<=Math.floor((p.x+reach)/size);x++)
            for(let y=Math.floor((p.y-reach)/size);y<=Math.floor((p.y+reach)/size);y++)
                for(const [a,b] of buckets.get(`${x},${y}`)??[]) {
                    const dx=b.x-a.x,dy=b.y-a.y,t=Math.max(0,Math.min(1,((p.x-a.x)*dx+(p.y-a.y)*dy)/(dx*dx+dy*dy||1)));
                    const d=(p.x-a.x-t*dx)**2+(p.y-a.y-t*dy)**2;
                    if(d<squared){squared=d;angle=Math.atan2(dy,dx);}
                }
        return {distance:Math.sqrt(squared),angle};
    };
}

/** City-wide plans are partitioned for rendering only. Hex edges do not shape growth. */
export class SettlementLayout {
    private tiles = new Map<string, CityTile>();
    private lakes = new Set<string>();
    private network!: TransportNetwork;
    private masters = new Map<string, CityPlan>();
    private districts = new Map<string, CityPlan>();
    private fingerprint = '';
    private nearby = new Map<string, CityTile[]>();
    private arteries = new Map<string, Point[][]>();
    private accessIndices = new Map<string, ReturnType<typeof streetIndex>>();
    revision = 0;

    setWorld(tiles: Map<string, CityTile>, network: TransportNetwork, lakes: Set<string>) {
        // Compare visible planning inputs, so an unchanged server refresh keeps every texture.
        const fingerprint = [...tiles.values()].map(t => `${keyOf(t)}:${t.terrain}:${t.owner}:${t.cityId}:${t.buildingCount}:${installationTypeFor(t.buildingKeys ?? [])}`).join('|')
            + [...network.paths.values()].map(p => `${p.id}:${p.length}`).join('|') + [...lakes].sort().join('|');
        this.tiles = tiles; this.network = network; this.lakes = lakes;
        if (fingerprint === this.fingerprint) return;
        this.fingerprint = fingerprint; this.revision++;
        this.masters.clear(); this.districts.clear(); this.nearby.clear(); this.arteries.clear(); this.accessIndices.clear();
    }

    private tileAt(p: Point) { const [q,r] = pixelToAxial(p.x,p.y,HEX_SIZE); return this.tiles.get(`${q},${r}`); }
    private candidates(tile: CityTile): CityTile[] {
        const cached=this.nearby.get(keyOf(tile)); if(cached) return cached;
        const result: CityTile[] = [];
        for(let dq=-2;dq<=2;dq++) for(let dr=-2;dr<=2;dr++) {
            if(Math.max(Math.abs(dq),Math.abs(dr),Math.abs(dq+dr))>2) continue;
            const city=this.tiles.get(`${tile.q+dq},${tile.r+dr}`);
            if(city?.cityId != null && city.terrain !== 0 && !this.lakes.has(keyOf(city)) && city.owner === tile.owner) result.push(city);
        }
        result.sort((a,b)=>a.q-b.q || a.r-b.r); this.nearby.set(keyOf(tile),result); return result;
    }
    private access(city: CityTile): Point[][] {
        const key=keyOf(city), cached=this.arteries.get(key); if(cached)return cached;
        const routes=new Map<string, Point[]>();
        for(const path of this.network.adjacent.get(`${key}:road`)??[]) {
            routes.set(path.id,path.points);
            const other=path.a===key?path.b:path.a;
            for(const next of this.network.adjacent.get(`${other}:road`)??[])routes.set(next.id,next.points);
        }
        const lines=[...routes.values()].map(points=>points.filter((_,i)=>i%8===0||i===points.length-1));
        this.arteries.set(key,lines);this.accessIndices.set(key,streetIndex(lines,12));return lines;
    }
    private sourceAt(p: Point): CityTile | undefined {
        const tile=this.tileAt(p);
        if(!tile || tile.terrain===0 || this.lakes.has(keyOf(tile))) return;
        if(tile.cityId==null && (!terrainForSuburbs.has(tile.terrain) || installationTypeFor(tile.buildingKeys ?? []))) return;
        let source: CityTile | undefined, best=1;
        for(const city of this.candidates(tile)) {
            const c=centre(city), d=Math.hypot(p.x-c.x,p.y-c.y)/radiusOf(city);
            if(d<best) {
                const roads=this.access(city);
                if(d*radiusOf(city)>18 && roads.length && this.accessIndices.get(keyOf(city))!(p).distance>7+(1-d)*4)continue;
                best=d; source=city;
            }
        }
        return source;
    }
    hasSettlement(tile: CityTile) {
        if(tile.terrain===0 || this.lakes.has(keyOf(tile)))return false;
        if(tile.cityId!=null) return true;
        if(!terrainForSuburbs.has(tile.terrain) || installationTypeFor(tile.buildingKeys ?? []) || this.lakes.has(keyOf(tile))) return false;
        return this.candidates(tile).some(city=>{const a=centre(tile),b=centre(city);return Math.hypot(a.x-b.x,a.y-b.y)<radiusOf(city)+HEX_SIZE;});
    }
    private master(city: CityTile): CityPlan {
        const key=keyOf(city), cached=this.masters.get(key); if(cached) return cached;
        const plan=empty(), c=centre(city), radius=radiusOf(city);
        // Neighboring districts use the same gently curved regional street coordinates.
        // Their lanes meet at chunk/city boundaries instead of ending at a hex edge.
        const angle=(hash(`streets:${city.owner ?? 'unowned'}`)-.5)*.9, co=Math.cos(angle), si=Math.sin(angle);
        const gx=c.x*co+c.y*si, gy=-c.x*si+c.y*co;
        const node=(x:number,y:number):Point=>{
            const nx=x+Math.sin(y*.14)*2.0+Math.sin(x*.07)*1.2;
            const ny=y+Math.sin(x*.17)*2.1+Math.sin(y*.09)*.6;
            return {x:nx*co-ny*si-c.x,y:nx*si+ny*co-c.y};
        };
        const world=(p:Point)=>({x:p.x+c.x,y:p.y+c.y});
        const belongs=(p:Point)=>this.sourceAt(world(p))?.cityId===city.cityId;
        for(const axis of [0,1]) {
            const fixed=axis===0?gy:gx, varying=axis===0?gx:gy;
            for(let v=Math.floor((fixed-radius-4)/5)*5;v<fixed+radius+4;v+=5) {
                const line:Point[]=[];
                for(let t=varying-radius-4;t<=varying+radius+4;t+=.65)line.push(axis===0?node(t,v):node(v,t));
                plan.streets.push(line);
            }
        }
        const exclusions: Point[][] = [...plan.streets];
        for(const path of this.network.paths.values()) {
            const a=path.points[0], b=path.points[path.points.length-1];
            if(Math.min(Math.hypot(a.x-c.x,a.y-c.y),Math.hypot(b.x-c.x,b.y-c.y))>radius+18) continue;
            const line=path.points.map(p=>({x:p.x-c.x,y:p.y-c.y}));
            exclusions.push(line);
            if(path.kind==='road') plan.streets.push(line);
        }
        const candidates: Array<Point & {seed:number}> = [];
        const nearStreet=streetIndex(plan.streets), nearExclusion=streetIndex(exclusions);
        for(let y=-radius;y<=radius;y+=.9) for(let x=-radius;x<=radius;x+=.9) {
            const value=hash(`${key}:${x.toFixed(1)}:${y.toFixed(1)}`);
            candidates.push({x:x+(value-.5)*.6,y:y+(hash(`y:${value}`)-.5)*.6,seed:value});
        }
        candidates.sort((a,b)=>a.seed-b.seed);
        const limit=Math.min(390,30+(city.buildingCount??0)*7);
        for(const p of candidates) {
            if(plan.roofs.length>=limit) break;
            const distance=Math.hypot(p.x,p.y), edge=distance/radius;
            if(edge>1 || !belongs(p)) continue;
            const near=nearStreet(p);
            const central=distance<12;
            const density=central?.94:Math.max(.08,.66*(1-edge));
            if(hash(`density:${p.seed}`)>density || near.distance> (central?3.4:2.4)) continue;
            const choice=hash(`type:${p.seed}`);
            const kind: Roof['kind'] = central && choice<.22?'apartment':central && choice<.36?'terrace':!central && choice<.07?'warehouse':choice>.78?'villa':choice>.5?'gable':'hip';
            const dimensions=kind==='apartment'?[1.45,2.05]:kind==='terrace'?[.95,2.3]:kind==='warehouse'?[2.3,3.5]:kind==='villa'?[1.45,1.65]:[.85+p.seed*.4,1.05+hash(`h:${p.seed}`)*.55];
            const [width,height]=dimensions, angle=near.angle, co=Math.cos(angle), si=Math.sin(angle);
            const corners=[[-1,-1],[-1,1],[1,1],[1,-1]].map(([x,y])=>({x:p.x+x*width/2*co-y*height/2*si,y:p.y+x*width/2*si+y*height/2*co}));
            if(near.distance<.45+Math.min(width,height)/2 || !corners.every(belongs)) continue;
            if(nearExclusion(p).distance<.38+Math.min(width,height)/2 || corners.some(v=>nearExclusion(v).distance<.38)) continue;
            if(plan.roofs.some(b=>Math.hypot(b.x-p.x,b.y-p.y)<(Math.hypot(width,height)+Math.hypot(b.width,b.height))*.45+.08)) continue;
            plan.roofs.push({x:p.x,y:p.y,width,height,angle,tone:Math.floor(hash(`tone:${p.seed}`)*4),kind,storeys:kind==='apartment'?3+Math.floor(p.seed*3):kind==='terrace'?3:1+Math.floor(p.seed*2)});
            if(!central || kind==='villa') {
                const garden={x:p.x-si*(height/2+.5),y:p.y+co*(height/2+.5)};
                if(belongs(garden)&&nearExclusion(garden).distance>.6)plan.gardens.push(garden);
            }
        }
        // Tree cover follows the inhabited fabric, without a circular hedge at the city edge.
        for(let i=0;i<2500;i++) {
            const a=hash(`${key}:tree:a:${i}`)*Math.PI*2, r=Math.sqrt(hash(`${key}:tree:r:${i}`))*radius;
            const p={x:Math.cos(a)*r,y:Math.sin(a)*r};
            if(!belongs(p) || nearExclusion(p).distance<.5)continue;
            if(!plan.roofs.some(b=>Math.hypot(b.x-p.x,b.y-p.y)<3))continue;
            if(plan.roofs.some(b=>Math.hypot(b.x-p.x,b.y-p.y)<Math.hypot(b.width,b.height)/2+.15))continue;
            plan.trees.push(p);
        }
        this.masters.set(key,plan); return plan;
    }
    plan(tile: CityTile): CityPlan {
        const key=keyOf(tile), cached=this.districts.get(key);if(cached)return cached;
        const result=empty(), c=centre(tile);
        const isHere=(p:Point)=>{const t=this.tileAt(p);return t && keyOf(t)===key;};
        for(const city of this.candidates(tile)) {
            const plan=this.master(city), origin=centre(city);
            const world=(p:Point)=>({x:p.x+origin.x,y:p.y+origin.y});
            const local=(p:Point)=>({x:p.x+origin.x-c.x,y:p.y+origin.y-c.y});
            for(const b of plan.roofs) if(isHere(world(b))) result.roofs.push({...b,...local(b)});
            for(const p of plan.gardens) if(isHere(world(p)))result.gardens.push(local(p));
            for(const p of plan.trees) if(isHere(world(p)))result.trees.push(local(p));
            for(const line of plan.streets) {
                let segment:Point[]=[];
                for(const p of line) {
                    const wp=world(p);
                    // A small overlap avoids cracks where a continuous road crosses a render chunk.
                    const inBounds=Math.abs(wp.x-c.x)<17 && Math.abs(wp.y-c.y)<15 && Math.abs(wp.x-c.x)*.866+Math.abs(wp.y-c.y)*.5<14.1;
                    if(inBounds&&this.sourceAt(wp)?.cityId===city.cityId) segment.push(local(p));
                    else {if(segment.length>1)result.streets.push(segment);segment=[];}
                }
                if(segment.length>1)result.streets.push(segment);
            }
        }
        this.districts.set(key,result);return result;
    }
    clear(){this.masters.clear();this.districts.clear();this.nearby.clear();this.arteries.clear();this.accessIndices.clear();this.fingerprint='';}
}
