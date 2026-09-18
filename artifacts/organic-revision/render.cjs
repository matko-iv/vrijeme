const fs = require('node:fs');
const path = require('node:path');
const root = 'C:/Users/Matija/Documents/GitHub/rpn-web';
const {chromium} = require(require.resolve('playwright', {paths:[root]}));
(async () => {
 const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'tiles.json')));
 const rivers = JSON.parse(fs.readFileSync(path.join(__dirname, 'rivers.json')));
 const tick = JSON.parse(fs.readFileSync(path.join(__dirname, 'tick.json')));
 const oldSource=fs.readFileSync(path.join(__dirname,'../organic-implementation/baseline/frontend/src/lib/game/organic-terrain.ts'),'utf8');
 const oldShader={vertex:oldSource.match(/const vertex = `([\s\S]*?)`;/)[1],fragment:oldSource.match(/const fragment = `([\s\S]*?)`;/)[1]};
 const browser = await chromium.launch({headless:true,args:['--use-angle=swiftshader']});
 try {
  const page = await browser.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:2});
  const errors=[]; page.on('console', m=>{if(m.text().startsWith('preview:'))console.log(m.text())}); page.on('pageerror', e=>errors.push(e.message));
  await page.route('**/__old_atlas.jpg',r=>r.fulfill({contentType:'image/jpeg',body:fs.readFileSync(path.join(__dirname,'../organic-implementation/baseline/frontend/static/tiles/organic/blue-marble-atlas.jpg'))}));
  await page.route('**/api/v1/**', r=>r.fulfill({status:401,body:'{}',contentType:'application/json'}));
  await page.route('**/__public_world_preview', r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html><body style="margin:0;background:#172427"></body></html>'}));
  await page.goto('http://127.0.0.1:5175/__public_world_preview');
  await page.evaluate(async ({data,rivers,tick,oldShader})=>{
   const url='/src/lib/game/map-renderer.ts', source=await(await fetch(url)).text();
   const {MapRenderer}=await import(url);
   const {Application,Filter,GlProgram,Assets}=await import(source.match(/from "([^"]*pixi__js[^"]*)"/)[1]);
   const app=new Application();await app.init({width:1600,height:1000,resolution:2,autoDensity:true,antialias:true,preference:'webgl',preserveDrawingBuffer:true});app.stop();document.body.append(app.canvas);
   const map=new MapRenderer(app);map.setWeatherEnabled(false);map.setWeatherSound(false);map.setMode('terrain');map.setSeason(tick.season);
   for(const name of ['icons','labels','borders'])map.setOverlayVisibility(name,false);
   map.loadTiles(data.tiles,data.nations,data.edges,data.hv_hexes,data.border_postures,data.customs_hexes);
   map.setRivers(rivers.edges,rivers.lakes,rivers.major_flow);
   console.log('preview: data loaded'); map.minZoomForBounds=()=>.1; map.centerOn(129,79);map.setZoom(2.6);map.renderTick(); console.log('preview: overview prepared');
   for(let i=0;i<200&&!map.organic.ready;i++)await new Promise(r=>setTimeout(r,50));
   if(!map.organic.ready)throw Error('NASA terrain failed to load');
   window.preview={map,app,Filter,GlProgram,Assets,oldShader};
  },{data,rivers,tick,oldShader});
  const views=[
   {id:'city',q:130,r:82,z:8,title:'Russom and its suburbs',description:'Actual Public World 1. Neighborhoods spread into suitable neighboring hexes along roads, rather than ending in isolated circles.'},
   {id:'streets',q:130,r:82,z:24,title:'Russom — building detail',description:'Detached houses, terraces, villas and apartments vary in footprint, roof form and height. The layout stays fixed when zooming.'},
   {id:'region',q:129,r:79,z:2.6,title:'Eastern cities',description:'The current world layout with continuous city districts and lower-density outskirts.'},
   {id:'mountains-before',q:122,r:15,z:5,title:'Mountains — previous rendering',description:'Previous shader and atlas, on the same live-world location and camera as the revised view. The source was stretched to a square, warped and blended with a rotated copy.'},
   {id:'mountains',q:122,r:15,z:5,title:'Mountains — revised rendering',description:'NASA Alps relief in square ground pixels. Unrotated windows of the photograph sit at random offsets, so there is no repeat, stretch, warp or mirroring.'},
   {id:'mountains-overview',q:138,r:38,z:1.4,title:'Mountain ranges of the northeast',description:'Glacial, temperate and tropical ranges side by side. Character follows the terrain beyond each range\'s foothills and shifts regionally, with glaciers gathering toward range cores.'},
   {id:'mountains-arid',q:73,r:93,z:4,title:'Arid range',description:'A range whose foothills give onto desert: Hindu Kush relief, brown ridges with snow only on the highest ground.'},
   {id:'mountains-temperate',q:140,r:42,z:4,title:'Temperate range',description:'Forest and grassland beyond the foothills: Alpine relief, with patches of glacier where the range is deepest.'},
   {id:'mountains-tropical',q:145,r:50,z:4,title:'Tropical range',description:'Jungle beyond the foothills: New Guinea highland relief, densely forested and snow-free, blending into temperate relief across the range.'},
   {id:'world',q:74.5,r:74.5,z:.15,title:'Public World 1',description:'The full current world with the revised terrain and settlements.'},
   {id:'border',q:91,r:86,z:4,title:'Forest meets desert',description:'Actual Public World 1 terrain. Each terrain claims ground with its own fractal noise, so the border wanders and interlocks instead of following hex edges.'},
   {id:'border-close',q:91,r:86,z:24,title:'Forest meets desert — maximum zoom',description:'The same border at 24× zoom, rendered at device resolution. NASA Landsat 45 m detail keeps the ground sharp where Blue Marble imagery alone would blur.'},
   {id:'farms',q:0,r:0,z:5,title:'Farmland among forest, plains and grassland',description:'A demonstration with known farm tiles; the public map hides Public World 1 farm locations. Farmland keeps the colour of the terrain it sits on and takes Kansas field structure from NASA Landsat.'},
   {id:'farms-close',q:0,r:0,z:16,title:'Farmland — close zoom',description:'Field boundaries and centre-pivot circles at 16× zoom, in the same palette as the surrounding plains.'}
  ];
  const only=process.env.VIEWS?.split(',');
  for(const v of views.filter(v=>!only||only.includes(v.id))){
   const stats=await page.evaluate(async v=>{const{map,app,Filter,GlProgram,Assets,oldShader}=window.preview;
    if(v.id==='mountains-before') {
     window.preview.currentFilter=map.organic.filter;
     const oldAtlas=await Assets.load('/__old_atlas.jpg');
     map.organic.filter=new Filter({glProgram:GlProgram.from(oldShader),resources:{uGround:map.organic.ground,uAtlas:oldAtlas.source,groundUniforms:window.preview.currentFilter.resources.groundUniforms},resolution:1,padding:0});
     map.organic.surface.filters=[map.organic.filter];
    } else if(window.preview.currentFilter) {
     const old=map.organic.filter;map.organic.filter=window.preview.currentFilter;map.organic.surface.filters=[map.organic.filter];old.destroy();delete window.preview.currentFilter;
    }
    if(v.id==='farms') {
     const tiles=[];
     for(let q=-14;q<=14;q++)for(let r=-14;r<=14;r++) {
      const x=q+r/2, terrain=x<-4?2:x<3?1:10, farm=Math.max(Math.abs(q),Math.abs(r),Math.abs(q+r))<=2&&!(q===2&&r===-2)&&terrain!==2;
      tiles.push([q,r,terrain,null,0,'demo',null,null,false,farm?3:0,farm?['farms']:[],null,null,null]);
     }
     map.loadTiles(tiles,{'demo':{country:'Material example',primary_color:'#7c8c57',secondary_color:'#c4bb83'}},[]);map.setRivers([],[],500);map.setSeason('summer');
    }
    map.centerOn(v.q,v.r);map.setZoom(v.z);map.renderTick();app.renderer.render(app.stage);return {zoom:map.zoom,visibleTiles:map.lastVisibleTiles,redrawMs:map.lastRedrawMs,roofs:map.cities.cache.get(`${v.q},${v.r}`)?.plan.roofs.length};},v);
   await page.screenshot({path:path.join(__dirname,`${v.id}.png`)});
   if(v.id==='world'&&stats.visibleTiles!==data.tiles.length)throw Error('World overview cropped live tiles'); console.log(v.id,JSON.stringify(stats));Object.assign(v,stats);
  }
  if(errors.length)throw Error(errors.join('\n'));
  const snapshot={source:'https://play.roleplay-nations.xyz/api/v1/map/tiles',fetchedAt:fs.statSync(path.join(__dirname,'tiles.json')).mtime.toISOString(),world:'Public World 1',hexes:data.tiles.length,cities:data.tiles.filter(t=>t[6]!=null).length,tick:tick.tick_count,season:tick.season,views};
  fs.writeFileSync(path.join(__dirname,'snapshot.json'),JSON.stringify(snapshot,null,2));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
