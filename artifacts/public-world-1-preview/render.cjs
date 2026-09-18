const fs = require('node:fs');
const path = require('node:path');
const root = 'C:/Users/Matija/Documents/GitHub/rpn-web';
const {chromium} = require(require.resolve('playwright', {paths:[root]}));
(async () => {
 const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'tiles.json')));
 const rivers = JSON.parse(fs.readFileSync(path.join(__dirname, 'rivers.json')));
 const tick = JSON.parse(fs.readFileSync(path.join(__dirname, 'tick.json')));
 const browser = await chromium.launch({headless:true,args:['--use-angle=swiftshader']});
 try {
  const page = await browser.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:2});
  const errors=[]; page.on('console', m=>{if(m.text().startsWith('preview:'))console.log(m.text())}); page.on('pageerror', e=>errors.push(e.message));
  await page.route('**/api/v1/**', r=>r.fulfill({status:401,body:'{}',contentType:'application/json'}));
  await page.route('**/__public_world_preview', r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html><body style="margin:0;background:#172427"></body></html>'}));
  await page.goto('http://127.0.0.1:5175/__public_world_preview');
  await page.evaluate(async ({data,rivers,tick})=>{
   const url='/src/lib/game/map-renderer.ts', source=await(await fetch(url)).text();
   const {MapRenderer}=await import(url);
   const {Application}=await import(source.match(/from "([^"]*pixi__js[^"]*)"/)[1]);
   const app=new Application();await app.init({width:1600,height:1000,resolution:2,autoDensity:true,antialias:true,preference:'webgl',preserveDrawingBuffer:true});app.stop();document.body.append(app.canvas);
   const map=new MapRenderer(app);map.setWeatherEnabled(false);map.setWeatherSound(false);map.setMode('terrain');map.setSeason(tick.season);
   for(const name of ['icons','labels','borders'])map.setOverlayVisibility(name,false);
   map.loadTiles(data.tiles,data.nations,data.edges,data.hv_hexes,data.border_postures,data.customs_hexes);
   map.setRivers(rivers.edges,rivers.lakes,rivers.major_flow);
   console.log('preview: data loaded'); map.minZoomForBounds=()=>.1; map.centerOn(74.5,74.5);map.setZoom(.15);map.renderTick(); console.log('preview: overview prepared');
   for(let i=0;i<200&&!map.organic.ready;i++)await new Promise(r=>setTimeout(r,50));
   if(!map.organic.ready)throw Error('NASA terrain failed to load');
   window.preview={map,app};
  },{data,rivers,tick});
  const views=[
   {id:'world',q:74.5,r:74.5,z:.15,title:'Public World 1',description:'The complete live world. Terrain and all 108 city locations are unchanged.'},
   {id:'region',q:129,r:79,z:2.6,title:'Cities of the eastern coast',description:'Paris, Toulouse, Metz and the Republic of Rats. Roads and rivers use the live world geometry.'},
   {id:'city',q:130,r:82,z:11,title:'City of Russom',description:'The largest current city by building count, at hex 130, 82. Irregular streets, individual roofs, gardens and tree cover.'},
   {id:'streets',q:130,r:82,z:24,title:'Russom — close-up',description:'24× zoom, captured at double pixel density. Roof tiles, hipped roofs, walls, chimneys, foliage and soft shadows.'},
   {id:'paris',q:125,r:75,z:16,title:'Paris',description:'A second real city, at hex 125, 75. Architecture and local streets are generated deterministically from its location.'}
  ];
  for(const v of views){
   const stats=await page.evaluate(v=>{const{map,app}=window.preview;map.centerOn(v.q,v.r);map.setZoom(v.z);map.renderTick();app.renderer.render(app.stage);return {zoom:map.zoom,visibleTiles:map.lastVisibleTiles,redrawMs:map.lastRedrawMs,roofs:map.cities.cache.get(`${v.q},${v.r}`)?.plan.roofs.length};},v);
   await page.screenshot({path:path.join(__dirname,`${v.id}.png`)});
   if(v.id==='world'&&stats.visibleTiles!==data.tiles.length)throw Error('World overview cropped live tiles'); console.log(v.id,JSON.stringify(stats));Object.assign(v,stats);
  }
  if(errors.length)throw Error(errors.join('\n'));
  const snapshot={source:'https://play.roleplay-nations.xyz/api/v1/map/tiles',fetchedAt:fs.statSync(path.join(__dirname,'tiles.json')).mtime.toISOString(),world:'Public World 1',hexes:data.tiles.length,cities:data.tiles.filter(t=>t[6]!=null).length,tick:tick.tick_count,season:tick.season,views};
  fs.writeFileSync(path.join(__dirname,'snapshot.json'),JSON.stringify(snapshot,null,2));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
