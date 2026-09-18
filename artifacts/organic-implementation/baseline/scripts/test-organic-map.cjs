// Browser integration checks use only generated public fixtures and mocked APIs.
const assert=require('node:assert/strict');
const path=require('node:path');
const fs=require('node:fs');
const {chromium}=require(require.resolve('playwright',{paths:[process.env.RPN_TEST_ROOT||path.resolve(__dirname,'..')]}));
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--use-angle=swiftshader']});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:960},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 page.on('console',m=>{if(m.type()==='error'&&/shader|program|glsl/i.test(m.text()))errors.push(m.text());});
 await page.route('**/api/v1/**',r=>r.fulfill({status:401,contentType:'application/json',body:'{}'}));
 await page.route('**/__organic_test',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html><body></body></html>'}));
 await page.goto((process.argv[2]||'http://127.0.0.1:5175')+'/__organic_test',{waitUntil:'domcontentloaded'});
 const results=await page.evaluate(async()=>{
  const url='/src/lib/game/map-renderer.ts';
  const {MapRenderer}=await import(url),source=await(await fetch(url)).text();
  const {Application}=await import(source.match(/from "([^"]*pixi__js[^"]*)"/)[1]);
  const scale=await import('/src/lib/game/map-scale.ts');
  const app=new Application();await app.init({width:1440,height:960,preference:'webgl',antialias:true,preserveDrawingBuffer:true});app.stop();
  const host=document.createElement('div');host.style.cssText='position:fixed;inset:0;z-index:999999;background:#16212a';document.body.append(host);host.append(app.canvas);
  const map=new MapRenderer(app);map.setWeatherEnabled(false);map.setWeatherSound(false);map.setMode('terrain');
  for(const key of ['icons','labels','borders'])map.setOverlayVisibility(key,false);
  const tiles=[],edges=[];
  for(let q=-30;q<65;q++)for(let r=-30;r<65;r++){
   const y=r+q/2;
   const coast=34+Math.sin(y*.16)*2;
   let terrain=q>coast?0:q<4?3:q<9?8:q<17?2:1;
   if(q===1&&r===19)terrain=11;
   if(q>=coast-1&&terrain!==0)terrain=7;
   const city=(q===23&&r===12)||(q===23&&r===11)||(q===18&&r===20)||(q===29&&r===5);
   tiles.push([q,r,terrain,null,0,'1',city?10000+q*100+r:null,city?'Organic city':null,false,city?20:0,city?['housing','road_network']:[],null,null,null]);
   if(q>=14&&q<31&&r===12)edges.push([q,r,q+1,r,'road']);
   if(q===23&&r>=3&&r<25)edges.push([q,r,q,r+1,'road']);
  }
  map.loadTiles(tiles,{'1':{country:'Study',primary_color:'#839977',secondary_color:'#d3caae'}},edges);
  map.centerOn(20,13);map.setZoom(1.6);map.renderTick();
  for(let i=0;i<200&&!map.organic.ready;i++)await new Promise(r=>setTimeout(r,50));
  if(!map.organic.ready)throw Error('Organic atlas did not load');
  map.renderTick();app.renderer.render(app.stage);
  const checks=[];function check(name,ok){if(!ok)throw Error(name);checks.push(name);}
  check('Organic shader is active',map.organic.container.visible);
  check('Legacy sprite artwork is not used',!map.tileContainer.visible);
  check('Cities are vector geometry',map.cities.cache.size>0);
  check('Every vehicle fits inside a lane',Object.values(scale.VEHICLE_DIMENSIONS).every(v=>v.width<scale.ROAD_WIDTH/2));
  const oldPlan=JSON.stringify(map.cities.cache.get('23,12').plan);
  map.centerOn(23,12);map.setZoom(20);map.renderTick();app.renderer.render(app.stage);
  check('Deep zoom enabled',map.zoom===20);
  check('Zoom keeps identical city layout',oldPlan===JSON.stringify(map.cities.cache.get('23,12').plan));
  check('High zoom draws roof and glazing detail',map.cities.cache.get('23,12').detail===2);
  const plan=map.cities.cache.get('23,12').plan;
  check('City has individual roofs and connected streets',plan.roofs.length>=15&&plan.streets.length>=4);
  check('Architectural dimensions are finite',plan.roofs.every(r=>[r.x,r.y,r.width,r.height,r.angle].every(Number.isFinite)));
  map.setMode('economic');map.renderTick();check('Analytical overlays disable Organic art',!map.organic.container.visible&&!map.cities.container.visible);
  map.setMode('terrain');map.setOverlayVisibility('textures',false);map.renderTick();check('Texture toggle still works',!map.organic.container.visible&&!map.cities.container.visible);
  map.setOverlayVisibility('textures',true);map.renderTick();
  const oldEntry=map.cities.cache.get('23,12');
  map.loadTiles(tiles,{'1':{country:'Study',primary_color:'#839977',secondary_color:'#d3caae'}},edges);map.renderTick();
  check('Unchanged map refresh reuses city geometry',map.cities.cache.get('23,12')===oldEntry);
  const atlasReady=map.organic.ready;map.organic.ready=false;map.redraw();
  check('Imagery fallback keeps vector cities above flat ground',map.baseGfx.visible&&map.cities.container.visible&&!map.tileContainer.visible);
  map.organic.ready=atlasReady;map.redraw();app.renderer.render(app.stage);
  window.organicTest={map,app,tiles,edges};
  return {checks,roofs:plan.roofs.length,visibleTiles:map.lastVisibleTiles,redrawMs:map.lastRedrawMs};
 });
 const output=path.resolve(process.argv[3]||'artifacts/organic-implementation/screenshots');fs.mkdirSync(output,{recursive:true});
 await page.screenshot({path:path.join(output,'city-detail.png')});
 await page.evaluate(()=>{const{map,app}=window.organicTest;map.centerOn(20,13);map.setZoom(1.6);map.renderTick();app.renderer.render(app.stage);});
 await page.screenshot({path:path.join(output,'organic-overview.png')});
 await page.evaluate(()=>{const{map,app}=window.organicTest;map.centerOn(1,19);map.setZoom(12);map.renderTick();app.renderer.render(app.stage);});
 await page.screenshot({path:path.join(output,'volcanic-detail.png')});
 await page.evaluate(()=>{const{map,app}=window.organicTest;map.setRivers([],[[23,12,1]],500);map.centerOn(23,12);map.renderTick();if(map.cities.container.children.some(c=>c===map.cities.cache.get('23,12')?.g))throw Error('Lake settlement was not hidden');map.destroy();app.destroy(true,true);});
 assert.deepEqual(errors,[]);console.log(JSON.stringify(results,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
