import json
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent
snapshot = json.loads((root / 'snapshot.json').read_text())
for view in snapshot['views']:
    with Image.open(root / (view['id'] + '.png')) as im:
        im.save(root / (view['id'] + '.webp'), quality=94, method=6)

html = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Public World 1 — Organic cities</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101917;color:#e6e9de;font:15px/1.5 system-ui,sans-serif}header{padding:22px 30px 15px;display:flex;align-items:center;justify-content:space-between;gap:20px}h1{font-size:24px;margin:0;font-weight:550;letter-spacing:-.7px}.eyebrow{color:#b4c594;letter-spacing:2px;font-size:11px;text-transform:uppercase}.meta{color:#9ca99d;font-size:12px;text-align:right}nav{display:flex;gap:8px;flex-wrap:wrap;padding:0 30px 18px}button,a{font:inherit}button{border:1px solid #435046;color:#cdd4c9;background:transparent;border-radius:6px;padding:8px 16px;cursor:pointer}button[aria-pressed=true]{background:#dbe3c9;color:#18231d;border-color:#dbe3c9}button:hover{border-color:#c3d0b4}main{margin:0 20px}#viewer{height:calc(100vh - 249px);min-height:370px;overflow:hidden;position:relative;background:#101817;border:1px solid #334138;border-radius:8px;touch-action:none;cursor:grab}#viewer.dragging{cursor:grabbing}#art{width:100%;height:100%;object-fit:contain;transform-origin:50% 50%;user-select:none;pointer-events:none}.tools{position:absolute;right:14px;bottom:14px;display:flex;gap:6px}.tools button,.tools a{background:#101b17e8;border:1px solid #62715e;color:#e6e9de;border-radius:5px;padding:6px 12px;text-decoration:none;font-size:12px}.caption{display:flex;justify-content:space-between;gap:24px;padding:13px 10px 0}.caption strong{font-weight:550;font-size:16px}.caption p{margin:3px 0;color:#aeb9ac;font-size:13px}#zoom{white-space:nowrap;color:#b6caa0;font-size:13px}footer{padding:14px 30px 20px;color:#84958a;font-size:11px}a{color:#bccdaf}@media(max-width:650px){header{padding:16px 18px;align-items:flex-start}h1{font-size:20px}.meta{font-size:10px;max-width:120px}nav{padding:0 18px 14px;gap:5px}nav button{padding:7px 10px;font-size:12px}main{margin:0 10px}#viewer{height:60vh;min-height:320px}.caption{display:block}.caption p{font-size:12px}footer{padding:12px 20px}}
</style>
<header><div><div class="eyebrow">Organic · renderer preview</div><h1>Public World 1</h1></div><div class="meta">22,500 hexes · 108 cities<br>15 September 2026 · Autumn · Tick 179</div></header>
<nav aria-label="Map views" id="tabs"></nav>
<main><div id="viewer"><img id="art" draggable="false" alt=""><div class="tools"><button id="reset">Fit view</button><a id="full" target="_blank" rel="noopener">Full resolution ↗</a></div></div><div class="caption"><div><strong id="title"></strong><p id="description"></p></div><span id="zoom"></span></div></main>
<footer>Captured from the current public world, with the revised local renderer. NASA terrain imagery and procedural architecture; no AI-generated city images. Scroll to magnify, drag to pan. <a href="snapshot.json">Snapshot details</a></footer>
<script>
const snapshot=SNAPSHOT;
const labels={world:'Whole world',region:'Eastern cities',city:'Russom',streets:'Street detail',paris:'Paris'};
const viewer=document.querySelector('#viewer'),art=document.querySelector('#art');let scale=1,x=0,y=0,drag=null;
function transform(){art.style.transform=`translate(${x}px,${y}px) scale(${scale})`}
function reset(){scale=1;x=y=0;transform()}
function select(id){const v=snapshot.views.find(v=>v.id===id);art.src=v.id+'.webp';art.alt=v.title+' in Public World 1 with organic city rendering';document.querySelector('#title').textContent=v.title;document.querySelector('#description').textContent=v.description;document.querySelector('#zoom').textContent=v.id==='world'?'All 22,500 hexes':v.zoom+'× map zoom · 3200 × 2000';document.querySelector('#full').href=v.id+'.png';for(const b of document.querySelectorAll('nav button'))b.setAttribute('aria-pressed',String(b.dataset.id===id));reset()}
for(const v of snapshot.views){const b=document.createElement('button');b.textContent=labels[v.id];b.dataset.id=v.id;b.onclick=()=>select(v.id);document.querySelector('#tabs').append(b)}
viewer.addEventListener('wheel',e=>{e.preventDefault();scale=Math.max(1,Math.min(3,scale*Math.exp(-e.deltaY*.001)));if(scale===1){x=0;y=0}transform()},{passive:false});
viewer.addEventListener('pointerdown',e=>{if(e.target.closest('.tools'))return;drag={x:e.clientX-x,y:e.clientY-y};viewer.setPointerCapture(e.pointerId);viewer.classList.add('dragging')});viewer.addEventListener('pointermove',e=>{if(drag){x=e.clientX-drag.x;y=e.clientY-drag.y;transform()}});for(const name of ['pointerup','pointercancel'])viewer.addEventListener(name,()=>{drag=null;viewer.classList.remove('dragging')});document.querySelector('#reset').onclick=reset;select('city');
</script></html>'''
(root / 'public-world-1-preview.html').write_text(html.replace('SNAPSHOT', json.dumps(snapshot, ensure_ascii=True)), encoding='utf-8')
print('Built Public World 1 preview with five captured renderer views.')
