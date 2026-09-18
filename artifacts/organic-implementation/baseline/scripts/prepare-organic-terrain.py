"""Build the real-imagery atlas. No generated/AI imagery is used.

Run: python scripts/prepare-organic-terrain.py --source-dir PATH_TO_BM_CROPS
The source URLs and original crops ship beside the atlas for reproducibility.
Requires Pillow; runtime browsers require no Python or third-party imagery service.
"""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument('--source-dir', type=Path)
args = parser.parse_args()
out = Path(__file__).resolve().parents[1] / 'frontend/static/tiles/organic'
source = args.source_dir or out / 'sources'
out.mkdir(parents=True, exist_ok=True)
sources = {
  1: ('plains_kansas', None), 2: ('forest_appalachia', None),
  3: ('mountain_alps', (0,230,1024,710)), 4: ('desert_sahara', None),
  5: ('tundra_yamal', None), 6: ('swamp_sudd', None),
  7: ('coastal_redsea', (0,0,400,1024)), 8: ('hills_ethiopia', (0,0,700,1024)),
  9: ('jungle_congo', None), 10: ('grassland_kazakh', None),
  11: ('mountain_alps', (0,230,1024,710)), 12: ('arctic_greenland', None),
}
atlas = Image.new('RGB', (2048,2048), (22,51,69))
for slot,(name,crop) in sources.items():
    im = Image.open(source / (name + '.jpg')).convert('RGB')
    if crop: im = im.crop(crop)
    im = im.resize((510,510), Image.Resampling.LANCZOS)
    # A one-pixel replicated gutter prevents adjacent atlas slots bleeding.
    tile = ImageOps.expand(im, 1)
    tile.paste(im.crop((0,0,510,1)), (1,0))
    tile.paste(im.crop((0,509,510,510)), (1,511))
    tile.paste(tile.crop((1,0,2,512)), (0,0))
    tile.paste(tile.crop((510,0,511,512)), (511,0))
    atlas.paste(tile, ((slot%4)*512,(slot//4)*512))
atlas.save(out / 'blue-marble-atlas.jpg', quality=95, subsampling=0)
(out / 'atlas.json').write_text(json.dumps({
    'credit':'NASA Earth Observatory, Blue Marble Next Generation',
    'atlasSize':2048,'slotSize':512,'gutter':1,
    'slots':{str(k):{'source':v[0]+'.jpg','crop':v[1]} for k,v in sources.items()},
    'water':'Procedural colours; shaded bathymetry is intentionally not sampled.',
    'volcanic':'Mountain imagery plus procedural dormant crater shading.',
},indent=2)+'\n',encoding='utf-8')
print(out / 'blue-marble-atlas.jpg')
