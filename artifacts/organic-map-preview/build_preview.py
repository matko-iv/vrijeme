"""Standalone design study; reads the previous artifact, never changes game assets."""
import base64
import io
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

SOURCE = Path(sys.argv[1]).resolve()
DEST = Path(__file__).resolve().parent

# Reuse the exact World 1 layout, source imagery and seeded terrain sampling.
original = (SOURCE / 'render.py').read_text(encoding='utf-8')
prefix = original.split('# C: warped borders')[0]
prefix = prefix.replace('HERE = Path(__file__).parent', 'HERE = SOURCE')
prefix = prefix.replace('OUT = HERE / "render"', 'OUT = DEST / "render"')
exec(compile(prefix, str(SOURCE / 'render.py'), 'exec'))

wx = fnoise(R * 1.6, 5, seed=1) * R * 0.55
wy = fnoise(R * 1.6, 5, seed=2) * R * 0.55
T_warp, _, _ = terrain_at(PX + wx, PY + wy)

# A continuous Alpine sample keeps ridges and valleys connected. Sample across
# world coordinates, rather than scattering a new snowy fragment every hex.
alps = np.asarray(Image.open(BM / 'mountain_alps.jpg').convert('RGB'), dtype=np.float32)
alps = alps[230:710, 0:1024]  # Exclude the northern lakes and southern lowlands.
yy, xx = np.mgrid[:H, :W].astype(np.float32)
u = xx * 0.94 + yy * 0.19 + fnoise(360, 3, 21) * 32
v = yy * 0.53 - xx * 0.17 + fnoise(300, 3, 22) * 23
mountains = np.stack([ndimage.map_coordinates(alps[:, :, c], [v, u], order=1, mode='reflect') for c in range(3)], axis=-1)
# Gentle tonal compression: preserve shaded relief without brilliant white dots.
luma = mountains @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
mountains = mountains * 0.80 + luma[..., None] * 0.20
mountains = mountains * 0.90 + np.array([11, 11, 9], dtype=np.float32)
base[3] = mountains

weights = {t: ndimage.gaussian_filter((T_warp == t).astype(np.float32), R * (0.25 if t == 3 else 0.16)) for t in present if t != 0}
total = sum(weights.values())
land_color = np.zeros((H, W, 3), dtype=np.float32)
for t, weight in weights.items():
    land_color += base[t] * (weight / np.maximum(total, 1e-6))[..., None]

# Water is an art-directed surface, not the original exaggerated bathymetry.
# Shelf depth varies gradually with distance to the actual coastline.
land_field = ndimage.gaussian_filter((T_warp != 0).astype(np.float32), 15)
land_mask = land_field > 0.5
ocean_dist = ndimage.distance_transform_edt(~land_mask).astype(np.float32)
shore_dist = ndimage.distance_transform_edt(land_mask).astype(np.float32)
coast_variation = np.clip(1 + fnoise(240, 3, 32) * 0.22, 0.7, 1.3)
shelf = np.exp(-ocean_dist / (47 * coast_variation))
deep = np.array([22, 51, 69], dtype=np.float32)
shallows = np.array([57, 103, 108], dtype=np.float32)
water = deep + shelf[..., None] * (shallows - deep)
water += fnoise(360, 3, 33)[..., None] * np.array([1.4, 2.2, 2.5], dtype=np.float32)

# Extend coastal ground beneath the smoothed shoreline, so smoothing cannot
# uncover black pixels where the old ocean mask had no land texture.
missing = (total < 0.03) & (land_field > 0.05)
if missing.any():
    nearest = ndimage.distance_transform_edt(total < 0.03, return_distances=False, return_indices=True)
    land_color[missing] = land_color[nearest[0][missing], nearest[1][missing]]
wet_edge = np.exp(-shore_dist / 3.0) * land_mask
land_color *= (1 - wet_edge * 0.10)[..., None]
land_alpha = np.clip((land_field - 0.485) / 0.03, 0, 1)
land_alpha = land_alpha * land_alpha * (3 - 2 * land_alpha)
revised = water * (1 - land_alpha[..., None]) + land_color * land_alpha[..., None]
im = save(revised, 'organic_refined')
draw_rivers(im, warp=(wx, wy))
im.save(OUT / 'organic_refined_full.png')
before = Image.open(SOURCE / 'render' / 'organic_full.png').convert('RGB')
grid = Image.open(SOURCE / 'render' / 'grid_full.png')

def uri(pic, width=None):
    if width:
        pic = pic.resize((width, round(pic.height * width / pic.width)), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    pic.save(stream, 'WEBP', quality=90, method=6)
    return 'data:image/webp;base64,' + base64.b64encode(stream.getvalue()).decode()

def detail(pic, box):
    return uri(pic.crop(box), 900)

html = (DEST / 'template.html').read_text(encoding='utf-8')
replacements = {
    'before': uri(before, 1800), 'after': uri(im, 1800), 'grid': uri(grid, 1800),
    'mountain_before': detail(before, (60, 340, 960, 940)),
    'mountain_after': detail(im, (60, 340, 960, 940)),
    'water_before': detail(before, (1260, 500, 2160, 1100)),
    'water_after': detail(im, (1260, 500, 2160, 1100)),
}
for key, value in replacements.items():
    html = html.replace('{{' + key + '}}', value)
assert '{{' not in html
(DEST / 'organic-map-preview.html').write_text(html, encoding='utf-8')

# Convenient full map and compact before/after contact sheet.
im.resize((1600, round(H * 1600 / W)), Image.Resampling.LANCZOS).save(DEST / 'organic-refined.jpg', quality=94)
sheet = Image.new('RGB', (1800, 720), '#101b1b')
d = ImageDraw.Draw(sheet)
font_path = Path('C:/Windows/Fonts/segoeui.ttf')
font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
for i, (title, pic) in enumerate([('ORIGINAL ORGANIC', before), ('REFINED ORGANIC', im)]):
    d.text((i * 900 + 24, 18), title, font=font, fill='#e8e9dc')
    sheet.paste(pic.resize((900, 659), Image.Resampling.LANCZOS), (i * 900, 61))
sheet.save(DEST / 'comparison.jpg', quality=92)
print(DEST / 'organic-map-preview.html')
