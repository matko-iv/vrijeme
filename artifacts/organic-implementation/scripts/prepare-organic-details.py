"""Prepare NASA mountain/field materials without geometric warping (Pillow + NumPy)."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageEnhance

out = Path(__file__).resolve().parents[1] / 'frontend/static/tiles/organic'
source = out / 'sources'

def periodic(image):
    """Remove only low-frequency boundary mismatch; preserve ridge/field geometry."""
    a = np.asarray(image, dtype=float)
    h, w = a.shape[:2]
    boundary = np.zeros_like(a)
    boundary[0] = a[-1] - a[0]
    boundary[-1] = -boundary[0]
    boundary[:, 0] += a[:, -1] - a[:, 0]
    boundary[:, -1] -= a[:, -1] - a[:, 0]
    denominator = 2*np.cos(2*np.pi*np.arange(h)/h)[:, None] + 2*np.cos(2*np.pi*np.arange(w)/w)[None, :] - 4
    denominator[0, 0] = 1
    smooth_fft = np.fft.fft2(boundary, axes=(0, 1)) / denominator[:, :, None]
    smooth_fft[0, 0] = 0
    smooth = np.fft.ifft2(smooth_fft, axes=(0, 1)).real
    return Image.fromarray(np.clip(a-smooth, 0, 255).astype('uint8'))

# Keep the original rectangular crop and its pixel aspect. No square stretching.
mountain = Image.open(source / 'mountain_alps.jpg').convert('RGB').crop((0, 230, 1024, 710))
periodic(mountain).save(out / 'mountain-relief.jpg', quality=97, subsampling=0)
fields = Image.open(source / 'agriculture_kansas_aster.jpg').convert('RGB').crop((600, 500, 1624, 1524))
fields = ImageEnhance.Color(fields).enhance(.64)
fields = ImageEnhance.Brightness(fields).enhance(.82)
periodic(fields).save(out / 'agricultural-fields.jpg', quality=96, subsampling=0)
(out / 'detail-materials.json').write_text(json.dumps({
    'mountains': {'source': 'sources/mountain_alps.jpg', 'crop': [0, 230, 1024, 710], 'size': [1024, 480], 'worldPeriod': [128, 60], 'processing': 'Periodic boundary colour correction only. No warp, shear, mirroring, or rotated duplicate.'},
    'fields': {'source': 'sources/agriculture_kansas_aster.jpg', 'crop': [600, 500, 1624, 1524], 'size': [1024, 1024], 'worldPeriod': [160, 160], 'page': 'https://science.nasa.gov/earth/earth-observatory/crop-circles-in-kansas-5772/', 'image': 'https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/images/imagerecords/5000/5772/kansas_AST_2001175_lrg.jpg?crop=faces%2Cfocalpoint&fit=clip&h=2481&w=2589', 'credit': 'NASA/GSFC/METI/ERSDAC/JAROS, and U.S./Japan ASTER Science Team', 'acquired': '2001-06-24', 'processing': 'Square crop, reduced saturation/brightness and periodic boundary colour correction. Representative terrain material, not geographic placement.'}
}, indent=2)+'\n', encoding='utf-8')
print('Prepared undistorted mountains and NASA agricultural fields.')
