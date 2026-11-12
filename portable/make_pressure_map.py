"""Render the GUI background: a synoptic surface-pressure chart.

Synthetic MSLP field (sum of Gaussian highs and lows, fixed seed), isobars
every 4 hPa labelled every 8 hPa, H/L centres with central pressure, a cold,
a warm and an occluded front, geostrophic wind barbs and a faint graticule on
a deep sea-slate gradient. Also writes the window/exe icon.

    python portable/make_pressure_map.py OUT_DIR
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon, Wedge  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

W, H, DPI = 1920, 1200, 100
SEA_TOP = "#0F2740"
SEA_BOTTOM = "#173E5E"
ISOBAR = "#A9D2F2"
HIGH = "#6CB4FF"
LOW = "#FF7A6B"
COLD = "#5B95FF"
WARM = "#FF6F61"
OCCLUDED = "#B98CF0"

CENTRES = [  # x, y (0..1), amplitude hPa, radius
    (0.20, 0.30, +17, 0.30),
    (0.86, 0.80, +12, 0.26),
    (0.60, 0.42, -24, 0.18),
    (0.33, 0.86, -11, 0.15),
    (0.96, 0.18, -9, 0.14),
]


def field(x, y):
    p = np.full_like(x, 1013.0)
    for cx, cy, amp, r in CENTRES:
        p += amp * np.exp(-((x - cx) ** 2 + ((y - cy) * 1.25) ** 2) / (2 * r ** 2))
    p += 1.6 * np.sin(3.1 * x + 1.3) * np.cos(2.4 * y - 0.4)
    return p


def front_path(points, n=400):
    pts = np.asarray(points, dtype=float)
    t = np.linspace(0, 1, len(pts))
    tt = np.linspace(0, 1, n)
    # Catmull-Rom-like smoothing through the control points
    xs = np.interp(tt, t, pts[:, 0])
    ys = np.interp(tt, t, pts[:, 1])
    k = np.ones(25) / 25
    xs = np.convolve(np.pad(xs, 12, mode="edge"), k, mode="valid")
    ys = np.convolve(np.pad(ys, 12, mode="edge"), k, mode="valid")
    return xs, ys


def draw_front(ax, points, kind, aspect):
    xs, ys = front_path(points)
    seg = np.hypot(np.diff(xs), np.diff(ys) * aspect)
    s = np.concatenate([[0], np.cumsum(seg)])
    color = {"cold": COLD, "warm": WARM, "occluded": OCCLUDED}[kind]
    ax.plot(xs, ys, color=color, lw=3.2, solid_capstyle="round", zorder=5, alpha=0.95)
    step = 0.045
    for i, at in enumerate(np.arange(step / 2, s[-1], step)):
        j = int(np.searchsorted(s, at))
        j = min(max(j, 1), len(xs) - 2)
        dx, dy = xs[j + 1] - xs[j - 1], (ys[j + 1] - ys[j - 1]) * aspect
        L = np.hypot(dx, dy)
        tx, ty = dx / L, dy / L
        nx, ny = -ty, tx           # left-hand normal (screen space)
        size = 0.012
        cx, cy = xs[j], ys[j]
        shape = kind
        if kind == "occluded":
            shape = "cold" if i % 2 == 0 else "warm"
        if shape == "cold":
            tri = [(cx - tx * size, cy - ty * size / aspect),
                   (cx + tx * size, cy + ty * size / aspect),
                   (cx + nx * size * 1.5, cy + ny * size * 1.5 / aspect)]
            ax.add_patch(Polygon(tri, closed=True, color=color, zorder=6, alpha=0.95))
        else:
            ang = np.degrees(np.arctan2(ny, nx))
            ax.add_patch(Wedge((cx, cy), size * 1.05, ang - 90, ang + 90, color=color,
                               zorder=6, alpha=0.95,
                               transform=ax.transData))


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    aspect = H / W
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # background gradient (top-left -> bottom-right)
    gy, gx = np.mgrid[0:1:300j, 0:1:480j]
    t = np.clip(0.55 * gx + 0.45 * (1 - gy), 0, 1)
    top = np.array(matplotlib.colors.to_rgb(SEA_TOP))
    bot = np.array(matplotlib.colors.to_rgb(SEA_BOTTOM))
    grad = top[None, None, :] * (1 - t[..., None]) + bot[None, None, :] * t[..., None]
    ax.imshow(grad, extent=(0, 1, 0, 1), origin="lower", zorder=0, aspect="auto")

    # graticule
    for gxv in np.linspace(0, 1, 13):
        ax.plot([gxv + 0.03 * np.sin(gxv * 3), gxv - 0.03], [0, 1], color="white",
                alpha=0.045, lw=0.8, zorder=1)
    for gyv in np.linspace(0, 1, 9):
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, gyv + 0.02 * np.sin(xs * np.pi), color="white", alpha=0.045, lw=0.8, zorder=1)

    x, y = np.meshgrid(np.linspace(0, 1, 480), np.linspace(0, 1, 300))
    p = field(x, y)
    levels = np.arange(984, 1036, 4)
    cs = ax.contour(x, y, p, levels=levels, colors=ISOBAR, linewidths=1.05, alpha=0.62, zorder=2)
    bold = [lv for lv in levels if lv % 8 == 0]
    ax.clabel(cs, levels=bold, fmt="%d", fontsize=9, inline=True, inline_spacing=6,
              colors=ISOBAR)

    # geostrophic barbs (wind parallel to isobars, low pressure on the left)
    bx, by = np.meshgrid(np.linspace(0.04, 0.96, 22), np.linspace(0.05, 0.95, 14))
    dpx = (field(bx + 1e-3, by) - field(bx - 1e-3, by)) / 2e-3
    dpy = (field(bx, by + 1e-3) - field(bx, by - 1e-3)) / 2e-3
    u, v = -dpy, dpx
    spd = np.hypot(u, v)
    scale = 55.0 / max(np.percentile(spd, 95), 1e-6)
    ax.barbs(bx, by, u * scale * 0.4, v * scale * 0.4, length=5.2, color="white",
             alpha=0.16, linewidth=0.8, zorder=3)

    # fronts
    # occlusion from the low to the triple point, then the cold and warm fronts
    draw_front(ax, [(0.615, 0.405), (0.64, 0.385), (0.665, 0.36)], "occluded", aspect)
    draw_front(ax, [(0.665, 0.36), (0.61, 0.28), (0.53, 0.20), (0.45, 0.10), (0.39, -0.02)],
               "cold", aspect)
    draw_front(ax, [(0.665, 0.36), (0.745, 0.34), (0.825, 0.30), (0.905, 0.24)], "warm", aspect)

    # pressure centres
    for cx, cy, amp, _ in CENTRES:
        hi = amp > 0
        col = HIGH if hi else LOW
        ax.text(cx, cy, "H" if hi else "L", color=col, fontsize=46, fontweight="bold",
                ha="center", va="center", zorder=7, family="DejaVu Sans")
        ax.text(cx + 0.002, cy - 0.068, f"{field(np.array(cx), np.array(cy)):.0f}",
                color=col, fontsize=13, ha="center", va="center", zorder=7, alpha=0.9,
                family="DejaVu Sans")
        ax.add_patch(Circle((cx, cy + 0.0), 0.0035, color=col, zorder=7, alpha=0.0))

    path = os.path.join(out_dir, "pressure_map.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)

    # soft vignette so panels and text stay readable
    img = Image.open(path).convert("RGB")
    vig = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(vig)
    d.ellipse([-W * 0.25, -H * 0.3, W * 1.25, H * 1.3], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(160))
    dark = Image.new("RGB", img.size, (6, 16, 28))
    img = Image.composite(img, dark, vig)
    img.save(path, optimize=True)
    print("->", path)

    # icon: an L with two isobar rings on the sea gradient
    ico = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(ico)
    d.rounded_rectangle([8, 8, 248, 248], radius=52, fill=(19, 50, 80, 255))
    for r, a in ((100, 150), (72, 190), (46, 230)):
        d.ellipse([128 - r, 128 - r * 0.86, 128 + r, 128 + r * 0.86],
                  outline=(169, 210, 242, a), width=7)
    try:
        font = ImageFont.truetype("bahnschrift.ttf", 92)
    except OSError:
        font = ImageFont.load_default()
    d.text((128, 132), "L", fill=(255, 122, 107, 255), font=font, anchor="mm")
    ico_path = os.path.join(out_dir, "app.ico")
    ico.save(ico_path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                              (128, 128), (256, 256)])
    ico.save(os.path.join(out_dir, "app_icon.png"))
    print("->", ico_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "gui", "assets"))
