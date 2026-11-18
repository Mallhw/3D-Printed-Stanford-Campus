"""Render a shaded preview of the height grid with landmarks and tile seams.

Run after build_terrain.py:  .venv/bin/python preview.py
With --layers it colours roads, water, lawns and buildings like Maps3D and
also writes preview_3d.png, a tilted view.
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")  # draw straight to a file; no window needed
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, to_rgb

from build_terrain import CACHE, GRID_MM, LAYERS, MODEL_MM, OUT, TILE_MM, TO_UTM

LANDMARKS = {
    "Hoover Tower": (37.4277, -122.1669), "The Dish": (37.4086, -122.1796),
    "Stadium": (37.4345, -122.1611), "Stern Hall": (37.4246, -122.1656),
    "Main Quad": (37.4275, -122.1703), "Lake Lagunita": (37.4222, -122.1765),
    "EVGR": (37.4262, -122.1575), "Hospital": (37.4345, -122.1770),
    "Clark Center": (37.4315, -122.1744), "Dinkelspiel": (37.4241, -122.1699),
    "Lakeside Dining": (37.4247, -122.1763),
}
# ground, green, water, road, building: the colours Maps3D uses on screen
PALETTE = np.array([to_rgb(c) for c in
                    ("#F5F0EF", "#CFE2C4", "#9AD4F2", "#6B727A", "#E8AC4A")])

top = np.load(CACHE / "top_mm.npy")
meta = json.loads((OUT / "map.json").read_text())
xmin, ymin, xmax, ymax = meta["bbox_utm"]

light = LightSource(315, 35)
shade = light.hillshade(top, vert_exag=3, dx=GRID_MM, dy=GRID_MM)
fig, ax = plt.subplots(figsize=(11, 11))
if LAYERS:
    colors = PALETTE[np.load(CACHE / "classes.npy")]
    # multiply colour by the shading so slopes and building edges still show
    ax.imshow(colors * (0.55 + 0.45 * shade[..., None]), extent=[0, MODEL_MM, 0, MODEL_MM])
else:
    ax.imshow(shade, cmap="gray", extent=[0, MODEL_MM, 0, MODEL_MM])
    ax.imshow(top, cmap="terrain", alpha=0.35, extent=[0, MODEL_MM, 0, MODEL_MM])
ax.axvline(TILE_MM, color="red", lw=1.5)
ax.axhline(TILE_MM, color="red", lw=1.5)
for name, (lat, lon) in LANDMARKS.items():
    x, y = TO_UTM.transform(lon, lat)
    if not (xmin < x < xmax and ymin < y < ymax):
        continue  # landmark is off this map
    mx = (x - xmin) / meta["span_m"] * MODEL_MM
    my = (y - ymin) / meta["span_m"] * MODEL_MM
    ax.plot(mx, my, "o", color="red", ms=5)
    ax.annotate(name, (mx, my), xytext=(4, 4), textcoords="offset points",
                color="white", fontsize=9, bbox=dict(fc="black", alpha=0.6, pad=1, lw=0))
ax.set_xlabel("mm (west to east)")
ax.set_ylabel("mm (south to north)")
ax.set_title(f"Stanford light map, 1:{meta['scale']:.0f}, red lines = tile seams")
fig.savefig(OUT / "preview.png", dpi=110, bbox_inches="tight")

if LAYERS:
    # tilted view; every 2nd vertex keeps matplotlib's slow 3D drawing bearable
    z, c = top[::2, ::2], colors[::2, ::2]
    r, k = z.shape
    X, Y = np.meshgrid(np.arange(k) * GRID_MM * 2, (r - 1 - np.arange(r)) * GRID_MM * 2)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, z, facecolors=light.shade_rgb(c, z, vert_exag=2, fraction=0.6),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((MODEL_MM, MODEL_MM, float(z.max())))   # true proportions
    ax.view_init(elev=42, azim=-110)
    ax.set_axis_off()
    fig.savefig(OUT / "preview_3d.png", dpi=130, bbox_inches="tight")
