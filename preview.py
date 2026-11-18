"""Render a shaded preview of the height grid with landmarks and tile seams.

Run after build_terrain.py:  .venv/bin/python preview.py
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")  # draw straight to a file; no window needed
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from build_terrain import CACHE, GRID_MM, MODEL_MM, OUT, TILE_MM, TO_UTM

LANDMARKS = {
    "Hoover Tower": (37.4277, -122.1669), "The Dish": (37.4086, -122.1796),
    "Stadium": (37.4345, -122.1611), "Stern Hall": (37.4246, -122.1656),
    "Main Quad": (37.4275, -122.1703), "Lake Lagunita": (37.4222, -122.1765),
    "EVGR": (37.4262, -122.1575), "Hospital": (37.4345, -122.1770),
    "Clark Center": (37.4315, -122.1744), "Dinkelspiel": (37.4241, -122.1699),
    "Lakeside Dining": (37.4247, -122.1763),
}

top = np.load(CACHE / "top_mm.npy")
meta = json.loads((OUT / "map.json").read_text())
xmin, ymin, xmax, ymax = meta["bbox_utm"]

shade = LightSource(315, 35).hillshade(top, vert_exag=3, dx=GRID_MM, dy=GRID_MM)
fig, ax = plt.subplots(figsize=(11, 11))
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
