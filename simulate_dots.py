"""Picture of the thin map with people shown as coloured dots.

Uses made-up people at well-known spots. It draws what the 64x64 LED
matrix would show, then dims it by the plastic thickness above each pixel.

Run after thin_map.py:  .venv/bin/python simulate_dots.py
"""

import json

import matplotlib
import numpy as np
from pyproj import CRS, Transformer
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_terrain import OUT

GRID = 0.3                       # must match thin_map.py
DOT_PIXELS = 2                   # each person lights a 2x2 block (8 mm) so it is easy to see
PEOPLE = {                       # name: (lat, lon, colour) - example positions only
    "A": (37.4246, -122.1656, (1.0, 0.25, 0.25)),   # Stern Hall
    "B": (37.4275, -122.1703, (0.25, 0.6, 1.0)),    # Main Quad
    "C": (37.4269, -122.1672, (0.3, 1.0, 0.4)),     # Green Library
    "D": (37.4315, -122.1744, (1.0, 0.85, 0.2)),    # Clark Center
    "E": (37.4247, -122.1763, (0.9, 0.4, 1.0)),     # Lakeside Dining
    "F": (37.4345, -122.1611, (1.0, 0.55, 0.1)),    # Stadium
    "G": (37.4296, -122.1698, (0.2, 1.0, 0.9)),     # The Oval
}


def latlon_to_pixel(lat, lon, m, to_local):
    """GPS position -> (row, col) on the LED matrix."""
    x, y = to_local.transform(lon, lat)
    x = x * m["mm_per_metre"] - m["model_centre_mm"][0]
    y = y * m["mm_per_metre"] - m["model_centre_mm"][1]
    half = m["panel_pixels"] * m["pixel_pitch_mm"] / 2
    return int((half - y) // m["pixel_pitch_mm"]), int((x + half) // m["pixel_pitch_mm"])


def main():
    thin = OUT / "thin"
    m = json.loads((thin / "map.json").read_text())
    thickness = np.load(thin / "thickness.npy")
    inside = np.isfinite(thickness)
    to_local = Transformer.from_crs("EPSG:4326", CRS.from_proj4(m["proj4"]), always_xy=True)
    n, pitch = m["panel_pixels"], m["pixel_pitch_mm"]

    def through_plastic(leds):
        """What you see on top of the plate for a given 64x64 LED picture."""
        rows, cols = thickness.shape
        # plate is centred on the panel; find which LED sits under each plate point
        y_mm = (rows - 1) * GRID / 2 - np.arange(rows) * GRID
        x_mm = np.arange(cols) * GRID - (cols - 1) * GRID / 2
        r = np.clip(((n * pitch / 2 - y_mm) // pitch).astype(int), 0, n - 1)
        c = np.clip(((x_mm + n * pitch / 2) // pitch).astype(int), 0, n - 1)
        img = leds[r][:, c]
        img = ndimage.gaussian_filter(img, sigma=(5, 5, 0))       # light spreads ~1.5 mm in the plastic
        img = img * np.exp(-np.where(inside, thickness, 99) / 0.9)[..., None]
        return np.clip(img * 2.6, 0, 1)

    dots = np.zeros((n, n, 3))
    for name, (lat, lon, colour) in PEOPLE.items():
        row, col = latlon_to_pixel(lat, lon, m, to_local)
        dots[row:row + DOT_PIXELS, col:col + DOT_PIXELS] = colour
        print(f"{name}: pixel row {row}, col {col}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10.5), facecolor="black")
    views = [("Dots only", dots), ("Dots + dim map glow", np.maximum(dots, 0.07))]
    for ax, (title, leds) in zip(axes, views):
        ax.imshow(through_plastic(leds))
        ax.set_title(title, color="white", fontsize=18)
        ax.set_axis_off()
    fig.savefig(thin / "dots_preview.png", dpi=80, bbox_inches="tight", facecolor="black")


if __name__ == "__main__":
    main()
