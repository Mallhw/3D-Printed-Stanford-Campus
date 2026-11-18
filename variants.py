"""Build several candidate map styles and render them side by side.

Nothing here is printable yet. Each style is just a height grid (the top
surface in mm) so the looks can be compared before choosing one to print.

Run:  .venv/bin/python variants.py
"""

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import requests
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from build_terrain import CACHE, MODEL_MM, OUT, TO_UTM, UA, building_height_m
from lidar import fetch_points

N = 800                                    # grid cells per side (0.4 mm each)
DEST = OUT / "variants"
AREAS = {                                  # name: (lat, lon, width in metres, height exaggeration)
    "full": (37.4285, -122.1724, 2160.0, 2.0),
    "zoom": (37.4265, -122.1690, 1280.0, 1.5),
}
ROAD_M = {"primary": 14, "secondary": 12, "tertiary": 10, "residential": 7,
          "unclassified": 7, "service": 5, "living_street": 6, "pedestrian": 5}
PATH_M = {"footway": 3, "path": 3, "cycleway": 3, "steps": 3, "track": 3}


class Area:
    def __init__(self, name):
        lat, lon, self.span, self.exag = AREAS[name]
        self.name = name
        cx, cy = TO_UTM.transform(lon, lat)
        h = self.span / 2
        self.bbox = (cx - h, cy - h, cx + h, cy + h)
        self.dx = self.span / N                         # metres per grid cell
        self.mm_per_m = MODEL_MM / self.span            # horizontal model scale
        self.z_mm_per_m = self.mm_per_m * self.exag

    def px(self, lons, lats):
        xs, ys = TO_UTM.transform(lons, lats)
        return [((x - self.bbox[0]) / self.dx, (self.bbox[3] - y) / self.dx)
                for x, y in zip(xs, ys)]

    def ring(self, geom):
        return self.px([p["lon"] for p in geom], [p["lat"] for p in geom])


# -------------------------------------------------------------- map data
def osm(area):
    path = CACHE / f"osm_{area.name}.json"
    if not path.exists():
        from pyproj import Transformer
        ll = Transformer.from_crs("EPSG:26910", "EPSG:4326", always_xy=True)
        w, s = ll.transform(area.bbox[0], area.bbox[1])
        e, n = ll.transform(area.bbox[2], area.bbox[3])
        box = f"({s - .001:.5f},{w - .001:.5f},{n + .001:.5f},{e + .001:.5f})"
        q = (f'[out:json][timeout:120];(way["building"]{box};relation["building"]{box};'
             f'way["highway"]{box};);out geom;')
        # the main server is often busy, so fall back to public mirrors
        servers = ["https://overpass-api.de/api/interpreter",
                   "https://overpass.private.coffee/api/interpreter",
                   "https://overpass.kumi.systems/api/interpreter"]
        for attempt, url in enumerate(servers * 2):
            try:
                r = requests.post(url, data={"data": q}, headers=UA, timeout=180)
                r.raise_for_status()
                json.loads(r.text)
                break
            except (requests.RequestException, ValueError):
                if attempt == len(servers) * 2 - 1:
                    raise
                time.sleep(5)
        path.write_text(r.text)
    return json.loads(path.read_text())["elements"]


def building_labels(area, elements):
    """Grid of building ids (0 = no building) and each id's tagged height."""
    img = Image.new("I", (N + 1, N + 1), 0)
    draw = ImageDraw.Draw(img)
    heights = [0.0]
    for el in sorted(elements, key=lambda e: e["type"] != "relation"):
        tags = el.get("tags", {})
        if "building" not in tags:
            continue
        bid = len(heights)
        if el["type"] == "relation":
            parts = [m for m in el.get("members", []) if len(m.get("geometry", [])) >= 3]
            if not parts:
                continue
            for inner, fill in ((False, bid), (True, 0)):
                for m in parts:
                    if (m.get("role") == "inner") == inner:
                        draw.polygon(area.ring(m["geometry"]), fill=fill)
        elif len(el.get("geometry", [])) >= 3:
            draw.polygon(area.ring(el["geometry"]), fill=bid)
        else:
            continue
        heights.append(building_height_m(tags))
    return np.array(img), np.array(heights)


def road_masks(area, elements):
    """Two true/false grids: roads (for cars) and paths (walking, bikes)."""
    masks = []
    for widths in (ROAD_M, PATH_M):
        img = Image.new("L", (N + 1, N + 1), 0)
        draw = ImageDraw.Draw(img)
        for el in elements:
            kind = el.get("tags", {}).get("highway")
            if el["type"] == "way" and kind in widths and len(el.get("geometry", [])) >= 2:
                # never thinner than 2 cells (0.8 mm) or the groove will not print
                w = max(2, round(widths[kind] / area.dx))
                draw.line(area.ring(el["geometry"]), fill=1, width=w, joint="curve")
        masks.append(np.array(img) > 0)
    return masks


def lidar_layers(area):
    """Ground height, building height and tree height (metres) per grid point."""
    x, y, z, c = fetch_points(area.bbox, CACHE / f"lidar_{area.name}.npz")
    col = np.rint((x - area.bbox[0]) / area.dx).astype(int)
    row = np.rint((area.bbox[3] - y) / area.dx).astype(int)
    ok = (col >= 0) & (col <= N) & (row >= 0) & (row <= N)
    flat = row[ok] * (N + 1) + col[ok]
    z, c = z[ok], c[ok]

    def highest(classes):
        g = np.full((N + 1) ** 2, -np.inf)
        m = np.isin(c, classes)
        np.maximum.at(g, flat[m], z[m])
        return g.reshape(N + 1, N + 1)

    ground = highest([2])                               # class 2 = bare ground
    missing = ~np.isfinite(ground)
    idx = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
    ground = ndimage.gaussian_filter(ground[tuple(idx)], 2)

    def above_ground(classes):
        h = highest(classes) - ground
        return np.where(np.isfinite(h) & (h > 2.0), h, 0.0)

    roofs = above_ground([6])                           # class 6 = buildings
    roofs = ndimage.median_filter(ndimage.grey_closing(roofs, size=3), size=3)
    trees = ndimage.gaussian_filter(above_ground([3, 4, 5]), 1.5)   # vegetation
    return ground, roofs, trees


# ---------------------------------------------------------------- styles
def blocks(area, labels, heights_m, roads, paths):
    """Flat ground, flat-roofed blocks, roads and paths cut in as grooves."""
    top = np.full(labels.shape, 1.0)
    top[paths] = 0.75
    top[roads] = 0.5
    built = labels > 0
    top[built] = 1.0 + heights_m[labels[built]] * area.z_mm_per_m
    return top


def lidar_roofs(area, roofs, roads, paths):
    top = np.full(roofs.shape, 1.0)
    top[paths] = 0.75
    top[roads] = 0.5
    return np.where(roofs > 0, 1.0 + roofs * area.z_mm_per_m, top)


def lidar_everything(area, ground, roofs, trees):
    return 1.0 + (ground - ground.min() + np.maximum(roofs, trees)) * area.z_mm_per_m


def backlit_plate(roofs, roads, paths):
    """Nearly flat plate: thin plastic glows bright, thick plastic stays dark."""
    t = np.full(roofs.shape, 1.4)
    t[paths] = 0.9
    t[roads] = 0.6
    t[roofs > 0] = 2.6
    return t


# --------------------------------------------------------------- renders
def pool2(z):
    """Halve the grid keeping the highest point of each 2x2 block."""
    z = z[: z.shape[0] // 2 * 2, : z.shape[1] // 2 * 2]
    return z.reshape(z.shape[0] // 2, 2, z.shape[1] // 2, 2).max(axis=(1, 3))


def draw_oblique(ax, top, zoom=1.75):
    z = pool2(top)
    r, c = z.shape
    X, Y = np.meshgrid(np.arange(c) * 0.8, (r - np.arange(r)) * 0.8)
    # plain light-and-shadow shading so every style looks like the same white plastic
    light = LightSource(300, 40).hillshade(z, vert_exag=2.5, dx=0.8, dy=0.8)
    grey = 0.25 + 0.75 * light
    ax.plot_surface(X, Y, z, facecolors=np.dstack([grey, grey, grey * 0.97]),
                    rcount=r, ccount=c, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((c, r, max(z.max(), 12) / 0.8), zoom=zoom)
    ax.view_init(elev=48, azim=-100)
    ax.set_axis_off()


def draw_lit(ax, thickness):
    glow = np.exp(-thickness / 0.9)                     # thinner plastic = brighter
    glow = (glow - glow.min()) / np.ptp(glow)
    ax.imshow(glow, cmap="afmhot", vmin=-0.15, vmax=1.1)
    ax.set_axis_off()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    full, zoom = Area("full"), Area("zoom")
    styles = []

    ms = OUT / "miniskyline" / "top_mm.npy"
    if ms.exists():
        styles.append(("A  Current: Mini Skyline, 2.2 km wide", np.load(ms), "3d"))

    el = osm(full)
    labels, heights = building_labels(full, el)
    styles.append(("B  Clean blocks + road grooves, 2.2 km wide",
                   blocks(full, labels, heights, *road_masks(full, el)), "3d"))

    # the zoomed area lies inside the full one, so the same map data covers it
    labels, heights = building_labels(zoom, el)
    roads, paths = road_masks(zoom, el)
    ground, roofs, trees = lidar_layers(zoom)
    # give each block its real height: the typical lidar roof height in its outline
    ids = np.unique(labels[labels > 0])
    real = heights.copy()
    measured = ndimage.median(roofs, labels, ids)
    real[ids] = np.where(measured > 2, measured, heights[ids])
    styles.append(("C  Clean blocks + road grooves, zoomed to 1.3 km",
                   blocks(zoom, labels, real, roads, paths), "3d"))
    styles.append(("D  Real roof shapes from laser scan, zoomed",
                   lidar_roofs(zoom, roofs, roads, paths), "3d"))
    styles.append(("E  Real roofs + trees + ground slope, zoomed",
                   lidar_everything(zoom, ground, roofs, trees), "3d"))
    styles.append(("F  Flat backlit plate (shown lit), zoomed",
                   backlit_plate(roofs, roads, paths), "lit"))

    fig = plt.figure(figsize=(24, 15), facecolor="white")
    for i, (title, top, kind) in enumerate(styles):
        np.save(DEST / f"{title[0]}_top_mm.npy", top)
        if kind == "3d":
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            draw_oblique(ax, top)
        else:
            ax = fig.add_subplot(2, 3, i + 1)
            draw_lit(ax, top)
        ax.set_title(title, fontsize=17, pad=2)
        print(f"{title}: height range {top.min():.1f}-{top.max():.1f} mm")
    fig.subplots_adjust(left=0, right=1, top=0.96, bottom=0, wspace=0, hspace=0.04)
    fig.savefig(DEST / "compare.png", dpi=80)

    # close-ups of the same spot (Main Quad and Hoover Tower) in each zoomed style
    fig = plt.figure(figsize=(21, 7.5), facecolor="white")
    for i, (title, top, kind) in enumerate(s for s in styles if s[0][0] in "CDE"):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        draw_oblique(ax, top[200:440, 300:600], zoom=1.15)
        ax.set_title(title, fontsize=16, pad=2)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0, wspace=0)
    fig.savefig(DEST / "closeup.png", dpi=80)


if __name__ == "__main__":
    main()
