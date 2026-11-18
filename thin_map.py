"""Turn a maps3d.io .3mf into one thin plate that glows on an LED matrix.

The plate lies flat on a 64x64 RGB matrix. Thin plastic glows bright and
thick plastic stays dim, so the map is drawn in thickness:

    roads, water   0.6 mm   brightest
    open ground    0.8 mm
    woods          1.0 mm
    buildings      0.8 mm + real building height   dimmest

Run:  .venv/bin/python -I thin_map.py <folder with the .3mf and georeference.json>
"""

import json
import re
import sys
import zipfile
from pathlib import Path

# -I (isolated mode, used because the .3mf is a downloaded file) stops Python
# adding this folder to the import path, so add it back by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from build_terrain import OUT, check_watertight, volume_cm3, write_stl

GRID = 0.3                 # mm between mesh points
GROUND, ROAD, WATER, WOODS = 0.8, 0.6, 0.6, 1.0     # plate thickness in mm
MAX_BUILDING = 8.0         # cap so Hoover Tower is not a fragile needle
PANEL_PX, PANEL_PITCH = 64, 4.0                     # 64x64 matrix, 4 mm per pixel
SAMPLE_STEP = 0.09
DEST = OUT / "thin"


def load_parts(path):
    data = zipfile.ZipFile(path).read("3D/3dmodel.model")
    starts = [(m.start(), m.group(1).decode()) for m in
              re.finditer(rb'<object id="\d+" name="([^"]*)"', data)]
    starts.append((len(data), None))
    vre = re.compile(rb'<vertex x="([-\d.e]+)" y="([-\d.e]+)" z="([-\d.e]+)"')
    tre = re.compile(rb'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"')
    parts = []
    for (s, name), (e, _) in zip(starts, starts[1:]):
        v = np.array(vre.findall(data[s:e]), dtype=np.float64)
        t = np.array(tre.findall(data[s:e]), dtype=np.int64)
        if len(v) and len(t):
            parts.append((name, v[t]))
    return parts


def rasterize(parts, lo, hi):
    """Highest surface at each grid point, and which part it belongs to."""
    cols = int(np.ceil((hi[0] - lo[0]) / GRID)) + 1
    rows = int(np.ceil((hi[1] - lo[1]) / GRID)) + 1
    z = np.full(rows * cols, -np.inf)
    owner = np.full(rows * cols, -1)
    for pi, (_, tris) in enumerate(parts):
        up = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 2] > 1e-9
        tris = tris[up]                                   # only faces you see from above
        edge = np.linalg.norm(tris[:, [1, 2, 0], :2] - tris[:, :, :2], axis=2).max(1)
        level = np.maximum(1, np.ceil(edge / SAMPLE_STEP)).astype(int)
        for n in np.unique(level):
            group = tris[level == n]
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            keep = i + j <= n
            w = np.stack([1 - (i[keep] + j[keep]) / n, i[keep] / n, j[keep] / n], 1)
            chunk = max(1, 1_500_000 // len(w))
            for k in range(0, len(group), chunk):
                p = np.einsum("pw,twc->tpc", w, group[k:k + chunk]).reshape(-1, 3)
                idx = (np.rint((hi[1] - p[:, 1]) / GRID).astype(int) * cols
                       + np.rint((p[:, 0] - lo[0]) / GRID).astype(int))
                order = np.argsort(p[:, 2])               # highest sample written last
                idx, pz = idx[order], p[order, 2]
                better = pz > z[idx]
                z[idx[better]] = pz[better]
                owner[idx[better]] = pi
    return z.reshape(rows, cols), owner.reshape(rows, cols)


def clean_outline(covered):
    """Which grid cells are inside the map, as one tidy shape."""
    inside = covered[:-1, :-1] & covered[:-1, 1:] & covered[1:, :-1] & covered[1:, 1:]
    inside = ndimage.binary_fill_holes(ndimage.binary_opening(inside, iterations=2))
    labels, n = ndimage.label(inside)
    if n > 1:                                             # keep the biggest island only
        inside = labels == 1 + np.argmax(ndimage.sum(inside, labels, range(1, n + 1)))
    while True:
        # two cells touching only at a corner make a mesh the slicer rejects; fill them in
        a, b, c, d = inside[:-1, :-1], inside[:-1, 1:], inside[1:, :-1], inside[1:, 1:]
        bad = (a & d & ~b & ~c) | (b & c & ~a & ~d)
        if not bad.any():
            return inside
        for part in (inside[:-1, :-1], inside[:-1, 1:], inside[1:, :-1], inside[1:, 1:]):
            part |= bad


def plate_mesh(thickness, inside):
    """Closed mesh: flat bottom at z=0, top at the given thickness, walls at the outline."""
    rows, cols = thickness.shape
    X, Y = np.meshgrid(np.arange(cols) * GRID, (rows - 1 - np.arange(rows)) * GRID)
    n = rows * cols
    verts = np.concatenate([np.stack([X, Y, thickness], -1).reshape(-1, 3),
                            np.stack([X, Y, np.zeros_like(X)], -1).reshape(-1, 3)])
    vid = np.arange(n).reshape(rows, cols)
    a, b, d, e = vid[:-1, :-1], vid[:-1, 1:], vid[1:, :-1], vid[1:, 1:]
    top = np.concatenate([np.stack([a, d, b], -1)[inside], np.stack([b, d, e], -1)[inside]])
    faces = [top, top[:, ::-1] + n]
    pad = np.pad(inside, 1)
    # a wall wherever a cell inside the map has an empty neighbour
    for (dr, dc), (p, q) in (((-1, 0), (a, b)), ((0, 1), (b, e)),
                             ((1, 0), (e, d)), ((0, -1), (d, a))):
        open_side = inside & ~pad[1 + dr: 1 + dr + inside.shape[0], 1 + dc: 1 + dc + inside.shape[1]]
        p, q = p[open_side], q[open_side]
        faces += [np.stack([p, q, p + n], -1), np.stack([q, q + n, p + n], -1)]
    return verts, np.concatenate(faces)


def main(folder):
    folder = Path(folder)
    model = next(folder.glob("*.3mf"))
    geo = json.loads((folder / "georeference.json").read_text())
    DEST.mkdir(parents=True, exist_ok=True)

    parts = load_parts(model)
    names = [p[0] for p in parts]
    every = np.concatenate([p[1].reshape(-1, 3) for p in parts])
    lo, hi = every.min(0), every.max(0)
    top, owner = rasterize(parts, lo, hi)
    ground, _ = rasterize([p for p in parts if p[0] == "terrain"], lo, hi)

    def owned(test):
        return np.isin(owner, [i for i, nm in enumerate(names) if test(nm)])

    building = owned(lambda nm: nm == "buildings")
    road = owned(lambda nm: nm.lower().startswith("roads") and nm != "roads_paths")
    road = ndimage.binary_dilation(road) & ~building     # at least 0.9 mm wide so it prints
    thickness = np.full(top.shape, GROUND)
    thickness[owned(lambda nm: nm == "landCover_WOOD")] = WOODS
    thickness[owned(lambda nm: nm == "water")] = WATER
    thickness[road] = ROAD
    known = np.isfinite(ground) & np.isfinite(top)
    height = np.zeros(top.shape)
    height[known] = top[known] - ground[known]
    height = np.clip(height, 0.4, MAX_BUILDING)
    thickness[building] = GROUND + height[building]

    inside = clean_outline(owner >= 0)
    verts, faces = plate_mesh(thickness, inside)
    assert check_watertight(faces), "plate is not watertight"
    write_stl(DEST / "thin_map.stl", verts, faces)
    vol = volume_cm3(verts, faces)
    w_mm, h_mm = hi[0] - lo[0], hi[1] - lo[1]
    print(f"thin_map.stl: {w_mm:.0f} x {h_mm:.0f} mm, {thickness[building].max():.1f} mm tall, "
          f"{len(faces):,} triangles, {vol * 1.24:.0f} g PLA")

    # 60 mm test piece around the Main Quad
    r0, c0 = round((hi[1] - 20) / GRID), round((-20 - lo[0]) / GRID)
    n = round(60 / GRID)
    cut = np.zeros_like(inside)
    cut[r0:r0 + n, c0:c0 + n] = inside[r0:r0 + n, c0:c0 + n]
    sv, sf = plate_mesh(thickness, cut)
    assert check_watertight(sf)
    sv[:, 0] -= c0 * GRID
    sv[:, 1] -= (thickness.shape[0] - 1 - r0 - n) * GRID
    write_stl(DEST / "thin_sample.stl", sv, sf)

    # Where the plate sits on the panel: centred, north up.
    panel_mm = PANEL_PX * PANEL_PITCH
    (DEST / "map.json").write_text(json.dumps({
        "proj4": geo["frame"]["proj4"],
        "mm_per_metre": geo["printScaleMmPerMeter"],
        "model_centre_mm": [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2],
        "panel_pixels": PANEL_PX, "pixel_pitch_mm": PANEL_PITCH,
        "note": "project lon/lat with proj4 to metres (x east, y north), multiply by "
                "mm_per_metre, subtract model_centre_mm, then "
                f"col = (x + {panel_mm / 2}) / pitch, row = ({panel_mm / 2} - y) / pitch",
    }, indent=2))

    np.save(DEST / "thickness.npy", np.where(np.pad(inside, ((0, 1), (0, 1))), thickness, np.nan))
    render(thickness, np.pad(inside, ((0, 1), (0, 1))), w_mm, h_mm)


def render(thickness, inside, w_mm, h_mm):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10.5), facecolor="white")
    shade = LightSource(315, 45).hillshade(thickness, vert_exag=4, dx=GRID, dy=GRID)
    unlit = np.where(inside, 0.35 + 0.65 * shade, 1.0)
    axes[0].imshow(unlit, cmap="gray", vmin=0, vmax=1, extent=[0, w_mm, 0, h_mm])
    axes[0].set_title("Unlit: white plastic", fontsize=16)
    glow = np.exp(-thickness / 0.9)                       # thinner plastic = brighter
    glow = (glow - glow[inside].min()) / np.ptp(glow[inside])
    lit = np.where(inside, glow, np.nan)
    axes[1].set_facecolor("black")
    axes[1].imshow(lit, cmap="afmhot", vmin=-0.2, vmax=1.1, extent=[0, w_mm, 0, h_mm])
    axes[1].set_title("Lit evenly from below (simulated)", fontsize=16)
    for ax in axes:
        ax.set_xlabel("mm")
    fig.savefig(DEST / "preview.png", dpi=80, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
