"""Turn a solid Mini Skyline .3mf into hollow LED tiles.

Keeps the model's top surface (roads, paths, landmarks), throws away the
solid inside, and rebuilds it as a thin skin over the same LED-cell walls
that build_terrain.py uses.

Run:  .venv/bin/python -I convert_3mf.py ~/Downloads/stanford-university.3mf
"""

import json
import re
import sys
import zipfile
from pathlib import Path

# -I (isolated mode, used because the .3mf is a downloaded file) stops Python
# adding this folder to the import path, so add it back by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import ndimage

from build_terrain import (CAVITY_MM, GRID_MM, LED_PITCH_MM, OUT, SKIN_MM,
                           check_watertight, solid_between, underside,
                           volume_cm3, wall_mask, write_stl)

WIDTH_MM = 320.0           # two 16-LED panels wide
PANEL_MM = 160.0
SAMPLE_STEP = 0.09         # mm in the source model; finer than half a grid cell
DEST = OUT / "miniskyline"


def load_top_faces(path):
    """All upward-facing triangles of the model as an (n, 3, 3) array."""
    with zipfile.ZipFile(path) as z:
        data = z.read("3D/3dmodel.model")
        meta = json.loads(z.read("Metadata/miniskylines_sources.json"))
    starts = [(m.start(), m.group(1)) for m in
              re.finditer(rb'<object id="\d+"[^>]*?name="([^"]*)"', data)]
    starts.append((len(data), None))
    vre = re.compile(rb'<vertex x="([-\d.e]+)" y="([-\d.e]+)" z="([-\d.e]+)"')
    tre = re.compile(rb'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"')
    out = []
    for (s, name), (e, _) in zip(starts, starts[1:]):
        if name == b"Base":
            continue       # the flat slab under the map; the LEDs go there instead
        seg = data[s:e]
        v = np.array(vre.findall(seg), dtype=np.float64)
        t = np.array(tre.findall(seg), dtype=np.int64)
        if len(v) and len(t):
            out.append(v[t])
    tris = np.concatenate(out)
    normal_z = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, 2]
    return tris[normal_z > 1e-9], meta["browserGeneration"]["sourceProvenance"]["bbox"]


def height_grid(tris, scale, rows, cols, half_w, half_h):
    """Highest surface point at every grid vertex (a z-buffer seen from above).

    Each triangle is covered with a lattice of sample points close enough
    together that no grid vertex under it is missed.
    """
    grid = np.full((rows + 1) * (cols + 1), -np.inf)
    edge = np.linalg.norm(tris[:, [1, 2, 0], :2] - tris[:, :, :2], axis=2).max(1)
    level = np.maximum(1, np.ceil(edge / SAMPLE_STEP)).astype(int)
    for n in np.unique(level):
        group = tris[level == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        a, b = i[keep] / n, j[keep] / n
        w = np.stack([1 - a - b, a, b], 1)                 # barycentric weights
        chunk = max(1, 2_000_000 // len(w))                # keep memory bounded
        for k in range(0, len(group), chunk):
            p = np.einsum("pw,twc->tpc", w, group[k:k + chunk]).reshape(-1, 3)
            c = np.rint((p[:, 0] * scale + half_w) / GRID_MM).astype(int)
            r = np.rint((half_h - p[:, 1] * scale) / GRID_MM).astype(int)
            ok = (c >= 0) & (c <= cols) & (r >= 0) & (r <= rows)
            np.maximum.at(grid, r[ok] * (cols + 1) + c[ok], p[ok, 2] * scale)
    grid = grid.reshape(rows + 1, cols + 1)
    missing = ~np.isfinite(grid)
    if missing.any():      # fill any vertex no sample landed on from its neighbour
        idx = ndimage.distance_transform_edt(missing, return_distances=False,
                                             return_indices=True)
        grid = grid[tuple(idx)]
    return grid, int(missing.sum())


def write_piece(name, top, bottom):
    w_r, w_c = wall_mask(top.shape[0] - 1), wall_mask(top.shape[1] - 1)
    b = np.where(w_r[:, None] | w_c[None, :], 0.0, bottom)
    verts, faces = solid_between(top, b)
    assert check_watertight(faces), f"{name} is not watertight"
    assert (top - b).min() > 0.5, f"{name}: skin collapsed"
    write_stl(DEST / name, verts, faces)
    vol = volume_cm3(verts, faces)
    size = f"{(top.shape[1] - 1) * GRID_MM:.0f} x {(top.shape[0] - 1) * GRID_MM:.0f}"
    print(f"{name}: {size} mm, {top.max():.1f} mm tall, {vol * 1.24:.0f} g PLA")


def main(path):
    DEST.mkdir(parents=True, exist_ok=True)
    tris, (west, south, east, north) = load_top_faces(path)
    xy = tris[:, :, :2].reshape(-1, 2)
    src_w, src_h = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
    # Scale so the height is a whole number of LED rows, then trim the width.
    rows_led = round(src_h * WIDTH_MM / src_w / LED_PITCH_MM)
    height_mm = rows_led * LED_PITCH_MM
    scale = height_mm / src_h
    cols, rows = round(WIDTH_MM / GRID_MM), round(height_mm / GRID_MM)
    z, n_missing = height_grid(tris, scale, rows, cols, WIDTH_MM / 2, height_mm / 2)

    top = z - z.min() + CAVITY_MM + SKIN_MM
    bottom = underside(top)
    np.save(DEST / "top_mm.npy", top)
    print(f"scaled {scale:.3f}x to {WIDTH_MM:.0f} x {height_mm:.0f} mm "
          f"({cols * GRID_MM / LED_PITCH_MM:.0f} x {rows_led} LEDs), "
          f"{len(tris)} top faces, {n_missing} grid points filled")

    # Split on the LED panel seams: the north row of tiles is one full panel tall.
    split = round(PANEL_MM / GRID_MM)
    for r0, r1, ns in ((0, split, "N"), (split, rows, "S")):
        for c0, c1, ew in ((0, split, "W"), (split, cols, "E")):
            sl = np.s_[r0:r1 + 1, c0:c1 + 1]
            write_piece(f"ms_tile_{ns}{ew}.stl", top[sl], bottom[sl])

    # 60 mm test piece around Memorial Church and Hoover Tower
    n = round(60 / GRID_MM)
    r0, c0 = round((height_mm - 160) / GRID_MM), round(180 / GRID_MM)
    sl = np.s_[r0:r0 + n + 1, c0:c0 + n + 1]
    drop = top[sl].min() - (CAVITY_MM + SKIN_MM)
    write_piece("ms_sample.stl", top[sl] - drop, bottom[sl] - drop)

    trim = (src_w * scale - WIDTH_MM) / 2 / (src_w * scale) * (east - west)
    (DEST / "map.json").write_text(json.dumps({
        "west": west + trim, "east": east - trim, "south": south, "north": north,
        "led_cols": round(WIDTH_MM / LED_PITCH_MM), "led_rows": rows_led,
        "note": "col = (lon - west) / (east - west) * led_cols; "
                "row = (north - lat) / (north - south) * led_rows",
    }, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
