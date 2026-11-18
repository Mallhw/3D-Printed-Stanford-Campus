"""Download USGS laser-scan (lidar) points for a small area.

The scan is the public Santa Clara County 2020 survey. It is stored as an
octree: a tree of boxes where each level down has points twice as dense.
We walk the tree, keep only boxes that touch our area, and stop at a depth
that gives about one point per square metre.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import laspy
import numpy as np
import requests
from pyproj import Transformer

ROOT = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_SantaClaraCounty_2020"
MAX_DEPTH = 10            # about 0.9 m between points at this level
UA = {"User-Agent": "terrain-model-builder/1.0"}
TO_MERC = Transformer.from_crs("EPSG:26910", "EPSG:3857", always_xy=True)
FROM_MERC = Transformer.from_crs("EPSG:3857", "EPSG:26910", always_xy=True)


def _get(url):
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    return r.content


def _node_box(cube, d, x, y, z):
    size = (cube[3] - cube[0]) / 2 ** d
    lo = (cube[0] + x * size, cube[1] + y * size, cube[2] + z * size)
    return lo, (lo[0] + size, lo[1] + size, lo[2] + size)


def _wanted_nodes(cube, query):
    """Keys of every octree box that overlaps the query box."""
    found, pending = [], ["0-0-0-0"]
    while pending:
        root = pending.pop()
        tree = json.loads(_get(f"{ROOT}/ept-hierarchy/{root}.json"))
        stack = [root]
        while stack:
            key = stack.pop()
            if key not in tree:
                continue
            d, x, y, z = map(int, key.split("-"))
            lo, hi = _node_box(cube, d, x, y, z)
            if any(hi[i] < query[i] or lo[i] > query[i + 3] for i in range(3)):
                continue
            if tree[key] == -1 and key != root:
                pending.append(key)          # this branch continues in another file
                continue
            found.append(key)
            if d < MAX_DEPTH:
                stack += [f"{d + 1}-{2 * x + i}-{2 * y + j}-{2 * z + k}"
                          for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    return found


def fetch_points(bbox_utm, cache_file):
    """Return (x, y, z, classification) arrays in UTM metres for the area."""
    cache_file = Path(cache_file)
    if cache_file.exists():
        d = np.load(cache_file)
        return d["x"], d["y"], d["z"], d["c"]
    xmin, ymin, xmax, ymax = bbox_utm
    mx, my = TO_MERC.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    query = (min(mx), min(my), -100.0, max(mx), max(my), 500.0)
    cube = json.loads(_get(f"{ROOT}/ept.json"))["bounds"]
    keys = _wanted_nodes(cube, query)

    def load(key):
        las = laspy.read(io.BytesIO(_get(f"{ROOT}/ept-data/{key}.laz")))
        return (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z),
                np.asarray(las.classification))

    with ThreadPoolExecutor(8) as pool:       # downloads are slow; do 8 at once
        parts = list(pool.map(load, keys))
    x, y, z, c = (np.concatenate(p) for p in zip(*parts))
    keep = (x >= query[0]) & (x <= query[3]) & (y >= query[1]) & (y <= query[4])
    ux, uy = FROM_MERC.transform(x[keep], y[keep])
    np.savez_compressed(cache_file, x=ux, y=uy, z=z[keep], c=c[keep])
    print(f"lidar: {len(keys)} boxes, {keep.sum():,} points")
    return ux, uy, z[keep], c[keep]
