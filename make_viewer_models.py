"""Write small STL copies of the models that GitHub can show in its 3D viewer.

The printable tiles are about 32 MB each, too big for the viewer, so this
saves each whole map at 1.2 mm detail instead of 0.4 mm and copies the
60 mm sample pieces as they are.

Run after build_terrain.py and convert_3mf.py:
    .venv/bin/python make_viewer_models.py
"""

import shutil
from pathlib import Path

import numpy as np

from build_terrain import CACHE, GRID_MM, OUT, write_stl

STEP = 3                                   # keep every 3rd grid point
MODELS = Path(__file__).parent / "models"


def coarse_surface(top):
    """Top surface, side skirts and a flat bottom, for viewing only."""
    r = (top.shape[0] - 1) // STEP * STEP
    c = (top.shape[1] - 1) // STEP * STEP
    # take the highest point in each block so thin buildings do not vanish
    padded = np.pad(top[:r + 1, :c + 1], ((0, STEP - 1), (0, STEP - 1)), mode="edge")
    blocks = padded.reshape(r // STEP + 1, STEP, c // STEP + 1, STEP)
    z = blocks.max(axis=(1, 3))
    rows, cols = z.shape
    step = GRID_MM * STEP
    X, Y = np.meshgrid(np.arange(cols) * step, (rows - 1 - np.arange(rows)) * step)
    verts = np.concatenate([np.stack([X, Y, z], -1).reshape(-1, 3),
                            np.stack([X, Y, np.zeros_like(z)], -1).reshape(-1, 3)])
    vid = np.arange(rows * cols).reshape(rows, cols)
    a, b, d, e = vid[:-1, :-1], vid[:-1, 1:], vid[1:, :-1], vid[1:, 1:]
    up = np.concatenate([np.stack([a, d, b], -1).reshape(-1, 3),
                         np.stack([b, d, e], -1).reshape(-1, 3)])
    loop = np.concatenate([vid[0, :-1], vid[:-1, -1], vid[-1, :0:-1], vid[:0:-1, 0]])
    nxt = np.roll(loop, -1)
    n = rows * cols
    side = np.concatenate([np.stack([loop, nxt, loop + n], -1),
                           np.stack([nxt, nxt + n, loop + n], -1)])
    corners = np.array([vid[0, 0], vid[0, -1], vid[-1, -1], vid[-1, 0]]) + n
    floor = np.array([[corners[0], corners[1], corners[2]],
                      [corners[0], corners[2], corners[3]]])
    return verts, np.concatenate([up, side, floor])


def main():
    MODELS.mkdir(exist_ok=True)
    jobs = [(CACHE / "top_mm.npy", "terrain_whole_map.stl"),
            (OUT / "miniskyline" / "top_mm.npy", "miniskyline_whole_map.stl")]
    for src, name in jobs:
        if src.exists():
            write_stl(MODELS / name, *coarse_surface(np.load(src)))
    for src, name in [(OUT / "sample.stl", "terrain_sample_60mm.stl"),
                      (OUT / "miniskyline" / "ms_sample.stl", "miniskyline_sample_60mm.stl")]:
        if src.exists():
            shutil.copyfile(src, MODELS / name)
    for f in sorted(MODELS.glob("*.stl")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
