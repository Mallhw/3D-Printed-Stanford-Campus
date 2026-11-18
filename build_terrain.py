"""Build printable Stanford terrain tiles that sit on a 32x32 LED matrix.

Pipeline: USGS 3DEP elevation + OpenStreetMap buildings -> one height grid
-> four 160 mm tiles (one per 16x16 LED panel) -> binary STL files.

Each tile is a thin translucent skin held up by a grid of walls on the LED
pitch. The walls do two jobs: they support the skin while printing (no
supports needed) and they stop one LED's light leaking into the next cell.

Run:  .venv/bin/python build_terrain.py
"""

import json
import math
from pathlib import Path

import numpy as np
import requests
import tifffile
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy import ndimage

# ----------------------------------------------------------------- config
CENTER_LAT, CENTER_LON = 37.4285, -122.1724  # centre of the Find My screenshot view
SPAN_M = 2160.0          # ground width of the square map (67.5 m per LED)
PANELS = 2               # 2x2 LED panels
LEDS_PER_PANEL = 16
LED_PITCH_MM = 10.0      # standard WS2812B 16x16 panel
GRID_MM = 0.4            # mesh resolution = one nozzle width; finer cannot print
Z_EXAG = 2.0             # campus is flat; exaggerate height so relief shows
SKIN_MM = 1.0            # translucent top thickness (5 layers at 0.2 mm)
CAVITY_MM = 5.0          # min air gap above the LEDs so light spreads in a cell
WALL_MM = 0.8            # baffle wall = two 0.4 mm perimeters
DEFAULT_BUILDING_M = 9.0
MIN_BUILDING_M = 5.0

HERE = Path(__file__).parent
CACHE = HERE / "data"                       # downloaded source data (gitignored)
OUT = Path.home() / "Documents" / "stanford-lightmap"   # durable outputs

TILE_MM = LEDS_PER_PANEL * LED_PITCH_MM     # 160
MODEL_MM = PANELS * TILE_MM                 # 320
N = round(MODEL_MM / GRID_MM)               # cells per side (800)
SCALE = SPAN_M * 1000.0 / MODEL_MM          # 12000 -> 1:12000
UA = {"User-Agent": "terrain-model-builder/1.0"}

# UTM zone 10N: a flat metre grid, so the map is not stretched like raw lat/lon
TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:26910", always_xy=True)
# cache files are named after the map area, so changing the area refetches
AREA = f"{CENTER_LAT}_{CENTER_LON}_{SPAN_M:.0f}"


def bbox_utm():
    cx, cy = TO_UTM.transform(CENTER_LON, CENTER_LAT)
    h = SPAN_M / 2
    return cx - h, cy - h, cx + h, cy + h


# ------------------------------------------------------------------ fetch
def fetch_dem(bbox):
    """Bare-earth elevation (metres) as a 1600x1600 array, row 0 = north."""
    path = CACHE / f"dem_{AREA}.tif"
    if not path.exists():
        r = requests.get(
            "https://elevation.nationalmap.gov/arcgis/rest/services/"
            "3DEPElevation/ImageServer/exportImage",
            params={
                "bbox": ",".join(f"{v:.2f}" for v in bbox),
                "bboxSR": 26910, "imageSR": 26910, "size": "1600,1600",
                "format": "tiff", "pixelType": "F32",
                "interpolation": "RSP_BilinearInterpolation", "f": "image",
            },
            headers=UA, timeout=180,
        )
        r.raise_for_status()
        path.write_bytes(r.content)
    return tifffile.imread(path).astype(np.float64)


def fetch_buildings(bbox):
    path = CACHE / f"buildings_{AREA}.json"
    if not path.exists():
        to_ll = Transformer.from_crs("EPSG:26910", "EPSG:4326", always_xy=True)
        w, s = to_ll.transform(bbox[0], bbox[1])
        e, n = to_ll.transform(bbox[2], bbox[3])
        box = f"({s - 0.001:.5f},{w - 0.001:.5f},{n + 0.001:.5f},{e + 0.001:.5f})"
        q = (f'[out:json][timeout:120];(way["building"]{box};'
             f'relation["building"]{box};);out geom tags;')
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": q}, headers=UA, timeout=180)
        r.raise_for_status()
        path.write_text(r.text)
    return json.loads(path.read_text())["elements"]


# ---------------------------------------------------------------- heights
def building_height_m(tags):
    for key, mult in (("height", 1.0), ("building:levels", 3.5)):
        try:
            return max(MIN_BUILDING_M, float(str(tags[key]).split()[0]) * mult)
        except (KeyError, ValueError, IndexError):
            pass
    return DEFAULT_BUILDING_M


def height_grid(bbox):
    """Elevation in metres at every mesh vertex, with buildings added."""
    xmin, ymin, xmax, ymax = bbox
    dem = ndimage.gaussian_filter(fetch_dem(bbox), 1.5)  # remove lidar noise
    # sample the 1600 px DEM at the (N+1) vertex positions
    idx = np.linspace(-0.5, dem.shape[0] - 0.5, N + 1)
    rows, cols = np.meshgrid(idx, idx, indexing="ij")
    ground = ndimage.map_coordinates(dem, [rows, cols], order=1, mode="nearest")

    dx = SPAN_M / N
    labels = Image.new("I", (N + 1, N + 1), 0)   # "I" = 32-bit ints: building ids
    draw = ImageDraw.Draw(labels)

    def ring(geom):
        xs, ys = TO_UTM.transform([p["lon"] for p in geom], [p["lat"] for p in geom])
        return [((x - xmin) / dx, (ymax - y) / dx) for x, y in zip(xs, ys)]

    heights = [0.0]
    elements = fetch_buildings(bbox)
    # relations first: their courtyard holes must not erase stand-alone buildings
    for el in sorted(elements, key=lambda e: e["type"] != "relation"):
        bid = len(heights)
        if el["type"] == "relation":
            parts = [m for m in el.get("members", []) if len(m.get("geometry", [])) >= 3]
            for role, fill in (("outer", bid), ("inner", 0)):
                for m in parts:
                    if m.get("role") == role:
                        draw.polygon(ring(m["geometry"]), fill=fill)
        elif len(el.get("geometry", [])) >= 3:
            draw.polygon(ring(el["geometry"]), fill=bid)
        else:
            continue
        heights.append(building_height_m(el.get("tags", {})))

    lab = np.array(labels)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    # flat roof = average ground under the footprint + building height
    base = ndimage.mean(ground, lab, ids)
    roof = np.zeros(len(heights))
    roof[ids] = base + np.array(heights)[ids]
    return np.where(lab > 0, np.maximum(ground, roof[lab]), ground), ground, len(ids)


# ------------------------------------------------------------------- mesh
def wall_mask(n_cells):
    """True on vertices that belong to a baffle wall along one axis."""
    i = np.arange(n_cells + 1)
    per = round(LED_PITCH_MM / GRID_MM)          # 25 vertices per LED cell
    half = round(WALL_MM / 2 / GRID_MM)          # 1 vertex either side
    k = i % per
    edge = round(WALL_MM / GRID_MM)
    return (k <= half) | (k >= per - half) | (i <= edge) | (i >= n_cells - edge)


def solid_between(top, bottom):
    """Watertight triangles for the solid between two height grids (mm).

    Row 0 is the north edge. bottom must be 0 around the whole perimeter.
    """
    r, c = top.shape
    xs = np.arange(c) * GRID_MM
    ys = (r - 1 - np.arange(r)) * GRID_MM
    X, Y = np.meshgrid(xs, ys)
    verts = np.concatenate([
        np.stack([X, Y, top], -1).reshape(-1, 3),
        np.stack([X, Y, bottom], -1).reshape(-1, 3),
    ])
    vid = np.arange(r * c).reshape(r, c)
    a, b, d, e = vid[:-1, :-1], vid[:-1, 1:], vid[1:, :-1], vid[1:, 1:]
    up = np.concatenate([np.stack([a, d, b], -1).reshape(-1, 3),
                         np.stack([b, d, e], -1).reshape(-1, 3)])
    down = up[:, ::-1] + r * c                   # same quads, flipped to face down
    # skirts: walk the perimeter clockwise (seen from above) and join top to bottom
    loop = np.concatenate([vid[0, :-1], vid[:-1, -1], vid[-1, :0:-1], vid[:0:-1, 0]])
    nxt = np.roll(loop, -1)
    side = np.concatenate([np.stack([loop, nxt, loop + r * c], -1),
                           np.stack([nxt, nxt + r * c, loop + r * c], -1)])
    return verts, np.concatenate([up, down, side])


def check_watertight(faces):
    """Every edge must be used once in each direction (closed, consistent)."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    fwd = np.unique(e, axis=0)
    back = np.unique(e[:, ::-1], axis=0)
    return len(fwd) == len(e) and np.array_equal(fwd, back)


def volume_cm3(verts, faces):
    p = verts[faces]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6000)


def write_stl(path, verts, faces):
    p = verts[faces].astype("<f4")
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, p
    with open(path, "wb") as f:
        f.write(b"stanford-lightmap".ljust(80, b" "))
        f.write(np.uint32(len(faces)).tobytes())
        f.write(rec.tobytes())


def underside(top):
    """Hollow the model: skin of constant thickness, walls down to the bed."""
    # erosion pulls the cavity in from steep faces so building walls are not paper-thin
    yy, xx = np.mgrid[-2:3, -2:3]
    disk = (yy * yy + xx * xx) <= 4
    return ndimage.grey_erosion(top, footprint=disk, mode="nearest") - SKIN_MM


# ------------------------------------------------------------------- main
def main():
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    bbox = bbox_utm()
    elev, ground, n_buildings = height_grid(bbox)
    floor_m = ground.min()
    top = CAVITY_MM + SKIN_MM + (elev - floor_m) * Z_EXAG * 1000.0 / SCALE
    bottom = underside(top)

    cells = round(TILE_MM / GRID_MM)             # 400
    w = wall_mask(cells)
    walls = w[:, None] | w[None, :]
    report = []
    for ty, ns in enumerate("NS"):
        for tx, ew in enumerate("WE"):
            sl = np.s_[ty * cells: (ty + 1) * cells + 1, tx * cells: (tx + 1) * cells + 1]
            t = top[sl]
            b = np.where(walls, 0.0, bottom[sl])
            verts, faces = solid_between(t, b)
            assert check_watertight(faces), f"tile {ns}{ew} is not watertight"
            assert (t - b).min() > 0.5, "skin collapsed somewhere"
            name = f"tile_{ns}{ew}.stl"
            write_stl(OUT / name, verts, faces)
            vol = volume_cm3(verts, faces)
            report.append((name, len(faces), t.max(), vol))

    # test coupon: four LED cells with different skin thickness, to pick SKIN_MM
    skins = [0.6, 0.8, 1.0, 1.4]
    per = round(LED_PITCH_MM / GRID_MM)
    ct = np.full((per + 1, per * 4 + 1), CAVITY_MM + max(skins))
    cb = ct - np.repeat(skins, per)[None, np.minimum(np.arange(per * 4 + 1), per * 4 - 1)]
    wr, wc = wall_mask(per), wall_mask(per * 4)
    cb = np.where(wr[:, None] | wc[None, :], 0.0, cb)
    cv, cf = solid_between(ct, cb)
    assert check_watertight(cf)
    write_stl(OUT / "test_coupon.stl", cv, cf)

    meta = {
        "crs": "EPSG:26910", "bbox_utm": bbox, "span_m": SPAN_M, "scale": SCALE,
        "leds_per_side": PANELS * LEDS_PER_PANEL,
        "metres_per_led": SPAN_M / (PANELS * LEDS_PER_PANEL),
        "z_exaggeration": Z_EXAG, "skin_mm": SKIN_MM,
        "elevation_range_m": [float(ground.min()), float(ground.max())],
        "note": "LED col = (x - xmin) // metres_per_led from west; "
                "row = (ymax - y) // metres_per_led from north",
    }
    (OUT / "map.json").write_text(json.dumps(meta, indent=2))
    np.save(CACHE / "top_mm.npy", top)

    print(f"scale 1:{SCALE:.0f}, {n_buildings} buildings, "
          f"ground {ground.min():.0f}-{ground.max():.0f} m")
    for name, nf, zmax, vol in report:
        # 1.24 g/cm3 is PLA density
        print(f"{name}: {nf} triangles, {zmax:.1f} mm tall, {vol:.0f} cm3 = {vol * 1.24:.0f} g PLA")


def latlon_to_led(lat, lon, meta):
    """Map a GPS fix to (row, col) on the LED matrix, or None if off the map."""
    x, y = TO_UTM.transform(lon, lat)
    xmin, ymin, xmax, ymax = meta["bbox_utm"]
    col = math.floor((x - xmin) / meta["metres_per_led"])
    row = math.floor((ymax - y) / meta["metres_per_led"])
    n = meta["leds_per_side"]
    return (row, col) if 0 <= row < n and 0 <= col < n else None


if __name__ == "__main__":
    main()
