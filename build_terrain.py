"""Build printable Stanford terrain tiles that sit on a 32x32 LED matrix.

Pipeline: USGS 3DEP elevation + OpenStreetMap buildings -> one height grid
-> four 160 mm tiles (one per 16x16 LED panel) -> binary STL files.

Each tile is a thin translucent skin held up by a grid of walls on the LED
pitch. The walls do two jobs: they support the skin while printing (no
supports needed) and they stop one LED's light leaking into the next cell.

Run:  .venv/bin/python build_terrain.py
      .venv/bin/python build_terrain.py --layers
      (Maps3D-style: adds roads, sunken water and raised green areas, uses
       the area picked in the Maps3D editor, writes to a maps3d-style folder)
      .venv/bin/python build_terrain.py --sample 180 110 60
      (a 60 mm test piece whose south-west corner is 180 mm east and
       110 mm north of the map's south-west corner)
"""

import json
import math
import sys
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

# ---- Maps3D-style layers (--layers). Heights are whole 0.2 mm print layers,
# because anything smaller than one layer cannot be printed.
# The flag is read here, not in main(), so preview.py sees the same settings
# when it imports this file.
LAYERS = "--layers" in sys.argv
if LAYERS:
    sys.argv.remove("--layers")             # leaves --sample's arguments in place
    # middle of the box drawn in the Maps3D editor (37.4189..37.43793 N,
    # 122.18218..122.15884 W); it fits inside the same 2.16 km square
    CENTER_LAT, CENTER_LON = 37.42842, -122.17051
    OUT = OUT / "maps3d-style"
GREEN_MM = 0.2           # lawns, parks, woods: one layer up
ROAD_MM = 0.4            # roads: two layers up, so they sit above the lawns
WATER_MM = -0.6          # water: three layers down ("inlaid" in Maps3D)
MIN_ROAD_MM = 0.8        # narrowest printable ridge = two nozzle widths
ROAD_WIDTH_M = {         # real-world widths by OpenStreetMap road type
    "motorway": 16, "trunk": 16, "primary": 14, "secondary": 12, "tertiary": 10,
    "residential": 8, "unclassified": 8, "living_street": 8,
    "pedestrian": 6, "service": 5,
}
GROUND, GREEN, WATER, ROAD, BUILDING = range(5)   # ids in the class grid

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


OVERPASS = (     # public OpenStreetMap query servers; any one of them will do
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)


def overpass(query):
    """Run an OpenStreetMap query; the free servers are often busy, so try each."""
    for url in OVERPASS:
        try:
            r = requests.post(url, data={"data": query}, headers=UA, timeout=90)
        except requests.RequestException:
            continue
        if r.ok and r.text.lstrip().startswith("{"):   # busy servers send an error page
            return r.text
    raise SystemExit("OpenStreetMap servers are busy; run again in a minute")


def fetch_buildings(bbox):
    path = CACHE / f"buildings_v2_{AREA}.json"
    if not path.exists():
        to_ll = Transformer.from_crs("EPSG:26910", "EPSG:4326", always_xy=True)
        w, s = to_ll.transform(bbox[0], bbox[1])
        e, n = to_ll.transform(bbox[2], bbox[3])
        box = f"({s - 0.001:.5f},{w - 0.001:.5f},{n + 0.001:.5f},{e + 0.001:.5f})"
        q = (f'[out:json][timeout:120];(way["building"]{box};'
             f'relation["building"]{box};);out geom;')
        path.write_text(overpass(q))
    return json.loads(path.read_text())["elements"]


def fetch_layers(bbox):
    """Roads, water and green areas from OpenStreetMap."""
    path = CACHE / f"layers_{AREA}.json"
    if not path.exists():
        to_ll = Transformer.from_crs("EPSG:26910", "EPSG:4326", always_xy=True)
        w, s = to_ll.transform(bbox[0], bbox[1])
        e, n = to_ll.transform(bbox[2], bbox[3])
        box = f"({s - 0.001:.5f},{w - 0.001:.5f},{n + 0.001:.5f},{e + 0.001:.5f})"
        areas = ('["natural"~"^(water|wood|grassland|scrub)$"]',
                 '["leisure"~"^(park|pitch|garden|golf_course|swimming_pool)$"]',
                 '["landuse"~"^(grass|meadow|forest|recreation_ground|village_green)$"]')
        parts = [f'way["highway"]{box};']
        parts += [f"{kind}{sel}{box};" for sel in areas for kind in ("way", "relation")]
        q = f'[out:json][timeout:120];({"".join(parts)});out geom;'
        path.write_text(overpass(q))
    return json.loads(path.read_text())["elements"]


def area_class(tags):
    if tags.get("natural") == "water" or tags.get("leisure") == "swimming_pool":
        return WATER
    if "highway" in tags:
        return None                      # roads are drawn as lines, not areas
    if tags.get("natural") or tags.get("leisure") or tags.get("landuse"):
        return GREEN
    return None


# ---------------------------------------------------------------- heights
def building_height_m(tags):
    for key, mult in (("height", 1.0), ("building:levels", 3.5)):
        try:
            h = float(str(tags[key]).split()[0]) * mult
            if h > 0:   # some buildings are tagged height=0 (e.g. the Stadium)
                return max(MIN_BUILDING_M, h)
        except (KeyError, ValueError, IndexError):
            pass
    return DEFAULT_BUILDING_M


def height_grid(bbox):
    """Elevation in metres at every mesh vertex, with buildings added.

    Also returns the layer relief in mm and a grid saying what each vertex is.
    """
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

    def paint(draw, el, fill):
        """Fill one OSM area on a drawing; False if it has no usable outline."""
        if el["type"] == "relation":
            parts = [m for m in el.get("members", []) if len(m.get("geometry", [])) >= 3]
            # anything not marked "inner" (a hole) counts as part of the area
            for inner, value in ((False, fill), (True, 0)):
                for m in parts:
                    if (m.get("role") == "inner") == inner:
                        draw.polygon(ring(m["geometry"]), fill=value)
            return bool(parts)
        if len(el.get("geometry", [])) >= 3:
            draw.polygon(ring(el["geometry"]), fill=fill)
            return True
        return False

    relief = np.zeros((N + 1, N + 1))            # mm added on top of the ground
    classes = np.zeros((N + 1, N + 1), np.uint8)
    if LAYERS:
        layers = sorted(fetch_layers(bbox), key=lambda e: e["type"] != "relation")
        green_img = Image.new("L", (N + 1, N + 1), 0)
        water_img = Image.new("I", (N + 1, N + 1), 0)     # one id per lake/pool
        n_water = 0
        for el in layers:
            kind = area_class(el.get("tags", {}))
            if kind == GREEN:
                paint(ImageDraw.Draw(green_img), el, 1)
            elif kind == WATER and paint(ImageDraw.Draw(water_img), el, n_water + 1):
                n_water += 1
        classes[np.array(green_img) > 0] = GREEN
        relief[classes == GREEN] = GREEN_MM
        wlab = np.array(water_img)
        wids = np.unique(wlab)
        wids = wids[wids > 0]
        if len(wids):
            # water is flat: level each lake at the average height of its bed
            level = np.zeros(n_water + 1)
            level[wids] = ndimage.mean(ground, wlab, wids)
            ground = np.where(wlab > 0, level[wlab], ground)
            classes[wlab > 0] = WATER
            relief[wlab > 0] = WATER_MM
        # roads: draw 1-pixel centre lines, then widen each road type by
        # measuring distance to the line (gives round corners and exact widths)
        for width_m in sorted(set(ROAD_WIDTH_M.values())):
            img = Image.new("L", (N + 1, N + 1), 0)
            d = ImageDraw.Draw(img)
            for el in layers:
                t = el.get("tags", {})
                kind = t.get("highway", "").removesuffix("_link")
                if (el["type"] != "way" or ROAD_WIDTH_M.get(kind) != width_m
                        or t.get("tunnel") == "yes" or t.get("area") == "yes"
                        or t.get("service") in ("parking_aisle", "driveway")):
                    continue
                d.line(ring(el["geometry"]), fill=1)
            half = max(width_m / dx, MIN_ROAD_MM / GRID_MM) / 2     # in cells
            road = ndimage.distance_transform_edt(np.array(img) == 0) <= half
            classes[road] = ROAD
            relief[road] = ROAD_MM

    heights = [0.0]
    elements = fetch_buildings(bbox)
    # relations first: their courtyard holes must not erase stand-alone buildings
    for el in sorted(elements, key=lambda e: e["type"] != "relation"):
        if paint(draw, el, len(heights)):
            heights.append(building_height_m(el.get("tags", {})))

    lab = np.array(labels)
    classes[lab > 0] = BUILDING
    relief[lab > 0] = 0.0                        # buildings already stand proud
    ids = np.unique(lab)
    ids = ids[ids > 0]
    # flat roof = average ground under the footprint + building height
    base = ndimage.mean(ground, lab, ids)
    roof = np.zeros(len(heights))
    roof[ids] = base + np.array(heights)[ids]
    elev = np.where(lab > 0, np.maximum(ground, roof[lab]), ground)
    return elev, ground, len(ids), relief, classes


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
    elev, ground, n_buildings, relief, classes = height_grid(bbox)
    floor_m = ground.min()
    # the cavity is deep enough that sunken water still leaves room for light
    top = CAVITY_MM + SKIN_MM + (elev - floor_m) * Z_EXAG * 1000.0 / SCALE + relief
    bottom = underside(top)

    if len(sys.argv) == 5 and sys.argv[1] == "--sample":
        write_sample(top, bottom, *(float(a) for a in sys.argv[2:5]))
        return

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
    np.save(CACHE / "classes.npy", classes)

    print(f"scale 1:{SCALE:.0f}, {n_buildings} buildings, "
          f"ground {ground.min():.0f}-{ground.max():.0f} m")
    for name, nf, zmax, vol in report:
        # 1.24 g/cm3 is PLA density
        print(f"{name}: {nf} triangles, {zmax:.1f} mm tall, {vol:.0f} cm3 = {vol * 1.24:.0f} g PLA")


def write_sample(top, bottom, x_mm, y_mm, size_mm):
    """Cut a small square out of the full model as a quick test print."""
    n = round(size_mm / GRID_MM)
    c0 = round(x_mm / GRID_MM)
    r0 = round((MODEL_MM - y_mm - size_mm) / GRID_MM)   # row 0 is north
    sl = np.s_[r0: r0 + n + 1, c0: c0 + n + 1]
    # drop the piece so its lowest ground sits just above the LED gap;
    # otherwise the sample prints millimetres of wall it does not need
    drop = top[sl].min() - (CAVITY_MM + SKIN_MM)
    t = top[sl] - drop
    w = wall_mask(n)
    b = np.where(w[:, None] | w[None, :], 0.0, bottom[sl] - drop)
    verts, faces = solid_between(t, b)
    assert check_watertight(faces), "sample is not watertight"
    assert (t - b).min() > 0.5, "skin collapsed somewhere"
    write_stl(OUT / "sample.stl", verts, faces)
    vol = volume_cm3(verts, faces)
    print(f"sample.stl: {size_mm:.0f} mm square, {t.max():.1f} mm tall, "
          f"{vol:.1f} cm3 = {vol * 1.24:.0f} g PLA")


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
