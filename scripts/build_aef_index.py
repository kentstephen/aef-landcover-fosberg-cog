"""The browser's AEF index: every 2025 COG that touches the HRRR grid, its bounds, and the byte ranges of its
coarse overview levels, so the map reads one range per file and level with no header round trip.

Writes web/public/aef-2025-conus.json:
  {"year", "base", "levels": [1024, 512, 256, 128, 64, 32, 16]  (level width in px; factor = 8192 / width),
   "block": 1024, "zones": {"13N": {"epsg": 32613, "n": <files>}, ...},
   "tiles": [[rel_path, west, south, east, north, zone, row_in_zone, utm_west, utm_south], ...]}
and per zone web/public/aef-2025-ranges-<zone>.bin, little endian, one record per file in zone row order:
  float64 start[7]            byte offset of the level's span (the 64 blocks are contiguous, not in band order)
  uint32 rel[7][64], count[7][64]   each band's block: offset relative to start, and its length (0, 0 = a
                                    sparse block, all nodata)
Record = 7 * 8 + 7 * 64 * 8 = 3640 bytes.

Header facts checked per file (block 1024 px, 64 samples, planar separate, zstd, no predictor); a file
that differs is left out with a warning. Resumable: finished files are kept in a cache under data/.
"""
import json
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import fsspec
import tifffile
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
INDEX = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
BASE = f"https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/{YEAR}/"
# The files kept are the ones that touch the HRRR grid (Lambert conformal, 1799 x 1059 px of 3 km):
# the ground where there is weather, into Canada and Mexico as far as the forecast goes. The box is
# only a first cut for the query; the grid's rectangle does the real test.
HRRR_PROJ = "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
HRRR_X0, HRRR_Y0, HRRR_DX, HRRR_W, HRRR_H = -2699020.142521929, 1588193.847443335, 3000, 1799, 1059
HRRR_BOX = (-135.0, 20.5, -60.0, 53.5)
CONUS = HRRR_BOX
LEVELS = [1024, 512, 256, 128, 64, 32, 16]
BLOCK = 1024
OUT_JSON = ROOT / "web" / "public" / f"aef-{YEAR}-conus.json"
OUT_DIR = ROOT / "web" / "public"
CACHE = ROOT / "data" / "aef_ranges_cache.jsonl"
WORKERS = 8


def read_ranges(rel):
    """Per level: (start, [(rel_offset, count) x 64]); None with a reason when the file is not as expected."""
    url = BASE + rel
    for attempt in range(4):
        try:
            with fsspec.open(url, "rb", block_size=256 * 1024) as f:
                tf = tifffile.TiffFile(f)
                by_width = {p.imagewidth: p for p in tf.pages}
                out = []
                for w in LEVELS:
                    p = by_width.get(w)
                    if p is None or p.tilewidth != BLOCK or p.samplesperpixel != 64 or p.planarconfig != 2 or p.compression != 50000 or p.predictor != 1:
                        return None, f"level {w} not as expected"
                    offs, cnts = p.dataoffsets, p.databytecounts
                    if len(offs) != 64:
                        return None, f"level {w} has {len(offs)} blocks"
                    # a block of all nodata is sparse (offset 0, count 0) and is left out of the span
                    real = [(o, c) for o, c in zip(offs, cnts) if c > 0]
                    if not real:
                        out.append((0, [(0, 0)] * 64))
                        continue
                    start = min(o for o, _ in real)
                    span = max(o + c for o, c in real) - start
                    total = sum(c for _, c in real)
                    if span > 1.5 * total + 65536:
                        return None, f"level {w} blocks spread over {span} bytes for {total} of data"
                    out.append((start, [(o - start if c > 0 else 0, c) for o, c in zip(offs, cnts)]))
                return out, None
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return None, f"read failed: {e}"
            time.sleep(2 * (attempt + 1))


def main():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    rows = con.execute(
        f"""
        select path, wgs84_west, wgs84_south, wgs84_east, wgs84_north, utm_zone, crs, utm_west, utm_south
        from read_parquet('{INDEX}')
        where year = {YEAR} and wgs84_east > {CONUS[0]} and wgs84_west < {CONUS[2]}
          and wgs84_north > {CONUS[1]} and wgs84_south < {CONUS[3]}
        order by utm_zone, path
        """
    ).fetchall()
    files = []
    tr = Transformer.from_crs("EPSG:4326", CRS.from_proj4(HRRR_PROJ), always_xy=True)
    xmax, ymin = HRRR_X0 + HRRR_DX * HRRR_W, HRRR_Y0 - HRRR_DX * HRRR_H
    for path, w, s, e, n, zone, crs, uw, us in rows:
        # the file touches the grid when any point of its outline falls inside the grid's rectangle
        # (the outline is sampled; a file is 82 km, the grid 5,400 x 3,200 km, so no edge slips between)
        k = 9
        ring = [(w + (e - w) * i / k, s) for i in range(k + 1)] + [(e, s + (n - s) * i / k) for i in range(k + 1)]
        ring += [(w + (e - w) * i / k, n) for i in range(k + 1)] + [(w, s + (n - s) * i / k) for i in range(k + 1)]
        ring += [((w + e) / 2, (s + n) / 2)]
        inside = False
        for lon, lat in ring:
            x, y = tr.transform(lon, lat)
            if HRRR_X0 <= x <= xmax and ymin <= y <= HRRR_Y0:
                inside = True
                break
        if not inside:
            continue
        files.append(dict(rel=path.split(f"/annual/{YEAR}/", 1)[1], bbox=[round(w, 4), round(s, 4), round(e, 4), round(n, 4)], zone=zone, epsg=int(crs.split(":")[1]), uw=uw, us=us))
    print(f"{len(files)} files", flush=True)

    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            r = json.loads(line)
            if r["ranges"] is not None:
                done[r["rel"]] = r["ranges"]
    todo = [f["rel"] for f in files if f["rel"] not in done]
    print(f"{len(done)} cached, {len(todo)} to read", flush=True)
    t0 = time.time()
    with CACHE.open("a") as cache, ThreadPoolExecutor(WORKERS) as ex:
        for i, (rel, (ranges, why)) in enumerate(zip(todo, ex.map(read_ranges, todo))):
            if ranges is None:
                print(f"skip {rel}: {why}", flush=True)
            done[rel] = ranges
            cache.write(json.dumps({"rel": rel, "ranges": ranges}) + "\n")
            if i % 50 == 0:
                print(f"{i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)

    zones = {}
    tiles = []
    for f in files:
        ranges = done.get(f["rel"])
        if ranges is None:
            continue
        z = zones.setdefault(f["zone"], {"epsg": f["epsg"], "n": 0, "buf": bytearray()})
        ri = z["n"]
        z["n"] += 1
        z["buf"] += struct.pack("<7d", *[r[0] for r in ranges])
        for _, blocks in ranges:
            z["buf"] += struct.pack("<128I", *[v for o, c in blocks for v in (o, c)])
        tiles.append([f["rel"], *f["bbox"], f["zone"], ri, f["uw"], f["us"]])
    for zone, z in zones.items():
        (OUT_DIR / f"aef-{YEAR}-ranges-{zone}.bin").write_bytes(bytes(z["buf"]))
    OUT_JSON.write_text(json.dumps({"year": YEAR, "base": BASE, "levels": LEVELS, "block": BLOCK, "zones": {k: {"epsg": v["epsg"], "n": v["n"]} for k, v in zones.items()}, "tiles": tiles}, separators=(",", ":")))
    print(f"{len(tiles)} tiles -> {OUT_JSON} ({OUT_JSON.stat().st_size / 1024:.0f} KB); zones {', '.join(f'{k} {v['n']}' for k, v in zones.items())}")


if __name__ == "__main__":
    sys.exit(main())
