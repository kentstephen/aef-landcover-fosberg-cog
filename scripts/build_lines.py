"""Build the transmission-line asset the browser draws and lights by FFWI.

Source: HIFLD Electric Power Transmission Lines (ArcGIS FeatureServer,
CORS *, 52k polylines, all >= 100 kV). Probed 2026-08-22; see CLAUDE.md.

Steps:
  1. pull every page (2000 features each) as GeoJSON into data/hifld/ (cached;
     delete the folder to refetch)
  2. project to the HRRR Lambert conformal conic (metres), Douglas-Peucker
     simplify (lines run straight between towers, so ~100 m keeps the shape)
  3. split every line into pieces no longer than --piece-km so each piece
     sits in about one 3 km HRRR pixel, and tag the piece with the pixel
     under its midpoint (twin of `lonLatToHrrrPixel` in web/src/hrrr/metadata.ts)
  4. write one Arrow IPC file, web/public/lines.arrow, one record batch,
     read zero-copy in the browser with apache-arrow and drawn by
     @geoarrow/deck.gl-layers GeoArrowPathLayer:
       geometry  geoarrow.linestring, list<fixed_size_list<float32, 2>> lon/lat
       px, py    int16   HRRR column / row under the piece's length-midpoint
       vc        uint8   VOLT_CLASS code
       km        float32 piece length
     No attribute table (MVP); owner / voltage can come later.

Run:
  uv run --with pyarrow --with shapely --with pyproj --with numpy --with requests \
      python scripts/build_lines.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import requests
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import transform as shp_transform

SERVICE = (
    "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/"
    "Electric_Power_Transmission_Lines/FeatureServer/0"
)
PAGE = 2000
OUT_FIELDS = "ID,TYPE,STATUS,OWNER,VOLTAGE,VOLT_CLASS,SOURCE"

# Twin of HRRR_PROJ4 / HRRR_GRID in web/src/hrrr/metadata.ts.
HRRR_PROJ4 = (
    "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 "
    "+x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
)
X0 = -2699020.142521929
Y0 = 1588193.847443335
DX = 3000.0
DY = -3000.0
WIDTH = 1799
HEIGHT = 1059

# VOLT_CLASS strings as served -> small code the app maps back to a label and width.
VOLT_CLASS_CODES = {
    "UNDER 100": 0,
    "100-161": 1,
    "220-287": 2,
    "345": 3,
    "500": 4,
    "735 AND ABOVE": 5,
    "DC": 6,
    "NOT AVAILABLE": 7,
}

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "hifld"
OUT = ROOT / "web" / "public"


def fetch_pages(session: requests.Session) -> list[dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    count = session.get(
        f"{SERVICE}/query",
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout=60,
    ).json()["count"]
    pages = math.ceil(count / PAGE)
    print(f"HIFLD: {count} features, {pages} pages")
    feats: list[dict] = []
    for p in range(pages):
        cache = RAW / f"page_{p:03d}.geojson"
        if cache.exists():
            gj = json.loads(cache.read_text())
        else:
            for attempt in range(5):
                try:
                    r = session.get(
                        f"{SERVICE}/query",
                        params={
                            "where": "1=1",
                            "outFields": OUT_FIELDS,
                            "outSR": 4326,
                            "geometryPrecision": 5,
                            "resultOffset": p * PAGE,
                            "resultRecordCount": PAGE,
                            "orderByFields": "OBJECTID",
                            "f": "geojson",
                        },
                        timeout=180,
                    )
                    r.raise_for_status()
                    gj = r.json()
                    if "error" in gj:
                        raise RuntimeError(gj["error"])
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  page {p}: {e}; retry", file=sys.stderr)
                    time.sleep(2 * (attempt + 1))
            else:
                raise SystemExit(f"page {p} failed")
            cache.write_text(json.dumps(gj, separators=(",", ":")))
            print(f"  page {p + 1}/{pages}: {len(gj['features'])} features, {cache.stat().st_size / 1e6:.1f} MB")
        feats.extend(gj["features"])
    return feats


def pixel_of(x: float, y: float) -> tuple[int, int]:
    col = math.floor((x - X0) / DX)
    row = math.floor((y - Y0) / DY)
    if col < 0 or col >= WIDTH or row < 0 or row >= HEIGHT:
        return -1, -1
    return col, row


def split_line(coords: np.ndarray, piece_m: float) -> list[np.ndarray]:
    """Cut a projected polyline into pieces of at most piece_m, equal length,
    keeping every original vertex. Returns a list of (k, 2) arrays."""
    seg = np.diff(coords, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    total = float(seg_len.sum())
    if total <= 0:
        return []
    n_pieces = max(1, math.ceil(total / piece_m))
    target = total / n_pieces
    pieces: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = [tuple(coords[0])]
    acc = 0.0
    for i in range(len(seg_len)):
        a = coords[i]
        b = coords[i + 1]
        L = float(seg_len[i])
        if L == 0:
            continue
        pos = 0.0
        while acc + (L - pos) >= target - 1e-9 and len(pieces) < n_pieces - 1:
            need = target - acc
            t = (pos + need) / L
            pt = (float(a[0] + (b[0] - a[0]) * t), float(a[1] + (b[1] - a[1]) * t))
            cur.append(pt)
            pieces.append(cur)
            cur = [pt]
            pos += need
            acc = 0.0
        acc += L - pos
        cur.append((float(b[0]), float(b[1])))
    if len(cur) >= 2:
        pieces.append(cur)
    return [np.asarray(p) for p in pieces]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance-m", type=float, default=100.0, help="Douglas-Peucker tolerance in metres")
    ap.add_argument("--piece-km", type=float, default=3.0, help="max piece length")
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "fosberg-hrrr build_lines (personal project)"
    feats = fetch_pages(session)

    fwd = Transformer.from_crs("EPSG:4326", HRRR_PROJ4, always_xy=True)
    inv = Transformer.from_crs(HRRR_PROJ4, "EPSG:4326", always_xy=True)

    piece_px: list[int] = []
    piece_py: list[int] = []
    piece_vc: list[int] = []
    piece_km: list[float] = []
    piece_xy: list[np.ndarray] = []
    class_km = np.zeros(8)

    n_vertices_in = 0
    n_vertices_out = 0
    skipped = 0
    for f in feats:
        geom = f.get("geometry")
        if not geom:
            skipped += 1
            continue
        g = shape(geom)
        if isinstance(g, LineString):
            parts = [g]
        elif isinstance(g, MultiLineString):
            parts = list(g.geoms)
        else:
            skipped += 1
            continue
        vc_name = str(f["properties"].get("VOLT_CLASS") or "NOT AVAILABLE").strip().upper()
        vc = VOLT_CLASS_CODES.get(vc_name, 7)
        for part in parts:
            n_vertices_in += len(part.coords)
            pm = shp_transform(fwd.transform, part).simplify(args.tolerance_m, preserve_topology=False)
            coords = np.asarray(pm.coords, dtype=np.float64)
            if len(coords) < 2:
                continue
            for pc in split_line(coords, args.piece_km * 1000):
                seg = np.diff(pc, axis=0)
                seg_len = np.hypot(seg[:, 0], seg[:, 1])
                km = float(seg_len.sum()) / 1000
                if km <= 0:
                    continue
                # midpoint by length, then the HRRR pixel under it
                half = km * 500
                acc = 0.0
                mid = pc[0]
                for i in range(len(pc) - 1):
                    L = float(seg_len[i])
                    if acc + L >= half:
                        t = (half - acc) / L if L > 0 else 0
                        mid = pc[i] + (pc[i + 1] - pc[i]) * t
                        break
                    acc += L
                px, py = pixel_of(float(mid[0]), float(mid[1]))
                if px < 0:
                    continue
                lon, lat = inv.transform(pc[:, 0], pc[:, 1])
                xy = np.empty(len(pc) * 2, dtype=np.float32)
                xy[0::2] = lon
                xy[1::2] = lat
                n_vertices_out += len(pc)
                piece_px.append(px)
                piece_py.append(py)
                piece_vc.append(vc)
                piece_km.append(km)
                piece_xy.append(xy)
                class_km[vc] += km

    n = len(piece_xy)
    offsets = np.zeros(n + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(a) // 2 for a in piece_xy])
    flat = np.concatenate(piece_xy) if piece_xy else np.zeros(0, np.float32)
    coords = pa.FixedSizeListArray.from_arrays(pa.array(flat, pa.float32()), 2)
    geom = pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), coords)
    schema = pa.schema(
        [
            pa.field("geometry", geom.type, metadata={"ARROW:extension:name": "geoarrow.linestring"}),
            pa.field("px", pa.int16()),
            pa.field("py", pa.int16()),
            pa.field("vc", pa.uint8()),
            pa.field("km", pa.float32()),
        ]
    )
    table = pa.Table.from_arrays(
        [geom, pa.array(piece_px, pa.int16()), pa.array(piece_py, pa.int16()), pa.array(piece_vc, pa.uint8()), pa.array(piece_km, pa.float32())],
        schema=schema,
    ).combine_chunks()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "lines.arrow"
    with pa.OSFile(str(out), "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
        writer.write_table(table, max_chunksize=n)

    total_km = float(class_km.sum())
    print(
        f"features: {len(feats)}, skipped {skipped}; {total_km:,.0f} km ({total_km * 0.621371:,.0f} mi)\n"
        f"pieces: {n}; vertices {n_vertices_in} -> {n_vertices_out}\n"
        f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)"
    )
    for name, code in VOLT_CLASS_CODES.items():
        if class_km[code]:
            print(f"  {code} {name:14s} {class_km[code] * 0.621371:9,.0f} mi")


if __name__ == "__main__":
    main()
