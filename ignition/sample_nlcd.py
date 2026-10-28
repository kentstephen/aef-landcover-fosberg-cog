# /// script
# requires-python = ">=3.12"
# dependencies = ["rasterio", "numpy", "pandas", "pyarrow", "pyproj"]
# ///
"""The land cover baseline at every point of cases.parquet: Annual NLCD of the year before the fire
(the source.coop mirror, 30 m Albers COG), the pixel class and the class fractions of the 5 x 5
window (150 m) around it, so the baseline sees the same neighborhood the embedding mean does. Windows
are read point by point (a 512 px block each, Byte) by WORKERS threads, each with its own dataset
handle; one part file per year and thread slice in data/ignition/nlcd_parts/, resumable. Peak RAM tiny.
The mirror ends at 2024, so points whose embedding year is 2025 (fires of 2026) take the 2024 map, one
year stale for the baseline.

    uv run ignition/sample_nlcd.py [--tag wide] [--workers 8]
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.env import Env
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "ignition" / "cases.parquet"
OUT = ROOT / "data" / "ignition" / "nlcd.parquet"
PARTS = ROOT / "data" / "ignition" / "nlcd_parts"
URL = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic/Annual_NLCD_LndCov_{year}_CU_C1V1.tif"
CLASSES = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
HALF = 2
LAST_YEAR = 2024


def _slice(job):
    """One year and one slice of its points (sorted by 512 px block): read the windows, write the part."""
    year, g, part = job
    out = []
    with Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIPLEX="YES", CPL_VSIL_CURL_CACHE_SIZE=200_000_000):
        with rasterio.open(URL.format(year=int(year))) as src:
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = tr.transform(g.lon.values, g.lat.values)
            rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
            for k in range(len(g)):
                r, c = int(rows[k]), int(cols[k])
                w = src.read(1, window=Window(c - HALF, r - HALF, 2 * HALF + 1, 2 * HALF + 1), boundless=True, fill_value=250)
                pix = int(w[HALF, HALF])
                vals = w[w != 250]
                frac = {f"nlcd_f{cl}": float((vals == cl).mean()) if vals.size else np.nan for cl in CLASSES}
                out.append(dict(row=int(g.index[k]), nlcd=pix, **frac))
    pd.DataFrame(out).to_parquet(part, index=False)
    return part.stem, len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-part", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="parts to read (a smoke test)")
    a = ap.parse_args()
    global CASES, OUT, PARTS
    if a.tag:
        CASES = CASES.with_name(f"cases_{a.tag}.parquet"); OUT = OUT.with_name(f"nlcd_{a.tag}.parquet"); PARTS = PARTS.with_name(f"nlcd_parts_{a.tag}")
    PARTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CASES)
    pts = df[(df.kind == "case") | df.kind.str.startswith("s")]
    # one job per year and slice of PER_PART points, the points sorted by 512 px block so a slice hits few blocks
    jobs = []
    for year, g in pts.groupby(pts.emb_year.clip(upper=LAST_YEAR)):
        with Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(URL.format(year=int(year))) as src:
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = tr.transform(g.lon.values, g.lat.values)
            rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        order = np.lexsort((np.asarray(cols) // 512, np.asarray(rows) // 512))
        g = g.iloc[order]
        for j in range(0, len(g), a.per_part):
            part = PARTS / f"{int(year)}_{j // a.per_part:04d}.parquet"
            if not part.exists():
                jobs.append((int(year), g.iloc[j:j + a.per_part], part))
    if a.limit: jobs = jobs[: a.limit]
    print(f"{len(jobs)} parts to read, {len(pts)} points", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (name, npts) in enumerate(ex.map(_slice, jobs)):
            el = time.time() - t0
            print(f"\r{n + 1}/{len(jobs)} part {name} pts {npts}, {el / (n + 1):.1f} s/part, eta {el / (n + 1) * (len(jobs) - n - 1) / 60:.0f} min", end="", flush=True)
    print()
    res = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")]).set_index("row").sort_index()
    res.to_parquet(OUT)
    print(f"wrote {OUT} {len(res)}; nodata {int((res.nlcd == 250).sum())}")
    print(res.nlcd.value_counts().head(12).to_string())


if __name__ == "__main__":
    sys.exit(main())
