# /// script
# requires-python = ">=3.12"
# dependencies = ["rasterio", "numpy", "pandas", "pyarrow", "pyproj"]
# ///
"""The land cover baseline at every point of cases.parquet: Annual NLCD of the year before the fire
(the source.coop mirror, 30 m Albers COG), the pixel class and the class fractions of the 5 x 5
window (150 m) around it, so the baseline sees the same neighborhood the embedding mean does. One
COG open per year, windows read point by point (a 512 px block each, Byte). Peak RAM tiny.

    uv run ignition/sample_nlcd.py
"""
import sys
import time
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
URL = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic/Annual_NLCD_LndCov_{year}_CU_C1V1.tif"
CLASSES = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
HALF = 2


def main():
    df = pd.read_parquet(CASES)
    pts = df[df.kind.isin(["case", "s_ctrl"])]
    out = []
    t0 = time.time()
    with Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIPLEX="YES", CPL_VSIL_CURL_CACHE_SIZE=200_000_000):
        for year, g in pts.groupby("emb_year"):
            with rasterio.open(URL.format(year=int(year))) as src:
                tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                xs, ys = tr.transform(g.lon.values, g.lat.values)
                rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
                order = np.lexsort((np.asarray(cols) // 512, np.asarray(rows) // 512))
                for k in order:
                    r, c = int(rows[k]), int(cols[k])
                    w = src.read(1, window=Window(c - HALF, r - HALF, 2 * HALF + 1, 2 * HALF + 1), boundless=True, fill_value=250)
                    pix = int(w[HALF, HALF])
                    vals = w[w != 250]
                    frac = {f"nlcd_f{cl}": float((vals == cl).mean()) if vals.size else np.nan for cl in CLASSES}
                    out.append(dict(row=int(g.index[k]), nlcd=pix, **frac))
                print(f"{int(year)}: {len(g)} points, {time.time() - t0:.0f} s so far", flush=True)
    res = pd.DataFrame(out).set_index("row").sort_index()
    res.to_parquet(OUT)
    print(f"wrote {OUT} {len(res)}; nodata {int((res.nlcd == 250).sum())}")
    print(res.nlcd.value_counts().head(12).to_string())


if __name__ == "__main__":
    sys.exit(main())
