# /// script
# requires-python = ">=3.12"
# dependencies = ["rasterio", "numpy", "pandas", "pyarrow", "geopandas", "shapely"]
# ///
"""Land area per state and NLCD class, from the Annual NLCD 2024 mosaic (source.coop, 30 m Albers)
read at its 480 m overview (factor 16: 10,000 x 6,560 px, 66 MB), states from the Census 20 m
cartographic boundaries rasterized onto the same grid. Each overview pixel is a nearest-neighbor
pick, so class shares are unbiased but a class's area has sampling noise of about 1 / sqrt(count).
Writes data/ignition/land_area.parquet: state, nlcd, km2.

    uv run ignition/land_area.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.env import Env
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parent.parent
STATES = ROOT / "data" / "ignition" / "states" / "cb_2023_us_state_20m.shp"
OUT = ROOT / "data" / "ignition" / "land_area.parquet"
URL = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic/Annual_NLCD_LndCov_2024_CU_C1V1.tif"
FACTOR = 16


def main():
    with Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(URL) as src:
            h, w = src.height // FACTOR, src.width // FACTOR
            arr = src.read(1, out_shape=(h, w))
            tr = src.transform * src.transform.scale(src.width / w, src.height / h)
            crs = src.crs
    print(f"overview {arr.shape}, pixel {tr.a:.0f} m, nodata {(arr == 250).mean():.1%}")
    st = gpd.read_file(STATES).to_crs(crs)
    st = st.reset_index(drop=True)
    sid = rasterize(((geom, i + 1) for i, geom in enumerate(st.geometry)), out_shape=arr.shape, transform=tr, fill=0, dtype="int32")
    px_km2 = abs(tr.a * tr.e) / 1e6
    ok = (arr != 250) & (sid > 0)
    key = sid[ok].astype(np.int64) * 1000 + arr[ok].astype(np.int64)
    u, n = np.unique(key, return_counts=True)
    df = pd.DataFrame({"state": ["US-" + st.STUSPS[int(k // 1000) - 1] for k in u], "nlcd": (u % 1000).astype(int), "km2": n * px_km2})
    df.to_parquet(OUT, index=False)
    tot = df.groupby("nlcd").km2.sum()
    print(f"land (not nodata, in a state) {df.km2.sum():,.0f} km2; open water {tot.get(11, 0):,.0f}")
    print((tot / tot.sum()).round(3).to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
