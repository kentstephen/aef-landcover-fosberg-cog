# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk", "xarray", "zarr>=3", "numpy", "pyproj"]
# ///
"""Probe the dynamical.org HRRR analysis store (hourly, 2014 to present) on AWS Open Data: open it
read-only, print the variables and time span, and pull the Fosberg inputs at one point for one day
(the Marshall Fire, Boulder County, 2021-12-30), computing FFWI hour by hour. One point, one day,
a few chunks: well under 1 GB of RAM.

    uv run ignition/probe_hrrr_analysis.py
"""
import sys
import time

import icechunk
import numpy as np
import xarray as xr
from pyproj import Transformer

BUCKET = "dynamical-noaa-hrrr"
PREFIX = "noaa-hrrr-analysis/v0.2.0.icechunk"
LCC = "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"


def ffwi(t_c, rh, wind_ms):
    t_f = t_c * 1.8 + 32
    h = np.clip(rh, 0, 100)
    m = np.where(
        h < 10,
        0.03229 + 0.281073 * h - 0.000578 * h * t_f,
        np.where(h < 50, 2.22749 + 0.160107 * h - 0.014784 * t_f, 21.0606 + 0.005565 * h * h - 0.00035 * h * t_f - 0.483199 * h),
    )
    mr = np.clip(m, 0, 30) / 30
    eta = 1 - 2 * mr + 1.5 * mr**2 - 0.5 * mr**3
    w = wind_ms * 2.2369363
    return eta * np.sqrt(1 + w * w) / 0.3002


def main():
    t0 = time.time()
    storage = icechunk.s3_storage(bucket=BUCKET, prefix=PREFIX, region="us-west-2", anonymous=True)
    repo = icechunk.Repository.open(storage)
    ds = xr.open_dataset(repo.readonly_session("main").store, engine="zarr", consolidated=False)
    print(f"opened in {time.time() - t0:.1f} s")
    print("dims:", dict(ds.sizes))
    print("vars:", list(ds.data_vars))
    print("coords:", list(ds.coords))
    print("time:", str(ds.time.values[0]), "to", str(ds.time.values[-1]), "step", str(ds.time.values[1] - ds.time.values[0]))
    for v in ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m", "wind_gust_surface"]:
        if v in ds:
            print(f"  {v}: chunks {ds[v].encoding.get('chunks')} shards {ds[v].encoding.get('shards')} dtype {ds[v].dtype}")

    # the point: Marshall Fire origin area, Boulder County; the day: 2021-12-30
    lon, lat = -105.23, 39.955
    x, y = Transformer.from_crs("EPSG:4326", LCC, always_xy=True).transform(lon, lat)
    sub = ds.sel(x=x, y=y, method="nearest").sel(time=slice("2021-12-30T00:00", "2021-12-31T00:00"))
    print("picked x, y:", float(sub.x), float(sub.y), "sizes", dict(sub.sizes))
    t1 = time.time()
    t = sub.temperature_2m.values
    rh = sub.relative_humidity_2m.values
    u = sub.wind_u_10m.values
    v = sub.wind_v_10m.values
    g = sub.wind_gust_surface.values
    print(f"read 5 vars x {t.size} hours in {time.time() - t1:.1f} s")
    w = np.hypot(u, v)
    f = ffwi(t, rh, w)
    fg = ffwi(t, rh, g)
    print("hour (UTC)         T degC   RH %   wind mph  gust mph   FFWI  FFWI(gust)")
    for i, ts in enumerate(sub.time.values):
        print(f"{str(ts)[:16]}  {t[i]:6.1f}  {rh[i]:5.0f}  {w[i]*2.237:8.1f}  {g[i]*2.237:8.1f}  {f[i]:6.1f}  {fg[i]:6.1f}")
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    sys.exit(main())
