# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk", "xarray", "zarr>=3", "numpy", "pandas", "pyarrow", "pyproj"]
# ///
"""The hour's weather at every row of cases.parquet, from the HRRR analysis (dynamical.org Icechunk on
AWS Open Data): T, RH, 10 m wind, gust, precipitation, and FFWI (sustained and gust), plus the day
before (max FFWI over the prior 24 h) and the 72 h precipitation. One weather chunk group at a time
(a 135 km square x 90 days, six variables, 105 MB float32); groups are written as they finish to
data/ignition/weather_parts/<group>.parquet, so a rerun picks up where it stopped. Groups are read
by WORKERS processes at once (each opens its own store session; the bucket takes parallel reads),
about 105 MB of float32 per group in flight, so 8 workers peak around 2 GB. One group alone is
about 4.4 s over the network.

    uv run ignition/sample_weather.py [--limit N] [--tag wide] [--workers 8]   # --tag reads cases_<tag>.parquet, writes weather_<tag>.parquet
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "ignition" / "cases.parquet"
PARTS = ROOT / "data" / "ignition" / "weather_parts"
OUT = ROOT / "data" / "ignition" / "weather.parquet"
CHUNK_T, CHUNK_XY = 2160, 45
VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m", "wind_gust_surface", "precipitation_surface"]


def ffwi(t_c, rh, wind_ms):
    t_f = t_c * 1.8 + 32
    h = np.clip(rh, 0, 100)
    m = np.where(h < 10, 0.03229 + 0.281073 * h - 0.000578 * h * t_f,
                 np.where(h < 50, 2.22749 + 0.160107 * h - 0.014784 * t_f, 21.0606 + 0.005565 * h * h - 0.00035 * h * t_f - 0.483199 * h))
    mr = np.clip(m, 0, 30) / 30
    eta = 1 - 2 * mr + 1.5 * mr**2 - 0.5 * mr**3
    w = wind_ms * 2.2369363
    return eta * np.sqrt(1 + w * w) / 0.3002


DS = None


def _open():
    global DS
    storage = icechunk.s3_storage(bucket="dynamical-noaa-hrrr", prefix="noaa-hrrr-analysis/v0.2.0.icechunk", region="us-west-2", anonymous=True)
    DS = xr.open_dataset(icechunk.Repository.open(storage).readonly_session("main").store, engine="zarr", consolidated=False)[VARS]


def _group(job):
    """One weather group: read the block, the hour's values at each row, write the part file."""
    g, rows, parts = job
    cx, cy, ct = int(rows.cx.iloc[0]), int(rows.cy.iloc[0]), int(rows.ct.iloc[0])
    sub = DS.isel(x=slice(cx * CHUNK_XY, (cx + 1) * CHUNK_XY), y=slice(cy * CHUNK_XY, (cy + 1) * CHUNK_XY), time=slice(ct * CHUNK_T, (ct + 1) * CHUNK_T))
    t1 = time.time()
    arr = {v: sub[v].values for v in VARS}
    li = (rows.ti.values - ct * CHUNK_T).astype(int)
    lx = (rows.hx.values - cx * CHUNK_XY).astype(int)
    ly = (rows.hy.values - cy * CHUNK_XY).astype(int)
    t = arr["temperature_2m"]; rh = arr["relative_humidity_2m"]; u = arr["wind_u_10m"]; v = arr["wind_v_10m"]; gu = arr["wind_gust_surface"]; pr = arr["precipitation_surface"]
    w = np.hypot(u, v)
    f = ffwi(t, rh, w)
    fg = ffwi(t, rh, gu)
    out = []
    for k in range(len(rows)):
        i, y, x = li[k], ly[k], lx[k]
        lo = max(0, i - 24)
        lo72 = max(0, i - 72)
        out.append(dict(
            row=int(rows.index[k]), t_c=float(t[i, y, x]), rh=float(rh[i, y, x]), wind_ms=float(w[i, y, x]), gust_ms=float(gu[i, y, x]),
            ffwi=float(f[i, y, x]), ffwi_gust=float(fg[i, y, x]),
            ffwi_max24=float(np.nanmax(f[lo:i + 1, y, x])) if i > lo else float(f[i, y, x]),
            precip72_mm=float(np.nansum(pr[lo72:i + 1, y, x])),
            hours_in_block=int(i),
        ))
    pd.DataFrame(out).to_parquet(parts / f"{g}.parquet", index=False)
    return g, len(rows), time.time() - t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    global CASES, PARTS, OUT
    if a.tag:
        CASES = CASES.with_name(f"cases_{a.tag}.parquet"); PARTS = PARTS.with_name(f"weather_parts_{a.tag}"); OUT = OUT.with_name(f"weather_{a.tag}.parquet")
    df = pd.read_parquet(CASES)
    PARTS.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem) for p in PARTS.glob("*.parquet")}
    groups = [g for g in df.group.unique() if int(g) not in done]
    if a.limit: groups = groups[: a.limit]
    print(f"{len(groups)} groups to read ({len(done)} done)")
    t0 = time.time()
    jobs = [(int(g), df[df.group == g], PARTS) for g in groups]
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_open) as ex:
        for n, (g, nrows, dt) in enumerate(ex.map(_group, jobs)):
            el = time.time() - t0
            print(f"\r{n + 1}/{len(groups)} group {g} rows {nrows} read {dt:.1f} s, {el / (n + 1):.2f} s/group, eta {el / (n + 1) * (len(groups) - n - 1) / 60:.0f} min", end="", flush=True)
    print()
    parts = [pd.read_parquet(p) for p in PARTS.glob("*.parquet")]
    allw = pd.concat(parts).set_index("row").sort_index()
    allw.to_parquet(OUT)
    print(f"wrote {OUT} {len(allw)} rows; ffwi nan {int(allw.ffwi.isna().sum())}")


if __name__ == "__main__":
    sys.exit(main())
