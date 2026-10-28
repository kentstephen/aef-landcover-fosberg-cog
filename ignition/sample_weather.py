# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk", "xarray", "zarr>=3", "numpy", "pandas", "pyarrow", "pyproj"]
# ///
"""The hour's weather at every row of cases.parquet, from the HRRR analysis (dynamical.org Icechunk on
AWS Open Data): T, RH, 10 m wind, gust, precipitation, and FFWI (sustained and gust), plus the day
before (max FFWI over the prior 24 h) and the 72 h precipitation. One weather chunk group at a time
(a 135 km square x 90 days, six variables, 105 MB float32); groups are written as they finish to
data/ignition/weather_parts/<group>.parquet, so a rerun picks up where it stopped. Peak RAM under
1 GB. About 5 s a group over the network.

    uv run ignition/sample_weather.py [--limit N]
"""
import argparse
import sys
import time
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_parquet(CASES)
    PARTS.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem) for p in PARTS.glob("*.parquet")}
    groups = [g for g in df.group.unique() if int(g) not in done]
    if a.limit: groups = groups[: a.limit]
    print(f"{len(groups)} groups to read ({len(done)} done)")
    storage = icechunk.s3_storage(bucket="dynamical-noaa-hrrr", prefix="noaa-hrrr-analysis/v0.2.0.icechunk", region="us-west-2", anonymous=True)
    ds = xr.open_dataset(icechunk.Repository.open(storage).readonly_session("main").store, engine="zarr", consolidated=False)[VARS]
    t0 = time.time()
    for n, g in enumerate(groups):
        rows = df[df.group == g]
        cx, cy, ct = int(rows.cx.iloc[0]), int(rows.cy.iloc[0]), int(rows.ct.iloc[0])
        sub = ds.isel(x=slice(cx * CHUNK_XY, (cx + 1) * CHUNK_XY), y=slice(cy * CHUNK_XY, (cy + 1) * CHUNK_XY), time=slice(ct * CHUNK_T, (ct + 1) * CHUNK_T))
        t1 = time.time()
        arr = {v: sub[v].values for v in VARS}
        nt = arr["temperature_2m"].shape[0]
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
        pd.DataFrame(out).to_parquet(PARTS / f"{int(g)}.parquet", index=False)
        el = time.time() - t0
        print(f"\r{n + 1}/{len(groups)} group {g} rows {len(rows)} read {time.time() - t1:.1f} s, {el / (n + 1):.1f} s/group, eta {el / (n + 1) * (len(groups) - n - 1) / 60:.0f} min", end="", flush=True)
    print()
    parts = [pd.read_parquet(p) for p in PARTS.glob("*.parquet")]
    allw = pd.concat(parts).set_index("row").sort_index()
    allw.to_parquet(OUT)
    print(f"wrote {OUT} {len(allw)} rows; ffwi nan {int(allw.ffwi.isna().sum())}")


if __name__ == "__main__":
    sys.exit(main())
