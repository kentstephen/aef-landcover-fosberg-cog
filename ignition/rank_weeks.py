# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk", "xarray", "zarr>=3", "numpy", "pandas"]
# ///
"""Rank the weeks of a year by the ignition head's weather term over the country, from the HRRR analysis
store (what happened, hourly). A thinned sample of the 135 km squares (every third in x and y, about 110 of
960), the whole window read square by square with a small thread pool, the term computed in numpy per hour
(the same fold as the browser: coef / scale on ffwi, t_c, rh, wind_ms minus their means), reduced per hour
to the fraction of sampled pixels above the gate (+1.0 logit), the mean term and the max. Then the top days
and the top 7-day windows by area-hours above the gate and by peak extent.

RAM: one square-chunk is 4 vars x 2160 x 45 x 45 float32 = 70 MB; 4 in flight plus the derived arrays is
about 600 MB; say under 1 GB. No output but a small CSV per hour and the printed tables.

    uv run ignition/rank_weeks.py --year 2026 --start 02-01 --end 10-27 --step 3 --workers 4
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "dynamical-noaa-hrrr"
PREFIX = "noaa-hrrr-analysis/v0.2.0.icechunk"
CHUNK_T, CHUNK_XY = 2160, 45
VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m"]
GATE = 1.0


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


def weather_fold():
    m = json.loads((ROOT / "models" / "ignition_logit.json").read_text())
    nb = len(m["mean"]) - len(m["weather"])
    wc = {name: m["coef"][0][nb + j] / m["scale"][nb + j] for j, name in enumerate(m["weather"])}
    wm = {name: m["mean"][nb + j] for j, name in enumerate(m["weather"])}
    return wc, wm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--start", default="02-01")
    ap.add_argument("--end", default="10-27")
    ap.add_argument("--step", type=int, default=3, help="every Nth square in x and y")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / ".claude" / "memory" / "rank_weeks.csv"))
    a = ap.parse_args()
    t0 = time.time()
    storage = icechunk.s3_storage(bucket=BUCKET, prefix=PREFIX, region="us-west-2", anonymous=True)
    ds = xr.open_dataset(icechunk.Repository.open(storage).readonly_session("main").store, engine="zarr", consolidated=False)[VARS]
    times = pd.DatetimeIndex(ds.time.values)
    lo = times.searchsorted(pd.Timestamp(f"{a.year}-{a.start}"))
    hi = min(times.searchsorted(pd.Timestamp(f"{a.year}-{a.end}") + pd.Timedelta(days=1)), len(times))
    ct0, ct1 = lo // CHUNK_T, (hi - 1) // CHUNK_T
    nx, ny = ds.sizes["x"], ds.sizes["y"]
    squares = [(cx, cy) for cy in range(0, (ny + CHUNK_XY - 1) // CHUNK_XY, a.step) for cx in range(0, (nx + CHUNK_XY - 1) // CHUNK_XY, a.step)]
    jobs = [(cx, cy, ct) for ct in range(ct0, ct1 + 1) for (cx, cy) in squares]
    wc, wm = weather_fold()
    n_hours = hi - lo
    above = np.zeros(n_hours)
    total = np.zeros(n_hours)
    tsum = np.zeros(n_hours)
    tmax = np.full(n_hours, -np.inf)
    print(f"{len(squares)} squares x {ct1 - ct0 + 1} time chunks = {len(jobs)} reads; hours {times[lo]} to {times[hi - 1]} ({n_hours} h)", flush=True)

    def read(job):
        cx, cy, ct = job
        t_lo, t_hi = max(ct * CHUNK_T, lo), min((ct + 1) * CHUNK_T, hi)
        sub = ds.isel(x=slice(cx * CHUNK_XY, (cx + 1) * CHUNK_XY), y=slice(cy * CHUNK_XY, (cy + 1) * CHUNK_XY), time=slice(t_lo, t_hi))
        arr = {v: sub[v].values for v in VARS}
        t = arr["temperature_2m"]
        rh = arr["relative_humidity_2m"]
        w = np.hypot(arr["wind_u_10m"], arr["wind_v_10m"])
        f = ffwi(t, rh, w)
        term = wc["ffwi"] * (f - wm["ffwi"]) + wc["t_c"] * (t - wm["t_c"]) + wc["rh"] * (rh - wm["rh"]) + wc["wind_ms"] * (w - wm["wind_ms"])
        ok = np.isfinite(term)
        term = np.where(ok, term, -np.inf)
        return (
            t_lo - lo,
            (term > GATE).sum(axis=(1, 2)),
            ok.sum(axis=(1, 2)),
            np.where(ok, term, 0).sum(axis=(1, 2)),
            term.max(axis=(1, 2)),
        )

    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(read, j) for j in jobs]
        for fut in as_completed(futs):
            o, ab, tot, ts, tm = fut.result()
            n = len(ab)
            above[o : o + n] += ab
            total[o : o + n] += tot
            tsum[o : o + n] += ts
            tmax[o : o + n] = np.maximum(tmax[o : o + n], tm)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} reads, {time.time() - t0:.0f} s", flush=True)

    df = pd.DataFrame({"time": times[lo:hi], "frac_above_gate": above / np.maximum(total, 1), "mean_term": tsum / np.maximum(total, 1), "max_term": tmax})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    df["day"] = df.time.dt.floor("D")
    daily = df.groupby("day").agg(peak_frac=("frac_above_gate", "max"), area_hours=("frac_above_gate", "sum"), mean_term=("mean_term", "mean"))
    print("\nTop 12 days by peak extent above the gate (fraction of sampled pixels at the worst hour):")
    print(daily.sort_values("peak_frac", ascending=False).head(12).round(3).to_string())
    print("\nTop 12 days by area-hours above the gate (fraction summed over the day's hours):")
    print(daily.sort_values("area_hours", ascending=False).head(12).round(3).to_string())
    wk = daily.rolling(7).sum().dropna()
    wk.index = wk.index - pd.Timedelta(days=6)
    wk = wk.rename(columns={"area_hours": "week_area_hours", "peak_frac": "sum_of_daily_peaks"})
    print("\nTop 10 seven-day windows (start day) by area-hours above the gate:")
    print(wk.sort_values("week_area_hours", ascending=False).head(10).round(3).to_string())
    print("\nTop 10 seven-day windows (start day) by the sum of daily peak extents:")
    print(wk.sort_values("sum_of_daily_peaks", ascending=False).head(10).round(3).to_string())
    print(f"\ndone in {time.time() - t0:.0f} s; per-hour table at {a.out}")


if __name__ == "__main__":
    sys.exit(main())
