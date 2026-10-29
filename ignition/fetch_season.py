# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk", "xarray", "zarr>=3", "numpy", "zstandard", "xarray-sql", "pandas", "pyarrow"]
# ///
"""A fire season as one frame per day: at every HRRR pixel, the day's worst hour of the weather part of
the ignition logit (models/ignition_logit.json: FFWI, temperature, humidity and wind through the head),
from the HRRR ANALYSIS (dynamical.org Icechunk on AWS Open Data, hourly, 3 km). A day runs 06Z to 06Z
(about midnight to midnight Mountain time), so an afternoon's peak falls inside its own day.

The analysis is chunked for time series (2160 hours x 45 x 45 pixels), so the pass goes square by square:
each worker reads one 135 km square for the whole window (four variables, about 210 MB of float32 for 270
days), folds it to days with one SQL query through xarray-sql (DataFusion: the square as a table of
(time, y, x) rows, the Fosberg index and the weather term as expressions, GROUP BY day, y, x), and writes a
small part file; parts are skipped on a rerun. The same reduction in numpy (reshape to (days, 24, y, x), max
over the hour axis) runs in the same time on a square and agrees to 1e-6; the query is kept because it says
what it does. Then `--assemble`
stacks the parts onto the grid and writes what the browser reads: one zstd-compressed byte plane per HRRR
shard (the 24 shards of 265 x 300 pixels the forecast reader uses), days-major, the weather term quantized
over TERM_LO..TERM_HI with 255 = missing, plus index.json. The FFWI and the hour of the worst hour are kept
in data/season/ for later, not shipped.

Peak RAM about 370 MB per worker (8 workers: about 3 GB); assembly about 1 GB.

    uv run ignition/fetch_season.py --smoke                # one square, timings, no files
    uv run ignition/fetch_season.py [--workers 8] [--limit N]
    uv run ignition/fetch_season.py --assemble
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HEAD = ROOT / "models" / "ignition_logit.json"
YEAR = 2026
START = np.datetime64(f"{YEAR}-02-01T06:00")
END = np.datetime64(f"{YEAR}-11-01T06:00")  # exclusive; cut to what the store has
DAY_START_UTC = 6
OUT_DIR = ROOT / "data" / "season" / str(YEAR)
PARTS = OUT_DIR / "parts"
WEB = ROOT / "web" / "public" / f"season-{YEAR}"
CHUNK_XY = 45
NY, NX = 1059, 1799
SHARD_H, SHARD_W = 265, 300
TERM_LO, TERM_HI = -6.0, 6.0
VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m"]


def weather_head():
    """The weather part of the ignition head folded to a * x + b per input, in the head's order."""
    m = json.loads(HEAD.read_text())
    n = 64
    a = {}
    b = 0.0
    for j, name in enumerate(m["weather"]):
        c = m["coef"][0][n + j] / m["scale"][n + j]
        a[name] = c
        b -= c * m["mean"][n + j]
    return a, b


def term_bytes(z):
    out = np.full(z.shape, 255, np.uint8)
    ok = np.isfinite(z)
    out[ok] = np.clip(np.round((z[ok] - TERM_LO) / (TERM_HI - TERM_LO) * 254), 0, 254).astype(np.uint8)
    return out


DS = None
T0 = T1 = 0
NDAYS = 0


def _open():
    global DS, T0, T1, NDAYS
    import icechunk
    import xarray as xr
    storage = icechunk.s3_storage(bucket="dynamical-noaa-hrrr", prefix="noaa-hrrr-analysis/v0.2.0.icechunk", region="us-west-2", anonymous=True)
    DS = xr.open_dataset(icechunk.Repository.open(storage).readonly_session("main").store, engine="zarr", consolidated=False)[VARS]
    times = DS.time.values
    T0 = int(np.searchsorted(times, START))
    assert times[T0] == START, (times[T0], START)
    end = min(END, times[-1] + np.timedelta64(1, "h"))
    NDAYS = int((end - START) / np.timedelta64(1, "D"))
    T1 = T0 + NDAYS * 24


def _square(job):
    """One 135 km square for the whole window: the day's worst weather term, its FFWI and hour."""
    cy, cx = job
    a, b = weather_head()
    t1 = time.time()
    sub = DS.isel(time=slice(T0, T1), y=slice(cy * CHUNK_XY, (cy + 1) * CHUNK_XY), x=slice(cx * CHUNK_XY, (cx + 1) * CHUNK_XY))
    t = sub.temperature_2m.values
    rh = sub.relative_humidity_2m.values
    w = np.hypot(sub.wind_u_10m.values, sub.wind_v_10m.values)
    read = time.time() - t1
    worst, fw, hour = reduce_sql(sub, a, b)
    term = term_bytes(worst)
    fb = np.full(fw.shape, 255, np.uint8)
    ok = np.isfinite(fw)
    fb[ok] = np.clip(np.round(fw[ok]), 0, 254).astype(np.uint8)
    hour[~np.isfinite(worst)] = 255
    return cy, cx, term, fb, hour, read, time.time() - t1


def reduce_sql(sub, a, b):
    """One square's hours to days by SQL: for each (day, y, x) the max of the weather term, and the hour and
    FFWI where it happened. `sub` is the in-memory xarray Dataset of the square; returns (days, y, x) arrays."""
    import xarray_sql as xql
    ctx = xql.Context()
    ctx.from_dataset("hrrr", sub, chunks=dict(time=2160))
    epoch0 = int((START - np.datetime64("1970-01-01T00:00")) / np.timedelta64(1, "s"))
    q = f"""
    WITH r AS (
      SELECT floor((date_part('epoch', time) - {epoch0}) / 86400.0) AS day,
             (date_part('epoch', time) - {epoch0}) / 3600 % 24 AS hour, y, x,
             temperature_2m AS t, least(greatest(relative_humidity_2m, 0), 100) AS h,
             sqrt(wind_u_10m * wind_u_10m + wind_v_10m * wind_v_10m) AS w,
             temperature_2m * 1.8 + 32 AS tf
      FROM hrrr
    ), f AS (
      SELECT day, hour, y, x, t, h, w,
        least(greatest(CASE WHEN h < 10 THEN 0.03229 + 0.281073 * h - 0.000578 * h * tf
                            WHEN h < 50 THEN 2.22749 + 0.160107 * h - 0.014784 * tf
                            ELSE 21.0606 + 0.005565 * h * h - 0.00035 * h * tf - 0.483199 * h END, 0), 30) / 30 AS mr
      FROM r
    ), g AS (
      SELECT day, hour, y, x, t, h, w,
        (1 - 2 * mr + 1.5 * mr * mr - 0.5 * mr * mr * mr) * sqrt(1 + (w * 2.2369363) * (w * 2.2369363)) / 0.3002 AS ffwi
      FROM f
    ), z AS (
      SELECT day, hour, y, x, ffwi, {a['ffwi']} * ffwi + {a['t_c']} * t + {a['rh']} * h + {a['wind_ms']} * w + {b} AS term
      FROM g
    )
    SELECT day, y, x, max(term) AS worst,
           first_value(hour ORDER BY term DESC NULLS LAST) AS hour,
           first_value(ffwi ORDER BY term DESC NULLS LAST) AS ffwi
    FROM z WHERE term IS NOT NULL GROUP BY day, y, x
    """
    res = ctx.sql(q).to_pandas()
    ys = sub.y.values
    xs = sub.x.values
    iy = np.searchsorted(-ys, -res.y.values) if ys[0] > ys[-1] else np.searchsorted(ys, res.y.values)
    ix = np.searchsorted(xs, res.x.values)
    d = res.day.astype(int).values
    shape = (NDAYS, len(ys), len(xs))
    worst = np.full(shape, np.nan, np.float32)
    fw = np.full(shape, np.nan, np.float32)
    hour = np.full(shape, 255, np.uint8)
    worst[d, iy, ix] = res.worst.values
    fw[d, iy, ix] = res.ffwi.values
    hour[d, iy, ix] = res.hour.values.astype(np.uint8)
    return worst, fw, hour


def run(a):
    PARTS.mkdir(parents=True, exist_ok=True)
    ny = -(-NY // CHUNK_XY)
    nx = -(-NX // CHUNK_XY)
    jobs = [(cy, cx) for cy in range(ny) for cx in range(nx) if not (PARTS / f"{cy:02d}_{cx:02d}.npz").exists()]
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} squares to read ({ny * nx - len(jobs)} done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_open) as ex:
        for n, (cy, cx, term, fb, hour, read, dt) in enumerate(ex.map(_square, jobs)):
            np.savez(PARTS / f"{cy:02d}_{cx:02d}.npz", term=term, ffwi=fb, hour=hour)
            el = time.time() - t0
            print(f"\r{n + 1}/{len(jobs)} square {cy},{cx} read {read:.1f} s total {dt:.1f} s, {el / (n + 1):.1f} s/square, eta {el / (n + 1) * (len(jobs) - n - 1) / 60:.0f} min", end="", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


def smoke():
    _open()
    print(f"window {START} .. {NDAYS} days, time index {T0}..{T1}", flush=True)
    cy, cx = 13, 13  # a square over the central Plains
    cy, cx, term, fb, hour, read, dt = _square((cy, cx))
    print(f"square {cy},{cx}: read {read:.1f} s, total {dt:.1f} s; term bytes {term.shape} missing {(term == 255).mean():.3f}")
    z = TERM_LO + term[term < 255].astype(float) / 254 * (TERM_HI - TERM_LO)
    print(f"term logit: min {z.min():.2f} median {np.median(z):.2f} p95 {np.percentile(z, 95):.2f} max {z.max():.2f}")
    worst_day = int(np.argmax(np.median(np.where(term == 255, 0, term), axis=(1, 2))))
    print(f"worst day of the square (median): day {worst_day} = {START + np.timedelta64(worst_day, 'D')}, hour index {np.bincount(hour[worst_day].ravel()[hour[worst_day].ravel() < 255]).argmax()} of the 06Z day")
    part = PARTS / f"{cy:02d}_{cx:02d}.npz"
    if part.exists():
        d = np.load(part)
        print(f"against the part on disk: term bytes differ at {(d['term'] != term).mean():.5f} of pixels, ffwi {(d['ffwi'] != fb).mean():.5f}, hour {(d['hour'] != hour).mean():.5f}")


def assemble():
    _open()
    ny = -(-NY // CHUNK_XY)
    nx = -(-NX // CHUNK_XY)
    term = np.full((NDAYS, NY, NX), 255, np.uint8)
    ffwi_all = np.full((NDAYS, NY, NX), 255, np.uint8)
    hour = np.full((NDAYS, NY, NX), 255, np.uint8)
    missing = 0
    for cy in range(ny):
        for cx in range(nx):
            p = PARTS / f"{cy:02d}_{cx:02d}.npz"
            if not p.exists():
                missing += 1
                continue
            d = np.load(p)
            ys = slice(cy * CHUNK_XY, min((cy + 1) * CHUNK_XY, NY))
            xs = slice(cx * CHUNK_XY, min((cx + 1) * CHUNK_XY, NX))
            term[:, ys, xs] = d["term"][:, : ys.stop - ys.start, : xs.stop - xs.start]
            ffwi_all[:, ys, xs] = d["ffwi"][:, : ys.stop - ys.start, : xs.stop - xs.start]
            hour[:, ys, xs] = d["hour"][:, : ys.stop - ys.start, : xs.stop - xs.start]
    print(f"{missing} squares missing; term missing fraction {(term == 255).mean():.4f}")
    np.savez_compressed(OUT_DIR / f"season_{YEAR}.npz", term=term, ffwi=ffwi_all, hour=hour)
    import zstandard
    WEB.mkdir(parents=True, exist_ok=True)
    rows = -(-NY // SHARD_H)
    cols = -(-NX // SHARD_W)
    total = 0
    cctx = zstandard.ZstdCompressor(level=12)
    for r in range(rows):
        for c in range(cols):
            ys = slice(r * SHARD_H, min((r + 1) * SHARD_H, NY))
            xs = slice(c * SHARD_W, min((c + 1) * SHARD_W, NX))
            block = np.ascontiguousarray(term[:, ys, xs])
            raw = block.tobytes()
            comp = cctx.compress(raw)
            (WEB / f"{r}-{c}.zst").write_bytes(comp)
            total += len(comp)
    manifest = dict(
        year=YEAR, start=str(START)[:10], dayStartUtcHour=DAY_START_UTC, days=NDAYS, termLo=TERM_LO, termHi=TERM_HI, missing=255,
        shard=dict(height=SHARD_H, width=SHARD_W), rows=rows, cols=cols, grid=dict(height=NY, width=NX),
        note="per HRRR pixel and day, the day's worst hour (06Z to 06Z) of the weather part of the ignition logit, bytes over termLo..termHi, days-major; one zstd file per shard",
    )
    (WEB / "index.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {rows * cols} shards, {total / 1e6:.1f} MB, to {WEB}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        return smoke()
    if a.assemble:
        return assemble()
    _open()
    run(a)


if __name__ == "__main__":
    sys.exit(main())
