"""Python parity / sanity check for the Fosberg Fire Weather Index.

Two modes:

  --fixed     print FFWI for the same fixed inputs as scripts/ffwi_check.mjs
              (compare the two tables; they must agree to 3 decimals)
  (default)   open the dynamical HRRR 48-hour icechunk repo, read one lead
              hour for a CONUS patch from the latest init, compute FFWI the
              same way the browser does, and print summary stats plus a few
              pixels you can compare against the app's tooltip.

Run:
  uv run --with numpy python scripts/verify_ffwi.py --fixed
  uv run --with "icechunk>=2" --with "zarr>=3" --with numpy python scripts/verify_ffwi.py --lead 12
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

MS_TO_MPH = 2.2369363


def ffwi(t_c, rh, wind_ms):
    """Vectorised twin of web/src/ffwi/compute.ts `ffwi`."""
    t_c = np.asarray(t_c, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    wind_ms = np.asarray(wind_ms, dtype=np.float64)
    t_f = t_c * 1.8 + 32
    h = np.clip(rh, 0, 100)
    m = np.where(
        h < 10,
        0.03229 + 0.281073 * h - 0.000578 * h * t_f,
        np.where(
            h < 50,
            2.22749 + 0.160107 * h - 0.014784 * t_f,
            21.0606 + 0.005565 * h * h - 0.00035 * h * t_f - 0.483199 * h,
        ),
    )
    m = np.clip(m, 0, 30)
    mr = m / 30
    eta = 1 - 2 * mr + 1.5 * mr**2 - 0.5 * mr**3
    u = wind_ms * MS_TO_MPH
    out = eta * np.sqrt(1 + u * u) / 0.3002
    bad = ~(np.isfinite(t_c) & np.isfinite(rh) & np.isfinite(wind_ms))
    return np.where(bad, np.nan, out)


FIXED = [
    (30, 10, 10),
    (35, 5, 15),
    (20, 50, 5),
    (10, 90, 2),
    (40, 8, 13.41),
    (25, 0, 13.41),
    (0, 30, 0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", action="store_true")
    ap.add_argument("--lead", type=int, default=12)
    ap.add_argument("--init", type=int, default=-1, help="init_time index (default latest)")
    args = ap.parse_args()

    if args.fixed:
        print("tC\trh\twind_ms\tffwi")
        for t, h, w in FIXED:
            print(f"{t}\t{h}\t{w}\t{float(ffwi(t, h, w)):.3f}")
        return 0

    import icechunk  # noqa: E402
    import zarr  # noqa: E402

    storage = icechunk.s3_storage(
        bucket="dynamical-noaa-hrrr",
        prefix="noaa-hrrr-forecast-48-hour/v0.1.0.icechunk",
        region="us-west-2",
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    g = zarr.open_group(repo.readonly_session("main").store, mode="r")
    n_init = g["temperature_2m"].shape[0]
    init = args.init if args.init >= 0 else n_init - 1
    it = g["init_time"][init]
    print(f"init idx {init} / {n_init}  init_time {np.datetime64(int(it), 's')}  lead +{args.lead} h")

    # One shard-aligned patch (rows 265..530, cols 900..1200): roughly the
    # central Plains / Rockies. Same slice the browser would fetch as a tile.
    ys, xs = slice(265, 530), slice(900, 1200)
    t = g["temperature_2m"][init, args.lead, ys, xs]
    rh = g["relative_humidity_2m"][init, args.lead, ys, xs]
    u = g["wind_u_10m"][init, args.lead, ys, xs]
    v = g["wind_v_10m"][init, args.lead, ys, xs]
    w = np.sqrt(u * u + v * v)
    f = ffwi(t, rh, w)
    ok = np.isfinite(f)
    print(f"patch {f.shape}: valid {ok.mean():.1%}  FFWI min {np.nanmin(f):.1f}  "
          f"mean {np.nanmean(f):.1f}  p95 {np.nanpercentile(f, 95):.1f}  max {np.nanmax(f):.1f}")
    print("sample pixels (hrrr_y, hrrr_x): tC  rh  wind_mph  ffwi")
    for dy, dx in [(0, 0), (100, 150), (200, 250), (264, 299)]:
        print(f"  ({265 + dy}, {900 + dx}): {t[dy, dx]:.1f}  {rh[dy, dx]:.0f}  "
              f"{w[dy, dx] * MS_TO_MPH:.0f}  {f[dy, dx]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
