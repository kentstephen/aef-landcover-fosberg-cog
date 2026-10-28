# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "numpy", "pyproj"]
# ///
"""Wider spatial controls for the cases in cases.parquet -> cases_wide.parquet.

The pilot's s_ctrl sits in the same 2.5 km mosaic chunk as the fire, and for natural fires the
ground read near chance there. These controls ask at what distance the ground starts to matter:
  s10_ctrl  a point 5 to 10 km from the fire, the fire's hour
  s30_ctrl  a point 20 to 30 km from the fire, the fire's hour
  r10_ctrl, r30_ctrl  those points at the case's t_ctrl hour (both differ)
Rows are numbered from 100000 so they never collide with cases.parquet. The samplers take
--tag wide; fit.py picks the file up when it exists.

    uv run ignition/make_controls_wide.py [--seed 2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_cases import CHUNK_T, CHUNK_XY, H, MOS_CHUNK, T0, W, hrrr_pixels, mosaic_pixels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "ignition" / "cases.parquet"
OUT = ROOT / "data" / "ignition" / "cases_wide.parquet"
RINGS = {"10": (5_000, 10_000), "30": (20_000, 30_000)}
BASE_ROW = 100_000


def offset(lon, lat, dist_m, bearing, rng=None):
    """Move a point dist_m along a bearing (radians) on a sphere."""
    r = 6_371_000.0
    la1, lo1 = np.radians(lat), np.radians(lon)
    d = dist_m / r
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(bearing))
    lo2 = lo1 + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(la1), np.cos(d) - np.sin(la1) * np.sin(la2))
    return np.degrees(lo2), np.degrees(la2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    df = pd.read_parquet(IN)
    cases = df[df.kind == "case"].set_index("case_id")
    t2 = df[df.kind == "t_ctrl"].set_index("case_id").ti
    keep = ["case_id", "cause", "size", "final_acres", "name", "fuel", "state", "year", "emb_year"]
    rows = []
    for ring, (lo, hi) in RINGS.items():
        n = len(cases)
        dist = np.sqrt(rng.uniform(lo * lo, hi * hi, n))  # uniform over the annulus area
        brg = rng.uniform(0, 2 * np.pi, n)
        lon2, lat2 = offset(cases.lon.values, cases.lat.values, dist, brg)
        base = cases.reset_index()[keep]
        for kind, ti in [(f"s{ring}_ctrl", cases.ti.values), (f"r{ring}_ctrl", t2.loc[cases.index].values)]:
            b = base.copy()
            b["kind"], b["lon"], b["lat"], b["ti"] = kind, lon2, lat2, ti.astype(int)
            b["dist_m"] = dist
            rows.append(b)
    out = pd.concat(rows, ignore_index=True)
    out["time"] = T0 + pd.to_timedelta(out.ti, unit="h")
    hx, hy = hrrr_pixels(out.lon.values, out.lat.values)
    out["hx"], out["hy"] = hx, hy
    inside = (hx >= 0) & (hx < W) & (hy >= 0) & (hy < H)
    out = out[inside].copy()
    out["cx"], out["cy"], out["ct"] = out.hx // CHUNK_XY, out.hy // CHUNK_XY, out.ti // CHUNK_T
    out["group"] = out.ct * 100000 + out.cy * 100 + out.cx
    mx, my = mosaic_pixels(out.lon.values, out.lat.values)
    out["mx"], out["my"] = mx, my
    out["mchunk"] = (my // MOS_CHUNK) * 100000 + (mx // MOS_CHUNK)
    out.index = np.arange(BASE_ROW, BASE_ROW + len(out))
    print(f"rows {len(out)} ({(~inside).sum()} off the HRRR grid dropped): weather groups {out.group.nunique()}, "
          f"mosaic chunks {out[out.kind.str.startswith('s')].mchunk.nunique()}")
    print(out.kind.value_counts().to_string())
    out.to_parquet(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
