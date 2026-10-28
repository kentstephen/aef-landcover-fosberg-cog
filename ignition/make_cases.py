# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "numpy", "pyproj"]
# ///
"""Cases and controls for the ignition study -> data/ignition/cases.parquet.

A case is a WFIGS wildfire with a known cause (Natural or Human), inside the HRRR grid, discovered
2019 to 2025 (so the embedding of the year before exists, 2018 to 2024). Each case gets three
controls, which together split "where" from "when":
  t_ctrl  same point, another hour in the same 90-day weather block (not within a day of the fire)
  s_ctrl  another point in the same 2.5 km embedding chunk (at least 300 m away), the fire's hour
  r_ctrl  that other point, at the t_ctrl hour
The pilot takes its cases by weather chunk (a 135 km square and a 90-day block), so the HRRR reads
stay few: NGROUPS squares drawn with probability ~ sqrt(fires in the square), up to PER_GROUP cases
each, half Natural and half Human as far as the square allows. Nothing heavy: a DataFrame.

    uv run ignition/make_cases.py [--ngroups 400] [--per-group 12] [--seed 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "ignition" / "wfigs.parquet"
OUT = ROOT / "data" / "ignition" / "cases.parquet"

LCC = "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
X0, Y0, DX, DY, W, H = -2699020.142521929, 1588193.847443335, 3000, -3000, 1799, 1059
T0 = pd.Timestamp("2014-10-01T00:00:00Z")
CHUNK_T, CHUNK_XY = 2160, 45
# the AEF mosaic: EPSG:4326, 256 px inner chunks
MOS_RES = 8.983111749910169e-5
MOS_X0, MOS_Y0 = -180.0, 83.68570533713473
MOS_CHUNK = 256


def hrrr_pixels(lon, lat):
    x, y = Transformer.from_crs("EPSG:4326", LCC, always_xy=True).transform(lon, lat)
    return np.floor((x - X0) / DX).astype(int), np.floor((y - Y0) / DY).astype(int)


def mosaic_pixels(lon, lat):
    return np.floor((lon - MOS_X0) / MOS_RES).astype(np.int64), np.floor((MOS_Y0 - lat) / MOS_RES).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngroups", type=int, default=400)
    ap.add_argument("--per-group", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    df = pd.read_parquet(IN)
    df = df[df.FireCause.isin(["Natural", "Human"])].copy()
    df["time"] = df.FireDiscoveryDateTime.dt.floor("h")
    df["year"] = df.time.dt.year
    df = df[(df.year >= 2019) & (df.year <= 2025)]
    df = df.rename(columns={"geom_lon": "lon", "geom_lat": "lat"})
    hx, hy = hrrr_pixels(df.lon.values, df.lat.values)
    df["hx"], df["hy"] = hx, hy
    df = df[(hx >= 0) & (hx < W) & (hy >= 0) & (hy < H)].copy()
    df["ti"] = ((df.time - T0) / pd.Timedelta("1h")).astype(int)
    df["cx"], df["cy"], df["ct"] = df.hx // CHUNK_XY, df.hy // CHUNK_XY, df.ti // CHUNK_T
    df["group"] = df.ct * 100000 + df.cy * 100 + df.cx
    print(f"eligible cases: {len(df)} ({(df.FireCause == 'Natural').sum()} natural, {(df.FireCause == 'Human').sum()} human), {df.group.nunique()} weather groups")

    counts = df.groupby("group").size()
    p = np.sqrt(counts.values)
    p /= p.sum()
    groups = rng.choice(counts.index.values, size=min(a.ngroups, len(counts)), replace=False, p=p)
    picks = []
    for g in groups:
        sub = df[df.group == g]
        half = a.per_group // 2
        nat = sub[sub.FireCause == "Natural"]
        hum = sub[sub.FireCause == "Human"]
        n_nat = min(len(nat), half + max(0, half - len(hum)))
        n_hum = min(len(hum), a.per_group - n_nat)
        if n_nat: picks.append(nat.sample(n_nat, random_state=int(rng.integers(1 << 31))))
        if n_hum: picks.append(hum.sample(n_hum, random_state=int(rng.integers(1 << 31))))
    cases = pd.concat(picks).reset_index(drop=True)
    cases["case_id"] = np.arange(len(cases))
    print(f"picked {len(cases)} cases ({(cases.FireCause == 'Natural').sum()} natural) in {len(groups)} groups")

    # controls
    rows = []
    mx, my = mosaic_pixels(cases.lon.values, cases.lat.values)
    for i, c in cases.iterrows():
        base = dict(case_id=c.case_id, cause=c.FireCause, size=c.IncidentSize, final_acres=c.FinalAcres,
                    name=c.IncidentName, fuel=c.PredominantFuelModel, state=c.POOState, year=int(c.year), emb_year=int(c.year) - 1)
        rows.append({**base, "kind": "case", "lon": c.lon, "lat": c.lat, "ti": int(c.ti)})
        # another hour in the same 90-day block, not within a day of the fire
        lo, hi = int(c.ct) * CHUNK_T, int(c.ct + 1) * CHUNK_T
        while True:
            t2 = int(rng.integers(lo, hi))
            if abs(t2 - int(c.ti)) > 24: break
        rows.append({**base, "kind": "t_ctrl", "lon": c.lon, "lat": c.lat, "ti": t2})
        # another point in the same 256 px mosaic chunk, at least 30 px (300 m) away
        cx0, cy0 = (mx[i] // MOS_CHUNK) * MOS_CHUNK, (my[i] // MOS_CHUNK) * MOS_CHUNK
        while True:
            px, py = cx0 + rng.integers(0, MOS_CHUNK), cy0 + rng.integers(0, MOS_CHUNK)
            if np.hypot(px - mx[i], py - my[i]) >= 30: break
        lon2 = MOS_X0 + (px + 0.5) * MOS_RES
        lat2 = MOS_Y0 - (py + 0.5) * MOS_RES
        rows.append({**base, "kind": "s_ctrl", "lon": lon2, "lat": lat2, "ti": int(c.ti)})
        rows.append({**base, "kind": "r_ctrl", "lon": lon2, "lat": lat2, "ti": t2})
    out = pd.DataFrame(rows)
    out["time"] = T0 + pd.to_timedelta(out.ti, unit="h")
    hx, hy = hrrr_pixels(out.lon.values, out.lat.values)
    out["hx"], out["hy"] = hx, hy
    out["cx"], out["cy"], out["ct"] = hx // CHUNK_XY, hy // CHUNK_XY, out.ti // CHUNK_T
    out["group"] = out.ct * 100000 + out.cy * 100 + out.cx
    mx, my = mosaic_pixels(out.lon.values, out.lat.values)
    out["mx"], out["my"] = mx, my
    out["mchunk"] = (my // MOS_CHUNK) * 100000 + (mx // MOS_CHUNK)
    # the point and its control share a chunk by construction; if the square edge made the HRRR group
    # differ (a control just over a chunk line) that only means one more weather group
    print(f"rows {len(out)}: weather groups {out.group.nunique()}, mosaic chunks {out.mchunk.nunique()} (x ~5 MB), point rows {len(out) // 2}")
    print(out.kind.value_counts().to_string())
    print(out.groupby('cause').year.value_counts().unstack(fill_value=0).to_string())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
