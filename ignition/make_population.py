# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "numpy", "pyproj", "geopandas", "shapely"]
# ///
"""A sample of REAL pixel-hours for the calibration: random points on CONUS land, random hours from
2019-01-01 to the last complete 90-day HRRR block, written in the cases format (kind "case", cause
"Population") as data/ignition/cases_pop.parquet, so sample_weather.py, sample_embeddings.py and
sample_nlcd.py read them unchanged (--tag pop). The head was fit case against control, so its 50% is
"as fire-like as a nearby control hour"; this sample is what it takes to turn that into ignitions
per area and time.

Stratified by weather group (a 135 km HRRR square x a 90-day block) so the weather reads stay few:
groups are drawn by uniform random land points and uniform random hours (so a group's chance is its
land area times its hours, and the sample is self-weighting), then PER_GROUP points and hours inside
each. The weight of every row is the same: CONUS land pixel-hours / rows.

    uv run ignition/make_population.py [--ngroups 400] [--per-group 50] [--seed 7]
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_cases import CHUNK_T, CHUNK_XY, DX, DY, H, LCC, T0, W, X0, Y0, finish, hrrr_pixels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATES = ROOT / "data" / "ignition" / "states" / "cb_2023_us_state_20m.shp"
OUT = ROOT / "data" / "ignition" / "cases_pop.parquet"
NOT_CONUS = {"AK", "HI", "PR", "AS", "GU", "MP", "VI"}
START = pd.Timestamp("2019-01-01T00:00:00Z")


def land_points(land, n, rng, bounds):
    """n points uniform over the land polygons (rejection sampling in an equal-area frame)."""
    minx, miny, maxx, maxy = bounds
    pts = []
    while sum(len(p) for p in pts) < n:
        xs = rng.uniform(minx, maxx, n * 2)
        ys = rng.uniform(miny, maxy, n * 2)
        g = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs=land.crs)
        inside = gpd.sjoin(gpd.GeoDataFrame(geometry=g), land[["STUSPS", "geometry"]], how="inner", predicate="within")
        pts.append(inside)
    out = pd.concat(pts).iloc[:n]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngroups", type=int, default=400)
    ap.add_argument("--per-group", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    st = gpd.read_file(STATES)
    st = st[~st.STUSPS.isin(NOT_CONUS)].to_crs("EPSG:5070")
    land_km2 = float(st.area.sum() / 1e6)
    t_start = int((START - T0) / pd.Timedelta("1h"))
    t_end = int((pd.Timestamp.now(tz="UTC") - T0) / pd.Timedelta("1h")) // CHUNK_T * CHUNK_T
    hours = t_end - t_start
    print(f"CONUS land {land_km2:,.0f} km2; hours {t_start} to {t_end} ({hours} h, {hours / 24 / 365.25:.2f} y)")

    # the groups: uniform land points x uniform hours, the first NGROUPS distinct (land x hours weighted)
    to_ll = Transformer.from_crs(st.crs, "EPSG:4326", always_xy=True)
    seeds = land_points(st, a.ngroups * 3, rng, st.total_bounds)
    lon, lat = to_ll.transform(seeds.geometry.x.values, seeds.geometry.y.values)
    hx, hy = hrrr_pixels(lon, lat)
    ti = rng.integers(t_start, t_end, len(seeds))
    g = pd.DataFrame({"cx": hx // CHUNK_XY, "cy": hy // CHUNK_XY, "ct": ti // CHUNK_T})
    g = g[(hx >= 0) & (hx < W) & (hy >= 0) & (hy < H)].drop_duplicates().iloc[: a.ngroups]
    print(f"{len(g)} groups")

    # the rows: PER_GROUP land points inside each square, each at a random hour of the block (within the window)
    to_lcc = Transformer.from_crs("EPSG:4326", LCC, always_xy=True)
    lcc_to_5070 = Transformer.from_crs(LCC, "EPSG:5070", always_xy=True)
    lcc_to_ll = Transformer.from_crs(LCC, "EPSG:4326", always_xy=True)
    rows = []
    for k, (cx, cy, ct) in enumerate(g.itertuples(index=False)):
        x0, x1 = X0 + cx * CHUNK_XY * DX, X0 + (cx + 1) * CHUNK_XY * DX
        y1, y0 = Y0 + cy * CHUNK_XY * DY, Y0 + (cy + 1) * CHUNK_XY * DY  # DY < 0
        got = []
        tries = 0
        while len(got) < a.per_group and tries < 40:
            tries += 1
            xs = rng.uniform(x0, x1, a.per_group * 2)
            ys = rng.uniform(y0, y1, a.per_group * 2)
            ex, ey = lcc_to_5070.transform(xs, ys)
            pts = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(ex, ey)], crs="EPSG:5070")
            pts["x"], pts["y"] = xs, ys
            inside = gpd.sjoin(pts, st[["STUSPS", "geometry"]], how="inner", predicate="within")
            got.extend(inside[["x", "y", "STUSPS"]].itertuples(index=False))
        if not got:
            continue  # a square with no land (should not happen: the seed point was on land)
        got = got[: a.per_group]
        lo, hi = max(int(ct) * CHUNK_T, t_start), min((int(ct) + 1) * CHUNK_T, t_end)
        for x, y, stusps in got:
            lo_, la_ = lcc_to_ll.transform(x, y)
            t = int(rng.integers(lo, hi))
            yr = int((T0 + pd.Timedelta(hours=t)).year)
            rows.append(dict(cause="Population", size=np.nan, final_acres=np.nan, name="", fuel="", state=f"US-{stusps}",
                             year=yr, emb_year=yr - 1, kind="case", lon=lo_, lat=la_, ti=t, dist_m=np.nan))
        if k % 50 == 0:
            print(f"\r{k + 1}/{len(g)} groups, {len(rows)} rows", end="", flush=True)
    print()
    out = pd.DataFrame(rows)
    out["case_id"] = np.arange(len(out))
    out, dropped = finish(out)
    if dropped:
        print(f"{dropped} points off the HRRR grid dropped")
    out["land_km2"] = land_km2
    out["hours"] = hours
    out["t_start"], out["t_end"] = t_start, t_end
    print(f"rows {len(out)}: weather groups {out.group.nunique()}, mosaic chunks {out.mchunk.nunique()}")
    print(out.year.value_counts().sort_index().to_string())
    print(out.state.value_counts().head(12).to_string())
    out.to_parquet(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
