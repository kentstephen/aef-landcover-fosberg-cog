# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "rasterio", "pyproj"]
# ///
"""Pull training material for the land cover head, one AEF tile at a time.

For each site in data/train_tiles.csv (first index row per site):
  - AEF 2025 embeddings at 40 m (COG overview factor 4; each overview pixel is a unit-length
    centroid of its 4x4 children), 64 bands int8, flipped to north-up.
  - Teachers warped onto the same tile:
      Esri / Impact Observatory 10 m annual LULC 2024 (s3://io-10m-annual-lulc), nearest, at 10 m
      USGS Annual NLCD 2024 land cover (30 m Albers, source.coop mirror), nearest, at 10 m
      Meta canopy v1 downsampled (0.00025 deg): avg height, p95 height, cover, averaged onto 40 m
Saves data/tiles/<site>.npz with the full tile (for preview maps) and a random sample of cells
(for fitting). Reruns skip tiles already on disk.

    uv run scripts/sample_landcover.py [--sites denver,sierra_foothills] [--n 300000]

Peak memory per tile: AEF 40 m int8 268 MB + two 10 m teacher rasters 134 MB + warp buffers,
about 0.6 GB. Tiles are processed one at a time and released.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.env import Env
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent.parent
TILES_CSV = ROOT / "data" / "train_tiles.csv"
OUT_DIR = ROOT / "data" / "tiles"

S3 = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop"
IO_S3 = "https://s3.us-west-2.amazonaws.com/io-10m-annual-lulc"
NLCD_URL = f"{S3}/kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic/Annual_NLCD_LndCov_2024_CU_C1V1.tif"
CHM = "/vsis3/dataforgood-fb-data/forests/v1/alsgedi_global_v6_float_epsg4326_v3_10deg/meta_chm_lat={lat}_lon={lon}_{p}.tif"
CHM_PARAMS = ("avg", "p95", "cover")
IO_YEAR = 2024  # Esri IO has no 2025 yet; 2024 labels against 2025 embeddings

FACTOR = 4           # 40 m
FULL = 8192
SIDE = FULL // FACTOR
NODATA = -128
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
           GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE", VSI_CACHE_SIZE=str(256 << 20))


def sites():
    rows = list(csv.DictReader(open(TILES_CSV)))
    seen = {}
    for r in rows:
        seen.setdefault(r["name"], r)  # first index row per site
    return seen


def aef_url(path):
    return path.replace("s3://us-west-2.opendata.source.coop", S3)


def io_band_letters(lat_s, lat_n):
    bands = "CDEFGHJKLMNPQRSTUVWX"  # 8 deg each from 80 S
    letter = lambda la: bands[min(19, max(0, int((la + 80) // 8)))]  # noqa: E731
    return sorted({letter(lat_s), letter(lat_n)})


def warp_into(src_path, dst, dst_transform, dst_crs, resampling, src_nodata=None, dst_nodata=0):
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            dst_transform=dst_transform, dst_crs=dst_crs,
            src_nodata=src_nodata if src_nodata is not None else src.nodata, dst_nodata=dst_nodata,
            resampling=resampling, num_threads=4, warp_mem_limit=512,
        )
    return dst


def one_tile(name, row, n_sample, rng):
    out = OUT_DIR / f"{name}.npz"
    if out.exists():
        print(f"[{name}] on disk, skipping")
        return
    t0 = time.time()
    url = aef_url(row["path"])
    with rasterio.open(url) as ds:
        assert ds.count == 64 and ds.width == FULL and ds.height == FULL, (ds.count, ds.width, ds.height)
        crs = ds.crs
        t = ds.transform
        assert t.e > 0, "expected the south-up AEF layout"
        west, south = t.c, t.f
        north = south + FULL * t.a
        # read the 40 m overview and flip to north-up
        aef = ds.read(out_shape=(64, SIDE, SIDE), resampling=Resampling.nearest)
        aef = np.ascontiguousarray(aef[:, ::-1, :])
        lon_w, lat_s, lon_e, lat_n = rasterio.warp.transform_bounds(crs, "EPSG:4326", *ds.bounds)
    t10 = Affine(10.0, 0.0, west, 0.0, -10.0, north)
    t40 = Affine(40.0, 0.0, west, 0.0, -40.0, north)
    valid = (aef != NODATA).all(axis=0)
    print(f"[{name}] AEF 40 m read {time.time() - t0:.1f} s, valid {valid.mean():.3f}, zone {row['utm_zone']}")

    # Esri IO 2024 at 10 m: one COG per zone and latitude band; a missing band is open sea
    zone = int(row["utm_zone"][:-1])
    esri10 = np.zeros((FULL, FULL), np.uint8)
    for b in io_band_letters(lat_s, lat_n):
        p = f"{IO_S3}/{zone:02d}{b}_{IO_YEAR}.tif"
        try:
            piece = warp_into(p, np.zeros((FULL, FULL), np.uint8), t10, crs, Resampling.nearest, src_nodata=0)
            esri10 = np.where(esri10 == 0, piece, esri10)
            del piece
        except rasterio.errors.RasterioIOError as e:
            print(f"[{name}]   Esri {zone:02d}{b} not there ({type(e).__name__})")
    print(f"[{name}] Esri IO {IO_YEAR} warped {time.time() - t0:.1f} s, covered {(esri10 > 0).mean():.3f}")

    # NLCD 2024 at 10 m (nearest from 30 m Albers)
    nlcd10 = warp_into(NLCD_URL, np.zeros((FULL, FULL), np.uint8), t10, crs, Resampling.nearest, src_nodata=250)
    print(f"[{name}] NLCD 2024 warped {time.time() - t0:.1f} s, covered {(nlcd10 > 0).mean():.3f}")

    # Meta canopy v1 downsampled onto 40 m (average); the 10 deg tile is named by its top-left corner
    chm = {}
    lat_top = int(np.ceil(lat_n / 10.0) * 10)
    lon_left = int(np.floor(lon_w / 10.0) * 10)
    tiles_needed = {(int(np.ceil(la / 10.0) * 10), int(np.floor(lo / 10.0) * 10)) for la in (lat_s, lat_n) for lo in (lon_w, lon_e)}
    for p in CHM_PARAMS:
        acc = np.full((SIDE, SIDE), np.nan, np.float32)
        for (la, lo) in sorted(tiles_needed):
            src = CHM.format(lat=f"{la:.1f}", lon=f"{lo:.1f}", p=p)
            try:
                piece = warp_into(src, np.full((SIDE, SIDE), np.nan, np.float32), t40, crs, Resampling.average, src_nodata=65535, dst_nodata=np.nan)
                acc = np.where(np.isnan(acc), piece, acc)
            except rasterio.errors.RasterioIOError as e:
                print(f"[{name}]   canopy {p} lat={la} lon={lo} not there ({type(e).__name__})")
        chm[p] = acc
    print(f"[{name}] canopy warped {time.time() - t0:.1f} s, covered {np.isfinite(chm['cover']).mean():.3f}; (tile corner {lat_top},{lon_left})")

    # sample cells: uniform over valid AEF cells
    idx = np.flatnonzero(valid)
    take = rng.choice(idx, size=min(n_sample, idx.size), replace=False)
    r, c = np.divmod(take, SIDE)
    emb = aef[:, r, c].T.copy()  # (N, 64) int8
    # teacher class fractions per 40 m cell from the 4x4 block of 10 m labels
    def block_values(a10):
        rr = (r[:, None] * FACTOR + np.arange(FACTOR))[:, :, None]
        cc = (c[:, None] * FACTOR + np.arange(FACTOR))[:, None, :]
        return a10[rr, cc].reshape(len(r), FACTOR * FACTOR)
    esri_blk = block_values(esri10)
    nlcd_blk = block_values(nlcd10)

    np.savez_compressed(
        out,
        name=name, crs=str(crs), transform40=np.array(t40.to_gdal(), np.float64), bounds_wgs84=np.array([lon_w, lat_s, lon_e, lat_n]),
        aef40=aef, valid=valid, esri10=esri10, nlcd10=nlcd10,
        chm_avg=chm["avg"], chm_p95=chm["p95"], chm_cover=chm["cover"],
        s_row=r.astype(np.int16), s_col=c.astype(np.int16), s_emb=emb,
        s_esri=esri_blk.astype(np.uint8), s_nlcd=nlcd_blk.astype(np.uint8),
        s_chm_avg=chm["avg"][r, c], s_chm_p95=chm["p95"][r, c], s_chm_cover=chm["cover"][r, c],
    )
    print(f"[{name}] saved {out.name} ({out.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="")
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    allsites = sites()
    names = [s for s in a.sites.split(",") if s] or list(allsites)
    rng = np.random.default_rng(a.seed)
    with Env(**ENV):
        for name in names:
            try:
                one_tile(name, allsites[name], a.n, rng)
            except Exception as e:  # keep going; the summary at the end says what failed
                print(f"[{name}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    print("done:", sorted(p.stem for p in OUT_DIR.glob("*.npz")))


if __name__ == "__main__":
    main()
