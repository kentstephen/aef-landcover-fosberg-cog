# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "rasterio", "icechunk>=1", "xarray", "zarr>=3", "scikit-learn", "pillow"]
# ///
"""Can the AEF embedding read CarbonPlan's Open Climate Risk fire layers?

Reads bp_2011 (Riley et al. 2025 burn probability, spread to non-burnable land), crps_scott
(Scott et al. 2024 conditional risk to potential structures) and rps_2011 (their product) from the
OCR Icechunk store over one sampled AEF tile, warps them onto the tile's 40 m grid, and fits ridge
heads from the embedding with a spatial split (west half fits, east half scores).

    uv run scripts/probe_carbonplan.py denver
"""
import sys
import time
from pathlib import Path

import icechunk
import numpy as np
import rasterio
import xarray as xr
from PIL import Image
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
VARS = ["bp_2011", "crps_scott", "rps_2011"]


def features(q):
    v = np.sign(q.astype(np.float32)) * (q.astype(np.float32) / 127.5) ** 2
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


def main(name):
    d = np.load(ROOT / "data" / "tiles" / f"{name}.npz")
    lon_w, lat_s, lon_e, lat_n = d["bounds_wgs84"]
    t40 = Affine.from_gdal(*d["transform40"])
    crs = str(d["crs"])
    side = d["valid"].shape[0]

    t0 = time.time()
    storage = icechunk.s3_storage(bucket="us-west-2.opendata.source.coop",
                                  prefix="carbonplan/carbonplan-ocr/output/fire-risk/tensor/production/v1.1.0/ocr.icechunk",
                                  region="us-west-2", anonymous=True)
    repo = icechunk.Repository.open(storage)
    ds = xr.open_dataset(repo.readonly_session("main").store, engine="zarr")
    print("variables:", list(ds.data_vars), "| dims", dict(ds.sizes), f"| opened {time.time() - t0:.1f} s")
    pad = 0.01
    asc = bool(ds.latitude.values[1] > ds.latitude.values[0])
    lat_slice = slice(lat_s - pad, lat_n + pad) if asc else slice(lat_n + pad, lat_s - pad)
    sub = ds.sel(latitude=lat_slice, longitude=slice(lon_w - pad, lon_e + pad))
    if asc:
        sub = sub.isel(latitude=slice(None, None, -1))  # north-up for the warp
    print("subset cells", dict(sub.sizes), "(store latitude ascending)" if asc else "")
    lat = sub.latitude.values
    lon = sub.longitude.values
    dy = float(lat[0] - lat[1])
    dx = float(lon[1] - lon[0])
    src_t = from_origin(float(lon[0]) - dx / 2, float(lat[0]) + dy / 2, dx, dy)

    out = {}
    for v in VARS:
        t1 = time.time()
        a = sub[v].values.astype(np.float32)
        print(f"  {v}: read {a.shape} in {time.time() - t1:.1f} s; nan {np.isnan(a).mean():.3f}; p50/p95/max {np.nanpercentile(a, [50, 95]).round(4)} {np.nanmax(a):.4f}")
        dst = np.full((side, side), np.nan, np.float32)
        reproject(a, dst, src_transform=src_t, src_crs="EPSG:4326", src_nodata=np.nan,
                  dst_transform=t40, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
        out[v] = dst
    np.savez_compressed(ROOT / "data" / f"carbonplan_{name}.npz", **out)

    # fit from the embedding: west half trains, east half scores (a spatial split, not a random one)
    r, c = d["s_row"], d["s_col"]
    X = features(d["s_emb"])
    west = c < side // 2
    for v in VARS:
        y = out[v][r, c]
        ok = np.isfinite(y)
        tr, te = ok & west, ok & ~west
        mean, scale = X[tr].mean(0), X[tr].std(0) + 1e-6
        reg = Ridge(alpha=1.0).fit((X[tr] - mean) / scale, y[tr])
        pred = reg.predict((X[te] - mean) / scale)
        r2 = 1 - ((pred - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
        rho = np.corrcoef(pred, y[te])[0, 1]
        print(f"  head 64 -> {v}: east-half R2 {r2:.2f}, corr {rho:.2f}; teacher mean {np.nanmean(y[te]):.4f}")
    # what each NLCD class (teacher mode of the sampled cells) carries, for reading the numbers
    nl = d["s_nlcd"]
    lab = np.array([np.bincount(row, minlength=100).argmax() for row in nl])
    print("  bp_2011 by NLCD class (mean):", {int(k): round(float(np.nanmean(out['bp_2011'][r, c][lab == k])), 4) for k in np.unique(lab) if k and (lab == k).sum() > 500})

    # picture: bp | crps | rps, each on its own gray ramp (p99 white), the teacher's view of this tile
    strips = []
    for v in VARS:
        a = out[v]
        top = np.nanpercentile(a, 99)
        g = np.nan_to_num(np.clip(a / max(top, 1e-9), 0, 1)) * 255
        strips.append(g.astype(np.uint8))
        strips.append(np.full((side, 8), 20, np.uint8))
    img = np.concatenate(strips[:-1], axis=1)
    p = ROOT / "data" / "previews" / f"{name}_carbonplan.png"
    Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2), Image.BILINEAR).save(p)
    print("  wrote", p.relative_to(ROOT), "(bp_2011 | crps_scott | rps_2011, gray, p99 = white)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "denver")
