# /// script
# requires-python = ">=3.12"
# dependencies = ["zarr>=3", "obstore", "numpy", "pandas", "pyarrow"]
# ///
"""The AlphaEarth embedding of the year before the fire at every point of cases.parquet, from the
10 m Zarr mosaic (tge-labs/aef-mosaic on source.coop, read from the S3 host). One inner chunk per
(mosaic chunk, year): 64 bands x 256 x 256 int8, about 5 MB over the wire, 4 MB in RAM; the shard
index is a small range read first. For each point in the chunk: the 10 m pixel (64 int8, as stored)
and the unit-normalized mean of the 15 x 15 window (150 m) around it. Parts written per chunk to
data/ignition/emb_parts/, resumable. Peak RAM well under 1 GB.

    uv run ignition/sample_embeddings.py [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from obstore.store import S3Store
from zarr.storage import ObjectStore

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "ignition" / "cases.parquet"
PARTS = ROOT / "data" / "ignition" / "emb_parts"
OUT = ROOT / "data" / "ignition" / "embeddings.parquet"
CHUNK = 256
YEAR0 = 2017
WIN = 7  # half window: 15 x 15 px


def dequant(q):
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_parquet(CASES)
    pts = df[df.kind.isin(["case", "s_ctrl"])].copy()
    pts["pkey"] = pts.mchunk.astype(str) + "_" + pts.emb_year.astype(str)
    PARTS.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in PARTS.glob("*.parquet")}
    keys = [k for k in pts.pkey.unique() if k not in done]
    if a.limit: keys = keys[: a.limit]
    print(f"{len(keys)} chunks to read ({len(done)} done), {len(pts)} points")
    store = ObjectStore(S3Store("us-west-2.opendata.source.coop", prefix="tge-labs/aef-mosaic", region="us-west-2", skip_signature=True,
                                client_options={"timeout": "60s", "connect_timeout": "10s"}), read_only=True)
    arr = zarr.open_array(store, path="embeddings", mode="r")
    print("mosaic", arr.shape, arr.dtype, "chunks", arr.chunks, "shards", arr.shards)
    t0 = time.time()
    nbytes = 0
    for n, k in enumerate(keys):
        rows = pts[pts.pkey == k]
        cx0 = int(rows.mx.iloc[0] // CHUNK) * CHUNK
        cy0 = int(rows.my.iloc[0] // CHUNK) * CHUNK
        ti = int(rows.emb_year.iloc[0]) - YEAR0
        t1 = time.time()
        block = arr[ti, :, cy0:cy0 + CHUNK, cx0:cx0 + CHUNK]  # (64, 256, 256) int8
        nbytes += block.nbytes
        out = []
        for i, r in rows.iterrows():
            x, y = int(r.mx - cx0), int(r.my - cy0)
            pix = block[:, y, x]
            y0, y1, x0, x1 = max(0, y - WIN), min(CHUNK, y + WIN + 1), max(0, x - WIN), min(CHUNK, x + WIN + 1)
            win = block[:, y0:y1, x0:x1].reshape(64, -1)
            ok = ~(win == -128).all(axis=0)
            if ok.any():
                m = dequant(win[:, ok]).mean(axis=1)
                nrm = float(np.linalg.norm(m))
                mean = (m / nrm) if nrm > 0 else m
            else:
                nrm, mean = float("nan"), np.full(64, np.nan, np.float32)
            out.append(dict(row=int(i), nodata=bool((pix == -128).all()), win_norm=nrm,
                            pix=pix.tobytes(), win_mean=mean.astype(np.float32).tobytes()))
        pd.DataFrame(out).to_parquet(PARTS / f"{k}.parquet", index=False)
        el = time.time() - t0
        print(f"\r{n + 1}/{len(keys)} chunk {k} pts {len(rows)} read {time.time() - t1:.1f} s, {el / (n + 1):.1f} s/chunk, {nbytes / 1e9:.1f} GB, eta {el / (n + 1) * (len(keys) - n - 1) / 60:.0f} min", end="", flush=True)
    print()
    parts = [pd.read_parquet(p) for p in PARTS.glob("*.parquet")]
    alle = pd.concat(parts).set_index("row").sort_index()
    alle.to_parquet(OUT)
    print(f"wrote {OUT} {len(alle)} points; nodata {int(alle.nodata.sum())}")


if __name__ == "__main__":
    sys.exit(main())
