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
from make_cases import finish, ring_rows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "ignition" / "cases.parquet"
OUT = ROOT / "data" / "ignition" / "cases_wide.parquet"
BASE_ROW = 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    df = pd.read_parquet(IN)
    out, dropped = finish(ring_rows(df, rng))
    out.index = np.arange(BASE_ROW, BASE_ROW + len(out))
    print(f"rows {len(out)} ({dropped} off the HRRR grid dropped): weather groups {out.group.nunique()}, "
          f"mosaic chunks {out[out.kind.str.startswith('s')].mchunk.nunique()}")
    print(out.kind.value_counts().to_string())
    out.to_parquet(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
