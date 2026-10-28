# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow"]
# ///
"""Every WFIGS wildfire ignition for the map, as one small binary the browser filters by the run's window:
uint32 n, then float32 lon[n], float32 lat[n], uint32 hour[n] (hours since 2018-01-01T00Z of discovery),
uint8 cause[n] (0 unknown, 1 natural, 2 human), uint8 size[n] (0 unknown, 1 under 1 acre, 2 under 10,
3 under 100, 4 under 1000, 5 larger). About 4 MB for 314k fires.

    uv run ignition/export_fires.py
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent.parent
OUT = D / "web" / "public" / "fires.bin"
EPOCH = pd.Timestamp("2018-01-01T00:00:00Z")


def main():
    w = pd.read_parquet(D / "data" / "ignition" / "wfigs.parquet")
    w = w[w.FireDiscoveryDateTime.notna() & w.geom_lon.notna() & w.geom_lat.notna()]
    w = w[(w.FireDiscoveryDateTime >= EPOCH)]
    w = w.sort_values("FireDiscoveryDateTime")
    hours = ((w.FireDiscoveryDateTime - EPOCH) / pd.Timedelta(hours=1)).astype(np.uint32)
    cause = w.FireCause.map({"Natural": 1, "Human": 2}).fillna(0).astype(np.uint8)
    size = w.IncidentSize.fillna(w.FinalAcres).fillna(w.DiscoveryAcres)
    sclass = np.select([size.isna(), size < 1, size < 10, size < 100, size < 1000], [0, 1, 2, 3, 4], 5).astype(np.uint8)
    n = len(w)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(w.geom_lon.to_numpy(np.float32).tobytes())
        f.write(w.geom_lat.to_numpy(np.float32).tobytes())
        f.write(hours.to_numpy().tobytes())
        f.write(cause.to_numpy().tobytes())
        f.write(sclass.tobytes())
    print(f"wrote {OUT}: {n} fires, {OUT.stat().st_size / 1e6:.1f} MB; cause {np.bincount(cause)}, size class {np.bincount(sclass)}")


if __name__ == "__main__":
    sys.exit(main())
