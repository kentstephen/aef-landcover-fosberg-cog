# /// script
# requires-python = ">=3.12"
# dependencies = ["requests", "pandas", "pyarrow"]
# ///
"""Pull NIFC WFIGS incident locations (wildfires, 2018 on) into data/ignition/wfigs.parquet:
discovery time, cause, size, point of origin, state, fuel model. Pages of 2000 over the
FeatureServer; about 250k rows, a few MB.

    uv run ignition/pull_wfigs.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ignition" / "wfigs.parquet"
URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations/FeatureServer/0/query"
FIELDS = [
    "OBJECTID", "IrwinID", "UniqueFireIdentifier", "IncidentName", "FireDiscoveryDateTime", "FireOutDateTime",
    "FireCause", "FireCauseGeneral", "FireCauseSpecific", "IncidentSize", "FinalAcres", "DiscoveryAcres",
    "InitialLatitude", "InitialLongitude", "POOState", "POOLandownerKind", "PredominantFuelGroup",
    "PredominantFuelModel", "IncidentTypeCategory", "IsValid",
]
WHERE = "IncidentTypeCategory='WF' AND FireDiscoveryDateTime>=DATE'2018-01-01'"


def main():
    rows = []
    offset = 0
    t0 = time.time()
    s = requests.Session()
    while True:
        r = s.get(URL, params={
            "where": WHERE, "outFields": ",".join(FIELDS), "orderByFields": "OBJECTID", "resultOffset": offset,
            "resultRecordCount": 2000, "returnGeometry": "true", "outSR": 4326, "f": "json",
        }, timeout=120)
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            raise SystemExit(d["error"])
        feats = d.get("features", [])
        for f in feats:
            a = f["attributes"]
            g = f.get("geometry") or {}
            a["geom_lon"] = g.get("x")
            a["geom_lat"] = g.get("y")
            rows.append(a)
        offset += len(feats)
        print(f"\r{offset} rows, {time.time() - t0:.0f} s", end="", flush=True)
        if not d.get("exceededTransferLimit") or not feats:
            break
    print()
    df = pd.DataFrame(rows)
    for c in ("FireDiscoveryDateTime", "FireOutDateTime"):
        df[c] = pd.to_datetime(df[c], unit="ms", utc=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} {len(df)} rows; discovery {df.FireDiscoveryDateTime.min()} to {df.FireDiscoveryDateTime.max()}")
    print(df.FireCause.value_counts(dropna=False).to_string())
    print("point of origin missing:", int(df.InitialLatitude.isna().sum()), "geometry missing:", int(df.geom_lat.isna().sum()))


if __name__ == "__main__":
    sys.exit(main())
