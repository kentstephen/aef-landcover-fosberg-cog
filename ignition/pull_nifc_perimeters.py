# /// script
# requires-python = ">=3.12"
# dependencies = ["requests"]
# ///
"""Every wildfire perimeter of one year from NIFC's WFIGS Interagency Perimeters service (the current
year's perimeters; MTBS lags a year), with the fire's discovery time, cause and acres, as
web/public/perimeters-<year>.geojson for the map (geometry simplified by the service to about 30 m).

    uv run ignition/pull_nifc_perimeters.py --year 2025
"""
import argparse
import json
import sys
from pathlib import Path

import requests

D = Path(__file__).resolve().parent.parent
URL = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query"
FIELDS = ["poly_IncidentName", "poly_GISAcres", "poly_IRWINID", "attr_FireDiscoveryDateTime", "attr_FireCause", "attr_IncidentSize", "attr_POOState"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    a = ap.parse_args()
    where = (f"attr_FireDiscoveryDateTime >= timestamp '{a.year}-01-01 00:00:00' AND attr_FireDiscoveryDateTime < timestamp "
             f"'{a.year + 1}-01-01 00:00:00' AND attr_IncidentTypeCategory = 'WF'")
    feats = []
    off = 0
    while True:
        r = requests.get(URL, params=dict(where=where, outFields=",".join(FIELDS), f="geojson", outSR=4326, resultOffset=off,
                                          resultRecordCount=2000, orderByFields="OBJECTID", maxAllowableOffset=0.0003, geometryPrecision=5), timeout=120)
        r.raise_for_status()
        j = r.json()
        got = j.get("features", [])
        for f in got:
            p = f["properties"]
            disc = p.get("attr_FireDiscoveryDateTime")
            f["properties"] = {"name": p.get("poly_IncidentName"), "acres": p.get("poly_GISAcres"), "irwin": p.get("poly_IRWINID"),
                               "discovered_ms": disc, "cause": p.get("attr_FireCause"), "size": p.get("attr_IncidentSize"), "state": p.get("attr_POOState")}
        feats.extend(got)
        print(f"{off}: +{len(got)} ({len(feats)})", flush=True)
        # pages come back short of 2000 when the response size limit bites, so only an empty page ends it
        if len(got) == 0:
            break
        off += len(got)
    feats = [f for f in feats if f.get("geometry")]
    out = D / "web" / "public" / f"perimeters-{a.year}.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
    acres = sum((f["properties"]["acres"] or 0) for f in feats)
    print(f"wrote {out}: {len(feats)} perimeters, {acres:,.0f} acres, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
