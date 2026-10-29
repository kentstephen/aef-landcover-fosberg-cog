# /// script
# requires-python = ">=3.12"
# dependencies = ["geopandas", "shapely", "pyogrio"]
# ///
"""The CONUS counties for the map: the Census 2023 cartographic boundaries at 1:5m (cb_2023_us_county_5m),
the 48 states and DC, as web/public/counties.json with id (GEOID), name, state (postal), land area (km2) and
the geometry at four decimals.

    uv run scripts/build_counties.py <path to cb_2023_us_county_5m.zip>
"""
import json
import sys
from pathlib import Path

import geopandas as gpd

D = Path(__file__).resolve().parent.parent
SKIP = {"02", "15", "60", "66", "69", "72", "78"}  # Alaska, Hawaii, the territories


def main():
    src = sys.argv[1]
    g = gpd.read_file(f"zip://{src}")
    g = g[~g.STATEFP.isin(SKIP)].to_crs(4326)
    feats = []
    for r in g.itertuples():
        geom = r.geometry.simplify(0.002, preserve_topology=True)
        gj = json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"]
        def rnd(c):
            return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 4), round(c[1], 4)]
        gj["coordinates"] = rnd(gj["coordinates"])
        b = geom.bounds
        feats.append({"type": "Feature", "id": r.GEOID, "properties": {"name": r.NAME, "state": r.STUSPS, "km2": round(r.ALAND / 1e6, 1),
                      "bbox": [round(v, 3) for v in b]}, "geometry": gj})
    out = D / "web" / "public" / "counties.json"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
    print(f"wrote {out}: {len(feats)} counties, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
