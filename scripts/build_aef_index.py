"""Write the browser's AEF tile index: every 2025 COG that touches CONUS, with its WGS84 bounds.

Reads tge-labs' aef_index.parquet with DuckDB and writes web/public/aef-2025-conus.json:
{"year": 2025, "base": "<S3 https prefix>", "tiles": [[ "<zone>/<file>", west, south, east, north ], ...]}
The S3 host serves many small requests without the gateway's throttling.
"""
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
INDEX = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
BASE = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2025/"
CONUS = (-125.0, 24.0, -66.9, 49.6)
OUT = ROOT / "web" / "public" / "aef-2025-conus.json"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
rows = con.execute(
    f"""
    select path, wgs84_west, wgs84_south, wgs84_east, wgs84_north
    from read_parquet('{INDEX}')
    where year = 2025 and wgs84_east > {CONUS[0]} and wgs84_west < {CONUS[2]}
      and wgs84_north > {CONUS[1]} and wgs84_south < {CONUS[3]}
    order by path
    """
).fetchall()
tiles = []
for path, w, s, e, n in rows:
    rel = path.split("/annual/2025/", 1)[1]
    tiles.append([rel, round(w, 4), round(s, 4), round(e, 4), round(n, 4)])
OUT.write_text(json.dumps({"year": 2025, "base": BASE, "tiles": tiles}, separators=(",", ":")))
print(f"{len(tiles)} tiles -> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
