# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb", "pandas", "pyarrow", "pytz"]
# ///
"""The burn perimeters of one year from the MTBS mirror on source.coop (cboettig/fire), joined to the WFIGS
ignition points by IRWIN id where there is one: data/ignition/perimeters_<year>.parquet (geometry as WKB,
EPSG:4326, with ignition date, acres, name, type, and the WFIGS discovery time, cause and point where
joined) and web/public/perimeters-<year>.geojson for the map (geometry simplified to about 30 m).

    uv run ignition/pull_perimeters.py --year 2024
"""
import argparse
import json
import sys
from pathlib import Path

import duckdb

D = Path(__file__).resolve().parent.parent
SRC = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/cboettig/fire/mtbs-perimeters-1984-2024.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    wf = D / "data" / "ignition" / "wfigs.parquet"
    con.execute(f"""
        CREATE TABLE p AS
        SELECT m.Event_ID AS event_id, m.irwinID AS irwin_id, m.Incid_Name AS name, m.Incid_Type AS type, m.BurnBndAc AS acres,
               m.Ig_Date AS ig_date, m.geom AS geom,
               w.FireDiscoveryDateTime AS discovered, w.FireCause AS cause, w.geom_lon AS lon, w.geom_lat AS lat
        FROM read_parquet('{SRC}') m
        LEFT JOIN read_parquet('{wf}') w ON w.IrwinID = '{{' || upper(m.irwinID) || '}}'
        WHERE year(m.Ig_Date) = {a.year}
    """)
    n, joined, acres = con.execute("SELECT COUNT(*), COUNT(discovered), SUM(acres) FROM p").fetchone()
    print(f"{a.year}: {n} perimeters, {joined} joined to a WFIGS point, {acres:,.0f} acres")
    print(con.execute("SELECT type, COUNT(*), SUM(acres) FROM p GROUP BY 1 ORDER BY 2 DESC").fetchall())
    print(con.execute("SELECT month(ig_date), COUNT(*) FROM p GROUP BY 1 ORDER BY 1").fetchall())
    out = D / "data" / "ignition" / f"perimeters_{a.year}.parquet"
    con.execute(f"COPY (SELECT * REPLACE (ST_AsWKB(geom) AS geom) FROM p) TO '{out}' (FORMAT PARQUET)")
    rows = con.execute("""
        SELECT event_id, name, type, acres, strftime(ig_date, '%Y-%m-%d'), discovered, cause,
               ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, 0.0003)) FROM p
    """).fetchall()
    feats = []
    for eid, name, typ, ac, igd, disc, cause, gj in rows:
        feats.append({"type": "Feature", "properties": {"id": eid, "name": name, "type": typ, "acres": ac, "ig_date": igd,
                      "discovered": disc.isoformat() if disc is not None else None, "cause": cause}, "geometry": json.loads(gj)})
    gout = D / "web" / "public" / f"perimeters-{a.year}.geojson"
    gout.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
    print(f"wrote {out} and {gout} ({gout.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
