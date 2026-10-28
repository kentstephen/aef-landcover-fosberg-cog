# aef-landcover-fosberg-cog

A browser-only fire risk map. Land cover is read from **AlphaEarth Foundations
embeddings** (64 bands, 10 m, public COGs and a Zarr mosaic on source.coop) by a
tiny linear model fit offline; the **Fosberg Fire Weather Index** is computed in
the browser from the NOAA HRRR 48 h forecast. The map lights the land cover in
viridis by the hour's fire weather and leaves calm areas transparent, so open
fuels come up as the forecast worsens. The classified land cover and the raw
weather field are each a toggle for checking.

**Status: working first build.** The map opens on the Marshall Fire day
(2021-12-30) over Boulder County. Zoomed out (zoom 9 to 12) the embeddings come
from the AEF COG overviews, one file per 82 km UTM tile; zoomed in from the
10 m Zarr mosaic. Every tile is classified once in a worker; the drawing is a
per-fragment lookup (class, fuel weight, the hour's FFWI). Fuel weights per
class are a first cut (`web/src/aef/fuel.ts`). See `CLAUDE.md`.

*Claude wrote the code and documentation under my supervision.*

## Model

```sh
uv run scripts/sample_landcover.py     # AEF 2025 at 40 m + NLCD 2024, Esri IO 2024, Meta canopy per tile
uv run scripts/fit_landcover_head.py   # logistic heads (NLCD, Esri) + canopy ridge -> models/*.json, previews
```

Features are the dequantized, unit-normalized embedding. AEF overview pixels are
unit-length centroids of their children, so the same head serves 10 m pixels and
every zoomed-out level.

## Web

```sh
cd web && npm install && npm run dev   # http://localhost:3000 (PORT=... to change)
uv run --with duckdb scripts/build_aef_index.py   # web/public/aef-2025-conus.json (the COG tile index)
npm run build                          # -> web/dist
npm run typecheck
```

## Data

- AlphaEarth Foundations embeddings, tge-labs on source.coop (CC BY 4.0).
- Annual NLCD 2024 (USGS, public domain), source.coop mirror.
- Esri / Impact Observatory 10 m annual land cover (CC BY 4.0).
- Meta / WRI canopy height v1 (CC BY 4.0).
- NOAA HRRR 48-hour forecast, dynamical.org (CC BY 4.0).
- Basemap CARTO dark matter, OpenStreetMap contributors.
