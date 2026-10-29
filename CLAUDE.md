# CLAUDE.md

Guidance for Claude Code in this repository.

## NO EM DASHES

Never use em dashes or en dashes in anything written for this project: prose,
code comments, commit messages, docs, UI copy. Use a comma, period, parentheses,
or colon, or restructure the sentence. Replace any you encounter while editing.
(`hrrr-fosberg.md` predates this rule and still contains some.)

## Purpose

A browser-only fire risk map built from two live cloud datasets and one tiny
model:

- **AlphaEarth Foundations (AEF) embeddings** (tge-labs on source.coop, 64
  bands int8, 10 m, 2017 to 2025) give the fuel: a small linear head, fit
  offline in Python, reads land cover from the embedding of each pixel.
- **The Fosberg Fire Weather Index (FFWI)**, computed in the browser from the
  NOAA HRRR 48 h forecast (dynamical.org Icechunk store), gives the hour's
  conditions.

The picture: the land cover, lit in viridis by the hour's FFWI, fully
transparent where conditions are calm, so open fuels come up as the forecast
worsens. The classified land cover can also be shown as land cover (a toggle,
for checking the model), and the raw FFWI field is its own viridis layer (a
toggle, plus hold the space bar to peek at it). No transmission lines for now.
No Google Earth Engine, ever. No server, no preprocessing beyond the model
weights; static site.

Order of work: Python first (the model, checked by looking at its land cover
maps), then the TypeScript map.

## Status (2026-10-28, week mode)

- The map plays a WEEK by default: one 12Z HRRR run per day, its first 24 hours, stitched hour by hour
  (`web/src/hrrr/week.ts`), opening on Los Angeles for January 5 to 11, 2025. The 48 h single run is the
  other time mode. Season playback is not built (the analysis store is time-chunked; see HANDOFF.md).
- Panel: Ignition / Off; the weather field fold, the land cover modes and grain are commented out.

## Status (2026-10-28, map built, reader rebuilt)

- `web/src/aef/` draws the ground lit by the hour's FFWI. Two readers, one head, one shader:
  - zoomed out (below zoom 12): a Web Mercator `RasterTileLayer` (`layers.ts`, affine tileset
    of XYZ levels) whose tiles come from `coarse.worker.ts`: per COG under the tile, the chosen
    overview level is ONE byte range (from the index), each band's zstd block decoded only as far
    as the rows the level uses, classified once per file and level (class planes kept, LRU
    256 MB per worker, embeddings dropped), resampled onto the tile by nearest pixel through
    UTM. Workers are assigned by longitude band so a file is decoded once. Measured over
    CONUS at zoom 4.3: cold 76 s (2153 range requests, 80 MB), warm 16 s (disk cache).
  - zoomed in (zoom 12+): the 10 m Zarr mosaic (`ZarrLayer`, `aef.worker.ts`).
  - `render.ts` shader: class -> color and fuel weight from a lookup texture, FFWI from a
    per-tile weather texture (49 leads, bilinear over HRRR pixels, `weather.ts`).
- Index: `scripts/build_aef_index.py` (DuckDB + tifffile, 8 threads, about 2 min) writes
  `web/public/aef-2025-conus.json` (1982 files, bounds, UTM origin) and one
  `aef-2025-ranges-<zone>.bin` per UTM zone (per file: the span of each coarse level, widths
  1024 to 16 px, and each band's block offset and length inside it; sparse blocks are 0, 0).
- HRRR shards (the Fosberg inputs) are kept on disk too (`hrrr/shard-disk.ts`), so a run looked
  at again does not refetch 440 MB.
- Panel: "Lit by FFWI" and "Off" only; the land cover classes mode and legend are commented out
  in `ControlPanel.tsx`. The HRRR field stays as "The weather" fold. Opens on the Marshall Fire
  (2021-12-30 12Z run, +6 h); the URL hash holds the view.
- Fuel weights per class (`web/src/aef/fuel.ts`) are a first cut, not tuned.
- Headless checks in `.claude/memory/`: `probe_tiles.py` (tiles, histograms, tooltip,
  screenshot), `probe_map.py` (requests by host, console), `probe_warm.py` (persistent profile,
  times "all tiles"), `probe_deck.py` (deck layer state).

## Model

- Features: `sign(q) * (q / 127.5)^2`, unit-normalized, then standardized
  with the mean and scale stored in the JSON. Overview pixels of the AEF COGs
  are unit-length centroids of their children, so one head serves 10 m pixels
  and every overview level.
- Heads: multinomial logistic 64 -> NLCD classes; 64 -> Esri IO classes;
  ridge 64 -> canopy cover, mean height, p95 height. All linear: a few vec4
  uniforms in a shader.
- Training cells are pure (at least 12 of the 16 children carry the majority
  class). Holdout tiles are never fit on; their accuracy is the honest number.
- Labels are 2024 (Esri IO has no 2025), embeddings are 2025.

## Data (all public, no tokens)

- AEF COGs: `https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/<year>/<zone>/<id>.tiff`
  (index `v1/annual/aef_index.parquet`, DuckDB). 8192 px, 10 m, 64 bands int8,
  nodata -128, stored south-up, overviews 2..8192. The S3 host, not the
  `data.source.coop` gateway, for many requests.
- AEF Zarr mosaic `tge-labs/aef-mosaic` (10 m, no pyramid): the close-in
  browser reader, copied from `../aef-tessera-overture/seasons-widget/src/`.
- Annual NLCD 2024: `kylebarron/usgs-landcover/annual-nlcd/c1/v1/cu/mosaic/Annual_NLCD_LndCov_2024_CU_C1V1.tif`
  on the same bucket (30 m Albers, Byte, nodata 250, COG).
- Esri / Impact Observatory LULC: `s3://io-10m-annual-lulc/{zone}{band}_{year}.tif`.
- Meta canopy v1 downsampled: `s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float_epsg4326_v3_10deg/`
  (uint16; heights in cm, cover as fraction x 1000; Python only, no CORS).
- HRRR 48 h forecast: dynamical.org Icechunk on AWS Open Data; see
  `docs/data-sources.md` and `hrrr-fosberg.md`.

## Commands

```sh
cd web && npm install && PORT=5372 npm run dev   # 5371 and 3000 are taken on this machine
npm run build && npm run typecheck
uv run --with duckdb --with tifffile --with fsspec --with aiohttp --with requests scripts/build_aef_index.py   # the COG index + byte ranges

uv run scripts/sample_landcover.py              # data/tiles/<site>.npz, one tile at a time
uv run scripts/fit_landcover_head.py            # models/*.json, data/previews/*.png
```

## Rules

- Protan-safe colors: never red vs green, never red alone. Green luminance
  ramps are fine. Land cover palettes in `fit_landcover_head.py` are the
  reference (greens for trees, purples for built, tans and yellows for open
  fuels, blues and teals for water and wetlands).
- Maps open full screen. No dark casing on strokes. American spelling.
- Do not coarsen resolution or change what gets read without asking.
- One reader at a time for remote COGs in Python; estimate peak RAM before a
  heavy job and say the number.
- Project memory and handoff notes live in `.claude/memory/` (gitignored).
