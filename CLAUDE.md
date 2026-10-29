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

## Status (2026-10-29, counties branch)

- Branch `counties`: a "Counties" ground mode, the pixel picture averaged by county live in the browser
  (`web/src/counties.ts`, `docs/counties.md`): the coarse workers sum exp(ground logit) per county and
  HRRR pixel over all 1,982 files at the 32 px level (109 s cold), the hour's weather is folded in per
  county, drawn as a GeoJsonLayer with the ground's ramp and scale, a ranked table of the counties in
  view in the panel. County shapes: `scripts/build_counties.py` -> `web/public/counties.json`. The
  number driving the shading (rate per area now) is open.

## Status (2026-10-29, after the calibration check)

- Opens on August 7 to 9, 2026 (3 days; the full week took too long to arrive). The time panel offers one window of up to 7 days; the single 48 h run mode is no longer offered (code path kept, timeMode "run").
- Tooltip reads the head's logit in real units (`web/src/aef/calibration.ts`: the offset from
  `models/ignition_calibration.json`, times the class's observed/expected ratio): "about 0.02 per
  1000 km2 per day, 2.2x the CONUS mean". The picture (ramp, gate) does not use the calibration.
- Burn side done (`ignition/burn_head.py`, `docs/burn-duration.md`, `models/burn_duration.json`): the
  embedding beats the NLCD class on duration and size (log size R2 0.28 vs 0.13; 100-acre AUC 0.81 vs 0.72).
  The class medians of hours from discovery to out (2 to 4 days, an agency clock) were tried as the hold
  hours and glued the picture; reverted to the fuel-lag first cut. Then `ignition/hour_memory.py`
  (`docs/hour-memory.md`): at the same point the past day's peak FFWI adds nothing to this hour's
  ignition chance (weight slightly negative), rain in the past 72 h subtracts. The afterglow now
  defaults to OFF; the hour is the signal. Hourly swing at a point is about half the spread across points.
- Ramps evened on lightness (`ramp.ts` evenLightness: mean of normal and protan L*, linear from each
  ramp's first stop to its last) so one slider position is the same brightness on every ramp.
- Nothing is written to disk for HRRR anymore: `hrrr/shard-disk.ts` is gone; the run mode holds its
  shards in memory like the week.

## Status (2026-10-28, streaming week)

- The map plays a WEEK by default (`web/src/hrrr/week.ts`): one 12Z HRRR run per day, its first 24 hours,
  stitched hour by hour, opening country-wide on March 10 to 16, 2026 over the 2025 embeddings. Shard-days
  are one queue ranked by the playhead (the day on screen first, then the days after it), 4 in flight; play runs over the days in hand and
  wraps there until the rest arrive; a strip of day cells under the slider shows in / loading / waiting.
  Only the shards under the view are fetched; the week writes nothing to disk (the raw shards are 23 MB
  each; writing a country-wide week put 4 GB in the browser process). Measured cold headless (one tab, all
  Chromium processes): a region (Oklahoma, zoom 6.8) all 7 days in 30 s, 2.1 GB; country-wide first days in
  12 s, all in 97 s, 2.7 GB peak 2.95 (was 6.9 GB). The app's floor before any week is about 2 GB (the
  coarse reader's workers hold up to 256 MB of class planes each). The 48 h single run is the other time
  mode (its shards stayed in memory only from 2026-10-29). Nothing is precomputed but the
  model weights; the season mode and its baked shards are gone.
- Memory probes: `.claude/memory/mem2.py <url> <seconds>` (RSS per Chromium process, heap, the week's
  queue), `.claude/memory/probe_week.py` (days in hand, tiles, tooltip, screenshot).
- Opens on July 26 to August 1, 2026 (ranked by `ignition/rank_weeks.py`: the head's weather term over a
  sample of the HRRR analysis, the week with the most area-hours above the gate). Ground ramps: Inferno
  (default), Viridis, Magma, Solar, Fire (`aef/ramp.ts`, 50 stops each, ends tuned to the dark map). The
  hour's weather term also draws as a gray field under everything (`fieldAlpha`), and the ground holds a past
  hour by land cover (hold hours in `aef/fuel.ts`: grass 3 h, shrub 6, evergreen 24, built 0). Keys: Space,
  F full screen, L basemap labels, Q tooltip. The collapsed card shows the hour and a ramp legend.
- Calibration check done (`ignition/calibrate.py`, `docs/ignition-calibration.md`): the head ranks pixel-hours well
  (top 10% hold 62% of WFIGS ignitions) but its rates are off by region and class (Mountain 2.4x and evergreen 2.9x
  under, cropland and dense built 4x over), since local controls cancel the base rate. Offset -14.84 turns the logit
  into ignitions per km2 per hour (`models/ignition_calibration.json`). The map is unchanged by it.
- Next: the same app in a marimo notebook with a Python kernel reading the HRRR analysis (virtual store)
  as another source of days; see `.claude/memory/PLAN_runtime_week.md` Phase B. A calibration plan for the
  head's probabilities is in the handoff (not started).
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
