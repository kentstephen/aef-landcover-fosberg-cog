# Data sources: verified facts (2026-08-21)

Everything here was probed with curl / python on 2026-08-21. Re-verify before
relying on anything older than a few months.

## HRRR forecast, 48 hour (dynamical.org)

| | |
|---|---|
| Icechunk v2 repo (AWS Open Data, CORS `*`) | `https://dynamical-noaa-hrrr.s3.us-west-2.amazonaws.com/noaa-hrrr-forecast-48-hour/v0.1.0.icechunk` |
| Zarr v3 mirror (deprecated upstream, still in sync) | `https://data.source.coop/dynamical/noaa-hrrr-forecast-48-hour/v0.1.0.zarr` |
| STAC | `https://stac.dynamical.org/noaa-hrrr-forecast-48-hour/collection.json` |
| Dims | `(init_time, lead_time, y, x)` = `(11847+, 49, 1059, 1799)` |
| Chunks / shards | inner `(1, 49, 265, 300)`, shard `(1, 49, 1060, 1800)`; bytes + blosc zstd-3 shuffle; sharding_indexed |
| Inits | every 6 h since 2018-07-13 12:00 UTC; `lead_time` float64 seconds 0..172800 |
| Grid | LCC sphere R=6371229, lat_0=38.5, lon_0=-97.5, sp 38.5/38.5; dx=dy=3000 m; x -2697520.14..2696479.86, y -1587306.15..1586693.85 (cell centers) |
| FFWI inputs | `temperature_2m` (degC), `relative_humidity_2m` (%), `wind_u_10m`, `wind_v_10m`, `wind_gust_surface` (m/s); fill NaN |
| License | CC BY 4.0 |

Why the other dynamical HRRR stores do not work in a browser:

- `noaa-hrrr-analysis` v0.2.0: chunks `(time=2160, y=45, x=45)`. One CONUS map
  is ~960 chunks x 17.5 MB uncompressed per variable. Time-series layout.
- `noaa-hrrr-forecast-18-hour-virtual` / `-48-hour-virtual`: map-shaped chunks
  `(1, 1, 1059, 1799)` but virtual references into GRIB2 on
  `s3://noaa-hrrr-bdp-pds` decoded by a `gribberish` zarr codec. zarrita has no
  GRIB codec. (The NOAA GRIB bucket itself does serve CORS.)

Fallback: MesoWest `s3://hrrrzarr` (zarr v2, CORS `*`, 2016-08-23 to present):
`sfc/YYYYMMDD/YYYYMMDD_HHz_{anl,fcst}.zarr/<level>/<var>/<level>/<var>/`,
anl chunks `(150, 150)`, fcst `(48|18, 150, 150)`, blosc lz4, float32 (TMP in K).
TMP/RH/UGRD/VGRD/GUST all present. Grid lat/lon at `grid/HRRR_chunk_index.zarr`.

## USFS Wildfire Risk to Communities via GeoPlatform ImageServer (live path for "potential")

Base: `https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/<service>/ImageServer`.
CORS: `Access-Control-Allow-Origin` echoes the request Origin. No key.
(`apps.fs.usda.gov/fsgisx01` and `/arcx` endpoints return 403.)

| service | content | pixel | range |
|---|---|---|---|
| `USFS_EDW_RMRS_WRC_WildfireHazardPotential` | WHP 2024 continuous index, 30 m, mosaic item `WHP_CONUS_reclassed` | U16 | 1..65535 |
| `USFS_EDW_RMRS_WRC_BurnProbability` | annual burn probability 2024, 30 m (scaled integer) | U16 | 0..1352 |
| `USFS_EDW_RMRS_WRC_RiskToPotentialStructures` | RPS 2024, 30 m | U16 | 0..1301 |
| `USFS_EDW_RMRS_WRC_ConditionalFlameLength` | CFL 2024, 30 m | U16 | 0..475 |
| `USFS_EDW_RMRS_WRC_FlameLengthExceedProb4ft` / `8ft` | FLEP | U16 | |
| `USFS_EDW_RMRS_WildfireHazardPotentialClassified` | WHP 2023 5-class, 270 m | U8 | 1..7 |
| `USFS_EDW_RMRS_WildfireHazardPotentialContinuous` | WHP continuous, 270 m | S32 | 0..144153 |

Also in the folder: `USFS_EDW_RMRS_ProbabilisticWildfireRisk_FLP1..6`, `..._BurnProbability`,
`USFS_QWRA_BurnProbability_CONUS`, `USFS_EDW_SAB_WildfireSuppressionDifficultyIndex*`,
population/building/housing layers, CWiRRZ.

Measured:
- `exportImage?f=image&format=lerc&pixelType=U16&bbox=<3857>&bboxSR=3857&imageSR=3857&size=W,H`
  CONUS 1800x1000: LERC 1.1 MB in 0.47 s, TIFF 2.7 MB in 0.58 s. 256 px tiles
  36-83 KB, ~0.2 s each, 4 in parallel 0.3 s wall. `interpolation=RSP_NearestNeighbor`
  and `noData=0` accepted. `f=json` returns an `href` to a server-side file (do not use).
- `getSamples` (multipoint): 5 points 1.3 s; 200 points 34 s and partial; caps near
  480 samples. Not usable for per-cell lookups.
- `legend?f=json` for WHP: Very Low / Low / Moderate / High / Very High.
- Rendering rule `{"rasterFunction":"WHP"}` via GET returned 400 in the probe
  (quoting); irrelevant, raw values are wanted.

## CarbonPlan Open Climate Risk (source.coop, CC BY 4.0, CORS `*`)

Bucket `us-west-2.opendata.source.coop`, prefix `carbonplan/carbonplan-ocr/`.
HTTPS hosts: `https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/<key>`
(path-style; CORS `*`) or `https://data.source.coop/carbonplan/carbonplan-ocr/<key>`.
Docs: https://open-climate-risk.readthedocs.io/en/latest/ (docs.carbonplan.org/ocr redirects).

| store | format | vars | shape / chunks | browser |
|---|---|---|---|---|
| `input/fire-risk/tensor/USFS/riley-et-al-2025/{2011,2047}-climate-run-30m-4326.icechunk` | icechunk v1 | `BP`, `FLP1..6`, 30 m 4326 (interp from 270 m) | `(97579, 208881)` f32, chunks `(6000, 4500)`, ~4 MB compressed / 108 MB decoded | no (ETL input) |
| `.../riley-et-al-2025/{2011,2047}-climate-run-270m-5070.icechunk` | icechunk v1 | same, native 270 m EPSG:5070 | `(11166, 17237)`, chunks `(6000, 4500)` | no |
| `output/fire-risk/tensor/production/v1.1.0/ocr.icechunk` | icechunk | `bp_2011`, `bp_2047`, `bp_2011_riley`, `bp_2047_riley`, `rps_2011/2047`, `crps_scott`, `rps_scott` | `(6000, 4500)` chunks | no |
| `output/fire-risk/pyramid/production/v0.13.2/pyramid.zarr` | zarr v2 ndpyramid, EPSG:3857, 512 px, levels 0-11, zlib | `wind_risk_2011`, `wind_risk_2047` only | web tiles | yes |
| `output/fire-risk/pyramid/qa/pyramid.zarr` | zarr v3, multiscales + `proj:code`, 4326, 362x362 chunks | `rps_2011`, `rps_2047` | web | yes (QA) |
| `input/fire-risk/tensor/conus404-ffwi/` | icechunk | `fosberg-fire-weather-index.icechunk` (hourly 1979-2022, 4 km), `-p95`, `-p99`, p99 wind-direction distribution/mode (+ 30m-4326 and reprojected variants), `winds.icechunk` | not probed | likely no |

No burn-probability web pyramid is published. Their FFWI/BP join (methods page):
FFWI hourly from CONUS404 per 4 km pixel; 99th percentile threshold; wind
directions on those hours binned to 8; elliptical filters (510 m major axis,
3 iterations, ~1.5 km) spread BP downwind into non-burnable land;
`RPS = BP x cRPS` (Scott et al. 2024, 30 m); 0-10 score by percentile bins.
`wind_risk` in the production pyramid is that score.

## FPA FOD ignition points (USFS RDS-2013-0009.7, 7th edition)

Short, Karen C. 2026. Spatial wildfire occurrence data for the United States,
1992-2024 [FPA_FOD_20260615]. 7th Edition. https://doi.org/10.2737/RDS-2013-0009.7

- ~2.66 M records, 1992-2024, NAD83 lat/lon, public domain.
- Only distribution: zips under `https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.7/`:
  `..._Data_Format3_GPKG.zip` (238 MB, contains `Data/FPA_FOD_20260615.gpkg` 1.08 GB),
  `..._Data_Format4_SQLITE.zip` (220 MB), GDB (158 MB), ACCDB (205 MB). HTTP range
  works, no CORS. No API, no CSV. Not fetchable from a browser.
- Key fields: `FOD_ID`, `FIRE_YEAR`, `DISCOVERY_DATE` (mm/dd/yyyy), `DISCOVERY_DOY`,
  `DISCOVERY_TIME` (hhmm, often missing), `NWCG_CAUSE_CLASSIFICATION`,
  `NWCG_GENERAL_CAUSE` (13), `FIRE_SIZE`, `FIRE_SIZE_CLASS` (A..G), `LATITUDE`,
  `LONGITUDE`, `STATE`, `COUNTY`, `OWNER_DESCR`.
- If ever used: offline bucket to H3 res 6 (DuckDB `h3` + `spatial` read the GPKG
  directly), ship a ~2-4 MB snappy parquet, read with hyparquet.

## Libraries (npm versions seen 2026-08-21)

deck.gl 9.3.10; @developmentseed/deck.gl-raster / deck.gl-zarr / deck.gl-geotiff
0.7.0 (0.8.0-beta.2 on main; peer deck.gl ^9.3, luma.gl ^9.3); zarrita 0.7.4;
icechunk-js 0.6.0 (pure TS, zarrita AsyncReadable, v1/v2 auto-detect, sharded
getRange); @earthmover/icechunk 2.0.3 (WASM, needs vite wasm plugins); h3-js
4.5.0; hyparquet 1.29.1; lerc 4.2.0 (Esri LERC decoder); @carbonplan/zarr-layer
0.9.0 (MapLibre CustomLayer, depends on @developmentseed/raster-reproject).
