# aef-landcover-fosberg-cog

**Live map: https://kentstephen.github.io/aef-landcover-fosberg-cog/** (static site, built from `main` by GitHub Actions; everything is read from the public cloud stores in the browser)

**Work in progress.** This is a demonstration, still being worked on, especially its speed. Everything is read
and computed in your browser, so it is heavy: expect your computer's fans while it loads and plays. Space pauses
playback; zooming in on a region loads far less than the whole country.

A browser-only map of where wildfires are more or less likely to start, hour by hour. Every pixel's
**AlphaEarth Foundations embedding** (64 bands, 10 m, public COGs and a Zarr mosaic on Source Cooperative)
goes through a small logistic head together with the hour's weather from the **NOAA HRRR forecast**
(the Fosberg Fire Weather Index, temperature, humidity and wind, computed in the browser). The head was fit
offline on 22,204 WFIGS wildfire ignitions (2019 to 2026, known cause) against control pixel-hours 20 to
30 km away at other hours. It is a risk outlook, not a detection: nothing on the map is an actual fire.

What the map shows:

- **The weather field**: the weather part of the head's score, drawn in gray under everything.
- **The ground**: the land, faint all the time and lit on a color ramp (Inferno by default) where the hour's
  weather lifts its score.
- **Counties**: the same picture averaged by county (the G key switches between pixels and counties).

The map plays a window of up to 7 days of daily 12Z HRRR runs (the first 24 hours of each, stitched hour by
hour). It opens country-wide on August 7 to 9, 2026; the date fields pick any window in the store, and
"Current forecast" jumps to the latest run. Only the forecast under the map's bounds is fetched: zoomed in on
a region, that region loads in seconds; the whole country takes about a minute for three days. The forecast is
decoded in a worker and kept in memory as a byte per pixel and hour; nothing is written to disk.

Keys: Space plays and pauses, G pixels or counties, F full screen, L basemap
labels, Q tooltip.

*Claude wrote the code and documentation under my supervision.*

## How it reads the embeddings

- Zoomed out (below zoom 12): the AEF COG overviews, one byte range per file and level (from an index built
  once by `scripts/build_aef_index.py`), decoded only as far as the level's rows, classified once per file,
  resampled onto Web Mercator tiles in workers.
- Zoomed in (zoom 12+): the 10 m Zarr mosaic.
- AEF overview pixels are unit-length centroids of their children, so one linear head serves 10 m pixels and
  every overview level. Drawing is a per-fragment lookup (the pixel's ground score, the hour's weather term).

## Models

```sh
# land cover heads (the tooltip's classes): AEF 2025 + NLCD 2024, Esri IO 2024, Meta canopy per tile
uv run scripts/sample_landcover.py
uv run scripts/fit_landcover_head.py        # -> models/landcover_*.json, models/canopy_ridge.json

# the ignition head: WFIGS cases and controls, their embeddings and HRRR analysis weather
uv run ignition/pull_wfigs.py
uv run ignition/make_cases.py
uv run ignition/sample_embeddings.py
uv run ignition/sample_weather.py
uv run ignition/export_head.py              # -> models/ignition_logit.json
uv run ignition/calibrate.py                # -> models/ignition_calibration.json
```

Write-ups in `docs/`: `ignition-calibration.md` (the head ranks pixel-hours well, its rates are off by region
and class; the tooltip's per-area rate comes from here), `hour-memory.md` (the past day's fire weather adds
nothing to this hour's chance; recent rain subtracts), `burn-duration.md`, `counties.md`, `data-sources.md`.

## Web

```sh
cd web && npm install && npm run dev   # http://localhost:3000 (PORT=... to change)
npm run build                          # -> web/dist
npm run typecheck
uv run --with duckdb --with tifffile --with fsspec --with aiohttp --with requests --with pyproj scripts/build_aef_index.py   # the COG index + byte ranges
uv run scripts/build_counties.py       # web/public/counties.json
```

## Data

- AlphaEarth Foundations embeddings, tge-labs on Source Cooperative (CC BY 4.0).
- NOAA HRRR 48-hour forecast and analysis, dynamical.org (CC BY 4.0).
- NIFC WFIGS incident locations and perimeters (public domain), to fit and check the model.
- Annual NLCD 2024 (USGS, public domain), Source Cooperative mirror.
- Esri / Impact Observatory 10 m annual land cover (CC BY 4.0).
- Meta / WRI canopy height v1 (CC BY 4.0).
- Census 2023 cartographic county boundaries (public domain).
- Basemap CARTO dark matter, OpenStreetMap contributors.
- Raster tiles drawn with deck.gl-raster (Development Seed).
