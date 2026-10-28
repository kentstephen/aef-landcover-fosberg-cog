# Plan: fully live FFWI forecast with a live potential layer, joined per H3 cell

Written 2026-08-21 before any app code; v1 built the same day (see CLAUDE.md for the as-built layout). One deviation: FFWI is computed on the CPU per tile and shared by raster and cells, so there is no GLSL twin. Preference: no preprocessing
unless unavoidable. Template: `../zarr-road-risk`.

## Three live sources, one browser

1. **FFWI raster** from the dynamical HRRR 48-hour store. `ZarrLayer`
   (`@developmentseed/deck.gl-zarr`) renders the native LCC grid reprojected on
   the GPU. Five bands uploaded per tile as `Texture2DArray` (49 lead hours
   each); a GLSL FFWI module writes the index; FilterRange hides low values;
   LinearRescale + Colormap. Lead hour is a uniform, so animation never refetches.
2. **Potential raster** from the GeoPlatform ImageServer (`exportImage`, raw U16
   LERC). Default Burn Probability, toggle to WHP. Fetched per viewport on
   debounced moveend at roughly screen resolution (cap 1024 px), decoded with
   `lerc`, kept as a Float32Array + affine in 3857. Optionally drawn as a
   BitmapLayer with its own ramp; always used for the join.
3. **H3 join, live**: `h3.polygonToCells(viewport, res)` with res by zoom (res 6
   finest). For each cell centroid: HRRR pixel via proj4 forward LCC
   (`floor((x - x0)/3000)`, rows from the top edge), sample the five bands from
   the side-channel chunk cache at the current lead, compute FFWI in TS; sample
   the potential raster at the centroid (3857 pixel); combine; color an
   `H3HexagonLayer`, pickable with a tooltip (FFWI, wind mph, RH, T, BP/WHP,
   lead hour, init time).

FFWI lives in two places that must stay in sync: `web/src/gpu/ffwi.ts` (GLSL)
and `web/src/ffwi/compute.ts` (TS). `scripts/verify_ffwi.py` is the Python
parity check (uv, icechunk).

## Join expressions (v1 = first; others are follow-ons)

- Gate: hex drawn in FFWI color only where potential >= floor (slider).
- Product: `BP * f(FFWI)`.
- Anomaly: `FFWI_today / FFWI_p99_cell` against CarbonPlan's CONUS404 p99
  climatology (needs a per-cell p99 asset; their store is not browser-shaped).
- Bivariate: potential class x FFWI lightness, protan-safe.

## Visual rules

- Render FFWI itself, not a proxy band. Hide FFWI below a threshold so the
  basemap shows where there is no danger. Quantized bands are an option.
- Ramp: luminance-monotone with a bright hot end (inferno / magma / thermal /
  hot / afmhot; viridis / cividis also fine). No red-vs-green meaning anywhere.
  Potential layer, if drawn, in a different hue family (blues / cool) so the
  two rasters do not merge.
- Basemap and ramp chosen together (dark + luminous, or light muted + dark).
- Nearest vs linear texture filtering for the 3 km pixels: decide, do not
  inherit by accident.
- Discrete lead stepping; no cross-dissolve between frames.

## Performance rules (from road-risk)

- One shard fetch per viewport tile per band delivers all 49 leads.
- Side channel: unique shards covering the cells in view, 5 bands each,
  progressive callbacks. Shard shape `(265, 300)` is hardcoded (zarrita only
  exposes the outer chunk).
- Debounce viewport-driven fetches (zarr tiles and exportImage alike).
- Hex count cap per frame; res by zoom keeps it in the low tens of thousands.

## Out of v1

- FPA FOD ignitions (would be the one static parquet).
- Init-time picker / retrospective mode (archive back to 2018-07-13).
- Gust as an alternate wind input (separate label, not "FFWI").
- Any CarbonPlan climatology join.
