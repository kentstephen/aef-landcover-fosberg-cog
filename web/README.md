# web: the fire risk map

Browser app (Vite + React + MapLibre + deck.gl + `@developmentseed/deck.gl-raster`). It reads the AlphaEarth
embeddings (COG overviews zoomed out, the 10 m Zarr mosaic zoomed in), runs the ignition head per pixel,
computes the Fosberg index and the head's weather term from the HRRR forecast in a worker, and draws the hour's
risk. See `../README.md`.

```sh
npm install
PORT=5372 npm run dev  # http://localhost:3000 by default (strictPort)
npm run build          # -> dist/
npm run typecheck
npm run ffwi-check     # node parity check of ffwi/compute.ts against fixed inputs
```

The only static data are the model weights, the COG index (`public/aef-2025-*`) and the county shapes
(`public/counties.json`). Everything else is fetched at runtime; see `../docs/data-sources.md`.

Where things live:

- `src/hrrr/week.ts`, `week.worker.ts`: the window of daily runs, fetched shard by shard under the view.
- `src/aef/`: the readers (`coarse.worker.ts`, `aef.worker.ts`), the heads (`head.ts`), the per-tile weather
  textures (`weather.ts`), the shader (`render.ts`), the ramps (`ramp.ts`).
- `src/counties.ts`: the county averages.
- `src/ui/ControlPanel.tsx`: the panel.
