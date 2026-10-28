# web: live FFWI x wildfire potential viewer

Browser app (Vite + React + deck.gl + `@developmentseed/deck.gl-raster`) that
computes the Fosberg Fire Weather Index from the dynamical NOAA HRRR 48-hour
forecast in the browser, reads USFS burn probability / hazard potential live
from the GeoPlatform ImageServer, and shows the two together as a per-hex
threat glow (BP x FFWI) over the animated FFWI raster.

```sh
npm install
npm run dev            # http://localhost:3000 (strictPort)
PORT=5371 npm run dev  # per-machine override
npm run build          # -> dist/ (base /fosberg-hrrr/)
npm run typecheck
npm run ffwi-check     # node parity check of ffwi/compute.ts against fixed inputs
```

No static data. Everything is fetched at runtime; see `../docs/`.
