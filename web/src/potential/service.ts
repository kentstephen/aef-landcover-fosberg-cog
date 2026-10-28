import * as Lerc from "lerc";
import lercWasmUrl from "lerc/lerc-wasm.wasm?url";

/**
 * USFS Wildfire Risk to Communities layers, read live from the federal
 * GeoPlatform ArcGIS ImageServer (CORS echoes Origin, no key). Verified
 * 2026-08-21, see docs/data-sources.md. `exportImage` with raw U16 LERC is
 * fast (CONUS 1800x1000 in ~0.5 s); `getSamples` is slow and capped, so all
 * per-cell lookups sample the exported raster client-side.
 */
const BASE =
  "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation";

export type PotentialLayerId = "bp" | "whp";

export type PotentialLayerDef = {
  id: PotentialLayerId;
  service: string;
  label: string;
  shortLabel: string;
  /** Raw U16 -> display value. */
  scale: number;
  unit: string;
  /** Display-unit range for the gate slider and the ramp. */
  displayMax: number;
  sliderStep: number;
  format: (display: number) => string;
  about: string;
};

export const POTENTIAL_LAYERS: Record<PotentialLayerId, PotentialLayerDef> = {
  bp: {
    id: "bp",
    service: "USFS_EDW_RMRS_WRC_BurnProbability",
    label: "Burn Probability (annual, WRC 2024)",
    shortLabel: "Burn probability",
    // U16 0..1352; a max of 13.5 %/yr is the only plausible reading of that
    // range, so 1e-4. Treat as an assumption until confirmed against the
    // published GeoTIFF.
    scale: 1e-4,
    unit: "%/yr",
    displayMax: 0.05,
    sliderStep: 0.0005,
    format: (v) => `${(v * 100).toFixed(2)} %/yr`,
    about:
      "Annual probability a pixel burns (USFS FSim, Wildfire Risk to Communities 2024, 30 m).",
  },
  whp: {
    id: "whp",
    service: "USFS_EDW_RMRS_WRC_WildfireHazardPotential",
    label: "Wildfire Hazard Potential (index, WRC 2024)",
    shortLabel: "Hazard potential",
    scale: 1,
    unit: "index",
    displayMax: 20000,
    sliderStep: 100,
    format: (v) => `${Math.round(v)}`,
    about:
      "Relative potential for high-intensity fire that is hard to control (USFS WHP 2024, 30 m, unitless index).",
  },
};

export type PotentialRaster = {
  layer: PotentialLayerId;
  width: number;
  height: number;
  /** Raw values, row-major from the top. */
  values: Uint16Array | Float32Array | Int32Array | Uint8Array | Int16Array;
  /** 1 = valid, 0 = nodata. May be null when everything is valid. */
  mask: Uint8Array | null;
  /** Web-mercator bbox [xmin, ymin, xmax, ymax] in metres. */
  bbox: [number, number, number, number];
  pxW: number;
  pxH: number;
  fetchedAt: number;
};

let lercReady: Promise<void> | null = null;
function ensureLerc(): Promise<void> {
  if (!lercReady) {
    lercReady = Lerc.load({ locateFile: () => lercWasmUrl });
  }
  return lercReady;
}

const R = 6378137;
export function lonLatToMercator(lon: number, lat: number): [number, number] {
  const x = (lon * Math.PI * R) / 180;
  const latR = Math.max(-85.05112878, Math.min(85.05112878, lat)) * (Math.PI / 180);
  const y = R * Math.log(Math.tan(Math.PI / 4 + latR / 2));
  return [x, y];
}

/**
 * Export the raw raster for a web-mercator bbox at the given pixel size.
 */
export async function fetchPotential(
  layer: PotentialLayerId,
  bbox: [number, number, number, number],
  width: number,
  height: number,
  signal?: AbortSignal,
): Promise<PotentialRaster> {
  await ensureLerc();
  const def = POTENTIAL_LAYERS[layer];
  const params = new URLSearchParams({
    f: "image",
    format: "lerc",
    pixelType: "U16",
    noData: "0",
    interpolation: "RSP_NearestNeighbor",
    bbox: bbox.join(","),
    bboxSR: "3857",
    imageSR: "3857",
    size: `${width},${height}`,
  });
  const url = `${BASE}/${def.service}/ImageServer/exportImage?${params}`;
  const resp = await fetch(url, { signal });
  if (!resp.ok) throw new Error(`exportImage ${def.service}: HTTP ${resp.status}`);
  const buf = await resp.arrayBuffer();
  const decoded = Lerc.decode(buf);
  const values = decoded.pixels[0]!;
  return {
    layer,
    width: decoded.width,
    height: decoded.height,
    values: values as PotentialRaster["values"],
    mask: decoded.mask,
    bbox,
    pxW: (bbox[2] - bbox[0]) / decoded.width,
    pxH: (bbox[3] - bbox[1]) / decoded.height,
    fetchedAt: Date.now(),
  };
}

/** Display-unit value at lon/lat, or NaN outside / nodata. */
export function samplePotential(
  r: PotentialRaster,
  lon: number,
  lat: number,
): number {
  const [mx, my] = lonLatToMercator(lon, lat);
  const col = Math.floor((mx - r.bbox[0]) / r.pxW);
  const row = Math.floor((r.bbox[3] - my) / r.pxH);
  if (col < 0 || col >= r.width || row < 0 || row >= r.height) return NaN;
  const i = row * r.width + col;
  if (r.mask && r.mask[i] === 0) return NaN;
  return (r.values[i] ?? 0) * POTENTIAL_LAYERS[r.layer].scale;
}

/**
 * Mean display-unit potential over a square window approximating a hex of
 * the given edge length (metres), centred on lon/lat. Nodata pixels count as
 * 0 (non-burnable / water); NaN only if the window is entirely off-raster.
 */
export function samplePotentialMean(
  r: PotentialRaster,
  lon: number,
  lat: number,
  edgeM: number,
): number {
  const [mx, my] = lonLatToMercator(lon, lat);
  const cx = Math.floor((mx - r.bbox[0]) / r.pxW);
  const cy = Math.floor((r.bbox[3] - my) / r.pxH);
  const mercPerGround = 1 / Math.max(0.2, Math.cos((lat * Math.PI) / 180));
  const rad = Math.min(24, Math.max(0, Math.round((edgeM * 0.85 * mercPerGround) / r.pxW)));
  const scale = POTENTIAL_LAYERS[r.layer].scale;
  let sum = 0;
  let count = 0;
  for (let y = cy - rad; y <= cy + rad; y++) {
    if (y < 0 || y >= r.height) continue;
    for (let x = cx - rad; x <= cx + rad; x++) {
      if (x < 0 || x >= r.width) continue;
      const i = y * r.width + x;
      count++;
      if (r.mask && r.mask[i] === 0) continue;
      sum += (r.values[i] ?? 0) * scale;
    }
  }
  return count === 0 ? NaN : sum / count;
}
