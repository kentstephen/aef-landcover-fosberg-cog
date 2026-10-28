import type { Device, Texture } from "@luma.gl/core";
import { toHalf } from "../hrrr/get-tile-data.js";
import { HRRR_LEAD_TIME_COUNT, lonLatToHrrrPixelF } from "../hrrr/metadata.js";
import type { FfwiTileCache } from "../hrrr/tile-cache.js";

/**
 * The hour's weather on a land cover tile: a grid of nodes over the tile (about one per 800 m, at
 * least 4 x 4), each the FFWI at its spot for all 49 lead hours, read from the shared cache as a
 * bilinear blend of the four HRRR pixels around it (the pixel is 3 km; blending, rather than taking
 * the nearest pixel, keeps the field continuous across tile edges and inside a tile). Uploaded as one
 * r16float texture array the shader samples at the fragment's uv; rebuilt when the cache changes
 * (a shard arrives, the fuel lag changes); the lead hour is a uniform.
 */
export type WeatherGrid = {
  cols: number;
  rows: number;
  /** Fractional HRRR pixel coordinates per node, row-major; NaN outside the grid. */
  fx: Float32Array;
  fy: Float32Array;
  /** The integer pixel under each node (for the side channel), -1 outside. */
  px: Int16Array;
  py: Int16Array;
};

/** Nodes at cell centers; `toLonLat` maps tile fractions (u right, v down) to lon/lat. */
export function makeGrid(extentM: number, toLonLat: (u: number, v: number) => [number, number]): WeatherGrid {
  const n = Math.max(4, Math.min(64, Math.round(extentM / 800)));
  const fx = new Float32Array(n * n);
  const fy = new Float32Array(n * n);
  const px = new Int16Array(n * n);
  const py = new Int16Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const k = j * n + i;
      const [lon, lat] = toLonLat((i + 0.5) / n, (j + 0.5) / n);
      const p = Number.isFinite(lon) && Number.isFinite(lat) ? lonLatToHrrrPixelF(lon, lat) : null;
      fx[k] = p ? p.x : NaN;
      fy[k] = p ? p.y : NaN;
      px[k] = p ? Math.floor(p.x) : -1;
      py[k] = p ? Math.floor(p.y) : -1;
    }
  }
  return { cols: n, rows: n, fx, fy, px, py };
}

/** What a weather texture holds per node and lead: the FFWI, or the weather part of the ignition logit. */
export type WeatherKind = "ffwi" | "ignition";

export function weatherTexture(device: Device, g: WeatherGrid, cache: FfwiTileCache, kind: WeatherKind = "ffwi"): Texture {
  const per = g.cols * g.rows;
  const data = new Uint16Array(HRRR_LEAD_TIME_COUNT * per);
  const nan = toHalf(NaN);
  const acc = new Float32Array(HRRR_LEAD_TIME_COUNT);
  const wsum = new Float32Array(HRRR_LEAD_TIME_COUNT);
  const s = new Float32Array(HRRR_LEAD_TIME_COUNT);
  for (let k = 0; k < per; k++) {
    const x = g.fx[k]!;
    const y = g.fy[k]!;
    if (!Number.isFinite(x)) {
      for (let l = 0; l < HRRR_LEAD_TIME_COUNT; l++) data[l * per + k] = nan;
      continue;
    }
    // the four pixel centers around the node, weighted by distance; a missing pixel drops out
    const cx = x - 0.5;
    const cy = y - 0.5;
    const x0 = Math.floor(cx);
    const y0 = Math.floor(cy);
    const tx = cx - x0;
    const ty = cy - y0;
    acc.fill(0);
    wsum.fill(0);
    for (let dy = 0; dy < 2; dy++) {
      for (let dx = 0; dx < 2; dx++) {
        const w = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty);
        if (w <= 0) continue;
        const xx = x0 + dx;
        const yy = y0 + dy;
        if (!cache.inGrid(xx, yy)) continue;
        if (kind === "ignition") cache.ignitionSeries(xx, yy, s);
        else cache.ffwiSeries(xx, yy, s);
        for (let l = 0; l < HRRR_LEAD_TIME_COUNT; l++) {
          const v = s[l]!;
          if (v === v) {
            acc[l] = acc[l]! + w * v;
            wsum[l] = wsum[l]! + w;
          }
        }
      }
    }
    for (let l = 0; l < HRRR_LEAD_TIME_COUNT; l++) data[l * per + k] = wsum[l]! > 0 ? toHalf(acc[l]! / wsum[l]!) : nan;
  }
  return device.createTexture({
    dimension: "2d-array",
    format: "r16float",
    width: g.cols,
    height: g.rows,
    depth: HRRR_LEAD_TIME_COUNT,
    mipLevels: 1,
    data,
    sampler: { minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
  });
}
