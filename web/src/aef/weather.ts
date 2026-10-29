import type { Device, Texture } from "@luma.gl/core";
import { toHalf } from "../hrrr/get-tile-data.js";
import { HRRR_LEAD_TIME_COUNT, lonLatToHrrrPixelF } from "../hrrr/metadata.js";
import type { FfwiTileCache } from "../hrrr/tile-cache.js";

/**
 * Where a weather texture's values come from: `depth` layers (the 49 leads of a run, or the hours of
 * a week) at every HRRR pixel, NaN where not in hand.
 */
export type WeatherSource = {
  depth: number;
  inGrid(x: number, y: number): boolean;
  series(x: number, y: number, out: Float32Array): void;
};

/** The run cache as a source: the FFWI, or the weather part of the ignition logit, per lead. */
export function cacheSource(cache: FfwiTileCache, kind: WeatherKind): WeatherSource {
  return {
    depth: HRRR_LEAD_TIME_COUNT,
    inGrid: (x, y) => cache.inGrid(x, y),
    series: kind === "ignition" ? (x, y, out) => cache.ignitionSeries(x, y, out) : (x, y, out) => cache.ffwiSeries(x, y, out),
  };
}

/**
 * The hour's weather on a land cover tile: a grid of nodes over the tile (about one per 800 m, at
 * least 4 x 4), each the FFWI at its spot for all 49 lead hours, read from the shared cache as a
 * bilinear blend of the four HRRR pixels around it (the pixel is 3 km; blending, rather than taking
 * the nearest pixel, keeps the field continuous across tile edges and inside a tile). Uploaded as one
 * r16float texture array the shader samples at the fragment's uv; rebuilt when the source changes
 * (a shard arrives, the fuel lag changes, a week replaces the run); the layer (hour) is a uniform.
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

export type WeatherTex = {
  tex: Texture;
  /** Every node on the grid had weather for every lead. */
  complete: boolean;
  /** The values per lead and node (lead-major), NaN where missing: what the texture holds, on the CPU. */
  nodes: Float32Array;
};

export function weatherTexture(device: Device, g: WeatherGrid, src: WeatherSource): WeatherTex {
  const per = g.cols * g.rows;
  const depth = src.depth;
  let complete = true;
  const nodes = new Float32Array(depth * per).fill(NaN);
  const data = new Uint16Array(depth * per);
  const nan = toHalf(NaN);
  const acc = new Float32Array(depth);
  const wsum = new Float32Array(depth);
  const s = new Float32Array(depth);
  for (let k = 0; k < per; k++) {
    const x = g.fx[k]!;
    const y = g.fy[k]!;
    if (!Number.isFinite(x)) {
      for (let l = 0; l < depth; l++) data[l * per + k] = nan;
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
        if (!src.inGrid(xx, yy)) continue;
        src.series(xx, yy, s);
        for (let l = 0; l < depth; l++) {
          const v = s[l]!;
          if (v === v) {
            acc[l] = acc[l]! + w * v;
            wsum[l] = wsum[l]! + w;
          }
        }
      }
    }
    for (let l = 0; l < depth; l++) {
      if (wsum[l]! > 0) {
        const v = acc[l]! / wsum[l]!;
        data[l * per + k] = toHalf(v);
        nodes[l * per + k] = v;
      } else {
        data[l * per + k] = nan;
        complete = false;
      }
    }
  }
  const tex = device.createTexture({
    dimension: "2d-array",
    format: "r16float",
    width: g.cols,
    height: g.rows,
    depth,
    mipLevels: 1,
    data,
    sampler: { minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
  });
  return { tex, complete, nodes };
}
