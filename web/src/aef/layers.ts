import type { Layer } from "@deck.gl/core";
import { apply, invert } from "@developmentseed/affine";
import { COGLayer, MosaicLayer, type GetTileDataOptions as CogTileOptions } from "@developmentseed/deck.gl-geotiff";
import { ZarrLayer, type GetTileDataOptions as ZarrTileOptions } from "@developmentseed/deck.gl-zarr";
import type { GeoTIFF, Overview } from "@developmentseed/geotiff";
import { epsgResolver } from "@developmentseed/proj";
import type { Device, Texture } from "@luma.gl/core";
import type * as zarr from "zarrita";
import type { FfwiTileCache } from "../hrrr/tile-cache.js";
import { lutTexture } from "./fuel.js";
import { type Classified, NUM_BANDS, type Teacher } from "./head.js";
import { type AefArray, MOSAIC_MIN_ZOOM, RES, X0, Y0, YEAR_IDX_2025 } from "./mosaic.js";
import type { AefPool } from "./pool.js";
import { rampTexture } from "./ramp.js";
import { fuelPipeline, type FuelMode } from "./render.js";
import { type AefSource, COG_MIN_ZOOM, openAefCog, utmConverter } from "./sources.js";
import { makeGrid, type WeatherGrid, weatherTexture } from "./weather.js";

/**
 * The land cover under the weather, two readers by zoom with the same head and the same shader:
 * zoomed out (COG_MIN_ZOOM to MOSAIC_MIN_ZOOM) the AlphaEarth COG overviews, one COGLayer per tile
 * file inside a MosaicLayer; zoomed in the 10 m Zarr mosaic. Each tile is classified once in a
 * worker (three class planes kept on the CPU for the tooltip, two class textures on the GPU) and
 * carries a small weather texture of FFWI for all 49 hours, rebuilt from the shared cache when it
 * changes. Drawing is a per-fragment lookup; switching the mode, the teacher or the hour reads nothing.
 */
export type AefTile = {
  key: string;
  kind: "cog" | "zarr";
  width: number;
  height: number;
  byteLength: number;
  classes: Classified;
  classTex: Record<Teacher, Texture>;
  grid: WeatherGrid;
  ffwiTex: Texture;
  bbox: [number, number, number, number];
  /** Pixel index under a lon/lat, or -1. */
  pixelAt: (lon: number, lat: number) => number;
  gone?: boolean;
};

export type AefLayerOpts = {
  mode: FuelMode;
  teacher: Teacher;
  leadTimeIdx: number;
  hideBelow: number;
  rescaleMax: number;
  opacity: number;
  beforeId?: string;
  visible: boolean;
};

/** Textures freed a moment later: a draw already queued may still bind them. */
function later(...tex: Texture[]) {
  setTimeout(() => tex.forEach((t) => t.destroy()), 1500);
}

export class AefLayers {
  readonly live = new Map<string, AefTile>();
  /** Tiles asked and not back yet, and how many have come back. */
  readonly work = { reading: 0, done: 0 };
  private device: Device | null = null;
  private ramp: Texture | null = null;
  private luts: Record<Teacher, Texture> | null = null;
  private opts: AefLayerOpts | null = null;
  private version = 0;
  private lastKey = "";

  constructor(
    private pool: AefPool,
    private cache: FfwiTileCache,
    private sources: AefSource[],
    private mosaic: { arr: AefArray; attrs: unknown },
    /** HRRR pixels a new tile needs the forecast for (the side channel fetches the shards). */
    private onNeedWeather: (px: Int16Array, py: Int16Array) => void,
    private redraw: () => void,
  ) {}

  private textures(device: Device) {
    if (this.device !== device || !this.ramp || !this.luts) {
      this.device = device;
      this.ramp = rampTexture(device);
      this.luts = { nlcd: lutTexture(device, "nlcd"), esri: lutTexture(device, "esri") };
    }
    return { ramp: this.ramp, luts: this.luts };
  }

  /** The weather changed (a shard arrived, the fuel lag moved): every tile's weather texture anew. */
  refreshWeather() {
    if (!this.device) return;
    for (const t of this.live.values()) {
      const old = t.ffwiTex;
      t.ffwiTex = weatherTexture(this.device, t.grid, this.cache);
      later(old);
    }
    this.version++;
    this.redraw();
  }

  private finish(
    device: Device,
    key: string,
    kind: AefTile["kind"],
    width: number,
    height: number,
    classes: Classified,
    toLonLat: (u: number, v: number) => [number, number],
    extentM: number,
    pixelAt: AefTile["pixelAt"],
  ): AefTile {
    this.textures(device);
    const tex = (plane: Uint8Array) =>
      device.createTexture({
        format: "r8unorm",
        width,
        height,
        mipLevels: 1,
        data: plane,
        sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
      });
    const grid = makeGrid(extentM, toLonLat);
    const corners = [toLonLat(0, 0), toLonLat(1, 0), toLonLat(0, 1), toLonLat(1, 1)];
    const bbox: AefTile["bbox"] = [
      Math.min(...corners.map((c) => c[0])), Math.min(...corners.map((c) => c[1])),
      Math.max(...corners.map((c) => c[0])), Math.max(...corners.map((c) => c[1])),
    ];
    const t: AefTile = {
      key, kind, width, height,
      byteLength: 3 * width * height + 2 * 49 * grid.cols * grid.rows,
      classes,
      classTex: { nlcd: tex(classes.nlcd), esri: tex(classes.esri) },
      grid,
      ffwiTex: weatherTexture(device, grid, this.cache),
      bbox,
      pixelAt,
    };
    const old = this.live.get(key);
    if (old) this.drop(old);
    this.live.set(key, t);
    this.onNeedWeather(grid.px, grid.py);
    return t;
  }

  private drop(t: AefTile) {
    t.gone = true;
    if (this.live.get(t.key) === t) this.live.delete(t.key);
    later(t.classTex.nlcd, t.classTex.esri, t.ffwiTex);
  }

  private async read<T>(f: () => Promise<T>): Promise<T> {
    this.work.reading++;
    try {
      const r = await f();
      this.work.done++;
      return r;
    } finally {
      this.work.reading--;
      this.redraw();
    }
  }

  // ---- the COG overviews
  private cogGetTileData = (sourceId: string) => async (image: GeoTIFF | Overview, o: CogTileOptions): Promise<AefTile> => {
    return this.read(async () => {
      const [tile] = await image.fetchTiles([[o.x, o.y]], { signal: o.signal, boundless: false, pool: o.pool });
      const a = tile!.array;
      if (a.layout !== "band-separate" || a.bands.length !== NUM_BANDS) {
        throw new Error(`AEF tile: expected ${NUM_BANDS} separate bands, got ${a.layout} x ${a.count}`);
      }
      const { width, height } = a;
      const n = width * height;
      const data = new Int8Array(NUM_BANDS * n);
      for (let b = 0; b < NUM_BANDS; b++) {
        const src = a.bands[b]!;
        const i8 = src instanceof Int8Array ? src : new Int8Array(src.buffer, src.byteOffset, src.length);
        if (i8.length === n) data.set(i8, b * n);
        else {
          // the block as stored is wider than the clipped tile: copy row by row
          const stride = image.tileWidth;
          for (let r = 0; r < height; r++) data.set(i8.subarray(r * stride, r * stride + width), b * n + r * width);
        }
      }
      const classes = await this.pool.classify(data, n, o.signal);
      const tf = a.transform;
      const inv = invert(tf);
      const conv = utmConverter(a.crs as number);
      const toLonLat = (u: number, v: number): [number, number] => {
        const [x, y] = apply(tf, u * width, v * height);
        const [lon, lat] = conv.forward([x, y]);
        return [lon!, lat!];
      };
      const pixelAt = (lon: number, lat: number) => {
        const [x, y] = conv.inverse([lon, lat]);
        const [c, r] = apply(inv, x!, y!);
        const ci = Math.floor(c);
        const ri = Math.floor(r);
        return ci < 0 || ri < 0 || ci >= width || ri >= height ? -1 : ri * width + ci;
      };
      return this.finish(o.device, `cog:${sourceId}:${image.width}:${o.x},${o.y}`, "cog", width, height, classes, toLonLat, width * Math.abs(tf[0]), pixelAt);
    });
  };

  // ---- the 10 m mosaic
  private zarrGetTileData = async (_arr: zarr.Array<"int8", zarr.Readable>, o: ZarrTileOptions): Promise<AefTile> => {
    return this.read(async () => {
      const ys = o.sliceSpec[2] as zarr.Slice;
      const xs = o.sliceSpec[3] as zarr.Slice;
      const py0 = ys.start ?? 0;
      const px0 = xs.start ?? 0;
      const { width, height } = o;
      const classes = await this.pool.zarr(YEAR_IDX_2025, py0, px0, height, width, o.signal);
      const toLonLat = (u: number, v: number): [number, number] => [X0 + (px0 + u * width) * RES, Y0 - (py0 + v * height) * RES];
      const pixelAt = (lon: number, lat: number) => {
        const ci = Math.floor((lon - X0) / RES) - px0;
        const ri = Math.floor((Y0 - lat) / RES) - py0;
        return ci < 0 || ri < 0 || ci >= width || ri >= height ? -1 : ri * width + ci;
      };
      return this.finish(o.device, `zarr:${py0},${px0}`, "zarr", width, height, classes, toLonLat, width * 10, pixelAt);
    });
  };

  private renderTile = (d: AefTile) => {
    const o = this.opts;
    if (!o || !this.device || d.gone) return { renderPipeline: [] };
    const { ramp, luts } = this.textures(this.device);
    return fuelPipeline({
      classTex: d.classTex[o.teacher],
      ffwiTex: d.ffwiTex,
      lutTex: luts[o.teacher],
      rampTex: ramp,
      mode: o.mode === "cover" ? 1 : 0,
      layerIndex: o.leadTimeIdx,
      hideBelow: o.hideBelow,
      rescaleMax: o.rescaleMax,
    });
  };

  private onUnload = (tile: { content?: AefTile | null }) => {
    if (tile.content) this.drop(tile.content);
  };

  layers(opts: AefLayerOpts): Layer[] {
    this.opts = opts;
    // debug hook for the headless probe (this instance is the one deck draws from)
    (globalThis as unknown as { __aef?: unknown }).__aef = { layers: this, cache: this.cache };
    const key = `${opts.mode}|${opts.teacher}|${opts.leadTimeIdx}|${opts.hideBelow}|${opts.rescaleMax}`;
    if (key !== this.lastKey) {
      this.lastKey = key;
      this.version++;
    }
    const v = this.version;
    const common = {
      renderTile: this.renderTile,
      opacity: opts.opacity,
      visible: opts.visible,
      updateTriggers: { renderTile: [v] },
      onTileUnload: this.onUnload,
      beforeId: opts.beforeId,
    };
    const cog = new MosaicLayer<AefSource, GeoTIFF>({
      id: "aef-cog",
      sources: this.sources,
      minZoom: COG_MIN_ZOOM,
      maxZoom: MOSAIC_MIN_ZOOM,
      maxRequests: 12,
      maxCacheSize: 160,
      visible: opts.visible,
      getSource: (s: AefSource) => openAefCog(s.url),
      onSourceUnload: (s: AefSource) => {
        for (const t of [...this.live.values()]) if (t.key.startsWith(`cog:${s.id}:`)) this.drop(t);
      },
      renderSource: (s: AefSource, { data }: { data?: GeoTIFF }) =>
        data
          ? new COGLayer<AefTile>({
              id: `aef-cog-${s.id}`,
              geotiff: data,
              epsgResolver,
              getTileData: this.cogGetTileData(s.id),
              refinementStrategy: "best-available",
              maxRequests: 4,
              ...common,
            } as never)
          : null,
    } as never);
    const fine = new ZarrLayer<zarr.Readable, "int8", AefTile>({
      id: "aef-zarr",
      node: this.mosaic.arr,
      metadata: this.mosaic.attrs,
      selection: { time: YEAR_IDX_2025, band: null },
      getTileData: this.zarrGetTileData,
      minZoom: MOSAIC_MIN_ZOOM,
      maxRequests: 6,
      maxCacheSize: 400,
      ...common,
    } as never);
    return [cog, fine];
  }

  /** The classes under a lon/lat from the tiles on hand (the finer reader first), or null. */
  at(lon: number, lat: number): { tile: AefTile; nlcd: number; esri: number; cover: number } | null {
    let best: { tile: AefTile; i: number } | null = null;
    for (const t of this.live.values()) {
      const [w, s, e, n] = t.bbox;
      if (lon < w || lon > e || lat < s || lat > n) continue;
      const i = t.pixelAt(lon, lat);
      if (i < 0) continue;
      if (!best || (t.kind === "zarr" && best.tile.kind === "cog") || (t.kind === best.tile.kind && t.width > best.tile.width)) best = { tile: t, i };
    }
    if (!best) return null;
    const { tile, i } = best;
    return { tile, nlcd: tile.classes.nlcd[i]!, esri: tile.classes.esri[i]!, cover: tile.classes.cover[i]! / 255 };
  }
}
