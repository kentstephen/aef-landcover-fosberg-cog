import { SourceCache, SourceChunk } from "@chunkd/middleware";
import { SourceView, type Source, type SourceMetadata } from "@chunkd/source";
import { SourceHttp } from "@chunkd/source-http";
import { GeoTIFF } from "@developmentseed/geotiff";
import proj4 from "proj4";
import { cacheGet, cachePut } from "./disk-cache.js";

/**
 * The AlphaEarth 2025 COGs that touch CONUS (web/public/aef-2025-conus.json, from tge-labs'
 * aef_index.parquet by scripts/build_aef_index.py): one 8192 px file per 82 km UTM tile, 64 bands
 * int8 band-separate in 1024 px blocks, overviews 2..8192 whose pixels are unit centroids of their
 * children, so the same head reads them. Fetched from the S3 host; every byte range read is kept on
 * disk (disk-cache.ts), the header's small reads included, so a revisit opens a file without the
 * network.
 */
export type AefSource = { id: string; url: string; bbox: [number, number, number, number] };

const COG_TTL = 30 * 864e5;
/** Zooms the COG overviews serve; above this the 10 m mosaic takes over (mosaic.ts). */
export const COG_MIN_ZOOM = 9;
/**
 * Overviews narrower than this are left out. The files go down to a 1 px image, and the tile layer
 * lays a full 1024 px tile over every level, so the coarsest levels put tile corners tens of
 * thousands of km outside the UTM zone, where proj4 returns null and the tileset fails. At 128 px
 * (640 m) a tile spans 655 km, within reach.
 */
export const MIN_OVERVIEW_PX = 128;

export async function loadAefIndex(): Promise<AefSource[]> {
  const r = await fetch(`${import.meta.env.BASE_URL}aef-2025-conus.json`);
  if (!r.ok) throw new Error(`AEF index: HTTP ${r.status}`);
  const j = (await r.json()) as { base: string; tiles: [string, number, number, number, number][] };
  return j.tiles.map(([rel, w, s, e, n]) => ({ id: rel, url: j.base + rel, bbox: [w, s, e, n] }));
}

/** An HTTP source whose ranges are kept on disk. */
class CachedHttp implements Source {
  readonly type = "http-cached";
  private inner: SourceHttp;
  constructor(url: string) {
    this.inner = new SourceHttp(url, {});
  }
  get url(): URL {
    return this.inner.url;
  }
  get metadata(): SourceMetadata | undefined {
    return this.inner.metadata;
  }
  head(): Promise<SourceMetadata> {
    return this.inner.head();
  }
  async fetch(offset: number, length?: number, options?: { signal: AbortSignal }): Promise<ArrayBuffer> {
    const key = `cog|${this.inner.url.href}|${offset}+${length ?? "end"}`;
    const have = await cacheGet(key);
    if (have) return have.buffer.slice(have.byteOffset, have.byteOffset + have.byteLength) as ArrayBuffer;
    const buf = await this.inner.fetch(offset, length, options);
    cachePut(key, new Uint8Array(buf), COG_TTL);
    return buf;
  }
}

const opened = new Map<string, Promise<GeoTIFF>>();

/** The file's header read once through a chunked view; tile data straight from the (cached) source. */
export function openAefCog(url: string): Promise<GeoTIFF> {
  let p = opened.get(url);
  if (!p) {
    p = (async () => {
      const source = new CachedHttp(url);
      await source.head();
      const view = new SourceView(source, [new SourceChunk({ size: 64 * 1024 }), new SourceCache({ size: 4 * 1024 * 1024 })]);
      const g = await GeoTIFF.open({ dataSource: source, headerSource: view });
      // the same file with the coarsest overviews hidden: an object over the real one (instanceof
      // and every getter still reach it) whose own `overviews` is the trimmed list
      const trimmed = Object.create(g) as GeoTIFF;
      Object.defineProperty(trimmed, "overviews", { value: g.overviews.filter((o) => o.width >= MIN_OVERVIEW_PX), enumerable: true });
      return trimmed;
    })();
    p.catch(() => opened.delete(url));
    opened.set(url, p);
  }
  return p;
}

const utmConverters = new Map<number, proj4.Converter>();

/** Forward and inverse between a UTM north zone (EPSG 326xx) and lon/lat, from the proj string itself. */
export function utmConverter(epsg: number): proj4.Converter {
  let c = utmConverters.get(epsg);
  if (!c) {
    c = proj4(`+proj=utm +zone=${epsg - 32600} +datum=WGS84 +units=m +no_defs`, "EPSG:4326");
    utmConverters.set(epsg, c);
  }
  return c;
}
