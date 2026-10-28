import { cacheGet, cachePut } from "../aef/disk-cache.js";
import type { WindMode } from "../ffwi/compute.js";
import type { FfwiTile } from "./tile-cache.js";

/**
 * A shard's Fosberg inputs kept on disk (IndexedDB, disk-cache.ts): the 49-lead moisture, wind,
 * humidity and temperature as the cache holds them, 6 bytes a pixel, about 23 MB a shard, so a
 * run looked at again is not fetched again (five float32 bands, about 18 MB over the wire, 78 MB
 * decoded, per shard). Keyed by run, wind mode and shard; 30 days.
 */
const TTL = 30 * 864e5;
const V = 1;

export const shardKey = (initTimeIdx: number, windMode: WindMode, row: number, col: number) => `hrrr|v${V}|${initTimeIdx}|${windMode}|${row},${col}`;

export async function shardFromDisk(key: string, n: number): Promise<Pick<FfwiTile, "mEq" | "wind" | "rh" | "tc"> | undefined> {
  const b = await cacheGet(key);
  if (!b || b.byteLength !== 6 * n) return undefined;
  const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  return {
    mEq: new Uint16Array(buf, 0, n),
    wind: new Uint16Array(buf, 2 * n, n),
    rh: new Uint8Array(buf, 4 * n, n),
    tc: new Uint8Array(buf, 5 * n, n),
  };
}

export function shardToDisk(key: string, t: Pick<FfwiTile, "mEq" | "wind" | "rh" | "tc">): void {
  const n = t.rh.length;
  const out = new Uint8Array(6 * n);
  out.set(new Uint8Array(t.mEq.buffer, t.mEq.byteOffset, 2 * n), 0);
  out.set(new Uint8Array(t.wind.buffer, t.wind.byteOffset, 2 * n), 2 * n);
  out.set(t.rh, 4 * n);
  out.set(t.tc, 5 * n);
  cachePut(key, out, TTL);
}
