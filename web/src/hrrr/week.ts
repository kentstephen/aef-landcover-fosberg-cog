import { ignitionWeather } from "../aef/head.js";
import { ffwiFromMoisture, M_SCALE, MS_TO_MPH, U16_NAN, WIND_SCALE, type WindMode } from "../ffwi/compute.js";
import { dateFromInitTimeIdx, HRRR_GRID, initTimeIdxFromDate } from "./metadata.js";
import { SideChannel } from "./side-channel.js";
import type { HrrrArrays } from "./store.js";
import { FfwiTileCache } from "./tile-cache.js";

/**
 * A run of days from the 48 h forecast store, one 12Z run per day, its first 24 hours each, stitched
 * into one hourly series (day 0 at 12Z through the last day at 11Z the next morning). The forecast
 * store is chunked by run and shard (49 leads x 265 x 300 px), so a day costs a whole shard fetch
 * (about 18 MB over the wire per shard) whatever is kept; what is kept is small: per shard and day,
 * the weather part of the ignition logit and the FFWI, a byte each per pixel and hour (3.8 MB per
 * shard-day). The raw run shards pass through a per-day FfwiTileCache and side channel (so the disk
 * cache and the fetch path are the ones the run mode uses) and are dropped once compacted. Days are
 * fetched one after another so the picture fills day by day and at most a few shards are decoding.
 */
export const RUN_HOUR_UTC = 12;
export const HOURS_PER_DAY = 24;
/** The byte range the weather term is kept over (logit); 255 = missing. */
const TERM_LO = -6;
const TERM_HI = 6;
export const termByte = (z: number) => (z === z ? Math.max(0, Math.min(254, Math.round(((z - TERM_LO) / (TERM_HI - TERM_LO)) * 254))) : 255);
export const termLogit = (b: number) => (b === 255 ? NaN : TERM_LO + (b / 254) * (TERM_HI - TERM_LO));
const ffwiByte = (f: number) => (f === f ? Math.max(0, Math.min(254, Math.round(f))) : 255);

export type WeekSpec = {
  /** The UTC date of day 0 (its 12Z run). */
  start: Date;
  days: number;
};

/** The init index of day d's run, or -1 when the store does not have it. */
export function weekInitIdx(spec: WeekSpec, d: number, initTimeCount: number): number {
  const t = new Date(spec.start.getTime());
  t.setUTCHours(RUN_HOUR_UTC, 0, 0, 0);
  t.setUTCDate(t.getUTCDate() + d);
  const idx = initTimeIdxFromDate(t, initTimeCount - 1);
  return dateFromInitTimeIdx(idx).getTime() === t.getTime() ? idx : -1;
}

/** The valid time of hour h of the week. */
export function weekHourDate(spec: WeekSpec, h: number): Date {
  const t = new Date(spec.start.getTime());
  t.setUTCHours(RUN_HOUR_UTC + h, 0, 0, 0);
  return t;
}

type WeekShard = { term: Uint8Array; ffwi: Uint8Array; width: number; rowStart: number; colStart: number };

type Day = { initIdx: number; cache: FfwiTileCache; sc: SideChannel; done: Set<string>; wanted: Set<string> };

let seq = 0;

export class WeekWeather {
  /** Tells one instance from another with the same spec (a tile's weather must come from the live one). */
  readonly id = ++seq;
  readonly depth: number;
  private days: Day[] = [];
  private shards = new Map<string, WeekShard>();
  version = 0;
  listeners = new Set<() => void>();
  private gone = false;
  /** Calls run one after another, so the whole week has at most one day's few shards in flight. */
  private queue: Promise<void> = Promise.resolve();

  constructor(arrays: HrrrArrays, readonly spec: WeekSpec, private windMode: WindMode, initTimeCount: number) {
    this.depth = spec.days * HOURS_PER_DAY;
    for (let d = 0; d < spec.days; d++) {
      const cache = new FfwiTileCache();
      const day: Day = { initIdx: weekInitIdx(spec, d, initTimeCount), cache, sc: new SideChannel(arrays, cache), done: new Set(), wanted: new Set() };
      cache.listeners.add(() => this.compact(d));
      this.days.push(day);
    }
  }

  /** Shard-days asked for and shard-days in hand, for the loading bar. */
  get progress(): { done: number; wanted: number } {
    let done = 0;
    let wanted = 0;
    for (const d of this.days) {
      done += d.done.size;
      wanted += d.wanted.size;
    }
    return { done, wanted };
  }

  inGrid(x: number, y: number): boolean {
    return x >= 0 && y >= 0 && x < HRRR_GRID.width && y < HRRR_GRID.height;
  }

  abort(): void {
    this.gone = true;
    for (const d of this.days) d.sc.abort();
  }

  /** The shards under these HRRR pixels for every day of the week, one day at a time. */
  ensure(px: Int16Array, py: Int16Array): Promise<void> {
    // mark them wanted now (the loading bar), fetch in turn
    for (const day of this.days) {
      if (day.initIdx < 0) continue;
      for (let i = 0; i < px.length; i++) {
        if (px[i]! < 0) continue;
        const key = FfwiTileCache.keyFor(px[i]!, py[i]!);
        if (!day.done.has(key) && !day.wanted.has(key)) {
          day.wanted.add(key);
          this.bump();
        }
      }
    }
    this.queue = this.queue.then(() => this.fetch(px, py)).catch(() => undefined);
    return this.queue;
  }

  private async fetch(px: Int16Array, py: Int16Array): Promise<void> {
    for (let d = 0; d < this.days.length && !this.gone; d++) {
      const day = this.days[d]!;
      if (day.initIdx < 0) continue;
      // the pixels whose shard this day still lacks
      const keep: number[] = [];
      for (let i = 0; i < px.length; i++) {
        const x = px[i]!;
        if (x < 0) continue;
        if (day.done.has(FfwiTileCache.keyFor(x, py[i]!))) continue;
        keep.push(i);
      }
      if (keep.length === 0) continue;
      const fx = new Int16Array(keep.length);
      const fy = new Int16Array(keep.length);
      keep.forEach((i, k) => {
        fx[k] = px[i]!;
        fy[k] = py[i]!;
      });
      await day.sc.ensure(fx, fy, day.initIdx, this.windMode);
    }
  }

  /** A raw run shard arrived for day d: keep its first 24 hours as bytes, drop the raw. */
  private compact(d: number): void {
    const day = this.days[d]!;
    for (const t of day.cache.entries()) {
      if (day.done.has(t.key)) continue;
      const per = t.width * t.height;
      const term = new Uint8Array(HOURS_PER_DAY * per);
      const ffwi = new Uint8Array(HOURS_PER_DAY * per);
      for (let h = 0; h < HOURS_PER_DAY; h++) {
        for (let p = 0; p < per; p++) {
          const i = h * per + p;
          const m = t.mEq[i]!;
          const w = t.wind[i]!;
          if (m === U16_NAN || w === U16_NAN) {
            term[i] = 255;
            ffwi[i] = 255;
            continue;
          }
          const mph = w / WIND_SCALE;
          const f = ffwiFromMoisture(m / M_SCALE, mph);
          ffwi[i] = ffwiByte(f);
          term[i] = termByte(ignitionWeather(f, (t.tc[i] ?? 100) - 100, t.rh[i] ?? 0, mph / MS_TO_MPH));
        }
      }
      this.shards.set(`${d}|${t.key}`, { term, ffwi, width: t.width, rowStart: t.rowStart, colStart: t.colStart });
      day.done.add(t.key);
      day.cache.delete(t.key);
    }
    this.bump();
  }

  private bump(): void {
    this.version++;
    for (const l of this.listeners) l();
  }

  private shard(d: number, x: number, y: number): WeekShard | undefined {
    return this.shards.get(`${d}|${FfwiTileCache.keyFor(x, y)}`);
  }

  /** The weather term for every hour of the week at one pixel (length depth), NaN where not in hand. */
  series(x: number, y: number, out: Float32Array): void {
    for (let d = 0; d < this.days.length; d++) {
      const s = this.shard(d, x, y);
      const o = d * HOURS_PER_DAY;
      if (!s) {
        out.fill(NaN, o, o + HOURS_PER_DAY);
        continue;
      }
      const base = (y - s.rowStart) * s.width + (x - s.colStart);
      const stride = s.term.length / HOURS_PER_DAY;
      for (let h = 0; h < HOURS_PER_DAY; h++) out[o + h] = termLogit(s.term[h * stride + base]!);
    }
  }

  /** The weather term at one pixel and hour of the week, NaN if not in hand. */
  termAt(x: number, y: number, h: number): number {
    const d = Math.floor(h / HOURS_PER_DAY);
    const s = this.shard(d, x, y);
    if (!s) return NaN;
    const stride = s.term.length / HOURS_PER_DAY;
    return termLogit(s.term[(h - d * HOURS_PER_DAY) * stride + (y - s.rowStart) * s.width + (x - s.colStart)]!);
  }

  /** The FFWI at one pixel and hour of the week, NaN if not in hand. */
  ffwiAt(x: number, y: number, h: number): number {
    const d = Math.floor(h / HOURS_PER_DAY);
    const s = this.shard(d, x, y);
    if (!s) return NaN;
    const stride = s.ffwi.length / HOURS_PER_DAY;
    const b = s.ffwi[(h - d * HOURS_PER_DAY) * stride + (y - s.rowStart) * s.width + (x - s.colStart)]!;
    return b === 255 ? NaN : b;
  }
}
