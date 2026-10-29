import { ignitionWeather } from "../aef/head.js";
import { ffwiFromMoisture, M_SCALE, MS_TO_MPH, U16_NAN, WIND_SCALE, type WindMode } from "../ffwi/compute.js";
import { dateFromInitTimeIdx, HRRR_GRID, HRRR_SHARD, initTimeIdxFromDate } from "./metadata.js";
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
 * cache and the fetch path are the ones the run mode uses) and are dropped once compacted. Shard-days
 * are one queue, ranked by the playhead: the view's shards for the day on screen first, then the days
 * after it in order, then the rest of the country; a few decode at once. The map plays over the days in
 * hand while the rest arrive.
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

type Day = { initIdx: number; cache: FfwiTileCache; sc: SideChannel; done: Set<string> };

/** One shard of one day, waiting its turn. `inView`: a tile on screen asked for it. */
type Job = { d: number; key: string; x: number; y: number; inView: boolean };

/** What the panel's day strip shows for each day. */
export type DayState = "in" | "loading" | "waiting" | "none";

/** Shards decoding at once over the whole week (each is about 18 MB over the wire and 80 MB while decoding). */
const IN_FLIGHT = 4;

let seq = 0;

export class WeekWeather {
  /** Tells one instance from another with the same spec (a tile's weather must come from the live one). */
  readonly id = ++seq;
  readonly depth: number;
  private days: Day[] = [];
  private shards = new Map<string, WeekShard>();
  /** Bumped when anything changed (a shard arrived, a job started, the view moved): the panel re-renders. */
  version = 0;
  listeners = new Set<() => void>();
  /** Shards (by key) whose planes arrived since the last `takeArrived`: the tiles over them need new textures. */
  private arrived = new Set<string>();
  private gone = false;
  /** Every shard-day of the week, keyed `${d}|${shard}`; done jobs leave the map. */
  private jobs = new Map<string, Job>();
  private inFlight = new Set<string>();
  /** The day the playhead is on: its shards go first, then the days after it in order. */
  private focusDay = 0;
  /** The shards the tiles on screen asked for (the view); `ensure` adds, `setView` replaces. */
  private viewKeys = new Set<string>();

  constructor(arrays: HrrrArrays, readonly spec: WeekSpec, private windMode: WindMode, initTimeCount: number) {
    this.depth = spec.days * HOURS_PER_DAY;
    // debug hook for the headless probes
    (globalThis as unknown as { __week?: unknown }).__week = this;
    for (let d = 0; d < spec.days; d++) {
      const cache = new FfwiTileCache();
      const day: Day = { initIdx: weekInitIdx(spec, d, initTimeCount), cache, sc: new SideChannel(arrays, cache), done: new Set() };
      cache.listeners.add(() => this.compact(d));
      this.days.push(day);
    }
  }

  /** The shards whose planes arrived since the last call (then cleared). */
  takeArrived(): Set<string> {
    const a = this.arrived;
    this.arrived = new Set();
    return a;
  }

  /** The job for shard `key` on day d, made on first ask. Only shards the view asked for are ever fetched. */
  private job(d: number, key: string): Job | undefined {
    if (this.days[d]!.initIdx < 0 || this.days[d]!.done.has(key)) return undefined;
    const id = `${d}|${key}`;
    let j = this.jobs.get(id);
    if (!j) {
      const [r, c] = key.split(",").map(Number) as [number, number];
      j = { d, key, x: c * HRRR_SHARD.width, y: r * HRRR_SHARD.height, inView: true };
      this.jobs.set(id, j);
    }
    return j;
  }

  /** Shard-days under the view: in hand, and asked for (the loading bar). */
  get progress(): { done: number; wanted: number } {
    let done = 0;
    let wanted = 0;
    for (const day of this.days) {
      if (day.initIdx < 0) continue;
      for (const key of this.viewKeys) {
        wanted++;
        if (day.done.has(key)) done++;
      }
    }
    return { done, wanted };
  }

  /** Each day of the week: in hand under the view, loading, waiting its turn, or not in the store. */
  dayStates(): DayState[] {
    return this.days.map((day, d) => {
      if (day.initIdx < 0) return "none";
      if (this.viewKeys.size === 0) return "waiting";
      let missing = 0;
      let loading = false;
      for (const key of this.viewKeys) {
        if (day.done.has(key)) continue;
        missing++;
        if (this.inFlight.has(`${d}|${key}`)) loading = true;
      }
      return missing === 0 ? "in" : loading ? "loading" : "waiting";
    });
  }

  /** Days whose shards under the view are all in hand (0 when nothing has been asked for yet). */
  daysReady(): number {
    return this.dayStates().filter((s) => s === "in").length;
  }

  /** Whether hour h's day is in hand under the view. */
  hourReady(h: number): boolean {
    const d = Math.floor(h / HOURS_PER_DAY);
    const day = this.days[d];
    if (!day || day.initIdx < 0) return false;
    for (const key of this.viewKeys) if (!day.done.has(key)) return false;
    return this.viewKeys.size > 0;
  }

  /**
   * The hour after h for the play loop: the next hour when its day is in hand under the view, else the
   * first hour of the next day that is (wrapping), else just the next hour (nothing is in yet: scrub gray).
   */
  nextHour(h: number): number {
    const n = (h + 1) % this.depth;
    if (this.hourReady(n)) return n;
    const d0 = Math.floor(n / HOURS_PER_DAY);
    for (let k = 0; k < this.days.length; k++) {
      const d = (d0 + k) % this.days.length;
      if (this.hourReady(d * HOURS_PER_DAY)) return d * HOURS_PER_DAY;
    }
    return n;
  }

  inGrid(x: number, y: number): boolean {
    return x >= 0 && y >= 0 && x < HRRR_GRID.width && y < HRRR_GRID.height;
  }

  abort(): void {
    this.gone = true;
    this.jobs.clear();
    for (const d of this.days) d.sc.abort();
  }

  /** The playhead moved: the day it is on comes first in the queue from now on. */
  focus(hour: number): void {
    this.focusDay = Math.max(0, Math.min(this.days.length - 1, Math.floor(hour / HOURS_PER_DAY)));
  }

  /** The view changed: the shards under these tiles are the view now; the others fall to the back. */
  setView(tiles: Iterable<{ px: Int16Array; py: Int16Array }>): void {
    const keys = new Set<string>();
    for (const t of tiles) for (let i = 0; i < t.px.length; i++) if (t.px[i]! >= 0) keys.add(FfwiTileCache.keyFor(t.px[i]!, t.py[i]!));
    const same = keys.size === this.viewKeys.size && [...keys].every((k) => this.viewKeys.has(k));
    if (same) return;
    this.viewKeys = keys;
    for (const j of this.jobs.values()) j.inView = keys.has(j.key);
    for (const key of keys) for (let d = 0; d < this.days.length; d++) this.job(d, key);
    this.bump();
    this.pump();
  }

  /** A tile asked for these HRRR pixels: their shards join the view for every day of the week. */
  ensure(px: Int16Array, py: Int16Array): void {
    let added = false;
    for (let i = 0; i < px.length; i++) {
      if (px[i]! < 0) continue;
      const key = FfwiTileCache.keyFor(px[i]!, py[i]!);
      if (this.viewKeys.has(key)) continue;
      this.viewKeys.add(key);
      added = true;
      for (let d = 0; d < this.days.length; d++) {
        const j = this.job(d, key);
        if (j) j.inView = true;
      }
    }
    if (added) this.bump();
    this.pump();
  }

  /** The view's shards first (a shard the view left waits), the playhead's day first, then the days after it. */
  private rank(j: Job): number {
    const n = this.days.length;
    return (j.inView ? 0 : 1000) + ((j.d - this.focusDay + n) % n);
  }

  /** Start the best waiting jobs until the in-flight cap is reached. */
  private pump(): void {
    while (!this.gone && this.inFlight.size < IN_FLIGHT) {
      let best: Job | null = null;
      let bestRank = Infinity;
      for (const j of this.jobs.values()) {
        const id = `${j.d}|${j.key}`;
        if (this.inFlight.has(id)) continue;
        const r = this.rank(j);
        if (r < bestRank) {
          bestRank = r;
          best = j;
        }
      }
      if (!best || !best.inView) return;
      void this.run(best);
    }
  }

  private async run(j: Job): Promise<void> {
    const id = `${j.d}|${j.key}`;
    this.inFlight.add(id);
    this.bump();
    const day = this.days[j.d]!;
    try {
      await day.sc.ensure(Int16Array.of(j.x), Int16Array.of(j.y), day.initIdx, this.windMode, 1);
    } catch {
      // the side channel logs; the job is dropped (a retry comes with the next ensure of a new instance)
    } finally {
      this.inFlight.delete(id);
      this.jobs.delete(id);
    }
    if (this.gone) return;
    this.bump();
    this.pump();
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
      this.arrived.add(t.key);
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

  /** WeatherSource.at: the term at one pixel and hour. */
  at = (x: number, y: number, h: number): number => this.termAt(x, y, h);

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
