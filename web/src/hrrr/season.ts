import { decompress } from "fzstd";
import { HRRR_GRID } from "./metadata.js";
import { FfwiTileCache } from "./tile-cache.js";

/**
 * A fire season as one frame per day, precomputed by ignition/fetch_season.py from the HRRR analysis:
 * at every HRRR pixel, the day's worst hour (06Z to 06Z) of the weather part of the ignition logit, a byte
 * over termLo..termHi (255 = missing), days-major, one zstd file per HRRR shard (the 24 shards of
 * 265 x 300 px the forecast reader uses). Shards are fetched as the tiles on screen ask for them, like
 * the week; a shard is about 4 MB over the wire and 21 MB in hand (269 days x 79,500 px). The one thing
 * on this map that is precomputed beyond the model weights.
 */
export type SeasonManifest = {
  year: number;
  /** The UTC date of day 0 (the day runs from dayStartUtcHour that date). */
  start: string;
  dayStartUtcHour: number;
  days: number;
  termLo: number;
  termHi: number;
  missing: number;
  shard: { height: number; width: number };
  rows: number;
  cols: number;
};

let seq = 0;

export class SeasonWeather {
  readonly id = ++seq;
  readonly depth: number;
  private shards = new Map<string, Uint8Array>();
  private wanted = new Set<string>();
  private inflight = new Map<string, Promise<void>>();
  version = 0;
  listeners = new Set<() => void>();
  private ctrl = new AbortController();

  constructor(readonly m: SeasonManifest, private base: string) {
    this.depth = m.days;
  }

  /** The valid day (its start) of day d. */
  dayDate(d: number): Date {
    const t = new Date(`${this.m.start}T00:00:00Z`);
    t.setUTCHours(this.m.dayStartUtcHour);
    t.setUTCDate(t.getUTCDate() + d);
    return t;
  }

  get progress(): { done: number; wanted: number } {
    return { done: this.shards.size, wanted: this.wanted.size };
  }

  inGrid(x: number, y: number): boolean {
    return x >= 0 && y >= 0 && x < HRRR_GRID.width && y < HRRR_GRID.height;
  }

  abort(): void {
    this.ctrl.abort();
  }

  /** The shards under these HRRR pixels. */
  async ensure(px: Int16Array, py: Int16Array): Promise<void> {
    const keys = new Set<string>();
    for (let i = 0; i < px.length; i++) {
      if (px[i]! < 0) continue;
      const key = FfwiTileCache.keyFor(px[i]!, py[i]!);
      if (this.shards.has(key) || this.inflight.has(key)) continue;
      keys.add(key);
    }
    if (keys.size === 0) return;
    for (const k of keys) {
      if (!this.wanted.has(k)) this.wanted.add(k);
    }
    this.bump();
    await Promise.all([...keys].map((k) => this.fetch(k)));
  }

  private fetch(key: string): Promise<void> {
    let p = this.inflight.get(key);
    if (p) return p;
    p = (async () => {
      const [r, c] = key.split(",");
      const resp = await fetch(`${this.base}season-${this.m.year}/${r}-${c}.zst`, { signal: this.ctrl.signal });
      if (!resp.ok) throw new Error(`season shard ${key}: ${resp.status}`);
      const bytes = decompress(new Uint8Array(await resp.arrayBuffer()));
      if (this.ctrl.signal.aborted) return;
      this.shards.set(key, bytes);
      this.bump();
    })()
      .catch((e) => {
        if (!this.ctrl.signal.aborted) console.warn("season shard failed", key, e);
      })
      .finally(() => this.inflight.delete(key));
    this.inflight.set(key, p);
    return p;
  }

  private bump(): void {
    this.version++;
    for (const l of this.listeners) l();
  }

  /** Byte index of (x, y) inside its shard at day 0, and the shard's bytes, or null. */
  private locate(x: number, y: number): { s: Uint8Array; base: number; per: number } | null {
    const s = this.shards.get(FfwiTileCache.keyFor(x, y));
    if (!s) return null;
    const { height, width } = this.m.shard;
    const row = Math.floor(y / height);
    const col = Math.floor(x / width);
    const h = Math.min(height, HRRR_GRID.height - row * height);
    const w = Math.min(width, HRRR_GRID.width - col * width);
    return { s, base: (y - row * height) * w + (x - col * width), per: h * w };
  }

  private logit(b: number): number {
    return b === this.m.missing ? NaN : this.m.termLo + (b / 254) * (this.m.termHi - this.m.termLo);
  }

  /** The weather term for every day at one pixel (length depth), NaN where not in hand. */
  series(x: number, y: number, out: Float32Array): void {
    const l = this.locate(x, y);
    if (!l) {
      out.fill(NaN);
      return;
    }
    for (let d = 0; d < this.depth; d++) out[d] = this.logit(l.s[d * l.per + l.base]!);
  }

  /** The weather term at one pixel and day, NaN if not in hand. */
  termAt(x: number, y: number, d: number): number {
    const l = this.locate(x, y);
    return l ? this.logit(l.s[d * l.per + l.base]!) : NaN;
  }
}

/** The season's manifest from the site, or null when the season is not built. */
export async function loadSeason(base: string, year: number): Promise<SeasonWeather | null> {
  const resp = await fetch(`${base}season-${year}/index.json`);
  if (!resp.ok) return null;
  const m = (await resp.json()) as SeasonManifest;
  return new SeasonWeather(m, base);
}
