import type { Classified } from "./head.js";

/**
 * The coarse reader's workers. Tiles go to a worker by longitude band so neighbors land on the
 * same worker and a file's level is decoded once; each worker keeps its own class planes.
 */
type W = { w: Worker; busy: number; waiting: Map<number, { ok: (c: Classified) => void; no: (e: Error) => void }> };
const aborted = () => Object.assign(new Error("aborted"), { name: "AbortError" });

export class CoarsePool {
  private ws: W[];
  private nextId = 0;

  constructor(baseUrl: string, size = Math.max(2, Math.min(4, (navigator.hardwareConcurrency || 4) - 2))) {
    this.ws = Array.from({ length: size }, () => {
      const w = new Worker(new URL("./coarse.worker.ts", import.meta.url), { type: "module" });
      w.postMessage({ type: "init", baseUrl });
      const ent: W = { w, busy: 0, waiting: new Map() };
      w.onmessage = (ev: MessageEvent<{ id: number; error?: { name: string; message: string }; nlcd?: Uint8Array; esri?: Uint8Array; cover?: Uint8Array }>) => {
        const { id, error, nlcd, esri, cover } = ev.data;
        const p = ent.waiting.get(id);
        if (!p) return;
        ent.waiting.delete(id);
        ent.busy--;
        if (error) p.no(Object.assign(new Error(error.message), { name: error.name }));
        else p.ok({ nlcd: nlcd!, esri: esri!, cover: cover! });
      };
      return ent;
    });
  }

  tile(z: number, x: number, y: number, level: number, lonCenter: number, signal?: AbortSignal): Promise<Classified> {
    if (signal?.aborted) return Promise.reject(aborted());
    // CONUS spans about 60 degrees; a band per worker
    const band = Math.max(0, Math.min(this.ws.length - 1, Math.floor(((lonCenter + 125) / 59) * this.ws.length)));
    const ent = this.ws[band]!;
    const id = ++this.nextId;
    ent.busy++;
    return new Promise<Classified>((ok, no) => {
      ent.waiting.set(id, { ok, no });
      ent.w.postMessage({ type: "tile", id, z, x, y, level });
      signal?.addEventListener(
        "abort",
        () => {
          if (!ent.waiting.has(id)) return;
          ent.waiting.delete(id);
          ent.busy--;
          ent.w.postMessage({ type: "abort", id });
          no(aborted());
        },
        { once: true },
      );
    });
  }
}
