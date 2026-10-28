import * as zarr from "zarrita";
import { classify, type Classified, NUM_BANDS } from "./head.js";
import { openAefMosaic, type AefArray } from "./mosaic.js";

/**
 * The head off the page's thread: a COG block handed over as band-major int8, or a mosaic chunk read
 * here (zarrita's zstd decode of a 4 MB chunk would otherwise hold the map). Each job answers with
 * the three class planes.
 */
type Job =
  | { type: "classify"; id: number; data: Int8Array; n: number }
  | { type: "zarr"; id: number; timeIdx: number; py0: number; px0: number; h: number; w: number }
  | { type: "abort"; id: number };

let mosaic: Promise<AefArray> | null = null;
const running = new Map<number, AbortController>();

function reply(id: number, c: Classified) {
  postMessage({ id, ...c }, { transfer: [c.nlcd.buffer, c.esri.buffer, c.cover.buffer] });
}
function fail(id: number, e: unknown) {
  const err = e as { name?: string; message?: string };
  postMessage({ id, error: { name: err?.name || "Error", message: err?.message || String(e) } });
}

self.onmessage = async (ev: MessageEvent<Job>) => {
  const m = ev.data;
  if (m.type === "abort") {
    running.get(m.id)?.abort();
    return;
  }
  if (m.type === "classify") {
    try {
      reply(m.id, classify(m.data, m.n));
    } catch (e) {
      fail(m.id, e);
    }
    return;
  }
  const ac = new AbortController();
  running.set(m.id, ac);
  try {
    if (!mosaic) mosaic = openAefMosaic().then((o) => o.arr);
    mosaic.catch(() => (mosaic = null));
    const arr = await mosaic;
    const c = await zarr.get(arr, [m.timeIdx, null, zarr.slice(m.py0, m.py0 + m.h), zarr.slice(m.px0, m.px0 + m.w)], { signal: ac.signal });
    if (c.shape.length !== 3 || c.shape[0] !== NUM_BANDS || c.shape[1] !== m.h || c.shape[2] !== m.w) {
      throw new Error(`Unexpected chunk shape [${c.shape.join(", ")}] for a ${m.w} x ${m.h} tile`);
    }
    reply(m.id, classify(c.data as Int8Array, m.w * m.h));
  } catch (e) {
    fail(m.id, e);
  } finally {
    running.delete(m.id);
  }
};
