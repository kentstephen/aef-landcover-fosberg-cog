import * as arrow from "apache-arrow";

/**
 * HIFLD transmission lines, preprocessed by scripts/build_lines.py into one
 * Arrow IPC file (web/public/lines.arrow, one record batch): the lines
 * simplified, cut into pieces of at most 3 km, each piece tagged with the
 * HRRR pixel under its length-midpoint. Columns: geometry
 * (geoarrow.linestring, float32 lon/lat), px, py (int16), vc (uint8
 * VOLT_CLASS code), km (float32). Read zero-copy; the geometry column is
 * handed to GeoArrowPathLayer as is.
 */
export type LineSet = {
  table: arrow.Table;
  n: number;
  px: Int16Array;
  py: Int16Array;
  /** VOLT_CLASS code: 1 = 100-161 kV, 2 = 220-287, 3 = 345, 4 = 500, 5 = 735+, 6 = DC, 7 = n/a. */
  vc: Uint8Array;
  km: Float32Array;
  totalKm: number;
  /** Unique (px, py) pixels the pieces sit on, and each piece's index into them. */
  pixelX: Int16Array;
  pixelY: Int16Array;
  pieceToPixel: Int32Array;
};

export const VOLT_CLASS_LABEL: Record<number, string> = {
  0: "under 100 kV",
  1: "100-161 kV",
  2: "220-287 kV",
  3: "345 kV",
  4: "500 kV",
  5: "735 kV and above",
  6: "DC",
  7: "voltage n/a",
};

export const KM_TO_MI = 0.621371;

export async function loadLines(signal?: AbortSignal): Promise<LineSet> {
  const resp = await fetch(`${import.meta.env.BASE_URL}lines.arrow`, { signal });
  if (!resp.ok) throw new Error(`lines.arrow HTTP ${resp.status}`);
  const table = arrow.tableFromIPC(await resp.arrayBuffer());
  if (table.batches.length !== 1) throw new Error(`lines.arrow: expected one record batch, got ${table.batches.length}`);
  const n = table.numRows;
  const col = <T>(name: string): T => {
    const c = table.getChild(name);
    if (!c) throw new Error(`lines.arrow: missing column ${name}`);
    return c.data[0]!.values as unknown as T;
  };
  const px = col<Int16Array>("px");
  const py = col<Int16Array>("py");
  const vc = col<Uint8Array>("vc");
  const km = col<Float32Array>("km");
  let totalKm = 0;
  for (let i = 0; i < n; i++) totalKm += km[i]!;

  // Pieces of one line share pixels (and neighbours often do); dedupe so the
  // per-run FFWI pass works per pixel, not per piece.
  const pieceToPixel = new Int32Array(n);
  const index = new Map<number, number>();
  const ux: number[] = [];
  const uy: number[] = [];
  for (let i = 0; i < n; i++) {
    const key = py[i]! * 2048 + px[i]!;
    let j = index.get(key);
    if (j === undefined) {
      j = ux.length;
      index.set(key, j);
      ux.push(px[i]!);
      uy.push(py[i]!);
    }
    pieceToPixel[i] = j;
  }
  return { table, n, px, py, vc, km, totalKm, pixelX: Int16Array.from(ux), pixelY: Int16Array.from(uy), pieceToPixel };
}
