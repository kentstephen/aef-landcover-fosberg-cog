// Parity check for web/src/ffwi/compute.ts against fixed inputs.
// Run: cd web && npm run ffwi-check   (node --experimental-strip-types)
// Compare the table with: uv run --with numpy python scripts/verify_ffwi.py --fixed
import { ffwi } from "../web/src/ffwi/compute.ts";

const cases = [
  // [tC, rh%, wind m/s]
  [30, 10, 10],
  [35, 5, 15],
  [20, 50, 5],
  [10, 90, 2],
  [40, 8, 13.41], // ~30 mph
  [25, 0, 13.41], // zero RH + 30 mph -> ~100
  [0, 30, 0],
];
console.log("tC\trh\twind_ms\tffwi");
for (const [t, h, w] of cases) {
  console.log(`${t}\t${h}\t${w}\t${ffwi(t, h, w).toFixed(3)}`);
}
