import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // Project Pages site: https://<user>.github.io/aef-landcover-fosberg-cog/. Production
  // asset URLs need the repo prefix; local dev stays at "/".
  base: command === "build" ? "/aef-landcover-fosberg-cog/" : "/",
  worker: { format: "es" },
  server: {
    // Standard dev port for anyone cloning the repo. Override per machine
    // with PORT (3000 is often taken; `PORT=5371 npm run dev`, see
    // CLAUDE.md). strictPort fails loudly.
    port: Number(process.env.PORT) || 3000,
    strictPort: true,
  },
  optimizeDeps: {
    exclude: ["lerc"],
  },
}));
