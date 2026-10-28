// Headless smoke test: loads the dev server, waits for data, screenshots,
// zooms in, reads the status line and the hover tooltip.
//
// Needs Playwright + a Chrome. This repo does not depend on Playwright; point
// NODE_PATH at any install (e.g. an npx cache) and use the system Chrome:
//
//   cd web && PORT=5371 npm run dev &
//   NODE_PATH=~/.npm/_npx/<hash>/node_modules node scripts/smoke.cjs /tmp/out http://localhost:5371/
//
// NOTE: the GeoPlatform WAF returns 403 (no CORS header) to a "HeadlessChrome"
// user agent, which surfaces as a CORS error on the potential layer. A normal
// Chrome UA (set below) is required; real browsers are unaffected.
const { chromium } = require("playwright");
const path = require("path");
(async () => {
  const out = process.argv[2] || ".";
  const url = process.argv[3] || "http://localhost:5371/";
  const browser = await chromium.launch({
    headless: true,
    channel: "chrome",
    args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
  });
  const ctx = await browser.newContext({
    viewport: { width: 1500, height: 950 },
    userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
  });
  const page = await ctx.newPage();
  const logs = [];
  page.on("console", (m) => { const t = m.text(); if (!/GL Driver|DevTools|\[vite\]/.test(t)) logs.push(`[${m.type()}] ${t.slice(0, 400)}`); });
  page.on("pageerror", (e) => logs.push(`[pageerror] ${String(e).slice(0, 400)}`));
  page.on("requestfailed", (r) => logs.push(`[reqfail] ${r.url().slice(0, 160)} :: ${r.failure()?.errorText}`));
  const status = async () => page.evaluate(() => {
    const el = [...document.querySelectorAll("p")].find((e) => /HRRR tiles/.test(e.textContent || ""));
    const pot = [...document.querySelectorAll("p")].find((e) => /^(fetching|live|not loaded)/.test(e.textContent || ""));
    return `${el?.textContent} | potential ${pot?.textContent}`;
  });
  await page.goto(url, { waitUntil: "load" });
  await page.waitForTimeout(20000);
  await page.screenshot({ path: path.join(out, "smoke-conus.png") });
  console.log("CONUS @20s:", await status());
  for (let i = 0; i < 3; i++) { await page.mouse.dblclick(880, 470); await page.waitForTimeout(900); }
  await page.waitForTimeout(9000);
  console.log("zoomed @+12s:", await status());
  await page.mouse.move(760, 600); await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(out, "smoke-zoom.png") });
  const tip = await page.evaluate(() => { const d = [...document.querySelectorAll("div")].find((e) => /^FFWI \d/.test(e.textContent || "")); return d ? d.parentElement?.textContent : "(no tooltip)"; });
  console.log("tooltip:", tip);
  console.log("--- console"); for (const l of logs.slice(0, 30)) console.log(l);
  await browser.close();
})().catch((e) => { console.error("SMOKE ERROR", e); process.exit(1); });
