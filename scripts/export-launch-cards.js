#!/usr/bin/env node
/**
 * Export the 5 social launch card canvases to PNG files.
 * Usage: node scripts/export-launch-cards.js
 */
const puppeteer = require(require('path').resolve(__dirname, '../frontend/node_modules/puppeteer'));
const fs = require('fs');
const path = require('path');

const HTML_PATH = path.resolve(__dirname, '../design/brand/social-launch-cards.html');
const OUT_DIR = path.resolve(__dirname, '../frontend/public/launch-cards');

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Load the HTML file with canvases
  await page.goto(`file://${HTML_PATH}`, { waitUntil: 'networkidle0' });

  // Wait for all 5 canvases to be rendered
  await page.waitForSelector('#card5');

  for (let i = 1; i <= 5; i++) {
    const dataUrl = await page.evaluate((cardId) => {
      const canvas = document.getElementById(cardId);
      return canvas.toDataURL('image/png');
    }, `card${i}`);

    // Strip the data URL prefix and write binary PNG
    const base64 = dataUrl.replace(/^data:image\/png;base64,/, '');
    const outFile = path.join(OUT_DIR, `launch-${i}.png`);
    fs.writeFileSync(outFile, Buffer.from(base64, 'base64'));
    console.log(`✓ Exported ${outFile}`);
  }

  await browser.close();
  console.log('\nDone! 5 launch card PNGs exported.');
})();
