import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { READ_ENV_BIC_BROWSER_RESULTS_FILE } from '#tools/shared/node-env';
import { withBrowserPackage, writeBrowserResults } from '../browser-smoke-harness.mjs';

const packageRoot = resolve(process.argv[2] ?? 'dist/package');
const phase = process.argv[3] ?? 'startup';
if (phase !== 'startup') {
  throw new Error('Speech browser validation supports startup only');
}
const profiles = JSON.parse(await readFile('config/profiles.json', 'utf8'));
const files = new Map([
  ['/smoke/browser-smoke-worker.mjs', fileURLToPath(new URL('./browser-smoke-worker.mjs', import.meta.url))],
  ['/smoke/startup-contract.mjs', fileURLToPath(new URL('./startup-contract.mjs', import.meta.url))],
]);
const results = await withBrowserPackage({
  packageRoot,
  files,
  run: async ({ browser, url }) => {
    const results = [];
    const page = await browser.newPage();
    await page.goto(url);
    for (const [profile, configuration] of Object.entries(profiles)) {
      const result = await page.evaluate(async ({ profile, pointerBytes }) => {
        const worker = new Worker('/smoke/browser-smoke-worker.mjs', { type: 'module' });
        let timer;
        try {
          return await new Promise((resolve, reject) => {
            timer = setTimeout(() => reject(new Error(`Speech Worker startup timed out: ${profile}`)), 120000);
            worker.onerror = event => reject(new Error(event.message || `Speech Worker startup failed: ${profile}`));
            worker.onmessage = ({ data }) => data.error ? reject(new Error(data.error)) : resolve(data.result);
            worker.postMessage({ profile, pointerBytes });
          });
        } finally {
          clearTimeout(timer);
          worker.terminate();
        }
      }, { profile, pointerBytes: configuration.memory64 ? 8 : 4 });
      results.push(result);
      console.log(JSON.stringify(result));
    }
    return results;
  },
});
await writeBrowserResults({ file: READ_ENV_BIC_BROWSER_RESULTS_FILE() || 'build/browser-results.json', results });
