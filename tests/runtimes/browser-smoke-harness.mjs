import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { READ_ENV_BIC_BROWSER_SESSION_JSON, READ_ENV_BIC_PLAYWRIGHT_MODULE } from '#tools/shared/node-env';

const mime = { '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' };

export function browserVariantsForPhase({ phase }) {
  switch (phase) {
  case 'startup': return ['browser'];
  case 'regression': return ['test'];
  case 'full': return ['browser', 'test'];
  default: throw new Error(`Unknown browser validation phase: ${phase}`);
  }
}

export async function writeBrowserResults({ file, results }) {
  const session = READ_ENV_BIC_BROWSER_SESSION_JSON();
  const output = session ? { session: JSON.parse(session), results } : results;
  await writeFile(file, JSON.stringify(output, null, 2) + '\n');
}

export async function withBrowserPackage({ packageRoot, files, run }) {
  const playwrightPath = resolve(READ_ENV_BIC_PLAYWRIGHT_MODULE() || '../../.tools/browser/node_modules/playwright/index.mjs');
  const { chromium } = await import(pathToFileURL(playwrightPath).href);
  const server = createServer(async (req, res) => {
    try {
      const pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://localhost').pathname);
      if (pathname === '/') {
        res.setHeader('Content-Type', 'text/html');
        res.end('<!doctype html><title>Runtime smoke test</title>');
        return;
      }
      const extraFile = files.get(pathname);
      const file = extraFile ?? resolve(packageRoot, '.' + pathname);
      if (extraFile === undefined && !file.startsWith(packageRoot + sep)) {
        res.writeHead(403).end();
        return;
      }
      const content = await readFile(file);
      res.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream');
      res.end(content);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  try {
    const browser = await chromium.launch({ headless: true });
    try {
      return await run({ browser, url: `http://127.0.0.1:${server.address().port}` });
    } finally {
      await browser.close();
    }
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
}
