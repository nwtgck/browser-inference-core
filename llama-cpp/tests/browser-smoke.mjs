/** CI-only checks of each shipped variant; synthetic inference and mocked GPU suspension. */
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const packageRoot = resolve(process.argv[2] || 'dist/package');
const modelFile = resolve(process.argv[3] || 'build/fixture.gguf');
const chatTestFile = resolve('tests/chat-surface.mjs');
const chatTemplate = await readFile('vendor/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja', 'utf8');
const playwrightPath = resolve('../.tools/browser/node_modules/playwright/index.mjs');
const { chromium } = await import(pathToFileURL(playwrightPath).href);
const mime = { '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' };
const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (pathname === '/') {
      res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>Core smoke test</title>'); return;
    }
    const file = pathname === '/fixture.gguf' ? modelFile : pathname === '/chat-surface.mjs' ? chatTestFile : resolve(packageRoot, '.' + pathname);
    if (file !== modelFile && file !== chatTestFile && !file.startsWith(packageRoot + sep)) { res.writeHead(403); res.end(); return; }
    res.setHeader('Content-Type', mime[extname(file)] || 'application/octet-stream');
    res.end(await readFile(file));
  } catch { res.writeHead(404); res.end(); }
});
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
const browser = await chromium.launch({ headless: true });
const results = [];
try {
  const url = `http://127.0.0.1:${server.address().port}`;
  // Exercise both suspension mechanisms on the shipped pointer widths and the bigint ABI through a real
  // suspension. A mocked missing adapter requires no physical GPU and does not
  // establish WebGPU inference or production-model support.
  for (const variant of ['browser', 'test']) {
    for (const profile of ['webgpu-wasm32-jspi', 'webgpu-wasm64-jspi', 'webgpu-wasm32-asyncify']) {
      const suspensionPage = await browser.newPage();
      await suspensionPage.goto(url);
      const suspensionResult = await suspensionPage.evaluate(async ({ profile, variant }) => {
        if (profile.endsWith('-jspi') && (typeof WebAssembly.Suspending !== 'function' || typeof WebAssembly.promising !== 'function')) {
          throw new Error('The CI browser must support JSPI');
        }
        let adapterRequests = 0;
        Object.defineProperty(navigator, 'gpu', {
          configurable: true,
          value: { requestAdapter: async () => {
            adapterRequests++;
            await new Promise(resolve => setTimeout(resolve, 0));
            return null;
          } },
        });
        const failures = [];
        addEventListener('unhandledrejection', event => failures.push(String(event.reason)));
        const { createCore } = await import('/examples/runtime/index.mjs');
        const core = await createCore({ profile, variant, moduleOptions: { print() {}, printErr() {} } });
        const expectedPointerBytes = profile.includes('wasm64') ? 8 : 4;
        if (core.pointerBytes !== expectedPointerBytes) throw new Error('Unexpected pointer width');
        // Keep this first: backend registry initialization must reach requestAdapter.
        const pending = core.api.ggml_backend_init_by_type(core.constant('GGML_BACKEND_DEVICE_TYPE_GPU'), 0n);
        if (!core.busy) throw new Error('The wrapper released the pending suspended call');
        let timer;
        try {
          const backend = await Promise.race([
            pending,
            new Promise((_, reject) => {
              timer = setTimeout(() => reject(new Error('Suspension did not finish')), 5000);
            }),
          ]);
          await new Promise(resolve => setTimeout(resolve, 0));
          if (adapterRequests !== 1 || backend !== 0n || core.busy || failures.length) {
            throw new Error(`Invalid suspension completion: ${JSON.stringify({ adapterRequests, backend: String(backend), busy: core.busy, failures })}`);
          }
          return { profile, variant, pointerBytes: core.pointerBytes, adapterRequests, mockedAdapter: true, suspension: true, passed: true };
        } finally { clearTimeout(timer); }
      }, { profile, variant });
      results.push(suspensionResult);
      console.log(JSON.stringify(suspensionResult));
      await suspensionPage.close();
    }
  }
  for (const variant of ['browser', 'test']) {
    for (const profile of ['cpu-wasm32', 'cpu-wasm64']) {
      const page = await browser.newPage();
      page.on('console', msg => { if (msg.type() === 'error') console.error(msg.text()); });
      await page.goto(url);
      const result = await page.evaluate(async ({ profile, variant, chatTemplate }) => {
        const { default: createNative } = await import(`/profiles/${profile}/${variant}/core.mjs`);
        const native = await createNative({ print() {}, printErr() {} });
        const { checkChatSurface } = await import('/chat-surface.mjs');
        const chatSurface = await checkChatSurface(native, chatTemplate);
        const { createCore, mountReadOnlyFile } = await import('/examples/runtime/index.mjs');
        const core = await createCore({ profile, variant, moduleOptions: { print() {}, printErr() {} } });
        await core.api.llama_backend_init();
        const bytes = new Uint8Array(await (await fetch('/fixture.gguf')).arrayBuffer());
        const mounted = mountReadOnlyFile(core, '/models/test.gguf', {
          size: bytes.length,
          read(view, position) { const chunk = bytes.subarray(position, position + view.length); view.set(chunk); return chunk.length; },
        }, { maxChunkBytes: 1024 });
        const params = core.allocRecord('llama_model_params');
        await core.api.llama_model_default_params(params);
        core.setField('llama_model_params', params, 'n_gpu_layers', 0);
        core.setField('llama_model_params', params, 'load_mode', core.constant('LLAMA_LOAD_MODE_NONE'));
        core.setField('llama_model_params', params, 'lazy_mode', core.constant('LLAMA_LAZY_MODE_OFF'));
        const path = core.utf8(mounted.path);
        const model = await core.api.llama_model_load_from_file(path, params);
        if (!model) throw new Error('Synthetic model load failed');
        const cp = core.allocRecord('llama_context_params');
        await core.api.llama_context_default_params(cp);
        for (const [key, value] of Object.entries({ n_ctx: 128, n_batch: 16, n_ubatch: 16, n_threads: 1, n_threads_batch: 1 })) {
          core.setField('llama_context_params', cp, key, value);
        }
        const ctx = await core.api.llama_init_from_model(model, cp);
        if (!ctx) throw new Error('Context creation failed');
        const token = core.alloc(4);
        const tokenView = core.bytes(token, 4);
        new DataView(tokenView.buffer, tokenView.byteOffset, 4).setInt32(0, 1, true);
        const batch = core.allocRecord('llama_batch');
        await core.api.llama_batch_get_one(batch, token, 1);
        if (await core.api.llama_decode(ctx, batch) !== 0) throw new Error('Decode failed');
        const sampler = await core.api.llama_sampler_init_greedy();
        const sampled = await core.api.llama_sampler_sample(sampler, ctx, -1);
        if (sampled < 0 || sampled >= 259) throw new Error('Invalid sampled token');
        const size = await core.api.llama_state_get_size(ctx);
        const state = core.alloc(size);
        const used = await core.api.llama_state_get_data(ctx, state, size);
        if (used <= 0n || await core.api.llama_state_set_data(ctx, state, used) !== used) throw new Error('State roundtrip failed');
        await core.api.llama_sampler_free(sampler);
        await core.api.llama_free(ctx);
        await core.api.llama_model_free(model);
        for (const pointer of [params, path, cp, token, batch, state]) core.free(pointer);
        mounted.remove();
        await core.api.llama_backend_free();
        return { profile, variant, syntheticModel: true, sampledToken: sampled, stateBytes: String(used), chatSurface, passed: true };
      }, { profile, variant, chatTemplate });
      results.push(result); await page.close();
    }
  }
  const resultFile = process.env.BIC_BROWSER_RESULTS_FILE || 'build/browser-results.json';
  const output = process.env.BIC_BROWSER_SESSION_JSON
    ? { session: JSON.parse(process.env.BIC_BROWSER_SESSION_JSON), results } : results;
  await writeFile(resultFile, JSON.stringify(output, null, 2)+'\n');
  console.log(JSON.stringify(results, null, 2));
} finally { await browser.close(); server.close(); }
