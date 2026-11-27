import { READ_ENV_BIC_BROWSER_RESULTS_FILE } from '#tools/shared/node-env';
/** CI-only checks of shipped Wasm with synthetic inference and mocked GPU suspension. */
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { browserVariantsForPhase, withBrowserPackage, writeBrowserResults } from '../browser-smoke-harness.mjs';

const packageRoot = resolve(process.argv[2] || 'dist/package');
const modelFile = resolve(process.argv[3] || 'build/fixture.gguf');
const phase = process.argv[4] ?? 'full';
const variants = browserVariantsForPhase({ phase });
const detailed = phase !== 'startup';
const files = new Map([['/fixture.gguf', modelFile]]);
let chatTemplate;
if (detailed) {
  files.set('/chat-surface.mjs', fileURLToPath(new URL('./chat-surface.mjs', import.meta.url)));
  files.set('/file-mount-lifetime.mjs', fileURLToPath(new URL('./file-mount-lifetime.mjs', import.meta.url)));
  chatTemplate = await readFile('vendor/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja', 'utf8');
}

async function smoke({ browser, url }) {
  const results = [];
  // Exercise both suspension mechanisms on the shipped pointer widths and the bigint ABI through a real
  // suspension. A mocked missing adapter requires no physical GPU and does not
  // establish WebGPU inference or production-model support.
  for (const variant of variants) {
    for (const profile of ['webgpu-wasm32-jspi', 'webgpu-wasm64-jspi', 'webgpu-wasm32-asyncify', 'webgpu-wasm64-jspi-gpu-profile']) {
      const suspensionPage = await browser.newPage();
      await suspensionPage.goto(url);
      const suspensionResult = await suspensionPage.evaluate(async ({ profile, variant }) => {
        if (profile.includes('-jspi') && (typeof WebAssembly.Suspending !== 'function' || typeof WebAssembly.promising !== 'function')) {
          throw new Error('The CI browser must support JSPI');
        }
        let adapterRequests = 0;
        let fallbackAdapterRequests = 0;
        const hostNavigator = {
          gpu: {
            requestAdapter: async () => {
              adapterRequests++;
              await new Promise(resolve => setTimeout(resolve, 0));
              return null;
            },
          },
        };
        const ambientGpu = variant === 'browser' ? {
          requestAdapter: async () => {
            fallbackAdapterRequests++;
            await new Promise(resolve => setTimeout(resolve, 0));
            return null;
          },
        } : hostNavigator.gpu;
        Object.defineProperty(navigator, 'gpu', {
          configurable: true,
          value: ambientGpu,
        });
        const failures = [];
        addEventListener('unhandledrejection', event => failures.push(String(event.reason)));
        const { createCore } = await import('/host/index.mjs');
        const moduleOptions = { print() {}, printErr() {} };
        if (variant === 'browser') {
          moduleOptions.hostNavigator = hostNavigator;
        }
        const core = await createCore({ profile, variant, moduleOptions });
        // Create the fallback instance before either acquisition. A shared
        // navigator binding would redirect the first instance to ambientGpu.
        const fallbackCore = variant === 'browser'
          ? await createCore({ profile, variant, moduleOptions: { print() {}, printErr() {} } })
          : undefined;
        const expectedPointerBytes = profile.includes('wasm64') ? 8 : 4;
        if (core.pointerBytes !== expectedPointerBytes) {
          throw new Error('Unexpected pointer width');
        }
        const fileRowsVersion = await core.api.bicore_file_backed_rows_version();
        if (fileRowsVersion !== (expectedPointerBytes === 8 ? 1 : 0)) {
          throw new Error('Unexpected file-backed rows capability');
        }

        async function checkMissingAdapter(selectedCore) {
          // Keep this first: backend registry initialization must reach requestAdapter.
          const pending = selectedCore.api.ggml_backend_init_by_type(selectedCore.constant('GGML_BACKEND_DEVICE_TYPE_GPU'), 0n);
          if (!selectedCore.busy) {
            throw new Error('The wrapper released the pending suspended call');
          }
          let timer;
          try {
            const backend = await Promise.race([
              pending,
              new Promise((_, reject) => {
                timer = setTimeout(() => reject(new Error('Suspension did not finish')), 5000);
              }),
            ]);
            await new Promise(resolve => setTimeout(resolve, 0));
            if (backend !== 0n || selectedCore.busy || failures.length) {
              throw new Error(`Invalid suspension completion: ${JSON.stringify({ backend: String(backend), busy: selectedCore.busy, failures })}`);
            }
          } finally {
            clearTimeout(timer);
          }
        }
        await checkMissingAdapter(core);
        if (adapterRequests !== 1 || fallbackAdapterRequests !== 0) {
          throw new Error('The native module did not use its selected host navigator');
        }
        if (fallbackCore) {
          await checkMissingAdapter(fallbackCore);
          if (adapterRequests !== 1 || fallbackAdapterRequests !== 1) {
            throw new Error('The independent module did not use the default navigator');
          }
        }
        if (navigator.gpu !== ambientGpu) {
          throw new Error('The native module changed the ambient GPU object');
        }
        return {
          profile,
          variant,
          pointerBytes: core.pointerBytes,
          fileRowsVersion,
          adapterRequests,
          mockedAdapter: true,
          hostNavigatorInjection: variant === 'browser',
          defaultNavigatorFallback: true,
          independentFactories: variant === 'browser',
          suspension: true,
          passed: true,
        };
      }, { profile, variant });
      results.push(suspensionResult);
      console.log(JSON.stringify(suspensionResult));
      await suspensionPage.close();
    }
  }
  for (const variant of variants) {
    for (const profile of ['cpu-wasm32']) {
      const page = await browser.newPage();
      page.on('console', msg => {
        if (msg.type() === 'error') {
          console.error(msg.text());
        }
      });
      await page.goto(url);
      const result = await page.evaluate(async ({ profile, variant, detailed, chatTemplate }) => {
        const { createCore, attachCore, schema, mountReadOnlyFile } = await import('/host/index.mjs');
        let chatSurface;
        if (detailed) {
          const { default: createNative } = await import(`/profiles/${profile}/${variant}/core.mjs`);
          const native = await createNative({ print() {}, printErr() {} });
          const { checkChatSurface } = await import('/chat-surface.mjs');
          const chatCore = attachCore(native, schema, { suspension: 'direct' });
          chatSurface = await checkChatSurface({ core: chatCore, template: chatTemplate });
        }
        const core = await createCore({ profile, variant, moduleOptions: { print() {}, printErr() {} } });
        if (core.pointerBytes !== 4) {
          throw new Error('Unexpected pointer width');
        }
        const fileRowsVersion = await core.api.bicore_file_backed_rows_version();
        if (fileRowsVersion !== 0) {
          throw new Error('Unexpected file-backed rows capability');
        }
        await core.api.llama_backend_init();
        const response = await fetch('/fixture.gguf');
        if (!response.ok) {
          throw new Error('Missing synthetic model');
        }
        const bytes = new Uint8Array(await response.arrayBuffer());
        const mounted = mountReadOnlyFile(core, '/models/test.gguf', {
          size: bytes.length,
          read(view, position) {
            const chunk = bytes.subarray(position, position + view.length);
            view.set(chunk);
            return chunk.length;
          },
        }, { maxChunkBytes: 1024 });
        let fileMountLifetime;
        if (detailed) {
          const { checkFileMountLifetime } = await import('/file-mount-lifetime.mjs');
          fileMountLifetime = checkFileMountLifetime(core, mountReadOnlyFile);
        }
        const params = core.allocRecord('llama_model_params');
        await core.api.llama_model_default_params(params);
        core.setField('llama_model_params', params, 'n_gpu_layers', 0);
        core.setField('llama_model_params', params, 'load_mode', core.constant('LLAMA_LOAD_MODE_NONE'));
        core.setField('llama_model_params', params, 'lazy_mode', core.constant(fileRowsVersion ? 'LLAMA_LAZY_MODE_ON' : 'LLAMA_LAZY_MODE_OFF'));
        const path = core.utf8(mounted.path);
        const model = await core.api.llama_model_load_from_file(path, params);
        if (!model) {
          throw new Error('Synthetic model load failed');
        }
        if (await core.api.bicore_model_file_backed_shard(model) !== -1) {
          throw new Error('Unexpected retained shard for the synthetic Llama model');
        }
        const cp = core.allocRecord('llama_context_params');
        await core.api.llama_context_default_params(cp);
        for (const [key, value] of Object.entries({ n_ctx: 128, n_batch: 16, n_ubatch: 16, n_threads: 1, n_threads_batch: 1 })) {
          core.setField('llama_context_params', cp, key, value);
        }
        const ctx = await core.api.llama_init_from_model(model, cp);
        if (!ctx) {
          throw new Error('Context creation failed');
        }
        const token = core.alloc(4);
        const tokenView = core.bytes(token, 4);
        new DataView(tokenView.buffer, tokenView.byteOffset, 4).setInt32(0, 1, true);
        const batch = core.allocRecord('llama_batch');
        await core.api.llama_batch_get_one(batch, token, 1);
        if (await core.api.llama_decode(ctx, batch) !== 0) {
          throw new Error('Decode failed');
        }
        const sampler = await core.api.llama_sampler_init_greedy();
        const sampled = await core.api.llama_sampler_sample(sampler, ctx, -1);
        if (sampled < 0 || sampled >= 259) {
          throw new Error('Invalid sampled token');
        }
        let stateBytes;
        if (detailed) {
          const size = await core.api.llama_state_get_size(ctx);
          const state = core.alloc(size);
          try {
            const used = await core.api.llama_state_get_data(ctx, state, size);
            if (used <= 0n || await core.api.llama_state_set_data(ctx, state, used) !== used) {
              throw new Error('State roundtrip failed');
            }
            stateBytes = String(used);
          } finally {
            core.free(state);
          }
        }
        await core.api.llama_sampler_free(sampler);
        await core.api.llama_free(ctx);
        await core.api.llama_model_free(model);
        for (const pointer of [params, path, cp, token, batch]) {
          core.free(pointer);
        }
        mounted.remove();
        await core.api.llama_backend_free();
        return {
          profile,
          variant,
          fileRowsVersion,
          syntheticModel: true,
          sampledToken: sampled,
          passed: true,
          ...(detailed ? { stateBytes, chatSurface, fileMountLifetime } : {}),
        };
      }, { profile, variant, detailed, chatTemplate });
      results.push(result);
      await page.close();
    }
  }
  return results;
}

const results = await withBrowserPackage({ packageRoot, files, run: smoke });
await writeBrowserResults({ file: READ_ENV_BIC_BROWSER_RESULTS_FILE() || 'build/browser-results.json', results });
console.log(JSON.stringify(results, null, 2));
