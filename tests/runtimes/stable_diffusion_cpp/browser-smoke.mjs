// Real Wasm/Worker/filesystem/callback tests plus small synthetic Qwen timestep
// and 3D convolution graphs. Optional SDCB_TEST_WEBGPU=1 checks WebGPU arithmetic;
// neither mode is a trained-model image-generation/quality test.
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { makeFixture } from './gguf-fixture.mjs';
import { makeModelIoFixtures } from './model-io-fixtures.mjs';
import { pathToFileURL } from 'node:url';
import { imageSmokeScope } from './browser-smoke-contract.mjs';
import { runArithmeticProbes } from './arithmetic-probes.mjs';
import { assertModuleSurface, assertPublicGguf } from './core-contract.mjs';
const root = path.resolve(process.argv[2] ?? 'dist/package');
const testWebGpu = process.env.SDCB_TEST_WEBGPU === '1';
const { chromium } = await import(pathToFileURL(path.resolve(process.env.BIC_PLAYWRIGHT_MODULE || '../../.tools/browser/node_modules/playwright/index.mjs')).href);
const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://localhost').pathname);
    if (pathname === '/') {
      res.setHeader('Content-Type', 'text/html');
      res.end('<!doctype html><title>Image core boundary test</title>');
      return;
    }
    const file = path.resolve(root, '.' + pathname);
    if (!file.startsWith(root + path.sep)) {
      res.writeHead(403).end();
      return;
    }
    res.setHeader('Content-Type', file.endsWith('.mjs') ? 'text/javascript' : file.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
    res.end(await readFile(file));
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const profiles = JSON.parse(await readFile('config/profiles.json', 'utf8'));
  const failures = [], results = [];
  for (const profile of Object.keys(profiles)) {
    for (const variant of ['browser', 'test']) {
      try {
        const result = await page.evaluate(async ({ profile, variant, fixtureSource, modelIoSource, testWebGpu, scope, arithmeticSource, moduleSurfaceSource, publicGgufSource }) => {
          const run = async ({ origin, base, variant, profile, testWebGpu, scope }) => {
            const { attachCore, schema, mountReadOnlyFile } = await import(origin + '/host/index.mjs');
            const create = (await import(base + 'core.mjs')).default;
            const response = await fetch(base + 'core.wasm');
            if (!response.ok) {
              throw Error('Missing Wasm');
            }
            const module = await create({
              wasmBinary: new Uint8Array(await response.arrayBuffer()),
              locateFile(name) {
                if (name !== 'core.wasm') {
                  throw Error('Unexpected side file');
                }
                return base + name;
              },
            });
            assertModuleSurface({ module });
            const core = attachCore(module, schema, { suspension: profile.endsWith('asyncify') ? 'asyncify' : 'jspi' });
            if (core.pointerBytes !== (profile.includes('wasm64') ? 8 : 4)) {
              throw Error('Wrong address width');
            }
            const params = core.allocRecord('sd_img_gen_params_t');
            await core.api.sd_img_gen_params_init(params);
            core.setField('sd_img_gen_params_t', params, 'seed', 9007199254741009n);
            core.setField('sd_img_gen_params_t', params, 'width', 1024);
            if (core.getField('sd_img_gen_params_t', params, 'seed') !== 9007199254741009n || core.getField('sd_img_gen_params_t', params, 'width') !== 1024) {
              throw Error('Caller parameter roundtrip failed');
            }
            core.free(params);
            const contextParams = core.allocRecord('sd_ctx_params_t');
            await core.api.sd_ctx_params_init(contextParams);
            if (core.getField('sd_ctx_params_t', contextParams, 'webgpu_bf16_type') !== core.constant('SD_TYPE_F32')) {
              throw Error('BF16 default must be F32');
            }
            core.setField('sd_ctx_params_t', contextParams, 'webgpu_bf16_type', core.constant('SD_TYPE_F16'));
            if (core.getField('sd_ctx_params_t', contextParams, 'webgpu_bf16_type') !== core.constant('SD_TYPE_F16')) {
              throw Error('BF16 policy roundtrip failed');
            }
            core.free(contextParams);
            // Public snapshot ABI and failure atomicity in every pointer-width/profile.
            for (const [record, getter] of [
              ['sd_runtime_info_t', 'sd_ctx_get_runtime_info'],
              ['sd_memory_info_t', 'sd_ctx_get_memory_info'],
              ['sd_ctx_params_t', 'sd_ctx_get_params'],
            ]) {
              const out = core.allocRecord(record), size = core.recordSize(record);
              try {
                const start = Number(out), length = Number(size);
                module.HEAPU8.fill(0xa5, start, start + length);
                if (await core.api[getter](0n, out, BigInt(length)) !== 0 ||
                    module.HEAPU8.subarray(start, start + length).some(byte => byte !== 0xa5)) {
                  throw Error('Snapshot failure changed caller output');
                }
                if (await core.api[getter](0n, 0n, 0n) !== 0) {
                  throw Error('Null snapshot accepted');
                }
              } finally {
                core.free(out);
              }
            }
            if (core.constant('SD_RUNTIME_INFO_VERSION') !== 1 || core.constant('SD_MEMORY_INFO_VERSION') !== 1) {
              throw Error('Snapshot version mismatch');
            }
            // Explicit public queries; do not assume a particular adapter or memory size.
            await assertPublicGguf({ core });
            const deviceTextSize = await core.api.sd_list_devices(0n, 0n);
            const deviceText = core.alloc(deviceTextSize + 1n);
            try {
              if (await core.api.sd_list_devices(deviceText, deviceTextSize + 1n) !== deviceTextSize) {
                throw Error('Device enumeration changed unexpectedly');
              }
              core.readUtf8(deviceText);
            } finally {
              core.free(deviceText);
            }
            const deviceCount = await core.api.ggml_backend_dev_count();
            if (deviceCount > 0n) {
              const device = await core.api.ggml_backend_dev_get(0n);
              const properties = core.allocRecord('ggml_backend_dev_props');
              try {
                await core.api.ggml_backend_dev_get_props(device, properties);
                if (!core.readUtf8(core.getField('ggml_backend_dev_props', properties, 'name'))) {
                  throw Error('Device properties have no name');
                }
              } finally {
                core.free(properties);
              }
            }
            module.FS.mkdir('/models');
            const reads = [];
            for (const gib of [0, 2, 4, 8]) {
              const fixture = makeFixture(gib, gib === 2 ? 2 : 3);
              const mounted = mountReadOnlyFile(core, '/models/probe.gguf', fixture.source, { maxChunkBytes: fixture.maxChunkBytes });
              try {
                const file = module.FS.open(mounted.path, 'r');
                try {
                  module.FS.llseek(file, fixture.offset, 0);
                  const value = new Uint8Array(4);
                  if (module.FS.read(file, value, 0, 4) !== 4 || value.join(',') !== '0,0,128,63') {
                    throw Error('FS large-offset read failed');
                  }
                } finally {
                  module.FS.close(file);
                }
                if (variant === 'test') {
                  const path = core.utf8(mounted.path);
                  const pointer = core.pointerBytes === 8 ? path : Number(path);
                  try {
                    fixture.setPhase('native-offset');
                    if (module._sdc_test_gguf_offset(pointer) !== BigInt(fixture.offset)) {
                      throw Error('Native GGUF offset was truncated');
                    }
                    fixture.setPhase('native-value');
                    if (module._sdc_test_gguf_value(pointer) !== 0x3f800000) {
                      throw Error('Native C++ large-offset read failed');
                    }
                  } finally {
                    core.free(path);
                  }
                }
                reads.push(fixture.summary());
              } catch (error) {
                throw Error(`${profile}/${variant}: ${String(error)}; read trace: ${JSON.stringify(fixture.summary())}`);
              } finally {
                mounted.remove();
              }
            }
            const modelIoReads = [];
            for (const gib of [0, 4, 20]) {
              const fixture = makeModelIoFixtures(gib);
              const mounted = [fixture.safetensors, ...fixture.shards].map(file => mountReadOnlyFile(core, file.path, file.source, { maxChunkBytes: 65536 }));
              const pointer = core.utf8(fixture.safetensors.path), shard = core.utf8(fixture.shards[0].path);
              try {
                if (variant === 'test') {
                  const arg = value => core.pointerBytes === 8 ? value : Number(value);
                  if (module._sdc_test_safetensors_offset(arg(pointer)) !== BigInt(fixture.safetensors.offset) || module._sdc_test_safetensors_value(arg(pointer)) !== 0x3f800000) {
                    throw Error('Native safetensors file offset/payload mismatch');
                  }
                  if (module._sdc_test_model_tensor_count(arg(shard)) !== 2) {
                    throw Error('Native GGUF shard assembly failed');
                  }
                  mounted.pop().remove();
                  if (module._sdc_test_model_tensor_count(arg(shard)) !== 0) {
                    throw Error('Incomplete GGUF group was accepted');
                  }
                  for (const file of fixture.originalShapes) {
                    const mountedShape = mountReadOnlyFile(core, file.path, file.source, { maxChunkBytes: 65536 });
                    const path = core.utf8(file.path), name = core.utf8('weight');
                    try {
                      for (let dimension = 0; dimension < 4; dimension++) {
                        const expected = file.expected ? BigInt(file.expected[dimension]) : (1n << 64n) - 1n;
                        if (BigInt.asUintN(64, module._sdc_test_gguf_shape(arg(path), arg(name), dimension)) !== expected) {
                          throw Error('Native GGUF original-shape validation mismatch');
                        }
                      }
                      if (file.numeric && module._sdc_test_gguf_original_shape_values(arg(path)) !== 1) {
                        throw Error('Native GGUF original-shape CPU arithmetic mismatch');
                      }
                    } finally {
                      core.free(path);
                      core.free(name);
                      mountedShape.remove();
                    }
                  }
                }
                modelIoReads.push(fixture.safetensors.summary());
              } finally {
                core.free(pointer);
                core.free(shard);
                for (const file of mounted.reverse()) {
                  file.remove();
                }
              }
            }
            if (typeof core.api.sd_set_graph_diagnostics !== 'function') {
              throw Error('Graph diagnostics API missing');
            }
            const logs = [], progress = [];
            const log = module.addFunction((level, text, data) => logs.push([level, core.readUtf8(BigInt(text)), Number(data)]), 'vipp');
            const update = module.addFunction((step, steps, time, data) => progress.push([step, steps, time, Number(data)]), 'viifp');
            try {
              await core.api.sd_set_log_callback(BigInt(log), 17n);
              await core.api.sd_set_progress_callback(BigInt(update), 19n);
              if (variant === 'test') {
                module._sdc_test_callbacks();
                if (!logs.at(-1)[1].includes('native callback probe') || logs.at(-1)[2] !== 17 || JSON.stringify(progress) !== '[[1,4,0.125,19]]') {
                  throw Error('Native callback ABI mismatch');
                }
                if (logs.some(entry => entry[1].includes('graph-stage-v1 '))) {
                  throw Error('Graph diagnostics enabled by default');
                }
                const stages = () => logs.filter(entry => entry[1].includes('graph-stage-v1 ')).map(entry => entry[1].match(/event=(\w+)/)?.[1]);
                try {
                  // Re-enabling starts a fresh window, even without an intervening disable.
                  for (let attempt = 0; attempt < 2; attempt++) {
                    await core.api.sd_set_graph_diagnostics(1);
                    module._sdc_test_callbacks();
                  }
                  if (JSON.stringify(stages()) !== JSON.stringify(['begin', 'end', 'begin', 'failed', 'begin', 'end', 'begin', 'failed'])) {
                    throw Error('Graph diagnostic gate/order mismatch');
                  }
                } finally {
                  await core.api.sd_set_graph_diagnostics(0);
                }
                const beforeDisabled = stages().length;
                module._sdc_test_callbacks();
                if (stages().length !== beforeDisabled) {
                  throw Error('Disabled graph diagnostics still emitted');
                }
              } else if (module._sdc_test_callbacks !== undefined || module._sdc_test_gguf_offset !== undefined || module._sdc_test_gguf_shape !== undefined || module._sdc_test_gguf_original_shape_values !== undefined || module._sdc_test_qwen_timestep !== undefined || module._sdc_test_bf16_weights !== undefined || module._sdc_test_graph_walk !== undefined || module._sdc_test_conv3d_bias !== undefined || module._sdc_test_webgpu_performance !== undefined) {
                throw Error('Test probe leaked');
              }
              await core.api.sd_set_log_callback(0n, 0n);
              await core.api.sd_set_progress_callback(0n, 0n);
              const count = logs.length + progress.length;
              if (variant === 'test') {
                module._sdc_test_callbacks();
              }
              if (count !== logs.length + progress.length) {
                throw Error('Callback unregistration failed');
              }
            } finally {
              await core.api.sd_set_log_callback(0n, 0n);
              await core.api.sd_set_progress_callback(0n, 0n);
              module.removeFunction(log);
              module.removeFunction(update);
            }
            const arithmetic = variant === 'test'
              ? await runArithmeticProbes({ module, core, backendNames: testWebGpu ? ['CPU', 'WebGPU'] : ['CPU'] })
              : { timestep: [], bf16Weights: [], graphWalk: undefined, conv3dBias: undefined, webgpuPerformance: undefined };
            return { passed: true, reads, modelIoReads, ...arithmetic, scope };
          };
          const source = `const makeFixture = ${fixtureSource}; const makeModelIoFixtures = ${modelIoSource}; const runArithmeticProbes = ${arithmeticSource}; const assertModuleSurface = ${moduleSurfaceSource}; const assertPublicGguf = ${publicGgufSource}; const run = ${run.toString()}; onmessage = async ({ data }) => { try { postMessage({ result: await run(data) }); } catch (error) { postMessage({ error: String(error.stack || error) }); } };`;
          const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
          const worker = new Worker(url, { type: 'module' });
          let timer;
          try {
            return await new Promise((resolve, reject) => {
              timer = setTimeout(() => reject(Error(`Worker smoke timed out: ${profile}/${variant}`)), 120000);
              worker.onerror = event => reject(Error(event.message));
              worker.onmessage = ({ data }) => data.error ? reject(Error(data.error)) : resolve({ profile, variant, ...data.result });
              worker.postMessage({ origin: location.origin, base: `${location.origin}/profiles/${profile}/${variant}/`, variant, profile, testWebGpu, scope });
            });
          } finally {
            clearTimeout(timer);
            worker.terminate();
            URL.revokeObjectURL(url);
          }
        }, { profile, variant, fixtureSource: makeFixture.toString(), modelIoSource: makeModelIoFixtures.toString(), arithmeticSource: runArithmeticProbes.toString(), moduleSurfaceSource: assertModuleSurface.toString(), publicGgufSource: assertPublicGguf.toString(), testWebGpu, scope: imageSmokeScope({ testWebGpu }) });
        console.log(JSON.stringify(result, null, 2));
        results.push(result);
        // The pipeline wrapper finalizes its tested package snapshot, not build/.
        // Preserve the standalone recorder behavior outside that explicit mode.
        if (!process.env.BIC_BROWSER_RESULTS_FILE) {
          const file = path.resolve('build', profile, variant, 'provenance.json');
          const provenance = JSON.parse(await readFile(file, 'utf8'));
          provenance.validation.browserSmoke = true;
          provenance.validation.browserSmokeScope = result.scope;
          await writeFile(file, JSON.stringify(provenance, null, 2) + '\n');
        }
      } catch (error) {
        const failure = { profile, variant, passed: false, error: String(error.stack || error) };
        failures.push(failure);
        console.error(JSON.stringify(failure, null, 2));
      }
    }
  }
  if (failures.length) {
    throw new Error(`${failures.length} image browser smoke configuration(s) failed; see per-profile diagnostics`);
  }
  if (process.env.BIC_BROWSER_RESULTS_FILE) {
    const output = process.env.BIC_BROWSER_SESSION_JSON
      ? { session: JSON.parse(process.env.BIC_BROWSER_SESSION_JSON), results } : results;
    await writeFile(process.env.BIC_BROWSER_RESULTS_FILE, JSON.stringify(output, null, 2) + '\n');
  }
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
