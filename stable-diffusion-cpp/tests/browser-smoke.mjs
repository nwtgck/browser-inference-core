// Real Wasm/Worker/filesystem/callback tests plus small synthetic Qwen timestep
// and 3D convolution graphs. Optional SDCB_TEST_WEBGPU=1 checks WebGPU arithmetic;
// neither mode is a trained-model image-generation/quality test.
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { makeFixture } from './gguf-fixture.mjs';
import { makeModelIoFixtures } from './model-io-fixtures.mjs';
import { pathToFileURL } from 'node:url';
const root = path.resolve(process.argv[2] ?? 'dist/package');
const testWebGpu = process.env.SDCB_TEST_WEBGPU === '1';
const { chromium } = await import(pathToFileURL(path.resolve('../.tools/browser/node_modules/playwright/index.mjs')).href);
const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url ?? '/', 'http://localhost').pathname);
    if (pathname === '/') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>Image core boundary test</title>'); return; }
    const file = path.resolve(root, '.' + pathname);
    if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    res.setHeader('Content-Type', file.endsWith('.mjs') ? 'text/javascript' : file.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
    res.end(await readFile(file));
  } catch { res.writeHead(404).end(); }
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
        const result = await page.evaluate(async ({ profile, variant, fixtureSource, modelIoSource, testWebGpu }) => {
          const run = async ({ origin, base, variant, profile, testWebGpu }) => {
            const { attachCore, schema, mountReadOnlyFile } = await import(origin + '/examples/runtime/index.mjs');
            const create = (await import(base + 'core.mjs')).default;
            const response = await fetch(base + 'core.wasm');
            if (!response.ok) throw Error('Missing Wasm');
            const module = await create({ wasmBinary: new Uint8Array(await response.arrayBuffer()),
              locateFile(name) { if (name !== 'core.wasm') throw Error('Unexpected side file'); return base + name; },
            });
            if (module._sdc_abi_version() !== 2 || module._sdc_model_io_capabilities() !== 3 || module._sdb_load !== undefined) throw Error('Wrong public surface');
            const core = attachCore(module, schema, { suspension: profile.endsWith('asyncify') ? 'asyncify' : 'direct' });
            if (core.pointerBytes !== (profile.includes('wasm64') ? 8 : 4)) throw Error('Wrong address width');
            const params = core.allocRecord('sd_img_gen_params_t');
            await core.api.sd_img_gen_params_init(params);
            core.setField('sd_img_gen_params_t', params, 'seed', 9007199254741009n);
            core.setField('sd_img_gen_params_t', params, 'width', 1024);
            if (core.getField('sd_img_gen_params_t', params, 'seed') !== 9007199254741009n || core.getField('sd_img_gen_params_t', params, 'width') !== 1024) throw Error('Caller parameter roundtrip failed');
            core.free(params);
            const contextParams = core.allocRecord('sd_ctx_params_t');
            await core.api.sd_ctx_params_init(contextParams);
            if (core.getField('sd_ctx_params_t', contextParams, 'webgpu_bf16_type') !== core.constant('SD_TYPE_F32')) throw Error('BF16 default must be F32');
            core.setField('sd_ctx_params_t', contextParams, 'webgpu_bf16_type', core.constant('SD_TYPE_F16'));
            if (core.getField('sd_ctx_params_t', contextParams, 'webgpu_bf16_type') !== core.constant('SD_TYPE_F16')) throw Error('BF16 policy roundtrip failed');
            core.free(contextParams);
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
                  if (module.FS.read(file, value, 0, 4) !== 4 || value.join(',') !== '0,0,128,63') throw Error('FS large-offset read failed');
                } finally { module.FS.close(file); }
                if (variant === 'test') {
                  const path = core.utf8(mounted.path);
                  const pointer = core.pointerBytes === 8 ? path : Number(path);
                  try {
                    fixture.setPhase('native-offset');
                    if (module._sdc_test_gguf_offset(pointer) !== BigInt(fixture.offset)) throw Error('Native GGUF offset was truncated');
                    fixture.setPhase('native-value');
                    if (module._sdc_test_gguf_value(pointer) !== 0x3f800000) throw Error('Native C++ large-offset read failed');
                  } finally { core.free(path); }
                }
                reads.push(fixture.summary());
              } catch (error) {
                throw Error(`${profile}/${variant}: ${String(error)}; read trace: ${JSON.stringify(fixture.summary())}`);
              } finally { mounted.remove(); }
            }
            const modelIoReads = [];
            for (const gib of [0, 4, 20]) {
              const fixture = makeModelIoFixtures(gib);
              const mounted = [fixture.safetensors, ...fixture.shards].map(file => mountReadOnlyFile(core, file.path, file.source, { maxChunkBytes: 65536 }));
              const pointer = core.utf8(fixture.safetensors.path), shard = core.utf8(fixture.shards[0].path);
              try {
                if (variant === 'test') {
                  const arg = value => core.pointerBytes === 8 ? value : Number(value);
                  if (module._sdc_test_safetensors_offset(arg(pointer)) !== BigInt(fixture.safetensors.offset) || module._sdc_test_safetensors_value(arg(pointer)) !== 0x3f800000) throw Error('Native safetensors file offset/payload mismatch');
                  if (module._sdc_test_model_tensor_count(arg(shard)) !== 2) throw Error('Native GGUF shard assembly failed');
                  mounted.pop().remove();
                  if (module._sdc_test_model_tensor_count(arg(shard)) !== 0) throw Error('Incomplete GGUF group was accepted');
                }
                modelIoReads.push(fixture.safetensors.summary());
              } finally { core.free(pointer); core.free(shard); for (const file of mounted.reverse()) file.remove(); }
            }
            const logs = [], progress = [];
            const log = module.addFunction((level, text, data) => logs.push([level, core.readUtf8(BigInt(text)), Number(data)]), 'vipp');
            const update = module.addFunction((step, steps, time, data) => progress.push([step, steps, time, Number(data)]), 'viifp');
            try {
              await core.api.sd_set_log_callback(BigInt(log), 17n);
              await core.api.sd_set_progress_callback(BigInt(update), 19n);
              if (variant === 'test') {
                module._sdc_test_callbacks();
                if (!logs.at(-1)[1].includes('native callback probe') || logs.at(-1)[2] !== 17 || JSON.stringify(progress) !== '[[1,4,0.125,19]]') throw Error('Native callback ABI mismatch');
              } else if (module._sdc_test_callbacks !== undefined || module._sdc_test_gguf_offset !== undefined || module._sdc_test_qwen_timestep !== undefined || module._sdc_test_bf16_weights !== undefined || module._sdc_test_graph_walk !== undefined || module._sdc_test_conv3d_bias !== undefined) throw Error('Test probe leaked');
              await core.api.sd_set_log_callback(0n, 0n);
              await core.api.sd_set_progress_callback(0n, 0n);
              const count = logs.length + progress.length;
              if (variant === 'test') module._sdc_test_callbacks();
              if (count !== logs.length + progress.length) throw Error('Callback unregistration failed');
            } finally {
              await core.api.sd_set_log_callback(0n, 0n);
              await core.api.sd_set_progress_callback(0n, 0n);
              module.removeFunction(log); module.removeFunction(update);
            }
            const timestep = [];
            const bf16Weights = [];
            const conv3dBias = variant === 'test' ? [] : undefined;
            const graphWalk = variant === 'test' ? module._sdc_test_graph_walk() === 1 : undefined;
            if (graphWalk === false) throw Error('Deep graph construction/compute propagation failed');
            if (variant === 'test') {
              for (const name of testWebGpu ? ['CPU', 'WebGPU'] : ['CPU']) {
                const pointer = core.utf8(name);
                const placement = [];
                const conversions = [];
                const placementLog = module.addFunction((_level, text) => {
                  const message = core.readUtf8(BigInt(text));
                  if (message.includes('browser-placement-v1 ')) placement.push(message);
                  if (message.includes('browser-weight-conversion-v1 ')) conversions.push(message);
                }, 'vipp');
                try {
                  await core.api.sd_set_log_callback(BigInt(placementLog), 0n);
                  // Unlike the file probes, a WebGPU compute/readback can suspend.
                  const code = await module.ccall('sdc_test_qwen_timestep', 'number',
                    [core.pointerBytes === 8 ? 'bigint' : 'number'],
                    [core.pointerBytes === 8 ? pointer : Number(pointer)], { async: true });
                  if (code !== 1) throw Error(`Synthetic Qwen timestep ${name} failed: ${code}`);
                  if (placement.length !== 1 || !placement[0].includes('bf16=1 inspected=1 cpu_bf16=1 webgpu_bf16=0 other_bf16=0') || placement[0].includes('probe')) throw Error('Missing or unsafe native BF16 placement summary');
                  const expectedWeights = name === 'WebGPU'
                    ? 'cpu_unsupported_bf16=1 webgpu_weights=1 host_weights=0 other_weights=0 webgpu_cpu_bf16=1 webgpu_cpu_bf16_use_bytes=2048'
                    : 'cpu_unsupported_bf16=0 webgpu_weights=0 host_weights=1 other_weights=0 webgpu_cpu_bf16=0 webgpu_cpu_bf16_use_bytes=0';
                  if (!placement[0].includes(expectedWeights)) throw Error('Original BF16 weight placement was lost across scheduler allocation');
                  timestep.push({ backend: name, passed: true });
                  placement.length = 0;
                  const converted = await module.ccall('sdc_test_bf16_weights', 'number',
                    [core.pointerBytes === 8 ? 'bigint' : 'number'],
                    [core.pointerBytes === 8 ? pointer : Number(pointer)], { async: true });
                  if (converted !== 1) throw Error(`Synthetic BF16 weight loading ${name} failed: ${converted}`);
                  if (placement.length !== 4 || placement.some(line => !line.includes(name === 'WebGPU'
                    ? 'bf16=0 inspected=0 cpu_bf16=0 webgpu_bf16=0 other_bf16=0'
                    : 'bf16=1 inspected=1 cpu_bf16=1 webgpu_bf16=0 other_bf16=0'))) throw Error('Unexpected converted-weight placement');
                  const expectedConversions = name === 'WebGPU' ? ['f32', 'f16', 'f32', 'f16'] : [];
                  if (conversions.length !== expectedConversions.length || expectedConversions.some((target, index) =>
                    !conversions[index].includes(`target=${target} tensors=2 source_bytes=2064 destination_bytes=${target === 'f32' ? 4128 : 2064} extra_bytes=${target === 'f32' ? 2064 : 0}`))) throw Error('BF16 unique-byte accounting failed');
                  bf16Weights.push({ backend: name, passed: true });
                  const convolution = await module.ccall('sdc_test_conv3d_bias', 'number',
                    [core.pointerBytes === 8 ? 'bigint' : 'number'],
                    [core.pointerBytes === 8 ? pointer : Number(pointer)], { async: true });
                  if (convolution !== 1) throw Error(`Synthetic 3D convolution bias ${name} failed: ${convolution}`);
                  conv3dBias.push({ backend: name, passed: true });
                } finally {
                  await core.api.sd_set_log_callback(0n, 0n);
                  module.removeFunction(placementLog);
                  core.free(pointer);
                }
              }
            }
            return { passed: true, reads, modelIoReads, timestep, bf16Weights, graphWalk, conv3dBias,
              scope: 'real-Wasm Worker, public records/callbacks, sparse GGUF/safetensors/shard I/O; test variants also check synthetic Qwen BF16 timestep and 3D convolution bias graph arithmetic on ' +
                (testWebGpu ? 'CPU and WebGPU' : 'CPU (no GPU inference)') + ', plus deep graph construction/selection; no trained-model image generation' };
          };
          const source = `const makeFixture = ${fixtureSource}; const makeModelIoFixtures = ${modelIoSource}; const run = ${run.toString()}; onmessage = async ({ data }) => { try { postMessage({ result: await run(data) }); } catch (error) { postMessage({ error: String(error.stack || error) }); } };`;
          const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
          const worker = new Worker(url, { type: 'module' });
          let timer;
          try {
            return await new Promise((resolve, reject) => {
              timer = setTimeout(() => reject(Error(`Worker smoke timed out: ${profile}/${variant}`)), 120000);
              worker.onerror = event => reject(Error(event.message));
              worker.onmessage = ({ data }) => data.error ? reject(Error(data.error)) : resolve({ profile, variant, ...data.result });
              worker.postMessage({ origin: location.origin, base: `${location.origin}/profiles/${profile}/${variant}/`, variant, profile, testWebGpu });
            });
          } finally {
            clearTimeout(timer); worker.terminate(); URL.revokeObjectURL(url);
          }
        }, { profile, variant, fixtureSource: makeFixture.toString(), modelIoSource: makeModelIoFixtures.toString(), testWebGpu });
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
  if (failures.length) throw new Error(`${failures.length} image browser smoke configuration(s) failed; see per-profile diagnostics`);
  if (process.env.BIC_BROWSER_RESULTS_FILE) {
    const output = process.env.BIC_BROWSER_SESSION_JSON
      ? { session: JSON.parse(process.env.BIC_BROWSER_SESSION_JSON), results } : results;
    await writeFile(process.env.BIC_BROWSER_RESULTS_FILE, JSON.stringify(output, null, 2) + '\n');
  }
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
