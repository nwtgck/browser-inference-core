// Real-Wasm ABI checks, with optional test-variant CPU synthetic arithmetic.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { attachCore } from '../examples/runtime/bindings.mjs';
import { runArithmeticProbes } from './arithmetic-probes.mjs';
import { mountReadOnlyFile } from '../examples/runtime/read-only-file.mjs';
import { makeModelIoFixtures } from './model-io-fixtures.mjs';

const [directory, suspension, pointerWidth, ...extra] = process.argv.slice(2);
const arithmeticEnabled = extra.length === 1 && extra[0] === '--arithmetic';
if (!directory || !['direct', 'asyncify'].includes(suspension) ||
    !['4', '8'].includes(pointerWidth) || (extra.length && !arithmeticEnabled)) {
  throw new TypeError('Usage: node node-core-contract.mjs <build-directory> <direct|asyncify> <4|8> [--arithmetic]');
}
if (arithmeticEnabled) {
  // A direct CPU probe must never discover or initialize a WebGPU device.
  const navigator = globalThis.navigator ?? {};
  Object.defineProperty(navigator, 'gpu', {
    get() { throw Error('Node CPU contracts must not access the GPU API'); },
  });
  if (!globalThis.navigator) Object.defineProperty(globalThis, 'navigator', { value: navigator });
}
const build = path.resolve(directory);
const { default: schema } = await import(pathToFileURL(path.join(build, 'generated/schema.mjs')).href);
const { default: createCore } = await import(pathToFileURL(path.join(build, 'runtime/core.mjs')).href);
const wasmBinary = new Uint8Array(await readFile(path.join(build, 'runtime/core.wasm')));
const module = await createCore({ wasmBinary, print: line => console.error(line) });
assert.equal(schema.abiVersion, 2);
assert.match(schema.schemaSha256, /^[0-9a-f]{64}$/);
assert.equal(module._sdc_abi_version(), 2);
assert.equal(module._sdc_model_io_capabilities(), 7);
const core = attachCore(module, schema, { suspension });
assert.equal(core.pointerBytes, Number(pointerWidth));
const checks = ['ABI, schema fingerprint and pointer width'];

function readPointer(storage) {
  const bytes = core.bytes(storage, core.pointerBytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return core.pointerBytes === 8 ? view.getBigUint64(0, true) : BigInt(view.getUint32(0, true));
}

for (const [record, initialize, generate, release] of [
  ['sd_img_gen_params_t', 'sd_img_gen_params_init', 'generate_image', 'free_sd_image_generation_result'],
  ['sd_vid_gen_params_t', 'sd_vid_gen_params_init', 'generate_video', 'free_sd_video_generation_result'],
]) {
  const params = core.allocRecord(record);
  const resultOut = core.alloc(core.pointerBytes);
  try {
    await core.api[initialize](params);
    const tiling = core.fieldAddress(record, params, 'vae_tiling_params');
    assert.equal(core.getField('sd_tiling_params_t', tiling, 'tile_size_w'), 0);
    assert.equal(core.getField('sd_tiling_params_t', tiling, 'tile_size_h'), 0);
    core.setField('sd_tiling_params_t', tiling, 'tile_size_w', 320);
    core.setField('sd_tiling_params_t', tiling, 'tile_size_h', 192);
    assert.equal(core.getField('sd_tiling_params_t', tiling, 'tile_size_w'), 320);
    assert.equal(core.getField('sd_tiling_params_t', tiling, 'tile_size_h'), 192);
    core.setField(record, params, 'seed', 9007199254741009n);
    assert.equal(core.getField(record, params, 'seed'), 9007199254741009n);

    // The null context must reject before any model, backend or result allocation.
    core.bytes(resultOut, core.pointerBytes).fill(0xa5);
    assert.equal(await core.api[generate](0n, params, resultOut), 0);
    assert.equal(readPointer(resultOut), 0n);
    assert.equal(await core.api[generate](0n, params, 0n), 0);
    await core.api[release](0n);
    checks.push(`${generate}: initialized records, pixel tile fields, i64 seed, null-context result reset and null free`);
  } finally {
    core.free(resultOut);
    core.free(params);
  }
}

const progress = core.allocRecord('sd_generation_progress_t');
try {
  const completed = (1n << 53n) + 1n;
  core.setField('sd_generation_progress_t', progress, 'completed', completed);
  core.setField('sd_generation_progress_t', progress, 'total', 0n);
  assert.equal(core.getField('sd_generation_progress_t', progress, 'completed'), completed);
  assert.equal(core.getField('sd_generation_progress_t', progress, 'total'), 0n);
  core.setField('sd_generation_progress_t', progress, 'completed', -1n);
  assert.equal(core.getField('sd_generation_progress_t', progress, 'completed'), -1n);
  assert.throws(() => core.setField('sd_generation_progress_t', progress, 'total', 1n << 63n), RangeError);
  await core.api.sd_set_generation_progress_callback(0n, 0n, 0n);
  checks.push('generation progress: exact signed i64 fields and null-context callback setter');
} finally {
  core.free(progress);
}

const preview = core.allocRecord('sd_preview_info_t');
try {
  core.bytes(preview, core.recordSize('sd_preview_info_t')).fill(0xa5);
  await core.api.sd_set_preview_callback(0n, core.constant('PREVIEW_NONE'), 0, 0, 0, 0n);
  await core.api.sd_get_preview_info(preview);
  assert.equal(core.getField('sd_preview_info_t', preview, 'sample_pass'), 0);
  assert.equal(core.getField('sd_preview_info_t', preview, 'total_steps'), 0);
  checks.push('preview info: native by-value record returned through caller storage');
} finally {
  core.free(preview);
}

for (const [record, getter] of [
  ['sd_runtime_info_t', 'sd_ctx_get_runtime_info'],
  ['sd_memory_info_t', 'sd_ctx_get_memory_info'],
  ['sd_ctx_params_t', 'sd_ctx_get_params'],
]) {
  const output = core.allocRecord(record);
  const size = core.recordSize(record);
  try {
    core.bytes(output, size).fill(0xa5);
    assert.equal(await core.api[getter](0n, output, BigInt(size)), 0);
    assert.ok(core.bytes(output, size).every(byte => byte === 0xa5));
    assert.equal(await core.api[getter](0n, 0n, 0n), 0);
  } finally {
    core.free(output);
  }
}
assert.equal(core.constant('SD_RUNTIME_INFO_VERSION'), 1);
assert.equal(core.constant('SD_MEMORY_INFO_VERSION'), 1);
checks.push('snapshot null-context failure leaves caller output unchanged');

if (arithmeticEnabled) {
  module.FS.mkdir('/models');
  for (const fixture of makeModelIoFixtures(0).originalShapes) {
    const mounted = mountReadOnlyFile(core, fixture.path, fixture.source, { maxChunkBytes: 65536 });
    const path = core.utf8(fixture.path), name = core.utf8('weight');
    const argument = value => core.pointerBytes === 8 ? value : Number(value);
    try {
      for (let dimension = 0; dimension < 4; dimension++) {
        const expected = fixture.expected ? BigInt(fixture.expected[dimension]) : (1n << 64n) - 1n;
        assert.equal(BigInt.asUintN(64, module._sdc_test_gguf_shape(argument(path), argument(name), dimension)), expected, fixture.path);
      }
      if (fixture.numeric) assert.equal(module._sdc_test_gguf_original_shape_values(argument(path)), 1, fixture.path);
    } finally { core.free(path); core.free(name); mounted.remove(); }
  }
  checks.push('GGUF original shapes: production loader dimensions, malformed metadata rejection and CPU matrix arithmetic');
}

const arithmetic = arithmeticEnabled
  ? await runArithmeticProbes({ module, core, backendNames: ['CPU'] }) : undefined;
if (arithmeticEnabled) checks.push('shared smoke runner: graph walk, Qwen timestep, all six BF16 weight cases, convolution bias and bounded graph/cache arithmetic');

console.log(JSON.stringify({
  passed: true,
  suspension,
  pointerBytes: core.pointerBytes,
  schemaSha256: schema.schemaSha256,
  wasmSha256: createHash('sha256').update(wasmBinary).digest('hex'),
  checks,
  arithmetic,
  scope: arithmeticEnabled
    ? 'Node real-Wasm CPU-only ABI, record and synthetic arithmetic/file/ownership contracts; no browser, GPU API access or trained-model inference'
    : 'Node real-Wasm CPU-only ABI and record contracts; no device queries, model contexts, compute or trained-model inference',
}, null, 2));
