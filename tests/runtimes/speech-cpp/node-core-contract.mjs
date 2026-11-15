// Real packaged Wasm interop checks. No browser, device queries or model files.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [directory, ...extra] = process.argv.slice(2);
if (!directory || extra.length) {
  throw new TypeError('Usage: node --experimental-wasm-jspi node-core-contract.mjs <speech-runtime-package>');
}
if (typeof WebAssembly.Suspending !== 'function' || typeof WebAssembly.promising !== 'function') {
  throw new Error('Speech JSPI checks require a Node runtime with WebAssembly.Suspending and WebAssembly.promising');
}
const navigator = globalThis.navigator ?? {};
Object.defineProperty(navigator, 'gpu', {
  get() {
    throw Error('Node speech ABI checks must not access the GPU API');
  },
});
if (!globalThis.navigator) {
  Object.defineProperty(globalThis, 'navigator', { value: navigator });
}
const root = path.resolve(directory);
const { attachCore, schema } = await import(pathToFileURL(path.join(root, 'host/index.mjs')).href);
const { default: createModule } = await import(pathToFileURL(path.join(root, 'profiles/webgpu-wasm64-jspi/test/core.mjs')).href);
const wasmBinary = new Uint8Array(await readFile(path.join(root, 'profiles/webgpu-wasm64-jspi/test/core.wasm')));
const module = await createModule({ wasmBinary });
const core = attachCore(module, schema, { suspension: 'jspi' });
assert.equal(core.pointerBytes, 8);
assert.equal(core.schema.runtime, 'speech-cpp');
assert.equal(await core.api.speech_api_version_major(), core.constant('SPEECH_API_VERSION_MAJOR'));
assert.equal(await core.api.speech_api_version_minor(), core.constant('SPEECH_API_VERSION_MINOR'));
assert.match(core.readUtf8(await core.api.speech_version()), /^\d+\.\d+\.\d+/u);
const checks = ['packaged module, schema fingerprint, memory64 pointers and upstream API version'];

function readPointer({ storage }) {
  const bytes = core.bytes(storage, core.pointerBytes);
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getBigUint64(0, true);
}

const output = core.alloc(core.pointerBytes);
let params = 0n;
try {
  core.bytes(output, core.pointerBytes).fill(0);
  assert.equal(await core.api.speech_load_params_new(output), core.constant('SPEECH_OK'));
  params = readPointer({ storage: output });
  assert.notEqual(params, 0n);
  assert.equal(await core.api.speech_load_params_set_threads(params, 1), core.constant('SPEECH_OK'));
  assert.equal(await core.api.speech_load_params_set_threads(params, 0), core.constant('SPEECH_ERROR_INVALID_ARGUMENT'));
  assert.equal(core.readUtf8(await core.api.speech_last_error_option()), 'threads');
  assert.match(core.readUtf8(await core.api.speech_last_error()), /threads/u);
  checks.push('opaque load-parameter allocation, pointer output, validation and synchronous error access');
  core.bytes(output, core.pointerBytes).fill(0xa5);
  assert.equal(await core.api.speech_dialogue_start(0n, 24000, 0n, 0n, output), core.constant('SPEECH_ERROR_INVALID_ARGUMENT'));
  assert.equal(readPointer({ storage: output }), 0n);
  await core.api.speech_dialogue_free(0n);
  await core.api.speech_request_cancel(0n);
  checks.push('Haru dialogue null-request rejection resets caller output; null release and cancellation');
} finally {
  if (params !== 0n) {
    await core.api.speech_load_params_free(params);
  }
  core.free(output);
}

const callbackData = 0x1_0000_0042n;
const received = [];
const callback = module.addFunction((level, text, userData) => {
  received.push({ level, text: core.readUtf8(text), userData });
}, 'vijj');
try {
  await core.api.speech_log_set(BigInt(callback), callbackData);
  assert.ok(core.schema.testFunctions?.some(fn => fn.name === 'spc_test_emit_log'));
  module._spc_test_emit_log();
  assert.deepEqual(received, [{ level: core.constant('SPEECH_LOG_INFO'), text: 'speech callback probe\n', userData: callbackData }]);
  checks.push('actual upstream log callback routes C to JavaScript with UTF-8 and untruncated memory64 user data');
} finally {
  await core.api.speech_log_set(0n, 0n);
  module.removeFunction(callback);
}

console.log(JSON.stringify({
  passed: true,
  runtime: 'speech-cpp',
  pointerBytes: core.pointerBytes,
  schemaSha256: core.schema.schemaSha256,
  checks,
  scope: 'Node real packaged Wasm ABI, opaque ownership, input validation and callback interop; no browser, GPU initialization or trained-model inference',
}, null, 2));
