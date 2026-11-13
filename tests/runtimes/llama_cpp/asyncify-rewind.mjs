/** Real Wasm suspension/rewind regression; no model or physical GPU is needed. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const profileRoot = resolve(process.argv[2] || 'dist/package/profiles/webgpu-wasm32-asyncify/test');
let adapterRequests = 0;
Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  value: { gpu: { requestAdapter: async () => { adapterRequests++; return null; } } },
});
const failures = [];
process.on('unhandledRejection', error => failures.push(error));
const { default: createNative } = await import(pathToFileURL(resolve(profileRoot, 'core.mjs')).href);
const native = await createNative({
  wasmBinary: await readFile(resolve(profileRoot, 'core.wasm')),
  print() {}, printErr() {},
});
// Keep this the first backend operation. Pre-initializing the registry with the
// zero-argument llama_backend_init would hide the missing bigint rewind arguments.
const pending = native.ccall('lcb_ggml_backend_init_by_type', 'bigint', ['number', 'bigint'], [1, 0n], { async: true });
let timer;
try {
  const result = await Promise.race([
    pending,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`Asyncify rewind did not finish: ${failures.map(String).join('; ')}`)), 5000);
    }),
  ]);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(adapterRequests, 1, 'The native call must actually suspend for an adapter request');
  assert.equal(result, 0n, 'No GPU adapter should produce a null backend');
  assert.deepEqual(failures, [], 'Rewinding must not produce an unhandled rejection');
  console.log('Asyncify bigint export suspended and rewound successfully (mock adapter, real Wasm).');
} finally {
  clearTimeout(timer);
}
