import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';
import { createAbiCaller, type NativeModule } from '#runtimes/shared/host/abi.mjs';

const directory = process.argv[2];
assert.ok(directory);
const factory = (await import(pathToFileURL(path.join(directory, 'fixture.mjs')).href)).default;
const module: NativeModule & { _ggml_fixture_i32(value: number): number } = await factory();
const { createLowLevelFunctions } = await import(pathToFileURL(path.join(directory, 'generated/functions.mjs')).href);
const schema: unknown = JSON.parse(readFileSync(path.join(directory, 'generated/schema.json'), 'utf8'));
let busy = false;
const gate = {
  enter() {
    if (busy) {
      throw new Error('Concurrent call');
    }
    busy = true;
  },
  leave() {
    busy = false;
  },
};
const caller = createAbiCaller({ module, schema, suspension: 'direct', gate });
const api = createLowLevelFunctions({ invoker: caller.invoker });

// A declared one-frame reference, not a copied historical function body. It
// performs the same scalar range check and awaits the same real Wasm export.
async function singleFrame(value: number): Promise<number> {
  gate.enter();
  try {
    if (!Number.isInteger(value) || value < -2147483648 || value > 2147483647) {
      throw new RangeError('i32');
    }
    return await module._ggml_fixture_i32(value);
  } finally {
    gate.leave();
  }
}

const samples: Record<string, number[]> = { singleFrame: [], abiCaller: [] };
for (let round = 0; round < 6; round++) {
  for (const [name, invoke] of [
    ['singleFrame', singleFrame],
    ['abiCaller', (value: number) => api.ggml_fixture_i32(value)],
  ] as const) {
    let result = 0;
    const iterations = round === 0 ? 20000 : 200000;
    const started = performance.now();
    for (let index = 0; index < iterations; index++) {
      result = await invoke(index);
    }
    assert.equal(result, iterations - 1);
    if (round) {
      samples[name]?.push((performance.now() - started) * 1e6 / iterations);
    }
  }
}
writeFileSync(1, JSON.stringify({ description: 'CPU JavaScript/Wasm call nanoseconds; no GPU inference or throughput measurement', node: process.version, samples }) + '\n');
