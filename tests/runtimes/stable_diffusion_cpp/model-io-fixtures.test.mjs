import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeModelIoFixtures } from './model-io-fixtures.mjs';
for (const gib of [0, 4, 20]) test(`sparse safetensors at ${gib} GiB and complete GGUF group`, () => {
  const fixtures = makeModelIoFixtures(gib);
  for (const file of [fixtures.safetensors, ...fixtures.shards]) {
    const header = new Uint8Array(8); assert.equal(file.source.read(header, 0), 8);
    const value = new Uint8Array(4); assert.equal(file.source.read(value, file.offset), 4);
    assert.deepEqual([...value], [0, 0, 128, 63]);
    assert.equal(file.source.read(value, file.size), 0);
    assert.throws(() => file.source.read(new Uint8Array(65537), 0), /uncapped/);
  }
  assert.equal(fixtures.shards.length, 2);
});
