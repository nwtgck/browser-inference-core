import test from 'node:test';
import assert from 'node:assert/strict';
import { makeFixture } from './gguf-fixture.mjs';

test('many small buffered reads do not imply a whole-model read', () => {
  const fixture = makeFixture(8);
  for (let i = 0; i < 256; ++i) fixture.source.read(new Uint8Array(4), 0);
  assert.equal(fixture.summary().calls, 256);
  assert.equal(fixture.summary().totalReadBytes, 1024);
  assert.equal(fixture.summary().violation, undefined);
});
test('uncapped reads fail at the source boundary, before allocation of padding data', () => {
  const fixture = makeFixture(8);
  assert.throws(() => fixture.source.read(new Uint8Array(4097), 0), /uncapped source read/);
});
test('a chunk-capped sequential traversal cannot hide behind a low call count', () => {
  const fixture = makeFixture(8);
  fixture.setPhase('native-offset');
  assert.throws(() => fixture.source.read(new Uint8Array(4), 1 << 20), /padding tensor/);
  assert.equal(fixture.summary().phase, 'native-offset');
});
test('repeated reads cannot exceed the independent total-byte budget', () => {
  const fixture = makeFixture(8);
  for (let i = 0; i < 64; ++i) fixture.source.read(new Uint8Array(4096), 0);
  assert.throws(() => fixture.source.read(new Uint8Array(1), 0), /total byte budget exceeded/);
  assert.equal(fixture.summary().totalReadBytes, 256 * 1024);
});
for (const gib of [0, 2, 4, 8]) test(`the scalar is readable without truncated positions at ${gib} GiB`, () => {
  const fixture = makeFixture(gib, gib === 2 ? 2 : 3);
  const destination = new Uint8Array(4);
  assert.equal(fixture.source.read(destination, fixture.offset), 4);
  assert.deepEqual([...destination], [0, 0, 128, 63]);
  assert.equal(fixture.summary().totalReadBytes, 4);
  assert.throws(() => fixture.source.read(new Uint8Array(4), fixture.offset + 1), /outside the virtual file/);
});
