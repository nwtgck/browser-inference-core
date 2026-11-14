import assert from 'node:assert/strict';
import { test } from 'node:test';
import { loadConfiguration, selectConfiguration } from '#runtimes/stable-diffusion-cpp/tools/config';
import { resolveSources } from '#runtimes/stable-diffusion-cpp/tools/prepare';
import { sourceFixture } from './source.test-support.js';

test('profiles retain all address widths, suspension modes and variant boundaries', async () => {
  const configuration = await loadConfiguration({ checkoutRoot: process.cwd() });
  assert.deepEqual(Object.keys(configuration.profiles), ['webgpu-wasm32-asyncify', 'webgpu-wasm32-jspi', 'webgpu-wasm64-jspi']);
  assert.deepEqual(Object.keys(configuration.variants), ['browser', 'test']);
  for (const [profile, flags] of Object.entries(configuration.profiles)) {
    for (const variant of Object.keys(configuration.variants)) {
      assert.deepEqual(selectConfiguration({ configuration, profile, variant }).profile, flags);
    }
  }
  assert.throws(() => selectConfiguration({ configuration, profile: '../escape', variant: 'browser' }), /Unknown/);
});

test('only the selected mode receives build-domain validation', async () => {
  const base = await loadConfiguration({ checkoutRoot: process.cwd() });
  const valid = { memory64: false, webgpu: true, jspi: true, asyncify: false, pthreads: false, maximumMemory: 4294967296 };
  for (const unsupported of [{ ...valid, asyncify: true }, { ...valid, maximumMemory: 17179869184 }]) {
    const configuration = { ...base, profiles: { valid, future: unsupported } };
    assert.equal(selectConfiguration({ configuration, profile: 'valid', variant: 'browser' }).profile, valid);
    assert.throws(() => selectConfiguration({ configuration, profile: 'future', variant: 'browser' }));
  }
});

test('static configuration cannot be mistaken for another source root and source paths remain contained', async () => {
  await using fixture = await sourceFixture();
  await assert.rejects(loadConfiguration({ checkoutRoot: fixture.repository }), /does not belong/);
  for (const unsafePath of ['../outside', '/outside', 'vendor/../outside']) {
    const declarations = { ...fixture.declarations, ggml: { ...fixture.declarations.ggml, path: unsafePath } };
    await assert.rejects(resolveSources({ declarations, source: fixture.source }), /Invalid relative source path/);
  }
});
