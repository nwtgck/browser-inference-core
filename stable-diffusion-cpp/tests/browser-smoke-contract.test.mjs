import assert from 'node:assert/strict';
import { test } from 'node:test';
import { imageSmokeContract, imageSmokeScope } from './browser-smoke-contract.mjs';

test('image smoke scope distinguishes optional WebGPU execution without model claims', () => {
  assert.ok(imageSmokeScope(false).includes('CPU (no GPU inference)'));
  assert.ok(imageSmokeScope(true).includes('CPU and WebGPU'));
  for (const enabled of [false, true]) {
    assert.ok(imageSmokeScope(enabled).includes('bounded convolution/attention and cache-copy'));
    assert.ok(imageSmokeScope(enabled).endsWith('no trained-model image generation'));
  }
});

test('all four arithmetic probes remain mandatory in test variants', () => {
  assert.equal(imageSmokeContract.schemaVersion, 1);
  assert.deepEqual(imageSmokeContract.sharedProbes, ['timestep', 'bf16Weights']);
  assert.deepEqual(imageSmokeContract.testOnlyProbes, ['conv3dBias', 'webgpuPerformance']);
});

test('a string or missing WebGPU flag cannot label CPU evidence as WebGPU', () => {
  for (const value of [undefined, null, 0, 1, '0', '1', 'false', 'true']) {
    assert.throws(() => imageSmokeScope(value), TypeError);
  }
});
