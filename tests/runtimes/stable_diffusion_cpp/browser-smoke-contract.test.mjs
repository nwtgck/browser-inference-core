import assert from 'node:assert/strict';
import { test } from 'node:test';
import { imageSmokeContract, imageSmokeScope } from './browser-smoke-contract.mjs';

test('image smoke scope distinguishes optional WebGPU execution without model claims', () => {
  assert.ok(imageSmokeScope({ variant: 'test', testWebGpu: false }).includes('CPU (no GPU inference)'));
  assert.ok(imageSmokeScope({ variant: 'test', testWebGpu: true }).includes('CPU and WebGPU'));
  for (const enabled of [false, true]) {
    const scope = imageSmokeScope({ variant: 'test', testWebGpu: enabled });
    assert.ok(scope.includes('bounded convolution/attention, softmax/RoPE, Qwen modulation/SwiGLU and cache-copy'));
    assert.ok(scope.includes('large-row arithmetic'));
    assert.ok(scope.includes('WebGPU mode also checks live-registry queries'));
    assert.ok(scope.endsWith('no trained-model image generation'));
  }
});

test('startup scope names the Worker and large file reads without claiming test-only probes', () => {
  const scope = imageSmokeScope({ variant: 'browser', testWebGpu: false });
  assert.match(scope, /Worker.*sparse GGUF reads at large offsets/u);
  assert.doesNotMatch(scope, /arithmetic|safetensors|WebGPU/u);
  assert.equal(imageSmokeScope({ variant: 'browser', testWebGpu: true }), scope);
});

test('all four arithmetic probes remain mandatory in test variants', () => {
  assert.equal(imageSmokeContract.schemaVersion, 1);
  assert.deepEqual(imageSmokeContract.sharedProbes, ['timestep', 'bf16Weights']);
  assert.deepEqual(imageSmokeContract.testOnlyProbes, ['conv3dBias', 'webgpuPerformance']);
});

test('a string or missing WebGPU flag cannot label CPU evidence as WebGPU', () => {
  for (const value of [undefined, null, 0, 1, '0', '1', 'false', 'true']) {
    assert.throws(() => imageSmokeScope({ variant: 'test', testWebGpu: value }), TypeError);
  }
});
