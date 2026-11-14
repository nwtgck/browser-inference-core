import assert from 'node:assert/strict';
import test from 'node:test';
import { decodeLlamaBrowserEnvelope } from '#dto/llama-cpp';
import { decodeStableDiffusionBrowserEnvelope } from '#dto/stable-diffusion-cpp';
import { checkedImageResults, addImageValidation } from '#tools/validation/image-results';
import { checkedLlamaResults, addLlamaValidation } from '#tools/validation/llama-results';
import { executionIdentity } from '#tools/validation/snapshot';
import { imageResults, llamaResults, readFixtureManifest, validationFixture } from './fixtures.test-support.js';

const profiles = ['cpu-wasm32', 'webgpu-wasm32-jspi'];
const variants = ['browser', 'test'];
const session = { ...executionIdentity({ runtime: 'llama-cpp', sourceCommit: 'a'.repeat(40), settings: {} }), snapshotSha256: 'b'.repeat(64), testInputsSha256: 'c'.repeat(64), testWebGpu: false };

test('image evidence requires every arithmetic probe on each backend in execution order', () => {
  for (const gpu of [false, true]) {
    const valid = imageResults({ profiles, variants, gpu });
    checkedImageResults({ profiles, variants, results: valid, testWebGpu: gpu });
    for (const key of ['timestep', 'bf16Weights', 'conv3dBias', 'webgpuPerformance'] as const) {
      for (const evidence of [[], [{ backend: 'WebGPU' as const, passed: true }], [{ backend: 'CPU' as const, passed: false }]]) {
        const results = valid.map((result) => result.variant === 'test' ? { ...result, [key]: evidence } : result);
        assert.throws(() => checkedImageResults({ profiles, variants, results, testWebGpu: gpu }), /evidence/u);
      }
      if (gpu) {
        const reversed = structuredClone(valid);
        for (const result of reversed) {
          if (result.variant === 'test') {
            result[key]?.reverse();
          }
        }
        assert.throws(() => checkedImageResults({ profiles, variants, results: reversed, testWebGpu: gpu }), /evidence/u);
      }
    }
    const wrongScope = structuredClone(valid);
    for (const result of wrongScope) {
      result.scope = 'old scope';
    }
    assert.throws(() => checkedImageResults({ profiles, variants, results: wrongScope, testWebGpu: gpu }), /scope/u);
    const wrongGraph = structuredClone(valid);
    for (const result of wrongGraph) {
      if (result.variant === 'test') {
        result.graphWalk = false;
      }
    }
    assert.throws(() => checkedImageResults({ profiles, variants, results: wrongGraph, testWebGpu: gpu }), /graph walk/u);
    for (const key of ['graphWalk', 'conv3dBias', 'webgpuPerformance'] as const) {
      const results = structuredClone(valid);
      for (const result of results) {
        if (result.variant === 'browser') {
          switch (key) {
          case 'graphWalk': result.graphWalk = true;
            break;
          case 'conv3dBias': result.conv3dBias = [];
            break;
          case 'webgpuPerformance': result.webgpuPerformance = [];
            break;
          default: {
            const exhaustive: never = key;
            throw new Error(String(exhaustive));
          }
          }
        }
      }
      assert.throws(() => checkedImageResults({ profiles, variants, results, testWebGpu: gpu }), /evidence/u);
    }
    const wrongBrowser = structuredClone(valid);
    for (const result of wrongBrowser) {
      if (result.variant === 'browser') {
        result.timestep = [{ backend: 'CPU', passed: true }];
      }
    }
    assert.throws(() => checkedImageResults({ profiles, variants, results: wrongBrowser, testWebGpu: gpu }), /evidence/u);
  }
});

test('all configured pairs must succeed exactly once with strict external boolean fields', () => {
  const valid = llamaResults({ profiles, variants });
  checkedLlamaResults({ profiles, variants, results: valid });
  const unknown = structuredClone(valid);
  const failed = structuredClone(valid);
  for (const result of unknown) {
    result.profile = 'unknown';
  }
  for (const result of failed) {
    result.passed = false;
  }
  for (const results of [valid.slice(1), [...valid, valid[0]], valid.map(() => valid[0]), unknown, failed]) {
    assert.throws(() => checkedLlamaResults({ profiles, variants, results: results.filter((result) => result !== undefined) }), /results|result/u);
  }
  for (const passed of [1, 'true', null]) {
    assert.throws(() => decodeLlamaBrowserEnvelope({ input: { session, results: valid.map((result) => ({ ...result, passed })) } }));
  }
  for (const key of ['graphWalk', 'passed'] as const) {
    assert.throws(() => decodeStableDiffusionBrowserEnvelope({ input: { session: { ...session, runtime: 'stable-diffusion-cpp' }, results: imageResults({ profiles, variants, gpu: false }).map((result) => ({ ...result, [key]: 'true' })) } }));
  }
});

test('llama evidence distinguishes synthetic CPU inference from GPU suspension and host routing', () => {
  const valid = llamaResults({ profiles, variants });
  for (const change of [{ syntheticModel: false }, { mockedAdapter: false }, { suspension: false }, { defaultNavigatorFallback: false }, { hostNavigatorInjection: true, independentFactories: true }]) {
    const results = structuredClone(valid);
    for (const result of results) {
      Object.assign(result, change);
    }
    assert.throws(() => checkedLlamaResults({ profiles, variants, results }), /scope/u);
  }
  for (const key of ['mockedAdapter', 'suspension', 'defaultNavigatorFallback', 'hostNavigatorInjection', 'independentFactories'] as const) {
    assert.throws(() => decodeLlamaBrowserEnvelope({ input: { session, results: valid.map((result) => ({ ...result, [key]: 1 })) } }));
  }
});

test('validation records clone verified evidence and preserve dirty and unrelated metadata', async () => {
  await using image = await validationFixture({ runtime: 'stable-diffusion-cpp' });
  const manifest = await readFixtureManifest({ fixture: image });
  assert.equal(manifest.runtime, 'stable-diffusion-cpp');
  if (manifest.runtime !== 'stable-diffusion-cpp') {
    throw new Error('Wrong fixture runtime');
  }
  const evidence = imageResults({ profiles: image.profiles, variants: image.variants, gpu: true });
  const first = manifest.profiles[image.profiles[0] ?? ''];
  assert.ok(first);
  const provenance = first.variants.test;
  assert.ok(provenance);
  provenance.sourceDirty = true;
  provenance.validation.realModelInference = true;
  manifest.futureMetadata = { nested: ['preserve'] };
  const updated = addImageValidation({ manifest, results: evidence });
  const result = evidence.find((item) => item.variant === 'test');
  assert.ok(result);
  result.timestep[0]!.passed = false;
  assert.equal(updated.profiles[image.profiles[0] ?? '']?.variants.test?.validation.browserSmokeEvidence?.timestep[0]?.passed, true);
  assert.equal(updated.profiles[image.profiles[0] ?? '']?.variants.test?.sourceDirty, true);
  assert.equal(updated.profiles[image.profiles[0] ?? '']?.variants.test?.validation.realModelInference, true);
  assert.deepEqual(updated.futureMetadata, manifest.futureMetadata);
  assert.equal(provenance.validation.browserSmoke, false);

  await using llama = await validationFixture({ runtime: 'llama-cpp' });
  const llamaManifest = await readFixtureManifest({ fixture: llama });
  if (llamaManifest.runtime !== 'llama-cpp') {
    throw new Error('Wrong fixture runtime');
  }
  const llamaEvidence = llamaResults({ profiles: llama.profiles, variants: llama.variants });
  for (const profile of llama.profiles) {
    for (const variant of llama.variants) {
      const provenance = llamaManifest.profiles[profile]?.variants[variant];
      assert.ok(provenance);
      provenance.sourceDirty = true;
      provenance.sourceStatusBeforeBuild = [' M README.md'];
    }
  }
  const next = addLlamaValidation({ manifest: llamaManifest, results: llamaEvidence });
  llamaEvidence[0]!.passed = false;
  const firstResult = llamaEvidence[0]!;
  assert.equal(next.profiles[firstResult.profile]?.variants[firstResult.variant]?.validation.syntheticUntrainedModel?.passed, true);
  assert.equal(next.profiles[firstResult.profile]?.variants[firstResult.variant]?.sourceDirty, true);
  assert.deepEqual(next.profiles[firstResult.profile]?.variants[firstResult.variant]?.sourceStatusBeforeBuild, [' M README.md']);
  assert.equal(llamaManifest.profiles[firstResult.profile]?.variants[firstResult.variant]?.validation.browserSmoke, false);
  assert.throws(() => checkedLlamaResults({ profiles: llama.profiles, variants: llama.variants, results: llamaEvidence.filter((result) => result.variant !== 'test') }), /results/u);
});
