import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { test } from 'node:test';
import { cachePlan, cacheRestoreDecision, configureCache } from '#tools/toolchain/cache';
import type { ToolchainConfiguration } from '#tools/toolchain/config';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

const toolchain: ToolchainConfiguration = {
  emsdkVersion: '6.0.9',
  emscriptenRelease: 'a'.repeat(40),
  nodeVersion: '22.16.0',
  emscriptenAsyncifyBigIntPatch: { sourceSha256: 'b'.repeat(64), patchedSha256: 'c'.repeat(64) },
  dawnTag: 'test',
  dawnSha256: 'd'.repeat(64),
  typescriptVersion: '6.0.3',
};
const options = {
  runtime: 'llama-cpp' as const,
  profile: 'cpu-wasm32',
  variant: 'browser',
  profileConfiguration: { memory64: false, jspi: false },
  variantConfiguration: { assertions: 0 },
  sourceCommit: 'e'.repeat(40),
  toolchain,
  ccacheVersion: 'ccache 4',
  platform: { os: 'Linux', arch: 'X64', environment: 'fixture' },
  workspace: '/work',
};

test('immutable cache snapshots reuse their SDK and object partitions across source commits', () => {
  const base = cachePlan(options);
  const next = cachePlan({ ...options, sourceCommit: 'f'.repeat(40) });
  assert.notEqual(base.emKey, next.emKey);
  assert.notEqual(base.ccKey, next.ccKey);
  assert.equal(base.emRestorePrefix, next.emRestorePrefix);
  assert.equal(base.ccRestorePrefix, next.ccRestorePrefix);
  assert.equal(cacheRestoreDecision({ requestedKey: next.emKey, matchedKey: base.emKey, outcome: 'success', restorePrefix: next.emRestorePrefix }), 'accept');
  const other = cachePlan({ ...options, profileConfiguration: { memory64: true, jspi: true } });
  assert.notEqual(base.ccRestorePrefix, other.ccRestorePrefix);
  assert.notEqual(base.emRestorePrefix, other.emRestorePrefix);
  assert.notEqual(base.emKey, other.emKey);
  assert.equal(cacheRestoreDecision({ requestedKey: next.emKey, matchedKey: other.emKey, outcome: 'success', restorePrefix: next.emRestorePrefix }), 'fresh');
  assert.equal(base.dawnKey, other.dawnKey);
});

test('runtime, profile and variant isolate compiler caches while the Dawn archive is shared by content', () => {
  const plans = [
    cachePlan(options),
    cachePlan({ ...options, runtime: 'stable-diffusion-cpp' }),
    cachePlan({ ...options, profile: 'webgpu-wasm32-jspi' }),
    cachePlan({ ...options, variant: 'test' }),
  ];
  for (const key of ['identity', 'emKey', 'emRestorePrefix', 'ccRestorePrefix', 'ccCachePath', 'ccStagingPath', 'emStagingPath'] as const) {
    assert.equal(new Set(plans.map(plan => plan[key])).size, plans.length, key);
  }
  assert.equal(new Set(plans.map(plan => plan.dawnKey)).size, 1);
});

test('toolchain, compiler, image, workspace and selected configuration changes partition compiler caches', () => {
  const base = cachePlan(options);
  const changed = [
    cachePlan({ ...options, toolchain: { ...toolchain, emsdkVersion: '6.0.10' } }),
    cachePlan({ ...options, toolchain: { ...toolchain, emscriptenRelease: '0'.repeat(40) } }),
    cachePlan({ ...options, toolchain: { ...toolchain, emscriptenAsyncifyBigIntPatch: { ...toolchain.emscriptenAsyncifyBigIntPatch, patchedSha256: '0'.repeat(64) } } }),
    cachePlan({ ...options, ccacheVersion: 'ccache 5' }),
    cachePlan({ ...options, platform: { ...options.platform, environment: 'another-image' } }),
    cachePlan({ ...options, platform: { ...options.platform, arch: 'ARM64' } }),
    cachePlan({ ...options, platform: { ...options.platform, os: 'another-os' } }),
    cachePlan({ ...options, workspace: '/another-workspace' }),
    cachePlan({ ...options, profileConfiguration: { memory64: true, jspi: false } }),
    cachePlan({ ...options, variantConfiguration: { assertions: 1 } }),
  ];
  for (const plan of changed) {
    assert.notEqual(plan.identity, base.identity);
    assert.notEqual(plan.emKey, base.emKey);
    assert.notEqual(plan.emRestorePrefix, base.emRestorePrefix);
    assert.notEqual(plan.ccRestorePrefix, base.ccRestorePrefix);
    assert.equal(plan.dawnKey, base.dawnKey);
  }
  const dawn = cachePlan({ ...options, toolchain: { ...toolchain, dawnSha256: '0'.repeat(64) } });
  assert.notEqual(dawn.dawnKey, base.dawnKey);
  assert.notEqual(dawn.emKey, base.emKey);
  assert.notEqual(dawn.emRestorePrefix, base.emRestorePrefix);
  assert.notEqual(dawn.ccRestorePrefix, base.ccRestorePrefix);
});

test('cache keys reject path injection and incomplete partition identity', () => {
  for (const profile of ['../outside', 'cpu\nINJECTED=value']) {
    assert.throws(() => cachePlan({ ...options, profile }), /Invalid cache path/u);
  }
  assert.throws(() => cachePlan({ ...options, variant: '../../outside' }), /Invalid cache path/u);
  assert.throws(() => cachePlan({ ...options, sourceCommit: 'HEAD' }), /Invalid source commit/u);
  assert.throws(() => cachePlan({ ...options, ccacheVersion: '' }), /Incomplete cache partition/u);
  assert.throws(() => cachePlan({ ...options, platform: { ...options.platform, environment: '' } }), /Incomplete cache partition/u);
  assert.throws(() => cachePlan({ ...options, workspace: '' }), /Incomplete cache partition/u);
});

test('only a successful exact or permitted partition restore is admitted', () => {
  const plan = cachePlan(options);
  assert.equal(cacheRestoreDecision({ requestedKey: plan.emKey, matchedKey: plan.emKey, outcome: 'success' }), 'accept');
  assert.equal(cacheRestoreDecision({ requestedKey: plan.emKey, matchedKey: plan.emKey, outcome: 'failure' }), 'fresh');
  assert.equal(cacheRestoreDecision({ requestedKey: plan.emKey, matchedKey: `${plan.emRestorePrefix}${'f'.repeat(40)}`, outcome: 'success', restorePrefix: plan.emRestorePrefix }), 'accept');
  assert.equal(cacheRestoreDecision({ requestedKey: plan.emKey, matchedKey: `${plan.emRestorePrefix}${'f'.repeat(40)}`, outcome: 'failure', restorePrefix: plan.emRestorePrefix }), 'fresh');
  assert.equal(cacheRestoreDecision({ requestedKey: plan.emKey, matchedKey: `${plan.emRestorePrefix}not-a-source`, outcome: 'success', restorePrefix: plan.emRestorePrefix }), 'fresh');
  assert.equal(cacheRestoreDecision({ requestedKey: plan.ccKey, matchedKey: `${plan.ccRestorePrefix}${'f'.repeat(40)}`, outcome: 'success', restorePrefix: plan.ccRestorePrefix }), 'accept');
  assert.equal(cacheRestoreDecision({ requestedKey: plan.ccKey, matchedKey: `${plan.ccRestorePrefix}not-a-source`, outcome: 'success', restorePrefix: plan.ccRestorePrefix }), 'fresh');
  assert.equal(cacheRestoreDecision({ requestedKey: plan.ccKey, matchedKey: 'another-key', outcome: 'success', restorePrefix: plan.ccRestorePrefix }), 'fresh');
  assert.equal(cacheRestoreDecision({ requestedKey: plan.ccKey, matchedKey: undefined, outcome: 'miss' }), 'fresh');
});

test('standard ccache policy is recreated independently of restored objects', async () => {
  await using temporary = await temporaryDirectory();
  const workRoot = temporary.root;
  await mkdir(join(workRoot, '.cache'));
  await writeFile(join(workRoot, '.cache/ccache.conf'), `\
sloppiness = time_macros
hard_link = true
base_dir = /old-workspace
`);
  const plan = cachePlan(options);
  const environment = await configureCache({ workRoot, plan });
  assert.notEqual(plan.ccStagingPath, plan.ccCachePath);
  assert.notEqual(plan.emStagingPath, plan.emCachePath);
  assert.equal(environment['CCACHE_DIR'], join(workRoot, plan.ccCachePath));
  assert.equal(environment['EM_CACHE'], join(workRoot, plan.emCachePath));
  assert.equal(environment['EM_COMPILER_WRAPPER'], 'ccache');
  const config = new Map<string, string>();
  for (const line of (await readFile(join(workRoot, '.cache/ccache.conf'), 'utf8')).trim().split('\n')) {
    const [name, value] = line.split('=').map(value => value.trim());
    assert.ok(name !== undefined && value !== undefined);
    config.set(name, value);
  }
  for (const [name, value] of [['compiler_check', 'content'], ['direct_mode', 'false'], ['sloppiness', ''], ['hard_link', 'false'], ['file_clone', 'false'], ['namespace', plan.identity]] as const) {
    assert.equal(config.get(name), value, name);
  }
  assert.equal(config.has('base_dir'), false);
});
