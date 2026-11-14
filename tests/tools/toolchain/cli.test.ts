import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createToolchainCli } from '#tools/toolchain/cli';
import { runCachePlan, writeCachePlan } from '#tools/toolchain/cli-options';
import { cachePlan } from '#tools/toolchain/cache';
import { loadToolchainConfiguration } from '#tools/toolchain/config';

test('prepare entry selects explicit roots and passes the dawn-only option', async () => {
  const calls: { checkoutRoot: string; toolsRoot: string; dawnOnly: boolean }[] = [];
  const cli = createToolchainCli({
    prepare: async (options) => {
      calls.push(options);
    },
  });
  await cli.parseAsync(['prepare', '--checkout-root', process.cwd(), '--tools-root', '/tmp/bicore-tools', '--dawn-only']);
  assert.deepEqual(calls, [{ checkoutRoot: process.cwd(), toolsRoot: '/tmp/bicore-tools', dawnOnly: true }]);
});

test('prepare failure reaches the caller without being replaced by the observer', async () => {
  const failure = new Error('original prepare failure');
  const cli = createToolchainCli({
    prepare: async () => {
      throw failure;
    },
  });
  await assert.rejects(cli.parseAsync(['prepare', '--checkout-root', process.cwd(), '--tools-root', '/tmp/bicore-tools']), error => error === failure);
});

test('selected runtime cache planning matches the shared partition and saves its JSON output', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-cache-plan-'));
  try {
    const checkoutRoot = process.cwd();
    const selection = {
      runtime: 'llama-cpp' as const,
      checkoutRoot,
      workRoot: root,
      profile: 'cpu-wasm32',
      variant: 'browser',
      profileConfiguration: { memory64: false, jspi: false },
      variantConfiguration: { assertions: 0 },
      sourceCommit: 'a'.repeat(40),
      ccacheVersion: 'ccache 4',
      environment: 'fixture',
      os: 'Linux',
      arch: 'X64',
    };
    const actual = await runCachePlan(selection);
    const toolchain = await loadToolchainConfiguration({ checkoutRoot });
    const expected = cachePlan({ ...selection, toolchain, platform: { os: selection.os, arch: selection.arch, environment: selection.environment }, workspace: root });
    assert.equal(actual.ccKey, expected.ccKey);
    assert.equal(actual.emKey, expected.emKey);
    assert.equal(actual.emRestorePrefix, expected.emRestorePrefix);
    assert.equal(actual.environment['CCACHE_DIR'], join(root, expected.ccCachePath));
    const output = join(root, 'plan.json');
    await writeCachePlan({ value: actual, output });
    assert.equal(await readFile(output, 'utf8'), `${JSON.stringify(actual, undefined, 2)}\n`);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
