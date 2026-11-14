import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { test } from 'node:test';
import ts from 'typescript';
import { loadToolchainConfiguration } from '#tools/toolchain/config';
import config from '#source/toolchain.json' with { type: 'json' };

test('toolchain values belong to the executing checkout and report the loaded compiler version', async () => {
  const checkoutRoot = process.cwd();
  const actual = await loadToolchainConfiguration({ checkoutRoot });
  assert.equal(actual.emscriptenRelease, config.emscriptenRelease);
  assert.equal(actual.typescriptVersion, ts.version);
  await assert.rejects(loadToolchainConfiguration({ checkoutRoot: resolve(checkoutRoot, 'other-checkout') }), /executing checkout/);
});
