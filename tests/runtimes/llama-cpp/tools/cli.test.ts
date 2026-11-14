import assert from 'node:assert/strict';
import test from 'node:test';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ProcessError, runProcess } from '#tools/shared/process';

test('runtime CLI prints a complete long failure once', async () => {
  const checkoutRoot = process.cwd();
  const profile = `CLI_FAILURE_BEGIN_${'x'.repeat(80 * 1024)}_CLI_FAILURE_END`;
  try {
    await runProcess({
      command: process.execPath,
      args: [fileURLToPath(import.meta.resolve('#runtimes/llama-cpp/tools/cli')), 'build',
        '--checkout-root', checkoutRoot, '--tooling-root', join(checkoutRoot, 'build/tooling'),
        '--output-root', join(checkoutRoot, 'build/unused-cli-failure'), '--toolchain-root', join(checkoutRoot, 'unused-toolchain'),
        '--profile', profile, '--local'],
      cwd: checkoutRoot,
      env: { kind: 'inherit', removePrefixes: [], overrides: { BIC_METRICS_MODE: 'off' } },
      capture: 'file',
    });
    assert.fail('An unknown profile must fail');
  } catch (error) {
    assert.ok(error instanceof ProcessError);
    assert.equal(error.exitCode, 1);
    assert.equal(error.stdout.length, 0);
    const expected = `Unknown llama.cpp profile/variant: ${profile}/browser\n`;
    assert.equal(error.stderr.length, Buffer.byteLength(expected));
    assert.equal(error.stderr.toString('utf8'), expected);
  }
});
