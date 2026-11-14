import { READ_ENV_BICORE_CHECKOUT_ROOT, READ_ENV_BICORE_TOOLING_ROOT, READ_ENV_BICORE_TOOLCHAIN_ROOT, READ_ENV_SDCB_TEST_DAWN_PACKAGE } from '#tools/shared/node-env';
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { verifyPreparedSources } from '#runtimes/stable-diffusion-cpp/tools/prepare';
import { cmakeArguments, prepareBuildInputs } from '#runtimes/stable-diffusion-cpp/tools/build';
import { loadConfiguration, selectConfiguration } from '#runtimes/stable-diffusion-cpp/tools/config';
import { regularFile } from '#tools/shared/files';
import { runProcess } from '#tools/shared/process';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

const checkoutRoot = path.resolve(READ_ENV_BICORE_CHECKOUT_ROOT() ?? process.cwd());
const toolingRoot = path.resolve(READ_ENV_BICORE_TOOLING_ROOT() ?? path.join(checkoutRoot, 'build/tooling'));
const toolchainRoot = path.resolve(READ_ENV_BICORE_TOOLCHAIN_ROOT() ?? path.join(checkoutRoot, '.tools'));
const dawnRoot = path.resolve(READ_ENV_SDCB_TEST_DAWN_PACKAGE() ?? path.join(toolchainRoot, 'emdawnwebgpu_pkg'));

// Native correctness and complete-TU compilation do not establish the generated
// JS/Wasm ABI or real shader execution. Those remain separate integration checks.
test('current image native contracts compile the production prepared sources', { timeout: 900_000 }, async () => {
  await regularFile({ path: path.join(dawnRoot, 'webgpu/include/webgpu/webgpu.h') });
  await regularFile({ path: path.join(toolingRoot, 'runtimes/stable-diffusion-cpp/tools/bindings.js') });
  await using temporary = await temporaryDirectory();
  const work = temporary.root;
  const configuration = await loadConfiguration({ checkoutRoot });
  const { prepared } = await prepareBuildInputs({
    checkoutRoot,
    snapshotRoot: path.join(work, 'inputs'),
    preparedRoot: path.join(work, 'prepared'),
    mode: { kind: 'local' },
    declarations: configuration.upstreams,
  });
  const selected = selectConfiguration({ configuration, profile: 'webgpu-wasm32-jspi', variant: 'browser' });
  const buildRoot = path.join(work, 'native');
  const command = cmakeArguments({
    context: { checkoutRoot, toolingRoot, outputRoot: work, toolchainRoot },
    inputs: prepared.inputs,
    buildRoot,
    preparedRoot: prepared.root,
    profile: selected.profile,
    variant: 'browser',
    nativeCheck: true,
    dawnRoot,
  });
  await runProcess({ command: 'cmake', args: command.slice(1), cwd: checkoutRoot, timeoutMs: 120_000 });
  await runProcess({ command: 'cmake', args: ['--build', buildRoot, '--parallel', '2'], cwd: checkoutRoot, timeoutMs: 600_000 });
  const result = await runProcess({ command: 'ctest', args: ['--test-dir', buildRoot, '--output-on-failure'], cwd: checkoutRoot, timeoutMs: 180_000 });
  assert.match(result.stdout.toString('utf8'), /100% tests passed/u);
  await verifyPreparedSources({ preparedRoot: prepared.root, facts: prepared.facts });

  const backend = path.join(prepared.inputs.upstreamRoots.ggml, 'src/ggml-webgpu/ggml-webgpu.cpp');
  const original = await readFile(backend);
  try {
    await writeFile(backend, Buffer.concat([original, Buffer.from('\n// } unrelated backend documentation\n')]));
    await runProcess({ command: 'cmake', args: ['--build', buildRoot, '--target', 'webgpu-policy-contract'], cwd: checkoutRoot, timeoutMs: 90_000 });
    await runProcess({ command: 'ctest', args: ['--test-dir', buildRoot, '-R', '^webgpu-policy-contract$', '--output-on-failure'], cwd: checkoutRoot, timeoutMs: 60_000 });
    await writeFile(backend, Buffer.concat([original, Buffer.from('\nstruct unused_contract_type { missing_contract_type member; };\n')]));
    await assert.rejects(runProcess({ command: 'cmake', args: ['--build', buildRoot, '--target', 'webgpu-policy-contract'], cwd: checkoutRoot, timeoutMs: 90_000 }), /missing_contract_type/u);
  } finally {
    await writeFile(backend, original);
  }
  await verifyPreparedSources({ preparedRoot: prepared.root, facts: prepared.facts });
});
