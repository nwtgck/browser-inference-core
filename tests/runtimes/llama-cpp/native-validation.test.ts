import { READ_ENV_BICORE_SOURCE_ROOT, READ_ENV_LCB_TEST_DAWN_PACKAGE } from '#tools/shared/node-env';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { test } from 'node:test';

import { prepareSources } from '#runtimes/llama-cpp/tools/prepare';
import { prepareBuildInputs } from '#runtimes/llama-cpp/tools/build';
import { directory, regularFile } from '#tools/shared/files';
import { runProcess } from '#tools/shared/process';
import { observeSource } from '#tools/source/capture';

const sourceRoot = resolve(READ_ENV_BICORE_SOURCE_ROOT() ?? process.cwd());
const dawn = resolve(READ_ENV_LCB_TEST_DAWN_PACKAGE() ?? join(sourceRoot, '.tools/emdawnwebgpu_pkg'));

// Native declarations and deterministic external API doubles are not GPU or
// browser execution. Required inputs must be explicit; never substitute history.
test('current native contracts use the production TypeScript preparation', { timeout: 480_000 }, async context => {
  await directory({ path: join(sourceRoot, 'runtimes/llama-cpp/vendor/llama.cpp') });
  await regularFile({ path: join(dawn, 'webgpu/include/webgpu/webgpu.h') });
  await using workspace = {
    path: await mkdtemp(join(tmpdir(), 'llama-native-contracts-')),
    async [Symbol.asyncDispose]() {
      await rm(this.path, { recursive: true, force: true });
    },
  };
  const work = workspace.path;
  const source = await observeSource({ checkoutRoot: sourceRoot, mode: { kind: 'local' } });
  const captured = await prepareBuildInputs({
    context: { checkoutRoot: sourceRoot, toolingRoot: join(sourceRoot, 'build/tooling'), outputRoot: work, toolchainRoot: join(sourceRoot, '.tools') },
    source,
    selection: { profile: 'webgpu-wasm32-jspi', variant: 'test', kind: 'native' },
    additionalInputs: ['tests/runtimes/llama-cpp/native', 'tests/runtimes/llama_cpp/mtmd-overlay'],
  });
  const { inputs } = captured;
  const upstream = inputs.upstreamRoots['llama'];
  assert.ok(upstream);
  const fixture = join(inputs.compileRoot, 'tests/runtimes/llama-cpp/native');
  let ordinaryShaders: Map<string, Buffer> | undefined;
  for (const gpuProfile of [false, true]) {
    await context.test(gpuProfile ? 'diagnostic current provider' : 'ordinary current provider', async () => {
      const mode = gpuProfile ? 'profile' : 'normal';
      const prepared = await prepareSources({
        inputs,
        output: join(work, mode, 'source'),
        options: { target: 'native', webgpu: true, gpuProfile, fileRows: false, qsa: false, mtmdMode: 'cpu' },
        python: 'python3',
      });
      assert.equal(prepared.receipt.options.target, 'native');
      assert.equal(prepared.receipt.selections[0]?.mode, 'upstream_native_context');
      assert.deepEqual(await readFile(join(prepared.root, 'src/llama-context.cpp')),
        await readFile(join(upstream, 'src/llama-context.cpp')));
      const build = join(work, mode, 'native');
      await runProcess({
        command: 'cmake',
        args: [
          '-S', fixture, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
          `-DLCB_COMPILE_ROOT=${inputs.compileRoot}`, `-DLCB_PREPARED_SOURCE=${prepared.root}`,
          `-DLCB_DAWN_PACKAGE=${dawn}`, `-DLCB_NATIVE_GPU_PROFILE=${gpuProfile ? 'ON' : 'OFF'}`,
          `-DLCB_NATIVE_CPU_CONTRACTS=${gpuProfile ? 'OFF' : 'ON'}`,
        ],
        cwd: sourceRoot,
        timeoutMs: 60_000,
      });
      await runProcess({ command: 'cmake', args: ['--build', build, '--parallel', '2'], cwd: sourceRoot, timeoutMs: 240_000 });
      const result = await runProcess({ command: 'ctest', args: ['--test-dir', build, '--output-on-failure'], cwd: sourceRoot, timeoutMs: 120_000 });
      assert.match(result.stdout.toString(), /100% tests passed/u);
      const shaderRoot = join(build, 'webgpu-provider-shaders');
      const shaders = new Map<string, Buffer>();
      for (const name of (await readdir(shaderRoot)).sort()) {
        const bytes = await readFile(join(shaderRoot, name));
        assert.ok(!bytes.includes('#include'), name);
        shaders.set(name, bytes);
      }
      assert.ok(shaders.size > 4);
      if (gpuProfile) {
        assert.deepEqual(shaders, ordinaryShaders);
      } else {
        ordinaryShaders = shaders;
      }
      for (const size of [32, 64, 128, 256]) {
        const shader = await readFile(join(shaderRoot, `q8-${size}.wgsl`), 'utf8');
        assert.ok(shader.includes(`@workgroup_size(${size})`));
        assert.ok(!shader.includes('#include'));
      }

      if (!gpuProfile) {
        for (const mtmdMode of ['cpu', 'webgpu'] as const) {
          const mtmdPrepared = mtmdMode === 'cpu' ? prepared : await prepareSources({
            inputs,
            output: join(work, 'mtmd-webgpu', 'source'),
            options: { target: 'wasm', webgpu: true, gpuProfile: false, fileRows: false, qsa: false, mtmdMode },
            python: 'python3',
          });
          const mtmdBuild = join(work, `mtmd-${mtmdMode}`, 'target');
          // The real upstream CMake target, with production include wiring.
          await runProcess({
            command: 'cmake',
            args: [
              '-S', join(inputs.compileRoot, 'tests/runtimes/llama_cpp/mtmd-overlay'), '-B', mtmdBuild, '-G', 'Ninja',
              `-DLCB_COMPILE_ROOT=${inputs.compileRoot}`, `-DLCB_PREPARED_SOURCE=${mtmdPrepared.root}`,
              `-DLCB_WEBGPU=${mtmdMode === 'webgpu' ? 'ON' : 'OFF'}`,
            ],
            cwd: sourceRoot,
            timeoutMs: 60_000,
          });
          const files = (await readFile(join(mtmdBuild, 'mtmd-sources.txt'), 'utf8')).trim().split(';');
          assert.equal(new Set(files).size, files.length);
          const mtmdRoot = join(mtmdPrepared.root, 'tools/mtmd');
          for (const file of files) {
            assert.ok(file.startsWith(`${mtmdRoot}/`), `mtmd source outside coherent tree: ${file}`);
            await regularFile({ path: file });
          }
          for (const name of ['mtmd.cpp', 'mtmd-model.cpp', 'mtmd-subcontext.cpp']) {
            assert.ok(files.includes(join(mtmdRoot, name)));
          }
          assert.deepEqual(await readFile(join(mtmdBuild, 'upstream-mtmd-headers.txt')),
            await readFile(join(mtmdBuild, 'mtmd-headers.txt')));
          const compdb = await readFile(join(mtmdBuild, 'compile_commands.json'), 'utf8');
          assert.ok(!compdb.includes(join(upstream, 'tools/mtmd')));
          assert.ok(!compdb.includes('mtmd-source-overlay'));
          assert.ok(!compdb.includes('mtmd-tts-overlay'));
          if (mtmdMode === 'webgpu') {
            const contracts = join(work, 'mtmd-webgpu', 'contracts');
            await runProcess({
              command: 'cmake',
              args: [
                '-S', join(fixture, 'cpu'), '-B', contracts, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
                `-DLCB_COMPILE_ROOT=${inputs.compileRoot}`, `-DLCB_PREPARED_SOURCE=${mtmdPrepared.root}`,
                '-DLCB_NATIVE_MTMD_CAPABILITY=1', '-DLCB_NATIVE_CHECKPOINT_WRITER=ON',
              ],
              cwd: sourceRoot,
              timeoutMs: 60_000,
            });
            await runProcess({ command: 'cmake', args: ['--build', contracts, '--target', 'mtmd-entry', 'mtmd-loader', 'mtmd-borrowed', 'checkpoint-writer', '--parallel', '2'], cwd: sourceRoot, timeoutMs: 180_000 });
            await runProcess({ command: 'ctest', args: ['--test-dir', contracts, '-R', '^(mtmd-(entry|loader|borrowed)|checkpoint-writer)$', '--output-on-failure'], cwd: sourceRoot, timeoutMs: 60_000 });
          }
        }
        const provider = join(prepared.root, 'ggml/src/ggml-webgpu/ggml-webgpu.cpp');
        const original = await readFile(provider);
        try {
          // Comments containing misleading braces are ordinary source input.
          await writeFile(provider, Buffer.concat([original, Buffer.from('\n// } unrelated provider documentation\n')]));
          await runProcess({ command: 'cmake', args: ['--build', build, '--target', 'webgpu-provider'], cwd: sourceRoot, timeoutMs: 90_000 });
          await runProcess({ command: 'ctest', args: ['--test-dir', build, '-R', '^webgpu-provider$', '--output-on-failure'], cwd: sourceRoot, timeoutMs: 60_000 });
          // Full-TU compilation must diagnose code outside the exercised helpers.
          await writeFile(provider, Buffer.concat([original, Buffer.from('\nstruct unreferenced_native_contract { missing_native_contract_type value; };\n')]));
          await assert.rejects(
            runProcess({ command: 'cmake', args: ['--build', build, '--target', 'webgpu-provider'], cwd: sourceRoot, timeoutMs: 90_000 }),
            /missing_native_contract_type/u,
          );
        } finally {
          await writeFile(provider, original);
        }
      }
    });
  }
});
