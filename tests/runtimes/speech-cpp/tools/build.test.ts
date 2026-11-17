import assert from 'node:assert/strict';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { cmakeArguments, resetCmakeSource } from '#runtimes/speech-cpp/tools/build';
import { loadConfiguration } from '#runtimes/speech-cpp/tools/config';
import { sourceFixture } from './source.test-support.js';

test('every profile and variant has explicit CMake source, ABI and suspension inputs', async () => {
  const { profiles, variants } = await loadConfiguration({ checkoutRoot: process.cwd() });
  const selections: string[] = [];
  for (const [profileName, profile] of Object.entries(profiles)) {
    for (const variant of Object.keys(variants)) {
      const workRoot = `/private/work/${profileName}/${variant}`;
      const context = { checkoutRoot: '/private/checkout', toolingRoot: '/private/emit', outputRoot: '/private/output', toolchainRoot: '/private/tools' };
      const inputs = { compileRoot: '/private/compiler', upstreamRoots: { 'speech': '/private/prepared/speech', ggml: '/private/prepared/ggml' } };
      const command = cmakeArguments({ context, inputs, buildRoot: workRoot, preparedRoot: '/private/prepared', profile, variant, nativeCheck: false, dawnRoot: '/private/dawn' });
      selections.push(workRoot);
      assert.ok(command.includes(`-DSPCB_VARIANT=${variant}`));
      assert.ok(command.includes(`-DSPCB_JSPI=${profile.jspi ? 'ON' : 'OFF'}`));
      assert.ok(command.includes(`-DSPCB_MEMORY64=${profile.memory64 ? 'ON' : 'OFF'}`));
      assert.ok(command.includes(`-DSPCB_MAXIMUM_MEMORY=${String(profile.maximumMemory)}`));
      assert.ok(command.includes('-DSPCB_CHECKOUT_ROOT=/private/checkout'));
      assert.ok(command.includes('-DSPCB_COMPILE_ROOT=/private/compiler'));
      assert.ok(command.includes('/private/compiler/runtimes/speech-cpp'));
      assert.ok(command.includes('-DSPCB_TOOLING_ROOT=/private/emit'));
      assert.ok(command.includes('-DSPCB_GGML_SOURCE=/private/prepared/ggml'));
      assert.ok(command.includes('-DSPCB_PROVENANCE=/private/prepared/patch-provenance.json'));
      assert.ok(command.includes('-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'));
      const native = cmakeArguments({ context, inputs, buildRoot: workRoot, preparedRoot: '/private/prepared', profile, variant, nativeCheck: true, dawnRoot: '/private/dawn' });
      assert.ok(native.includes('-DCMAKE_C_COMPILER=clang'));
      assert.ok(native.includes('-DCMAKE_CXX_COMPILER=clang++'));
    }
  }
  assert.equal(new Set(selections).size, 2);
});

test('moving selected CMake sources retires configuration while retaining compiled runtime', async () => {
  await using fixture = await sourceFixture();
  const buildRoot = path.join(fixture.root, 'build');
  await mkdir(path.join(buildRoot, 'CMakeFiles'), { recursive: true });
  await mkdir(path.join(buildRoot, 'runtime'));
  await writeFile(path.join(buildRoot, 'CMakeCache.txt'), 'CMAKE_HOME_DIRECTORY:INTERNAL=/former/source\n');
  await writeFile(path.join(buildRoot, 'CMakeFiles/stale'), 'old configuration');
  await writeFile(path.join(buildRoot, 'runtime/core.wasm'), 'previous binary');
  await resetCmakeSource({ buildRoot, compileRoot: '/selected' });
  await assert.rejects(stat(path.join(buildRoot, 'CMakeCache.txt')), /ENOENT/);
  await assert.rejects(stat(path.join(buildRoot, 'CMakeFiles')), /ENOENT/);
  assert.equal(await readFile(path.join(buildRoot, 'runtime/core.wasm'), 'utf8'), 'previous binary');
  await writeFile(path.join(buildRoot, 'CMakeCache.txt'), 'CMAKE_HOME_DIRECTORY:INTERNAL=/selected/runtimes/speech-cpp\n');
  await resetCmakeSource({ buildRoot, compileRoot: '/selected' });
  assert.ok((await stat(path.join(buildRoot, 'CMakeCache.txt'))).isFile());
});
