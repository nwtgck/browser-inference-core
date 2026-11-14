import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdir, readFile, writeFile, rm, chmod } from 'node:fs/promises';
import { join } from 'node:path';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { checkSdkDeclarationCompiler } from '#runtimes/llama-cpp/tools/build';

test('declaration preflight executes the SDK-local compiler with the configured SDK Node', async () => {
  await using temporary = await temporaryDirectory();
  const buildRoot = join(temporary.root, 'build');
  const emscriptenRoot = join(temporary.root, 'emscripten');
  const binaries = join(emscriptenRoot, 'node_modules/typescript/bin');
  const globalBin = join(temporary.root, 'global-bin');
  await mkdir(buildRoot);
  await mkdir(binaries, { recursive: true });
  await mkdir(globalBin);
  const globalCompiler = join(globalBin, 'tsc');
  await writeFile(globalCompiler, '#!/bin/sh\nexit 0\n');
  await chmod(globalCompiler, 0o755);
  await writeFile(join(buildRoot, 'CMakeCache.txt'), `CMAKE_CROSSCOMPILING_EMULATOR:UNINITIALIZED=${process.execPath}\r\n`);
  const marker = join(temporary.root, 'sdk-call.json');
  const compiler = join(binaries, 'tsc');
  await writeFile(compiler, `\
const fs = require('node:fs');
fs.writeFileSync(process.env.SDK_TSC_MARKER, JSON.stringify({ arguments: process.argv.slice(2), cwd: process.cwd(), node: process.execPath }));
console.log('Version fixture');
`);
  const env = { ...process.env, PATH: globalBin, SDK_TSC_MARKER: marker };
  await checkSdkDeclarationCompiler({ emscriptenRoot, buildRoot, env });
  assert.deepEqual(JSON.parse(await readFile(marker, 'utf8')), { arguments: ['--version'], cwd: buildRoot, node: process.execPath });
  const log = await readFile(join(buildRoot, 'declaration-compiler.log'), 'utf8');
  assert.equal(log.split('\n')[0], JSON.stringify([process.execPath, compiler, '--version']));
  await rm(compiler);
  await assert.rejects(checkSdkDeclarationCompiler({ emscriptenRoot, buildRoot, env }), /SDK npm dependencies.*npm ci --prefix/su);
  await writeFile(compiler, 'process.exitCode = 7;\n');
  await assert.rejects(checkSdkDeclarationCompiler({ emscriptenRoot, buildRoot, env }), /SDK npm dependencies/su);
});
