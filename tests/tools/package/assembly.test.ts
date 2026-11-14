import test from 'node:test';
import assert from 'node:assert/strict';
import { constants } from 'node:fs';
import { access, chmod, readFile, rename, rm } from 'node:fs/promises';
import { delimiter, join } from 'node:path';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { buildPackage } from '#runtimes/llama-cpp/tools/package';
import { verifyRuntimePackage as verifyLlama } from '#runtimes/llama-cpp/tools/verify';
import { verifyRuntimePackage as verifyImage } from '#runtimes/stable-diffusion-cpp/tools/verify';
import { freezePackage } from '#tools/publication/artifacts';
import { assemblePackage, validatePackage } from '#tools/package/aggregate';
import { artifactDirectories } from '#tools/shared/contracts';
import { decodePackageMetadata } from '#dto/shared';
import { makeRuntime, packageFixture, rehashRuntime, sourceRoot, fixtureSource, write } from './fixtures.test-support.js';

test('runtime assembly preserves its previous valid output on payload, schema, notice and npm failures', async () => {
  await using temporary = await temporaryDirectory();
  const root = temporary.root;
  await makeRuntime({ root, runtime: 'llama-cpp', selection: 'minimal' });
  const licenseRoot = join(root, 'llama.cpp');
  await write({ path: join(licenseRoot, 'LICENSE'), data: 'Explicit synthetic standalone fixture notice' });
  const destination = join(root, 'assembled');
  const buildRoot = join(root, 'build-inputs/llama-cpp');
  const options = { sourceRoot, buildRoot, destination, profiles: ['cpu-wasm32'], licenseRoots: [licenseRoot], checkNpmPack: false };
  await buildPackage(options);
  const original = await readFile(join(destination, 'manifest.json'));
  for (const relative of ['cpu-wasm32/test/runtime/core.wasm', 'cpu-wasm32/browser/generated/schema.json']) {
    const path = join(buildRoot, relative);
    const saved = path + '.saved';
    await rename(path, saved);
    await assert.rejects(buildPackage(options));
    assert.deepEqual(await readFile(join(destination, 'manifest.json')), original);
    await rename(saved, path);
  }
  await rm(join(licenseRoot, 'LICENSE'));
  await assert.rejects(buildPackage(options), /No license notices/u);
  assert.deepEqual(await readFile(join(destination, 'manifest.json')), original);
  await write({ path: join(licenseRoot, 'LICENSE'), data: 'Explicit synthetic standalone fixture notice' });
  const command = join(root, 'bin/npm');
  await write({
    path: command,
    data: `\
#!/bin/sh
exit 42
`,
  });
  await chmod(command, 0o755);
  const beforePath = process.env.PATH;
  using restore = {
    [Symbol.dispose]() {
      if (beforePath === undefined) {
        delete process.env.PATH;
      } else {
        process.env.PATH = beforePath;
      }
    },
  };
  process.env.PATH = join(root, 'bin') + ':' + (beforePath ?? '');
  await assert.rejects(buildPackage({ ...options, checkNpmPack: true }), /exited with 42/u);
  assert.deepEqual(await readFile(join(destination, 'manifest.json')), original);
});

test('aggregate activation requires both independently valid runtime inputs', async () => {
  await using fixture = await packageFixture();
  const original = await readFile(join(fixture.directory, 'manifest.json'));
  const image = join(fixture.root, 'package-inputs/stable-diffusion-cpp');
  await rename(image, image + '.saved');
  await assert.rejects(assemblePackage({ sourceRoot, inputs: join(fixture.root, 'package-inputs'), destination: fixture.directory, checkNpmPack: false }));
  assert.deepEqual(await readFile(join(fixture.directory, 'manifest.json')), original);
});

test('rehashing package metadata cannot introduce build hooks, peers or bundled dependencies', async () => {
  await using fixture = await packageFixture();
  const path = join(fixture.directory, artifactDirectories['llama-cpp'], 'package.json');
  const original = JSON.parse(await readFile(path, 'utf8')) as object;
  for (const [field, value] of [['scripts', { preinstall: 'unexpected' }], ['dependencies', { extra: '1' }], ['devDependencies', { extra: '1' }], ['optionalDependencies', { extra: '1' }], ['peerDependencies', { extra: '1' }], ['peerDependenciesMeta', { extra: {} }], ['bundleDependencies', ['extra']], ['bundledDependencies', ['extra']], ['workspaces', ['extra']]] as const) {
    await write({ path, data: JSON.stringify({ ...original, [field]: value }) });
    await rehashRuntime({ directory: fixture.directory, runtime: 'llama-cpp', change: () => {} });
    await assert.rejects(validatePackage({ sourceRoot, directory: fixture.directory, requireClean: true, checkNpmPack: false }), /install-time/u);
  }
});

test('standalone aggregate checks all three npm package boundaries while deferred assembly performs no pack', async () => {
  await using fixture = await packageFixture();
  let npm: string | undefined;
  for (const directory of (process.env.PATH ?? '').split(delimiter)) {
    const candidate = join(directory, 'npm');
    try {
      await access(candidate, constants.X_OK);
      npm = candidate;
      break;
    } catch { /* Try the next executable search directory. */ }
  }
  assert(npm, 'A real npm executable is required');
  const command = join(fixture.root, 'bin/npm');
  const log = join(fixture.root, 'packs.log');
  await write({ path: join(fixture.root, 'bin/package.json'), data: '{"type":"commonjs"}' });
  await write({
    path: command,
    data: `\
#!${process.execPath}
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
fs.appendFileSync(process.env.BICORE_TEST_NPM_LOG, JSON.stringify({ cwd: process.cwd(), name: JSON.parse(fs.readFileSync(path.join(process.cwd(), 'package.json'), 'utf8')).name }) + '\\n');
const child = spawn(process.env.BICORE_TEST_NPM_REAL, process.argv.slice(2), { stdio: 'inherit' });
child.on('error', error => { console.error(error); process.exitCode = 1; });
child.on('close', code => { process.exitCode = code === null ? 1 : code; });
`,
  });
  await chmod(command, 0o755);
  const previous = new Map(['PATH', 'BICORE_TEST_NPM_LOG', 'BICORE_TEST_NPM_REAL'].map(name => [name, process.env[name]]));
  using restore = {
    [Symbol.dispose]() {
      for (const [name, value] of previous) {
        if (value === undefined) {
          delete process.env[name];
        } else {
          process.env[name] = value;
        }
      }
    },
  };
  process.env.PATH = join(fixture.root, 'bin') + delimiter + (process.env.PATH ?? '');
  process.env.BICORE_TEST_NPM_LOG = log;
  process.env.BICORE_TEST_NPM_REAL = npm;
  await assemblePackage({ sourceRoot, inputs: join(fixture.root, 'package-inputs'), destination: join(fixture.root, 'deferred'), checkNpmPack: false });
  await assert.rejects(readFile(log), { code: 'ENOENT' });
  await validatePackage({ sourceRoot, directory: fixture.directory, requireClean: true, checkNpmPack: true });
  const packed = (await readFile(log, 'utf8')).trim().split('\n').map(line => JSON.parse(line) as { name: string });
  assert.deepEqual(packed.map(item => item.name), ['llama-cpp-browser-core', 'stable-diffusion-cpp-browser-core', 'browser-inference-core']);
  await write({ path: log, data: '' });
  await verifyLlama({ sourceRoot, directory: join(fixture.root, 'package-inputs/llama-cpp'), requireClean: true, checkNpmPack: true });
  await verifyImage({ sourceRoot, directory: join(fixture.root, 'package-inputs/stable-diffusion-cpp'), requireClean: true, checkNpmPack: true });
  let sourceChecks = 0;
  await freezePackage({
    sourceRoot,
    packageDirectory: fixture.directory,
    work: join(fixture.root, 'publication'),
    receipt: join(fixture.root, 'frozen.json'),
    branch: 'artifacts',
    expectedSource: fixtureSource,
    verifySource: async ({ sourceCommit, manifests }) => {
      sourceChecks++;
      const image = Object.values(Object.values(manifests['stable-diffusion-cpp'].profiles)[0]!.variants)[0]!;
      // Explicit synthetic report facts; this test counts package boundaries, not source correctness.
      return { sourceCommit, runtimes: { 'llama-cpp': { upstream: manifests['llama-cpp'].llamaCommit, preparations: {}, verifiedVariants: [] }, 'stable-diffusion-cpp': { upstreams: manifests['stable-diffusion-cpp'].upstreams, preparation: image.preparedSources, patches: [], verifiedVariants: [] } } };
    },
  });
  const handoff = (await readFile(log, 'utf8')).trim().split('\n').map(line => JSON.parse(line) as { name: string });
  assert.deepEqual(handoff.map(item => item.name), ['llama-cpp-browser-core', 'stable-diffusion-cpp-browser-core', 'browser-inference-core']);
  assert.equal(sourceChecks, 1);
  const root = join(fixture.directory, artifactDirectories['llama-cpp']);
  const metadata = decodePackageMetadata({ input: JSON.parse(await readFile(join(root, 'package.json'), 'utf8')) });
  await write({ path: join(root, 'package.json'), data: JSON.stringify({ ...metadata, private: true, files: ['api/'] }) });
  await rehashRuntime({ directory: fixture.directory, runtime: 'llama-cpp', change: () => {} });
  await assert.rejects(validatePackage({ sourceRoot, directory: fixture.directory, requireClean: true, checkNpmPack: true }), /npm pack tree mismatch/u);
});
