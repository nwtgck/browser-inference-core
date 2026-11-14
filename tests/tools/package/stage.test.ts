import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { stageProfile } from '#tools/package/stage';
import { apiFiles } from '#tools/package/runtime';
import { write } from './fixtures.test-support.js';
import type { LlamaProvenance } from '#dto/llama-cpp';

function stageProvenance(): LlamaProvenance {
  return { runtime: 'llama-cpp', profile: 'cpu-wasm32', variant: 'browser', sourceCommit: 'a'.repeat(40), sourceDirty: true, configuration: {}, variantConfiguration: {}, toolchain: {}, llamaCommit: 'b'.repeat(40), emccVersion: 'fixture', nodeVersion: 'fixture', sourceInputs: { mode: 'local', publishable: false, sourceCommit: 'a'.repeat(40), files: {} }, upstreamBuildInfo: { commit: 'fixture', number: 0 }, preparedSources: { formatVersion: 1, options: { target: 'wasm', webgpu: false, gpuProfile: false, fileRows: false, qsa: true, mtmdMode: 'cpu' }, upstream: { files: 0, sha256: '0'.repeat(64) }, inputs: {}, runtimeInputs: {}, patches: [], selections: [], shaders: {}, outputs: {} }, compiledInputs: { translationUnits: {}, targetSources: {} }, cmakeCommand: [], buildCache: {}, validation: { compiled: true, browserSmoke: false, realModelInference: false } };
}

test('staging installs exact compiled/generated/provenance/notices and retains the previous group on failure', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-stage-'));
  try {
    const generatedDirectory = join(root, 'generated');
    for (const name of apiFiles) {
      await write({ path: join(generatedDirectory, name), data: 'generated ' + name });
    }
    const runtimeFiles = ['core.mjs', 'core.wasm', 'core.d.ts'].map(name => ({ name, source: join(root, 'compiled', name) }));
    for (const file of runtimeFiles) {
      await write({ path: file.source, data: 'compiled ' + file.name });
    }
    const noticeRoots = ['emscripten', 'emdawnwebgpu_pkg'].map(name => join(root, name));
    for (const source of noticeRoots) {
      await write({ path: join(source, 'LICENSE'), data: 'Complete original fixture notice\n' });
    }
    const options = { buildDirectory: join(root, 'staged'), generatedDirectory, runtimeFiles, provenance: stageProvenance(), noticeRoots, noticeDestination: join(root, 'notices') };
    await stageProfile(options);
    assert.equal(await readFile(join(options.buildDirectory, 'runtime/abi.json'), 'utf8'), 'generated schema.json');
    assert.equal(await readFile(join(options.buildDirectory, 'runtime/core.wasm'), 'utf8'), 'compiled core.wasm');
    assert.equal(await readFile(join(options.noticeDestination, 'emscripten/LICENSE'), 'utf8'), 'Complete original fixture notice\n');
    await rm(join(generatedDirectory, 'schema.json'));
    const changed = stageProvenance();
    changed.profile = 'changed';
    await assert.rejects(stageProfile({ ...options, provenance: changed }));
    assert.equal(await readFile(join(options.buildDirectory, 'provenance.json'), 'utf8'), JSON.stringify(options.provenance, undefined, 2) + '\n');
    assert.equal(await readFile(join(options.noticeDestination, 'emscripten/LICENSE'), 'utf8'), 'Complete original fixture notice\n');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('staging rejects duplicate or missing members, duplicate notice roots and linked notices', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-stage-'));
  try {
    const runtimeFiles = ['core.mjs', 'core.wasm', 'core.d.ts'].map(name => ({ name, source: join(root, name) }));
    for (const file of runtimeFiles) {
      await write({ path: file.source, data: file.name });
    }
    const generatedDirectory = join(root, 'generated');
    for (const name of apiFiles) {
      await write({ path: join(generatedDirectory, name), data: name });
    }
    const options = { buildDirectory: join(root, 'staged'), generatedDirectory, runtimeFiles, provenance: stageProvenance(), noticeRoots: [] as string[], noticeDestination: undefined as string | undefined };
    await assert.rejects(stageProfile({ ...options, runtimeFiles: [runtimeFiles[0]!, runtimeFiles[0]!] }), /exactly once/u);
    const notice = join(root, 'notices/emscripten');
    await write({ path: join(notice, 'LICENSE'), data: 'fixture notice' });
    await assert.rejects(stageProfile({ ...options, noticeRoots: [notice, notice], noticeDestination: join(root, 'notice-output') }), /distinct names/u);
    await rm(join(notice, 'LICENSE'));
    await symlink(join(root, 'core.mjs'), join(notice, 'LICENSE'));
    await assert.rejects(stageProfile({ ...options, noticeRoots: [notice], noticeDestination: join(root, 'notice-output') }), /Linked native notice/u);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
