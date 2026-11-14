import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { observeSource } from '#tools/source/capture';

import { compileArguments, type CompileEntry } from '#tools/shared/compiler-inputs';
import { annotateBrowserModule, prepareBuildInputs, verifyCompileConfiguration } from '#runtimes/llama-cpp/tools/build';
import type { Profile } from '#runtimes/llama-cpp/tools/config';

function entries({ profile }: { profile: Profile }): CompileEntry[] {
  const result: CompileEntry[] = ['mtmd-model.cpp', 'mtmd-subcontext.cpp', 'mtmd.cpp'].map(file => ({
    file,
    directory: '/tmp',
    arguments: ['clang++', '-D', `LCB_MTMD_WEBGPU_BF16_RUNTIME=${profile.webgpu ? 1 : 0}`, '-c', file],
  }));
  if (profile.webgpu) {
    result.push({ file: 'ggml-webgpu.cpp', directory: '/tmp', arguments: ['clang++', '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS', ...(profile.gpuProfile === true ? ['-DGGML_WEBGPU_GPU_PROFILE=1'] : []), '-c', 'ggml-webgpu.cpp'] });
  }
  if (profile.gpuProfile === true) {
    result.push({ file: 'webgpu-q8-diagnostic.cpp', directory: '/tmp', arguments: ['clang++', '-DLCB_WEBGPU_Q8_DIAGNOSTIC=1', '-c', 'webgpu-q8-diagnostic.cpp'] });
  }
  return result;
}

const gpu: Profile = { memory64: true, webgpu: true, jspi: true, asyncify: false, pthreads: false, maximumMemory: 17179869184, gpuProfile: true };

test('compile verification checks actual definitions and diagnostic sources', () => {
  const selected = entries({ profile: gpu });
  verifyCompileConfiguration({ entries: selected, profile: gpu });
  assert.throws(() => verifyCompileConfiguration({ entries: selected.filter(entry => entry.file !== 'webgpu-q8-diagnostic.cpp'), profile: gpu }));
  assert.throws(() => verifyCompileConfiguration({ entries: selected.map(entry => entry.file === 'ggml-webgpu.cpp' ? { ...entry, arguments: ['clang++', '-DGGML_WEBGPU_GPU_PROFILE=0'] } : entry), profile: gpu }));
  assert.throws(() => verifyCompileConfiguration({ entries: [...selected, selected[0]!], profile: gpu }));
  assert.throws(() => verifyCompileConfiguration({ entries: selected.map(entry => entry.file === 'mtmd.cpp' ? { ...entry, arguments: ['clang++', '-DLCB_MTMD_WEBGPU_BF16_RUNTIME=0'] } : entry), profile: gpu }));
});

test('BF16 compile definitions remain unambiguous in argument arrays and shell commands', () => {
  for (const webgpu of [false, true]) {
    const profile: Profile = { ...gpu, webgpu, gpuProfile: false, jspi: webgpu };
    const selected = entries({ profile });
    verifyCompileConfiguration({ entries: selected, profile });
    const commands = selected.map(({ arguments: arguments_, ...entry }) => {
      assert.ok(arguments_);
      return { ...entry, command: arguments_.join(' ') };
    });
    verifyCompileConfiguration({ entries: commands, profile });
  }
  const selected = entries({ profile: gpu });
  const valid = '-DLCB_MTMD_WEBGPU_BF16_RUNTIME=1';
  for (const invalid of [
    [], ['-DLCB_MTMD_WEBGPU_BF16_RUNTIME=0'], ['-DLCB_MTMD_WEBGPU_BF16_RUNTIME'],
    [valid, '-DLCB_MTMD_WEBGPU_BF16_RUNTIME=0'], [valid, valid],
    [valid, '-U', 'LCB_MTMD_WEBGPU_BF16_RUNTIME'], [valid, '-D', 'LCB_MTMD_WEBGPU_BF16_RUNTIME=0'],
    [valid, '-DLCB_MTMD_RETAIN_BF16=1'], [valid, '-U', 'LCB_MTMD_RETAIN_BF16'],
  ]) {
    const modified = selected.map(entry => entry.file.startsWith('mtmd') ? { ...entry, arguments: ['clang++', ...invalid] } : entry);
    assert.throws(() => verifyCompileConfiguration({ entries: modified, profile: gpu }));
  }
  for (const filename of ['mtmd.cpp', 'mtmd-model.cpp', 'mtmd-subcontext.cpp']) {
    const original = selected.find(entry => entry.file === filename);
    assert.ok(original);
    assert.throws(() => verifyCompileConfiguration({ entries: selected.filter(entry => entry !== original), profile: gpu }));
    assert.throws(() => verifyCompileConfiguration({ entries: [...selected, original], profile: gpu }));
  }
});

test('actual backend compile keeps batch parameter uploads enabled in normal and diagnostic profiles', () => {
  for (const gpuProfile of [false, true]) {
    const profile = { ...gpu, gpuProfile };
    const selected = entries({ profile });
    for (const uploads of [
      ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'], ['-D', 'GGML_WEBGPU_BATCH_PARAM_UPLOADS=1'],
    ]) {
      verifyCompileConfiguration({ profile, entries: selected.map(entry => entry.file === 'ggml-webgpu.cpp' ? { ...entry, arguments: ['clang++', ...uploads, ...(gpuProfile ? ['-DGGML_WEBGPU_GPU_PROFILE=1'] : [])] } : entry) });
    }
    for (const uploads of [
      [], ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS=0'],
      ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS', '-UGGML_WEBGPU_BATCH_PARAM_UPLOADS'],
      ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS=1', '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS=1'],
    ]) {
      assert.throws(() => verifyCompileConfiguration({ profile, entries: selected.map(entry => entry.file === 'ggml-webgpu.cpp' ? { ...entry, arguments: ['clang++', ...uploads, ...(gpuProfile ? ['-DGGML_WEBGPU_GPU_PROFILE=1'] : [])] } : entry) }), /Batch parameter uploads/u);
    }
  }
});

test('compile arguments preserve quoted paths and separate definitions', () => {
  assert.deepEqual(compileArguments({ entry: { file: 'a.cpp', directory: '/tmp', command: 'clang++ -D NAME=1 -I"/tmp/path with spaces" -c a.cpp' } }), ['clang++', '-D', 'NAME=1', '-I/tmp/path with spaces', '-c', 'a.cpp']);
  assert.throws(() => compileArguments({ entry: { file: 'a.cpp', directory: '/tmp', command: 'clang++ "unfinished' } }));
});

test('browser module annotation is idempotent and rejects an unknown Wasm boundary', () => {
  const input = 'const wasm = new URL("core.wasm", import.meta.url);';
  const output = annotateBrowserModule({ source: input });
  assert.equal(output, 'const wasm = new URL(/* @vite-ignore */"core.wasm", import.meta.url);');
  assert.equal(annotateBrowserModule({ source: output }), output);
  assert.throws(() => annotateBrowserModule({ source: `${input}\n${input}` }));
  assert.throws(() => annotateBrowserModule({ source: 'new URL("renamed.wasm", import.meta.url)' }));
});

test('compiler capture excludes Node tooling and preserves warm input timestamps', { timeout: 120000 }, async () => {
  await using output = {
    path: await mkdtemp(join(tmpdir(), 'llama-compile-inputs-')),
    async [Symbol.asyncDispose]() {
      await rm(this.path, { recursive: true, force: true });
    },
  };
  const checkoutRoot = process.cwd();
  const context = { checkoutRoot, toolingRoot: join(checkoutRoot, 'build/tooling'), outputRoot: output.path, toolchainRoot: join(output.path, 'unused-toolchain') };
  const selection = { profile: 'cpu-wasm32', variant: 'browser', kind: 'native' as const, jobs: 1, fresh: false, includeToolchainNotices: false, configureOnly: true, python: 'python3' };
  const state = await observeSource({ checkoutRoot, mode: { kind: 'local' } });
  const first = await prepareBuildInputs({ context, source: state, selection, additionalInputs: [] });
  assert.equal(first.snapshot.source.publishable, false);
  assert.equal(Object.keys(first.snapshot.files).some(path => path.endsWith('.ts') || path.includes('node_modules') || path === 'package.json'), false);
  assert.ok(Object.keys(first.snapshot.files).some(path => path.endsWith('/native/abi/metadata.cpp')));
  assert.ok(Object.keys(first.snapshot.files).includes('tools/abi/fingerprint.h.in'));
  assert.ok(Object.keys(first.snapshot.files).includes('runtimes/llama-cpp/host/emscripten/host-navigator-pre.js'));
  assert.ok(Object.keys(first.snapshot.files).includes('tools/shared/TargetSources.cmake'));
  const header = join(first.inputs.upstreamRoots['llama']!, 'include/llama.h');
  assert.ok((await readFile(header, 'utf8')).includes('llama_model'));
  const before = (await stat(header)).mtimeMs;
  const second = await prepareBuildInputs({ context, source: state, selection, additionalInputs: [] });
  assert.deepEqual(second.snapshot.files, first.snapshot.files);
  assert.equal((await stat(header)).mtimeMs, before);
});
