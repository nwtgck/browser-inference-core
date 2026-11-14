import assert from 'node:assert/strict';
import { cp, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { runtimePath, upstreamPath } from '#runtimes/llama-cpp/tools/config';
import { prepareSources, verifyPreparedSources } from '#runtimes/llama-cpp/tools/prepare';
import { patchDirectory, webgpuPatches } from '#runtimes/llama-cpp/tools/recipe';

const sourceRoot = process.cwd();

test('concrete normal/profile recipes preserve order and isolate dedicated templates', () => {
  const normal = webgpuPatches({ mode: 'normal' });
  const profile = webgpuPatches({ mode: 'profile' });
  for (const patches of [normal, profile]) {
    assert.equal(patches.length, new Set(patches.map(patch => patch.file)).size);
    assert.ok(patches.findIndex(patch => patch.file === 'llama-sampler-single-sync.patch') < patches.findIndex(patch => patch.file.includes('matmul-vec4-layout')));
    assert.equal(patches.find(patch => patch.file === 'ggml-webgpu-iq3s-row-split.patch')?.target, 'iq3s');
    assert.equal(patches.find(patch => patch.file === 'ggml-webgpu-aligned-word-load.patch')?.target, 'word-template');
  }
  assert.equal(normal.some(patch => patch.file.endsWith('-profile.patch')), false);
  assert.ok(profile.some(patch => patch.file === 'ggml-webgpu-q8-diagnostic.patch'));
  assert.ok(profile.findIndex(patch => patch.file === 'ggml-webgpu-q8-diagnostic.patch') < profile.findIndex(patch => patch.file === 'ggml-webgpu-matmul-vec4-layout-profile.patch'));
});

test('coherent source preparation preserves previous output and unchanged mtimes', { timeout: 120000 }, async () => {
  await using directory = {
    path: await mkdtemp(join(tmpdir(), 'llama-source-test-')),
    async [Symbol.asyncDispose]() {
      await rm(this.path, { recursive: true, force: true });
    },
  };
  const temporary = directory.path;
  const ownedSource = join(temporary, 'owned');
  const runtime = join(ownedSource, runtimePath);
  await cp(join(sourceRoot, runtimePath, 'native'), join(runtime, 'native'), { recursive: true });
  await cp(join(sourceRoot, runtimePath, patchDirectory), join(runtime, patchDirectory), { recursive: true });
  const upstream = join(temporary, 'upstream');
  await cp(join(sourceRoot, upstreamPath, 'src'), join(upstream, 'src'), { recursive: true });
  await cp(join(sourceRoot, upstreamPath, 'tools/mtmd'), join(upstream, 'tools/mtmd'), { recursive: true });
  await cp(join(sourceRoot, upstreamPath, 'common/chat.cpp'), join(upstream, 'common/chat.cpp'), { recursive: true });
  const changedHeader = join(upstream, 'tools/mtmd/mtmd-model.h');
  await writeFile(changedHeader, `${await readFile(changedHeader, 'utf8')}\n// harmless upstream comment\n`);
  const args = { inputs: { compileRoot: ownedSource, upstreamRoots: { llama: upstream } }, output: join(temporary, 'prepared'), options: { target: 'native' as const, webgpu: false, gpuProfile: false, fileRows: true, qsa: true, mtmdMode: 'cpu' as const }, python: 'python3' };
  const first = await prepareSources(args);
  await verifyPreparedSources({ prepared: first });
  const modelIdentity = first.receipt.outputs['src/models/qwen4exp.cpp'];
  assert.ok(modelIdentity);
  await verifyPreparedSources({ prepared: { ...first, receipt: { ...first.receipt, outputs: { ...first.receipt.outputs, 'src/models/qwen4exp.cpp': { ...modelIdentity, note: 'unrelated future metadata' } } } } });
  assert.ok((await readFile(join(first.root, 'tools/mtmd/mtmd-model.h'), 'utf8')).includes('// harmless upstream comment'));
  const model = join(first.root, 'src/models/qwen4exp.cpp');
  const modelHeader = join(first.root, 'src/models/models.h');
  const modelBuilder = join(first.root, 'tools/mtmd/models/qwen3vl.cpp');
  assert.ok((await readFile(model, 'utf8')).includes('file_backed'));
  assert.ok((await readFile(modelHeader, 'utf8')).includes('file_backed'));
  assert.deepEqual(await readFile(modelBuilder), await readFile(join(sourceRoot, upstreamPath, 'tools/mtmd/models/qwen3vl.cpp')));
  const before = await stat(model);
  const second = await prepareSources(args);
  assert.deepEqual(second.receipt, first.receipt);
  assert.equal((await stat(model)).mtimeMs, before.mtimeMs);
  const receiptBytes = await readFile(join(args.output, 'source-preparation.json'));
  await writeFile(join(runtime, patchDirectory, 'qwen4exp-all-pools-selection.patch'), 'invalid patch');
  await assert.rejects(prepareSources(args));
  assert.deepEqual(await readFile(join(args.output, 'source-preparation.json')), receiptBytes);
  assert.equal((await stat(model)).mtimeMs, before.mtimeMs);
  await verifyPreparedSources({ prepared: first });
  await cp(join(sourceRoot, runtimePath, patchDirectory, 'qwen4exp-all-pools-selection.patch'), join(runtime, patchDirectory, 'qwen4exp-all-pools-selection.patch'));
  await writeFile(join(upstream, 'src/models/file-backed-rows.h'), '// upstream now owns this name');
  await assert.rejects(prepareSources(args), /now exists upstream/u);
  assert.deepEqual(await readFile(join(args.output, 'source-preparation.json')), receiptBytes);
  await writeFile(model, `${await readFile(model, 'utf8')}\n// mutated after preparation\n`);
  await assert.rejects(verifyPreparedSources({ prepared: first }), /changed after preparation/u);
});
