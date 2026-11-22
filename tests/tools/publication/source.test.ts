import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyCompiledInputs, verifyLlamaPreparationReceipt } from '#tools/publication/source';
import type { PreparationReceipt } from '#dto/llama-cpp';
import type { FileIdentity } from '#dto/shared';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { captureTreeFingerprint } from '#tools/source/fingerprint';
import { localGit } from '#tools/publication/git';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

test('compiled input evidence binds actual translation units and original Unity target sources to the prepared tree', () => {
  const preparedFile: FileIdentity = { bytes: 42, sha256: 'a'.repeat(64) };
  const ownedFile = { bytes: 24, sha256: 'b'.repeat(64) };
  const compiled = { translationUnits: { 'build/unity.cpp': { bytes: 100, sha256: 'c'.repeat(64) } }, targetSources: { runtime: { 'prepared/runtime.cpp': preparedFile, 'compile/bridge.cpp': ownedFile } } };
  assert.doesNotThrow(() => verifyCompiledInputs({ compiled, source: { 'bridge.cpp': ownedFile }, prepared: { 'runtime.cpp': preparedFile } }));
  assert.throws(() => verifyCompiledInputs({ compiled, source: { 'bridge.cpp': preparedFile }, prepared: { 'runtime.cpp': preparedFile } }), /source snapshot/u);
  assert.throws(() => verifyCompiledInputs({ compiled, source: { 'bridge.cpp': ownedFile }, prepared: {} }), /absent from prepared/u);
  assert.throws(() => verifyCompiledInputs({ compiled: { translationUnits: compiled.translationUnits, targetSources: {} }, source: {}, prepared: {} }), /Missing actual prepared/u);
  preparedFile['futureMetadata'] = 'metadata';
  assert.doesNotThrow(() => verifyCompiledInputs({ compiled, source: { 'bridge.cpp': ownedFile }, prepared: { 'runtime.cpp': { bytes: 42, sha256: 'a'.repeat(64) } } }));
});

test('source preparation compares known facts while preserving unrelated artifact metadata', () => {
  const expected: PreparationReceipt = { formatVersion: 1, options: { target: 'wasm', webgpu: false, gpuProfile: false, fileRows: false, qsa: true, mtmdMode: 'cpu' }, upstream: { files: 1, sha256: 'a'.repeat(64) }, inputs: { 'runtime.cpp': { bytes: 42, sha256: 'b'.repeat(64) } }, runtimeInputs: {}, patches: [{ file: 'change.patch', target: 'llama', reason: 'required implementation', mode: 'apply', sha256: 'c'.repeat(64), files: [{ path: 'runtime.cpp', beforeSha256: 'd'.repeat(64), afterSha256: 'b'.repeat(64) }] }], selections: [], shaders: {}, outputs: { 'runtime.cpp': { bytes: 42, sha256: 'b'.repeat(64) } } };
  const recorded = structuredClone(expected);
  recorded['futureMetadata'] = { note: 'preserved' };
  recorded.options['futureMetadata'] = true;
  recorded.upstream['futureMetadata'] = 'extra';
  recorded.inputs['runtime.cpp']!['futureMetadata'] = 'extra';
  recorded.patches[0]!['futureMetadata'] = 1;
  recorded.patches[0]!.files[0]!['futureMetadata'] = 'extra';
  assert.doesNotThrow(() => verifyLlamaPreparationReceipt({ recorded, expected }));
  assert.equal(recorded['futureMetadata']['note'], 'preserved');
  recorded.outputs['runtime.cpp']!.sha256 = '0'.repeat(64);
  assert.throws(() => verifyLlamaPreparationReceipt({ recorded, expected }), /Prepared llama sources/u);
  recorded.outputs['runtime.cpp']!.sha256 = expected.outputs['runtime.cpp']!.sha256;
  recorded.options.qsa = false;
  assert.throws(() => verifyLlamaPreparationReceipt({ recorded, expected }), /Prepared llama sources/u);
});

test('source fingerprint runs and publishes the same committed recipe at the selected source head', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = temporary.root;
  const reproduce = await readFile('.github/scripts/treefingerprint.sh', 'utf8');
  await mkdir(join(sourceRoot, '.github/scripts'), { recursive: true });
  await writeFile(join(sourceRoot, '.github/scripts/treefingerprint.sh'), reproduce);
  await writeFile(join(sourceRoot, 'source.txt'), 'committed input');
  await localGit({ cwd: sourceRoot, args: ['init', '-q'] });
  await localGit({ cwd: sourceRoot, args: ['config', 'user.name', 'Fixture'] });
  await localGit({ cwd: sourceRoot, args: ['config', 'user.email', 'fixture@example.invalid'] });
  await localGit({ cwd: sourceRoot, args: ['add', '--all'] });
  const tree = (await localGit({ cwd: sourceRoot, args: ['write-tree'] })).toString('utf8').trim();
  const sourceCommit = (await localGit({ cwd: sourceRoot, args: ['commit-tree', tree], input: 'Source fixture\n' })).toString('utf8').trim();
  // update-ref is a source-fixture mutation, not a publication operation.
  const { runProcess } = await import('#tools/shared/process');
  await runProcess({ command: 'git', args: ['update-ref', 'HEAD', sourceCommit], cwd: sourceRoot });
  const observed = await captureTreeFingerprint({ sourceRoot, sourceCommit });
  assert.match(observed.value, /^[a-f0-9]{64}$/u);
  assert.equal(observed.reproduce, reproduce);
  await writeFile(join(sourceRoot, '.github/scripts/treefingerprint.sh'), 'exit 99\n');
  await writeFile(join(sourceRoot, 'source.txt'), 'dirty local change');
  assert.deepEqual(await captureTreeFingerprint({ sourceRoot, sourceCommit }), observed);
  await assert.rejects(captureTreeFingerprint({ sourceRoot, sourceCommit: '0'.repeat(40) }), /selected commit/u);
});
