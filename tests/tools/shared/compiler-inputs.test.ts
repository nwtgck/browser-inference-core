import assert from 'node:assert/strict';
import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { compileArguments, recordCompilerInputs } from '#tools/shared/compiler-inputs';
import { fileIdentity } from '#tools/shared/files';
import { runProcess } from '#tools/shared/process';
import { temporaryDirectory } from './fixtures.test-support.js';

test('compiler input identities follow real compilation database and target facts with specific root ownership', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = join(temporary.root, 'source');
  const preparedRoot = join(sourceRoot, 'prepared');
  const buildRoot = join(temporary.root, 'build');
  await mkdir(preparedRoot, { recursive: true });
  await mkdir(join(buildRoot, 'target-sources'), { recursive: true });
  const file = join(preparedRoot, 'unit.cpp');
  await writeFile(file, '// ordinary complete fixture\n');
  await writeFile(join(buildRoot, 'compile_commands.json'), JSON.stringify([{ directory: preparedRoot, file: 'unit.cpp', arguments: ['clang++', '-c', 'unit.cpp'], futureMetadata: 'accepted' }]));
  await writeFile(join(buildRoot, 'target-sources', 'core.txt'), `${preparedRoot}\nunit.cpp\n`);
  await writeFile(join(buildRoot, 'target-sources', 'core.txt.declared'), `${preparedRoot}\nunit.cpp\n`);
  const result = await recordCompilerInputs({ buildRoot, roots: { source: sourceRoot, prepared: preparedRoot, build: buildRoot } });
  const identity = await fileIdentity({ path: file });
  assert.deepEqual(result.translationUnits, { 'prepared/unit.cpp': identity });
  assert.deepEqual(result.targetSources, { core: { 'prepared/unit.cpp': identity } });
  await writeFile(join(buildRoot, 'compile_commands.json'), JSON.stringify([{ directory: temporary.root, file: 'outside.cpp', arguments: ['clang++', '-c', 'outside.cpp'] }]));
  await assert.rejects(recordCompilerInputs({ buildRoot, roots: { prepared: preparedRoot } }), /outside the selected/u);
});

test('compdb argument arrays are retained and command strings use the shared decoder', () => {
  assert.deepEqual(compileArguments({ entry: { directory: '/source', file: 'a.cpp', arguments: ['clang++', '-DVALUE=a b'] } }), ['clang++', '-DVALUE=a b']);
  assert.deepEqual(compileArguments({ entry: { directory: '/source', file: 'a.cpp', command: 'clang++ -DVALUE="a b"' } }), ['clang++', '-DVALUE=a b']);
  assert.throws(() => compileArguments({ entry: { directory: '/source', file: 'a.cpp' } }), /requires arguments/u);
});

test('CMake declaration facts keep real rule-named and generated inputs without internal rule bookkeeping', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = join(temporary.root, 'source');
  const buildRoot = join(temporary.root, 'build');
  await cp(resolve('tests/tools/shared/fixtures/target-sources'), sourceRoot, { recursive: true });
  await mkdir(join(buildRoot, 'generated'), { recursive: true });
  const generated = join(buildRoot, 'generated', 'schema.json');
  await writeFile(generated, '{"fixture":true}\n');
  const authoredRule = join(buildRoot, 'generated', 'authored.rule');
  await writeFile(authoredRule, 'declared generated input\n');
  await writeFile(join(buildRoot, 'generated', 'implicit-header.h'), 'late generated input\n');
  await runProcess({ command: 'cmake', args: ['-S', sourceRoot, '-B', buildRoot, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', `-DTARGET_SOURCES_HELPER=${resolve('tools/shared/TargetSources.cmake')}`], cwd: temporary.root });
  // This project has no compiler or translation units; only target declarations
  // are under test. The real compiler database boundary is covered above.
  await writeFile(join(buildRoot, 'compile_commands.json'), '[]\n');
  const result = await recordCompilerInputs({ buildRoot, roots: { source: sourceRoot, build: buildRoot } });
  assert.deepEqual(result.translationUnits, {});
  assert.deepEqual(Object.keys(result.targetSources.core ?? {}).sort(), ['build/generated/authored.rule', 'build/generated/implicit-header.h', 'build/generated/schema.json', 'source/conditional.txt', 'source/input, with space.txt', 'source/legitimate.rule']);
  await rm(authoredRule);
  await assert.rejects(recordCompilerInputs({ buildRoot, roots: { source: sourceRoot, build: buildRoot } }), /Missing|missing/u);
  await writeFile(authoredRule, 'declared generated input\n');
  await rm(generated);
  await assert.rejects(recordCompilerInputs({ buildRoot, roots: { source: sourceRoot, build: buildRoot } }), /Missing|missing/u);
});
