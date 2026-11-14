import assert from 'node:assert/strict';
import { chmod, copyFile, mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { inspectAst, type Compilation } from '#tools/abi/clang';
import { readClangJson } from '#tools/abi/clang-json';

const processFixture = fileURLToPath(new URL('./json-process.test-support.js', import.meta.url));

test('Clang filtered scope retains global C facts and rejects ambiguous aliases', async () => {
  const work = await mkdtemp(path.join(os.tmpdir(), 'bicore-clang-scope-'));
  await using cleanup = {
    async [Symbol.asyncDispose]() {
      await rm(work, { recursive: true, force: true });
    },
  };
  const source = fileURLToPath(new URL('../../../../../tests/tools/abi/fixtures/scope.cpp', import.meta.url));
  const compilation: Compilation = { command: 'clang++-18', args: ['-std=c++17', '-c', source, '-o', path.join(work, 'scope.o')], directory: work, file: source };
  const scope = { filters: ['mtmd', 'int32_t', 'size_t'], prefixes: ['mtmd_'], aliases: ['int32_t', 'size_t'] };
  const entries = await inspectAst({ compilation, scope });
  assert.ok(entries.some(({ node }) => node.name === 'mtmd_identity'));
  assert.ok(entries.some(({ node }) => node.name === 'mtmd_params' && node.completeDefinition));
  assert.ok(entries.some(({ node }) => node.name === 'int32_t' && node.kind === 'TypedefDecl'));
  assert.ok(!entries.some(({ node }) => node.name === 'mtmd_cpp_deleter'));
  assert.ok(!entries.some(({ node }) => node.name === 'mtmd_helper_cpp_owner'));
  await assert.rejects(inspectAst({ compilation: { ...compilation, args: [...compilation.args, '-DBICORE_AMBIGUOUS_ALIAS'] }, scope }), /Ambiguous compiler typedef size_t/u);
  await assert.rejects(inspectAst({ compilation, scope: { ...scope, filters: ['int32_t', 'size_t'] } }), /filtered declarations are incomplete.*mtmd_/u);
});

test('standard JSON sequence normalization preserves strings and compiler failures', async () => {
  const work = await mkdtemp(path.join(os.tmpdir(), 'bicore-clang-json-'));
  await using cleanup = {
    async [Symbol.asyncDispose]() {
      await rm(work, { recursive: true, force: true });
    },
  };
  const documents: unknown[] = [];
  await readClangJson({ command: process.execPath, args: [processFixture, 'compiler', 'valid'], cwd: work, onDocument: ({ input }) => documents.push(input) });
  assert.deepEqual(documents, [{ kind: 'FunctionDecl', name: 'a', value: '日本語 } \\"\n' }, { kind: 'FunctionDecl', name: 'b' }]);
  for (const [mode, reason] of [
    ['compiler_error', /actual compiler diagnostic/u],
    ['json_eof', /jq:.*exited with|parse error/us],
    ['utf8', /Invalid UTF-8 in Clang AST/u],
    ['utf8_eof', /Incomplete UTF-8 at EOF in Clang AST/u],
  ] as const) {
    await assert.rejects(readClangJson({ command: process.execPath, args: [processFixture, 'compiler', mode], cwd: work, onDocument: () => undefined }), reason);
  }
});

test('JSONL consumer rejects normalizer failure, invalid UTF-8 and truncated JSON', async () => {
  const work = await mkdtemp(path.join(os.tmpdir(), 'bicore-jq-errors-'));
  await using cleanup = {
    async [Symbol.asyncDispose]() {
      await rm(work, { recursive: true, force: true });
    },
  };
  const jq = path.join(work, 'jq');
  await copyFile(processFixture, jq);
  await chmod(jq, 0o755);
  using environment = {
    path: process.env.PATH,
    mode: process.env.BICORE_TEST_JQ_MODE,
    [Symbol.dispose]() {
      if (this.path === undefined) {
        delete process.env.PATH;
      } else {
        process.env.PATH = this.path;
      }
      if (this.mode === undefined) {
        delete process.env.BICORE_TEST_JQ_MODE;
      } else {
        process.env.BICORE_TEST_JQ_MODE = this.mode;
      }
    },
  };
  process.env.PATH = path.join(work, 'missing');
  await assert.rejects(readClangJson({ command: process.execPath, args: [processFixture, 'compiler', 'valid'], cwd: work, onDocument: () => undefined }), /spawn jq ENOENT/u);
  process.env.PATH = work + path.delimiter + (environment.path ?? '');
  for (const [mode, reason] of [['exit', /actual jq diagnostic/u], ['utf8', /UTF-8 in jq JSONL/u], ['json_eof', /JSON/u]] as const) {
    process.env.BICORE_TEST_JQ_MODE = mode;
    await assert.rejects(readClangJson({ command: process.execPath, args: [processFixture, 'compiler', 'valid'], cwd: work, onDocument: () => undefined }), reason);
  }
});
