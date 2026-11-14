import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { test } from 'node:test';
import { ESLint, Linter } from 'eslint';
import * as parser from '@typescript-eslint/parser';
import ts from 'typescript';
import config, { rule } from '#lint/force-switch-for-union';

const root = resolve(import.meta.dirname, '../../..');
const ruleId = 'local-rules-switch/force-switch-for-union';
const cases = [
  { name: 'function parameter', code: "export function check(value: 'a' | 'b') { if (value === 'a') {} }", count: 1, condition: "value === 'a'" },
  { name: 'object property', code: "export function check(value: { kind: 'a' | 'b' }) { if (value.kind === 'a') {} }", count: 1, condition: "value.kind === 'a'" },
  { name: 'left literal', code: "export function check(value: 'a' | 'b') { if ('a' === value) {} }", count: 1 },
  { name: 'discriminated union', code: "type Action = { kind: 'create'; data: string } | { kind: 'delete'; id: number }; export function check(value: Action) { if (value.kind === 'create') { value.data; } }", count: 1 },
  { name: 'ternary property', code: "export function check(value: { mode: 'dark' | 'light' }) { return value.mode === 'dark' ? 'Dark' : 'Light'; }", count: 1 },
  { name: 'unique symbol member', code: "const symbol = Symbol('value'); export function check(value: 'a' | 'b' | typeof symbol) { if (value === 'a') {} }", count: 1 },
  { name: 'symbol member', code: "export function check(value: 'a' | symbol) { if (value === 'a') {} }", count: 1 },
  { name: 'nullable member', code: "export function check(value: 'a' | null) { if (value === 'a') {} }", count: 1 },
  { name: 'optional member', code: "export function check(value: 'a' | undefined) { if (value === 'a') {} }", count: 1 },
  { name: 'static template literal', code: "export function check(value: 'a' | 'b') { if (value === `a`) {} }", count: 1 },
  { name: 'function result', code: "declare function getValue(): 'a' | 'b'; export function check() { if (getValue() === 'a') {} }", count: 1 },
  { name: 'else-if narrowing', code: "export function check(value: 'a' | 'b' | 'c') { if (value === 'a') {} else if (value === 'b') {} }", count: 2 },
  { name: 'imported union', code: "import type { Mode } from './switch-union-import.fixture.js'; export function check(value: Mode) { if (value === 'ready') {} }", count: 1 },
  { name: 'logical expression', code: "export function check(value: 'a' | 'b', flag: boolean) { if (value === 'a' && flag) {} }", count: 0 },
  { name: 'plain string', code: "export function check(value: string) { if (value === 'a') {} }", count: 0 },
  { name: 'typeof expression', code: "export function check(value: unknown) { if (typeof value === 'string') {} }", count: 0 },
  { name: 'boolean', code: 'export function check(value: boolean) { if (value === true) {} }', count: 0 },
  { name: 'mixed number member', code: "export function check(value: 'a' | 1) { if (value === 'a') {} }", count: 0 },
  { name: 'mixed boolean member', code: "export function check(value: 'a' | boolean) { if (value === 'a') {} }", count: 0 },
  { name: 'single literal', code: "export function check(value: 'a') { if (value === 'a') {} }", count: 0 },
  { name: 'two variable operands', code: "export function check(left: 'a' | 'b', right: 'a' | 'b') { if (left === right) {} }", count: 0 },
  { name: 'interpolated template operand', code: "export function check(value: 'a' | 'b', suffix: string) { if (value === `a${suffix}`) {} }", count: 0 },
  { name: 'exhaustive switch', code: "export function check(value: 'a' | 'b') { switch (value) { case 'a': break; case 'b': break; default: { const exhaustive: never = value; return exhaustive; } } }", count: 0 },
  { name: 'incomplete switch is outside this rule', code: "export function check(value: 'a' | 'b') { switch (value) { case 'a': break; } }", count: 0 },
];
for (const operator of ['==', '!==', '!=']) {
  cases.push({ name: `comparison ${operator}`, code: `export function check(value: 'a' | 'b') { if (value ${operator} 'a') {} }`, count: 1 });
}

const filenames = cases.map((_, index) => resolve(root, `tools/switch-union-memory-fixture-${index}.ts`));
const importedFilename = resolve(root, 'tools/switch-union-import.fixture.ts');
const sources = new Map(filenames.map((filename, index) => [filename, cases[index].code]));
sources.set(importedFilename, "export type Mode = 'ready' | 'waiting';");
const configuration = ts.readConfigFile(resolve(root, 'tsconfig.tools.json'), ts.sys.readFile);
assert.equal(configuration.error, undefined);
const project = ts.parseJsonConfigFileContent(configuration.config, ts.sys, root);
assert.deepEqual(project.errors, []);
assert.equal(project.options.strict, true);
const compilerOptions = { ...project.options, noEmit: true };
const host = ts.createCompilerHost(compilerOptions);
const readFile = host.readFile.bind(host);
const fileExists = host.fileExists.bind(host);
host.readFile = filename => sources.get(filename) ?? readFile(filename);
host.fileExists = filename => sources.has(filename) || fileExists(filename);
const program = ts.createProgram([...sources.keys()], compilerOptions, host);
const linter = new Linter({ cwd: root });

test('configuration uses typed production scopes and preserves test exceptions', () => {
  assert.deepEqual(config.files, ['00-dto/**/*.ts', 'tools/**/*.ts', 'runtimes/*/tools/**/*.ts']);
  assert.deepEqual(config.ignores, ['**/*.test.ts', '**/*.spec.ts']);
  assert.equal(rule.meta.fixable, undefined);
  assert.equal(rule.meta.hasSuggestions, undefined);
  assert.equal(rule.meta.docs.url, undefined);
  assert.match(rule.meta.messages.preferSwitch, /does not verify switch exhaustiveness/u);
});

for (const [index, item] of cases.entries()) {
  test(`${item.name}: ${item.count} reports from actual TypeScript types`, () => {
    const source = program.getSourceFile(filenames[index]);
    assert.ok(source);
    assert.deepEqual(program.getSyntacticDiagnostics(source), []);
    assert.deepEqual(program.getSemanticDiagnostics(source), []);
    const messages = linter.verify(item.code, [{
      files: ['**/*.ts'],
      languageOptions: { parser, parserOptions: { programs: [program] } },
      plugins: { 'local-rules-switch': { rules: { 'force-switch-for-union': rule } } },
      rules: { [ruleId]: 'error' },
    }], { filename: filenames[index] });
    assert.equal(messages.some(message => message.fatal), false, JSON.stringify(messages));
    assert.equal(messages.length, item.count, JSON.stringify(messages));
    for (const message of messages) {
      assert.equal(message.ruleId, ruleId);
      assert.equal(message.messageId, 'preferSwitch');
      if (item.condition) {
        assert.ok(message.message.includes(`instead of \`${item.condition}\``));
      }
    }
  });
}

test('real flat configuration applies only where TypeScript type information exists', async () => {
  const eslint = new ESLint({ cwd: root, overrideConfigFile: resolve(root, 'eslint.config.js') });
  for (const filename of ['tools/shared/files.ts', 'runtimes/llama-cpp/tools/config.ts', 'runtimes/stable-diffusion-cpp/tools/config.ts', '00-dto/shared/artifacts.ts']) {
    const applied = await eslint.calculateConfigForFile(filename);
    assert.equal(applied.rules[ruleId][0], 2, filename);
    assert.equal(applied.rules['local-rules-test-resource/test-resource-management'][0], 2, filename);
    assert.equal(applied.rules['local-rules-named-args/require-named-args'][0], 2, filename);
  }
  for (const filename of ['tests/tools/shared/files-json.test.ts', 'tools/example.test.ts', 'tools/example.spec.ts', 'runtimes/llama-cpp/host/index.mjs', 'runtimes/example/host/future.ts', 'tools/future.mjs']) {
    assert.equal((await eslint.calculateConfigForFile(filename)).rules[ruleId], undefined, filename);
  }
  for (const filePath of ['runtimes/example/host/runtime.js', 'runtimes/example/host/future.ts', 'tools/future.mjs']) {
    const [result] = await eslint.lintText("export const value = 'a'; if (value === 'a') {}", { filePath });
    assert.equal(result.messages.some(message => message.fatal), false, JSON.stringify(result.messages));
    assert.equal(result.messages.some(message => message.ruleId === ruleId), false);
  }
});
