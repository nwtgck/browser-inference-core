import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Linter } from 'eslint';
import * as parser from '@typescript-eslint/parser';
import { rule } from '#lint/no-test-support-imports';

function lint({ code, filename = 'tools/runtime.ts' }) {
  const messages = new Linter().verify(code, [{
    files: ['**/*.{ts,tsx,mts,cts,js,mjs,cjs}'],
    languageOptions: { parser, parserOptions: { ecmaVersion: 'latest', sourceType: 'module' } },
    plugins: { local: { rules: { imports: rule } } },
    rules: { 'local/imports': 'error' },
  }], { filename });
  assert.equal(messages.some(message => message.fatal), false, JSON.stringify(messages));
  return messages;
}

const forbidden = [
  'import { helper } from "./helper.test-support";',
  'import type { Helper } from "./helper.test-support";',
  'export { helper } from "./helper.test-support.ts";',
  'export * from "#tests/tools/helper.test-support";',
  'import "#tests/tools/fixture";',
  'import "./tests/fixture.js";',
  'import("./helper.test-support.ts?raw");',
  'const suffix = ".test-support"; import(`./helper${suffix}`);',
  'const prefix = "./helper"; import(prefix + ".test-support");',
  'require("./helper.test-support");',
  'const load = require; load("./helper.test-support");',
  'importScripts("./helper.test-support.ts");',
  'require.resolve("./helper.test-support");',
  'audioWorklet.addModule("./helper.test-support.ts");',
  'import helper = require("./helper.test-support");',
  'type T = import("./helper.test-support").Helper;',
  'new Worker("./helper.test-support.ts");',
  'new SharedWorker(new URL("./helper.test-support.ts", import.meta.url));',
  'new URL("./helper.test-support.ts", import.meta.url);',
  'import "./test-utils";',
  'import "./test-support/index";',
  'import "./helper.test-helpers";',
  'import "./helper.test.ts";',
  'import "./helper.spec.ts";',
];

for (const filename of ['tools/runtime.ts', 'tools/runtime.mjs', 'runtimes/example/host/runtime.js', 'runtimes/example/host/future.ts']) {
  test(`blocks test dependencies from implementation: ${filename}`, () => {
    for (const code of forbidden) {
      assert.ok(lint({ code, filename }).some(message => message.messageId === 'forbidden'), code);
    }
  });
}

for (const filename of ['tests/tools/example.test.ts', 'tests/tools/example.test-support.ts', '.github/tests/example.test.ts', '.github/tests/example.test-support.ts']) {
  test(`preserves test-only dependency chains: ${filename}`, () => {
    for (const code of forbidden) {
      assert.deepEqual(lint({ code, filename }), [], code);
    }
  });
}

for (const filename of ['tools/example.test.mjs', 'tools/test-support/helper.ts', 'tools/example.test-helpers.ts', 'tools/example.test-support.tsx']) {
  test(`does not grant permission to a lookalike test filename: ${filename}`, () => {
    assert.ok(lint({ code: forbidden[0], filename }).some(message => message.messageId === 'forbidden'));
  });
}

test('static import tracing respects lexical aliases and shadowed loaders', () => {
  for (const code of [
    'import "./runtime";',
    'import "./contest-support";',
    'import "./test-supporting";',
    'function run(require: (value: string) => void) { require("./helper.test-support"); }',
    'const require = localLoader; require("./helper.test-support");',
    'const path = "./helper.test-support"; function run(path: string) { import(path); }',
  ]) {
    assert.deepEqual(lint({ code }), [], code);
  }
});
