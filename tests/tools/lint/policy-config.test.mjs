import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { test } from 'node:test';
import { ESLint, Linter } from 'eslint';
import * as parser from '@typescript-eslint/parser';
import { rule as resource } from '#lint/test-resource-management';
import { rule as imports } from '#lint/no-test-support-imports';
import { isTestFile, isTestSupportFile } from '#lint/test-resource-ast';

const root = resolve(import.meta.dirname, '../../..');
const ruleId = 'local-rules-test-resource/test-resource-management';
const importRuleId = 'local-rules-test-imports/no-test-support-imports';
const namedRuleId = 'local-rules-named-args/require-named-args';
const eslint = new ESLint({ cwd: root, overrideConfigFile: resolve(root, 'eslint.config.js') });

test('test permissions depend on exact physical suffixes across platforms', () => {
  assert.equal(isTestFile({ filename: 'C:\\runtime\\example.test.ts' }), true);
  assert.equal(isTestSupportFile({ filename: 'C:\\runtime\\example.test-support.ts' }), true);
  for (const filename of ['example.test-support.tsx', 'example.test-support.ts?raw', 'example.test-support.ts/runtime.ts', 'example.test.mjs']) {
    assert.equal(isTestFile({ filename }), false, filename);
    assert.equal(isTestSupportFile({ filename }), false, filename);
  }
});

test('the real configuration covers production tooling and browser JS and future TS', async () => {
  for (const file of ['tools/shared/process.ts', 'tools/future.mjs', 'tools/future.cjs', 'runtimes/llama-cpp/host/index.mjs', 'runtimes/example/host/runtime.js', 'runtimes/example/host/future.ts', 'eslint-local-rules/test-resource-ast.js']) {
    const config = await eslint.calculateConfigForFile(file);
    assert.equal(config.rules[ruleId][0], 2, file);
    assert.equal(config.rules[importRuleId][0], 2, file);
  }
  assert.equal((await eslint.calculateConfigForFile('tools/cli.ts')).rules[namedRuleId][0], 2);
  assert.equal((await eslint.calculateConfigForFile('tests/tools/lint/policy-config.test.mjs')).rules[namedRuleId], undefined);
  assert.equal(await eslint.calculateConfigForFile('runtimes/example/vendor/upstream/runtime.ts'), undefined);
});

test('the real configuration enforces the boundary without requiring a TS project for browser files', async () => {
  for (const filePath of ['tools/future.mjs', 'runtimes/example/host/runtime.js', 'runtimes/example/host/future.ts']) {
    const [result] = await eslint.lintText('using resource = make(); Symbol.dispose; import "./helper.test-support";', { filePath });
    assert.equal(result.messages.some(message => message.fatal), false, JSON.stringify(result.messages));
    for (const id of ['usingOnlyInTest', 'protocolOnlyInTest', 'forbidden']) {
      assert.ok(result.messages.some(message => message.messageId === id), `${filePath}: ${id}`);
    }
  }
});

test('virtual *.test.ts filenames do not grant production or support using permission', () => {
  for (const filename of ['tools/runtime.mjs', 'runtimes/example/host/future.ts', 'tests/example.test-support.ts']) {
    const messages = new Linter().verify('ignored', [{
      files: [filename],
      plugins: {
        extract: {
          processors: {
            script: {
              preprocess: () => [{ filename: 'virtual.test.ts', text: 'using value = make(); Symbol.dispose; import "./helper.test-support";' }],
              postprocess: messages => messages.flat(),
            },
          },
        },
      },
      processor: 'extract/script',
    }, {
      files: ['**/*.ts'],
      languageOptions: { parser },
      plugins: { policy: { rules: { resource, imports } } },
      rules: { 'policy/resource': 'error', 'policy/imports': 'error' },
    }], { filename, filterCodeBlock: () => true });
    assert.equal(messages.some(message => message.fatal), false, JSON.stringify(messages));
    assert.ok(messages.some(message => message.messageId === 'usingOnlyInTest'), filename);
    if (!filename.endsWith('.test-support.ts')) {
      assert.ok(messages.some(message => message.messageId === 'protocolOnlyInTest'), filename);
      assert.ok(messages.some(message => message.messageId === 'forbidden'), filename);
    }
  }
});
