import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { test } from 'node:test';
import vm from 'node:vm';
import { ESLint, Linter } from 'eslint';
import * as parser from '@typescript-eslint/parser';
import ts from 'typescript';
import objectLayout from '#lint/object-layout';
import functionSpacing from '#lint/function-spacing';
import statementLayout from '#lint/statement-layout';
import { getProtectedLines, intersectsProtectedLines } from '#lint/layout-directive-safety';

const root = resolve(import.meta.dirname, '../../..');
const language = {
  files: ['**/*.{js,jsx,mjs,cjs,ts,tsx,mts,cts}'],
  languageOptions: { parser, sourceType: 'module', parserOptions: { ecmaFeatures: { jsx: true } } },
  linterOptions: { reportUnusedDisableDirectives: 'off' },
};
const objects = [language, objectLayout];
const functions = [language, functionSpacing];
const statements = [language, statementLayout];
const together = [language, objectLayout, functionSpacing, statementLayout];

function stripPositions(value) {
  if (Array.isArray(value)) {
    return value.map(stripPositions);
  }
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value)
      .filter(([key]) => !['loc', 'range', 'parent', 'tokens', 'comments'].includes(key))
      .map(([key, entry]) => [key, stripPositions(entry)]));
  }
  return value;
}

function parsed(code) {
  return parser.parse(code, { range: true, loc: true, tokens: true, comment: true, jsx: true });
}

function commentsAndLiterals(code) {
  const ast = parsed(code);
  return {
    comments: ast.comments.map(comment => code.slice(...comment.range)),
    literals: ast.tokens.filter(token => ['String', 'RegularExpression', 'Template', 'JSXText'].includes(token.type))
      .map(token => [token.type, token.value]),
  };
}

function documentationBindings(code) {
  const source = ts.createSourceFile('fixture.tsx', code, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const bindings = [];
  function visit(node) {
    const comments = ts.getJSDocCommentsAndTags(node).map(comment => comment.getText(source));
    if (comments.length > 0) {
      bindings.push([node.kind, comments]);
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  return bindings;
}

function expectFixed({ input, output, config, filename = 'fixture.ts', sameAst = true }) {
  const linter = new Linter();
  const result = linter.verifyAndFix(input, config, { filename });
  assert.deepEqual(result.messages, []);
  assert.equal(result.output, output);
  assert.deepEqual(commentsAndLiterals(result.output), commentsAndLiterals(input));
  assert.deepEqual(documentationBindings(result.output), documentationBindings(input));
  if (sameAst) {
    assert.deepEqual(stripPositions(parsed(result.output)), stripPositions(parsed(input)));
  }
  assert.deepEqual(linter.verifyAndFix(result.output, config, { filename }), { fixed: false, output, messages: [] });
}

for (const [name, input, output] of [
  ['inline boundaries', 'const value = {a:1,b:2,};', 'const value = { a: 1, b: 2 };'],
  ['nested objects', 'const value = {a:{b:1},...other};', 'const value = { a: { b: 1 }, ...other };'],
  ['computed keys and ternaries', 'const value = {[(a ? b : c)]:((value)),"a,b}":1};', 'const value = { [(a ? b : c)]: ((value)), "a,b}": 1 };'],
  ['multiline boundary and trailing comma', 'const value = {\n  a:1,b:2};', 'const value = {\n  a: 1,\n  b: 2,\n};'],
  ['comment contents', 'const value = {a:1, /* keep  two spaces */b:2};', 'const value = { a: 1, /* keep  two spaces */ b: 2 };'],
  ['colon comment contents', 'const value = {a/* key */:/* value */1};', 'const value = { a /* key */: /* value */ 1 };'],
  ['literal contents', 'const value = {text:"a:1,b:2",pattern:/[{},:]/,raw:`a:1,b:2`};', 'const value = { text: "a:1,b:2", pattern: /[{},:]/, raw: `a:1,b:2` };'],
  ['empty objects', 'const value = {\n};', 'const value = {};'],
  ['ordinary exported objects', 'export const TEST_ONLY = {};', 'export const TEST_ONLY = {};'],
]) {
  test(`object layout preserves the program: ${name}`, () => expectFixed({ input, output, config: objects }));
}

test('object layout does not format patterns, blocks, imports or TS type literals', () => {
  const input = 'import {a,b} from "module"; type Shape = { a:number;b:string; }; const {a,b}=source; if (ready) {}';
  expectFixed({ input, output: input, config: objects });
});

test('object layout uses the existing physical newline style', () => {
  for (const newline of ['\n', '\r\n', '\r', '\u2028', '\u2029']) {
    const input = 'const value = {\n  a:1,b:2};'.replaceAll('\n', newline);
    const output = 'const value = {\n  a: 1,\n  b: 2,\n};'.replaceAll('\n', newline);
    expectFixed({ input, output, config: objects });
  }
});

for (const input of [
  'const value = {/** @deprecated old */ a:1,\nb:2};',
  'const value = {a:1\n/** @internal */,b:2};',
]) {
  test(`object layout reports documentation attachment without a fix: ${input}`, () => {
    const result = new Linter().verifyAndFix(input, objects, { filename: 'fixture.ts' });
    assert.equal(result.output, input);
    assert.equal(result.fixed, false);
    assert.ok(result.messages.some(message => message.messageId === 'commentAttachment'));
  });
}

for (const [name, input, output] of [
  ['top-level definitions', 'const before = 1;\nfunction read() {}\nconst after = 2;', 'const before = 1;\n\nfunction read() {}\n\nconst after = 2;'],
  ['arrow function initializer', 'const before = 1;\nconst read = () => 1;\nconst after = 2;', 'const before = 1;\n\nconst read = () => 1;\n\nconst after = 2;'],
  ['class methods', 'class Reader {\n  value = 1;\n  read() {}\n  write() {}\n}', 'class Reader {\n  value = 1;\n\n  read() {}\n\n  write() {}\n}'],
  ['overload group', 'const before = 1;\nfunction read(value: string): string;\nfunction read(value: number): number;\nfunction read(value: string | number) { return value; }\nconst after = 2;', 'const before = 1;\n\nfunction read(value: string): string;\nfunction read(value: number): number;\nfunction read(value: string | number) { return value; }\n\nconst after = 2;'],
  ['attached documentation', 'const before = 1;\n/** documentation */\nfunction read() {}', 'const before = 1;\n\n/** documentation */\nfunction read() {}'],
  ['accessor pair', 'class Reader {\n  value = 1;\n  get current() { return this.value; }\n  set current(value) { this.value = value; }\n  clear() {}\n}', 'class Reader {\n  value = 1;\n\n  get current() { return this.value; }\n  set current(value) { this.value = value; }\n\n  clear() {}\n}'],
]) {
  test(`function spacing preserves the program: ${name}`, () => expectFixed({ input, output, config: functions }));
}

test('function spacing leaves local functions, objects, type members and existing blank lines alone', () => {
  for (const input of [
    'function outer() {\n  const before = 1;\n  function local() {}\n  const after = 2;\n}',
    'const value = {\n  read() {},\n  write: () => {},\n};',
    'interface Reader {\n  read(): void;\n  write(): void;\n}',
    'function read() {}\n\n\nfunction write() {}',
  ]) {
    expectFixed({ input, output: input, config: functions });
  }
});

test('function spacing does not split inline definitions or move trailing documentation', () => {
  for (const [input, messageId] of [
    ['function read() {} function write() {}', 'inline'],
    ['function read() {} /** @deprecated trailing */\nfunction write() {}', 'commentAttachment'],
  ]) {
    const result = new Linter().verifyAndFix(input, functions, { filename: 'fixture.ts' });
    assert.equal(result.output, input);
    assert.equal(result.fixed, false);
    assert.ok(result.messages.some(message => message.messageId === messageId));
  }
});

for (const [name, input, output] of [
  ['independent statements', 'first(); second(); third();', 'first();\nsecond();\nthird();'],
  ['type declarations', 'type A = string; interface B {}', 'type A = string;\ninterface B {}'],
  ['empty semicolons', 'first();;; second();', 'first();;;\nsecond();'],
  ['switch branches', 'switch (value) { case 1: first(); break; case 2: second(); break; }', 'switch (value) {\ncase 1: first();\n  break;\ncase 2: second();\n  break;\n}'],
]) {
  test(`statement layout separates ${name}`, () => expectFixed({ input, output, config: statements }));
}

test('statement layout adds explicit control blocks without changing execution', () => {
  const input = 'let total = 0; for (let index = 0; index < 3; index++) if (index !== 1) total += index; globalThis.result = total;';
  const result = new Linter().verifyAndFix(input, statements, { filename: 'fixture.js' });
  assert.deepEqual(result.messages, []);
  assert.match(result.output, /for \([^\n]+\) \{\n/u);
  assert.match(result.output, /if \([^\n]+\) \{\n/u);
  const before = {}, after = {};
  vm.runInNewContext(input, before);
  vm.runInNewContext(result.output, after);
  assert.equal(after.result, before.result);
  expectFixed({ input: result.output, output: result.output, config: statements });
});

test('whitespace wrappers preserve template, JSX, comment and literal payloads', () => {
  const input = 'const text = `first  \n  second  \n`;\n/*\n  comment  \n*/\nconst element = <pre>\n  payload  \n</pre>;\nconst after = 1;  \n';
  const expected = input.replace('const after = 1;  \n', 'const after = 1;\n');
  expectFixed({ input, output: expected, config: statements, filename: 'fixture.tsx' });
});

test('statement layout keeps PURE annotation and Annex B declaration fixes manual', () => {
  for (const input of [
    'if (ready) /* @__PURE__ */ run();',
    'if (ready) function declaration() {}',
  ]) {
    const config = [{ ...language, languageOptions: { parser, sourceType: 'script' } }, statementLayout];
    const result = new Linter().verifyAndFix(input, config, { filename: 'fixture.js' });
    assert.equal(result.output, input);
    assert.equal(result.fixed, false);
    assert.ok(result.messages.length > 0);
  }
});

function collectProtected(code) {
  let intervals;
  const messages = new Linter().verify(code, [{
    ...language,
    linterOptions: { noInlineConfig: true },
    plugins: {
      collect: {
        rules: {
          ranges: {
            create(context) {
              return {
                Program() {
                  intervals = getProtectedLines({ sourceCode: context.sourceCode });
                },
              };
            },
          },
        },
      },
    },
    rules: { 'collect/ranges': 'error' },
  }], { filename: 'fixture.ts' });
  assert.equal(messages.some(message => message.fatal), false, JSON.stringify(messages));
  return intervals;
}

test('directive protection recognizes compiler prefixes and counted coverage spans', () => {
  for (const directive of ['expect-error', 'ignore']) {
    for (const suffix of ['TS2304', '_reason', '123']) {
      for (const comment of [`// @ts-${directive}${suffix}`, `/// @ts-${directive}${suffix}`, `/* @ts-${directive}${suffix} */`, `/* explanation\n * @ts-${directive}${suffix} */`]) {
        assert.deepEqual(collectProtected(`${comment}\nconst value = missing;`), [{ start: 1, end: comment.split('\n').length + 1 }]);
      }
    }
  }
  for (const tool of ['c8', 'v8', 'node:coverage']) {
    assert.deepEqual(collectProtected(`/* ${tool} ignore next 3 */\nconst first = 1;\nconst value = {};\nconst last = 2;\nconst safe = 3;`), [{ start: 1, end: 4 }]);
  }
  assert.deepEqual(collectProtected('// ordinary prose mentions @ts-ignore\nconst value = "node:coverage ignore next 3";'), []);
});

test('directive lookup merges intervals and agrees with a linear intersection', () => {
  assert.deepEqual(collectProtected('/* c8 ignore next 3 */\nconst first = 1;\n// @ts-ignore\nconst value = {};\nconst safe = {};'), [{ start: 1, end: 4 }]);
  const intervals = [{ start: 2, end: 4 }, { start: 7, end: 9 }, { start: 14, end: 17 }];
  for (let start = 1; start <= 20; start++) {
    for (let end = start; end <= 20; end++) {
      assert.equal(intersectsProtectedLines({ node: { loc: { start: { line: start }, end: { line: end } } }, intervals }), intervals.some(interval => interval.start <= end && start <= interval.end));
    }
  }
});

test('all formatters preserve directive targets while formatting independent code', () => {
  for (const directive of ['// @ts-expect-errorTS2304', '// @ts-ignore_reason', '/* node:coverage ignore next 3 */', '/* c8 ignore next 3 */', '/* v8 ignore next 3 */']) {
    const protectedCode = directive.startsWith('//')
      ? `${directive}\nconst value = {a:1,\nb:2};`
      : `${directive}\nconst first = 1;\nconst value = {a:1,\nb:2};`;
    const input = `${protectedCode}\nconst safe = {a:1,b:2};`;
    const result = new Linter().verifyAndFix(input, together, { filename: 'fixture.ts' });
    const protectedLines = protectedCode.split('\n');
    assert.deepEqual(result.output.split('\n').slice(0, protectedLines.length).map(line => line.trim()), protectedLines.map(line => line.trim()));
    assert.ok(result.output.endsWith('const safe = { a: 1, b: 2 };'));
    assert.ok(result.messages.every(message => !message.fix));
  }
});

test('formatters converge together without changing a generated-source string', () => {
  const input = 'const declaration = `export const abi: {abiVersion:3;runtime:"llama-cpp";};`;\nconst read = () => ({value:1,other:2});\nif (ready) { first(); second(); }';
  const output = 'const declaration = `export const abi: {abiVersion:3;runtime:"llama-cpp";};`;\n\nconst read = () => ({ value: 1, other: 2 });\n\nif (ready) {\n  first();\n  second();\n}';
  expectFixed({ input, output, config: together });
});

test('ported rules do not claim to normalize TS declaration member delimiters', () => {
  const input = 'export declare const abi: { abiVersion:3;runtime:"llama-cpp";pointerBytes:4|8; };';
  expectFixed({ input, output: input, config: together, filename: 'abi.d.ts' });
});

test('real configuration covers host declarations and ordinary scripts without Vue or CSS', async () => {
  const eslint = new ESLint({ cwd: root, overrideConfigFile: resolve(root, 'eslint.config.js') });
  for (const path of ['tools/abi/generate.ts', 'runtimes/shared/host/abi.d.ts', 'runtimes/llama-cpp/host/index.d.ts', 'runtimes/llama-cpp/host/index.mjs', 'tests/tools/lint/layout.test.mjs', 'runtimes/example/host/future.ts']) {
    const config = await eslint.calculateConfigForFile(path);
    for (const id of ['local-rules-object-layout/object-layout', 'local-rules-function-spacing/function-spacing', 'local-rules-statement-layout/statement-layout']) {
      assert.equal(config.rules[id][0], 2, `${path}: ${id}`);
    }
    for (const id of ['brace-style', 'indent', 'no-trailing-spaces']) {
      assert.equal(config.rules[id][0], 0, `${path}: ${id}`);
    }
    assert.deepEqual(config.rules['local-rules-statement-layout/indent'], [1, 2]);
  }
  for (const path of ['future.css', 'future.vue']) {
    assert.equal(await eslint.calculateConfigForFile(path), undefined);
  }
});
