import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { test } from 'node:test';
import { Linter } from 'eslint';
import * as parser from '@typescript-eslint/parser';
import ts from 'typescript';
import config, { rule } from '#lint/require-named-args';

const root = resolve(import.meta.dirname, '../../..');
const filename = resolve(root, 'tools/named-args-memory-fixture.ts');

function lint({ code, typed = false }) {
  const compilerOptions = {
    target: ts.ScriptTarget.ESNext,
    module: ts.ModuleKind.NodeNext,
    moduleResolution: ts.ModuleResolutionKind.NodeNext,
    strict: true,
    noEmit: true,
    skipLibCheck: true,
    types: ['node'],
    typeRoots: [resolve(root, 'node_modules/@types')],
    lib: ['lib.esnext.d.ts', 'lib.dom.d.ts'],
  };
  const host = ts.createCompilerHost(compilerOptions);
  const read = host.readFile.bind(host);
  const exists = host.fileExists.bind(host);
  host.readFile = file => file === filename ? code : read(file);
  host.fileExists = file => file === filename || exists(file);
  const program = typed ? ts.createProgram([filename], compilerOptions, host) : undefined;
  const messages = new Linter({ cwd: root }).verify(code, [{
    files: ['**/*.ts'],
    languageOptions: { parser, parserOptions: { ecmaVersion: 'latest', sourceType: 'module', ...(program ? { programs: [program] } : {}) } },
    plugins: { local: { rules: { named: rule } } },
    rules: { 'local/named': 'error' },
  }], { filename });
  assert.equal(messages.some(message => message.fatal), false, JSON.stringify(messages));
  return messages;
}

test('owned callable scope excludes tests without disabling other policies', () => {
  assert.deepEqual(config.files, ['00-dto/**/*.ts', 'tools/**/*.ts', 'runtimes/*/tools/**/*.ts']);
  assert.deepEqual(config.ignores, ['**/*.test.ts', '**/*.spec.ts']);
  assert.equal(rule.meta.fixable, undefined);
  assert.equal(rule.meta.hasSuggestions, undefined);
});

for (const code of [
  'function read() {}',
  'function read({ path }: { path: string }) {}',
  'type Listener = ({ value }: { value: number }) => void;',
  'type A = { value: string }; function read({ value }: { value: A["value"] }) {}',
  'function read({ value }: { value: string } | { value: number }) {}',
  'function isString(value: unknown): value is string { return typeof value === "string"; }',
  'function html(parts: TemplateStringsArray, ...values: unknown[]) { return parts[0]; }',
]) {
  test(`allows an owned named or language-defined signature: ${code}`, () => {
    assert.deepEqual(lint({ code }), []);
  });
}

for (const code of [
  'function read(path: string) {}',
  'function read(path: string, mode: string) {}',
  'function read(options: { path: string }) {}',
  'function read({ path }: { path: string } = { path: "." }) {}',
  'type Args = { path: string }; function read({ path }: Args) {}',
  'type Listener = (event: Event) => void;',
  'type Options = {}; function read({}: Options) {}',
  'function read(_args: Record<never, never>) {}',
  'function write(value: string) {}',
  'let resolveResult: (result: string) => void;',
]) {
  test(`rejects an owned positional or hidden signature: ${code}`, () => {
    assert.ok(lint({ code }).length > 0);
  });
}

test('inline outer shapes expose alias mistakes without banning cohesive values', () => {
  const code = 'type Args = { path: string }; function bad({ path }: Args) {}';
  assert.equal(lint({ code })[0].messageId, 'requireInlineNamedArgsType');
  assert.deepEqual(lint({ code: 'type Value = { path: string }; function good({ value }: { value: Value }) {}' }), []);
});

test('local direct callbacks do not inherit an external exemption from call syntax', () => {
  const code = `\
declare function consume(callback: (value: number) => void): void;
consume(value => String(value));
`;
  assert.ok(lint({ code, typed: true }).some(message => message.line === 2));
});

test('real Array and Promise callbacks preserve their native positional contracts', () => {
  const code = `\
type Item = { id: string };
const items: Item[] = [];
items.map(({ id }: Item, index) => id + index);
Promise.resolve(1).then(value => String(value));
new Promise<string>((resolve, reject) => { resolve('ok'); reject(new Error()); });
`;
  assert.deepEqual(lint({ code, typed: true }), []);
});

test('stored Node and native Promise callbacks prove ownership through types', () => {
  const code = `\
import type { RequestListener } from 'node:http';
const listener: RequestListener = (request, response) => { response.end(request.url); };
let finish: ReturnType<typeof Promise.withResolvers<boolean>>['resolve'];
finish = value => {};
`;
  assert.deepEqual(lint({ code, typed: true }), []);
});

test('an external parameter type does not make a local assignment callback external', () => {
  const code = `\
type Listener = (event: Event) => void;
let listener: Listener;
listener = event => {};
`;
  assert.ok(lint({ code, typed: true }).some(message => message.line === 3 && message.messageId === 'requireNamedArgsAssignment'));
});

test('standard Readonly is verified while local wrappers cannot hide argument fields', () => {
  assert.deepEqual(lint({ code: 'export {}; function read({ value }: Readonly<{ value: string }>) {}', typed: true }), []);
  for (const code of [
    'export {}; type Readonly<T> = T; function read({ value }: Readonly<{ value: string }>) {}',
    'type Wrapper<T> = T; function read({ value }: Wrapper<{ value: string }>) {}',
    'type Args = { value: string }; function read({ value }: Readonly<Args>) {}',
  ]) {
    assert.equal(lint({ code, typed: true }).at(-1).messageId, 'requireInlineNamedArgsType');
  }
});

test('native Web Streams callbacks are type verified instead of name exempted', () => {
  assert.deepEqual(lint({ code: 'new ReadableStream<string>({ start(controller) { controller.enqueue("ok"); } });', typed: true }), []);
  const code = `\
export {}; declare class ReadableStream { constructor(source: { start(value: number): void }); }
new ReadableStream({ start(value) {} });
`;
  assert.ok(lint({ code, typed: true }).some(message => message.line === 2));
});

test('external inherited methods do not exempt constructors or new local methods', () => {
  const code = `\
class Receiver implements EventListenerObject {
  constructor(value: string) {}
  handleEvent(event: Event): void {}
  local(value: string): void {}
}
`;
  const messages = lint({ code, typed: true });
  assert.ok(messages.some(message => message.line === 2));
  assert.equal(messages.some(message => message.line === 3), false);
  assert.ok(messages.some(message => message.line === 4));
});

test('abstract and ambient methods follow the same owned signature contract', () => {
  const messages = lint({ code: 'abstract class Reader { abstract read(value: string): void; } declare class Writer { write(value: string): void; }' });
  assert.equal(messages.length, 2);
});

test('type evidence is required even for an externally named callback', () => {
  assert.ok(lint({ code: 'new ReadableStream({ start(controller) {} });' }).length > 0);
  assert.ok(lint({ code: 'const resolveResult = value => value;' }).length > 0);
});
