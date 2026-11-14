import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Linter } from 'eslint';
import * as parser from '@typescript-eslint/parser';
import { rule } from '#lint/test-resource-management';

function lint({ code, filename = 'tools/runtime.ts' }) {
  const messages = new Linter().verify(code, [{
    files: ['**/*.{ts,tsx,mts,cts,js,mjs,cjs}'],
    languageOptions: { parser, parserOptions: { ecmaVersion: 'latest', sourceType: 'module' } },
    plugins: { local: { rules: { resource: rule } } },
    rules: { 'local/resource': 'error' },
  }], { filename });
  assert.equal(messages.some(message => message.fatal), false, JSON.stringify(messages));
  return messages;
}

const runtimeAccesses = [
  'Symbol.dispose;', 'Symbol.asyncDispose;', 'Symbol["dispose"];', 'Symbol[`async${"Dispose"}`];',
  'const key = "dis"; Symbol[key + "pose"];', 'Symbol[chooseKey()];',
  'const { [chooseKey()]: value } = Symbol;', 'Reflect.get(Symbol, chooseKey());',
  'let S = Symbol; S.dispose;', 'var S = globalThis.Symbol; S.asyncDispose;',
  'const get = Reflect.get; get(Symbol, "dispose");',
  'const { get: read } = Reflect; read(Symbol, "asyncDispose");',
  'const { getOwnPropertyDescriptor: read } = Object; read(Symbol, "dispose");',
  'const S = Symbol; S.dispose;', 'const S = globalThis.Symbol; S.asyncDispose;',
  'const root = window; const { Symbol: S } = root; S.dispose;',
  'const { dispose: close } = Symbol;', 'let close; ({ asyncDispose: close } = Symbol);',
  '(Symbol as SymbolConstructor)!.dispose;', 'Reflect.get(Symbol, "dispose");',
  'const O = Object; O.getOwnPropertyDescriptor(Symbol, "asyncDispose");',
  'type Key = typeof Symbol.dispose;', 'type Key = (typeof Symbol)["asyncDispose"];',
  'new DisposableStack();', 'new globalThis.AsyncDisposableStack();',
  'const C = SuppressedError; new C(first, second);',
];

const polyfills = [
  'import "core-js";', 'import "core-js/stable";', 'import "core-js/actual/index.js";', 'import "core-js-pure/full";',
  'import fallback = require("core-js/proposals/explicit-resource-management");',
  'const fallback = "core-js/actual/symbol/dispose"; import(fallback);',
  'Symbol[chooseKey()] = Symbol();', 'Reflect.set(Symbol, chooseKey(), Symbol());',
  'Object.defineProperties(Symbol, { [chooseKey()]: { value: Symbol() } });',
  'Object.assign(globalThis, { Symbol: replacement });',
  'Object.defineProperties(globalThis, { SuppressedError: { value: replacement } });',
  'let S = Symbol; S.dispose = Symbol();',
  'const define = Object.defineProperty; define(Symbol, "dispose", { value: Symbol() });',
  'const { set: write } = Reflect; write(Symbol, "dispose", Symbol());',
  'const lookup = Symbol.for; lookup("Symbol.dispose");',
  'import("core-js/proposals/explicit-resource-management");',
  'require("core-js/actual/symbol/dispose");',
  'globalThis.Symbol = replacement;', 'const root = globalThis; root.Symbol = replacement;',
  'Object.defineProperty(globalThis, "Symbol", { value: replacement });', 'Reflect.set(globalThis, "Symbol", replacement);',
  'globalThis.DisposableStack = class {};', 'runtime.stubGlobal("Symbol", { dispose: Symbol() });',
  'runtime.stubGlobal("AsyncDisposableStack", class {});',
  'Symbol.dispose ??= Symbol();', 'Symbol["asyncDispose"] = Symbol();', 'const S = Symbol; S.dispose = Symbol();',
  'Reflect.set(Symbol, "dispose", Symbol());', 'Object.defineProperty(Symbol, "asyncDispose", { value: Symbol() });',
  'Object.defineProperties(Symbol, { dispose: { value: Symbol() } });', 'Object.assign(Symbol, { asyncDispose: Symbol() });',
  'Object.assign(Symbol, descriptors);', 'delete Symbol.dispose;', 'Symbol.for("Symbol.dispose");', 'Symbol("Symbol.asyncDispose");',
  'import "core-js/proposals/explicit-resource-management";',
  'Object.defineProperty(globalThis, "DisposableStack", { value: class {} });',
];

for (const filename of ['tools/runtime.ts', 'tools/runtime.mjs', 'runtimes/example/host/runtime.js', 'runtimes/example/host/future.ts']) {
  test(`blocks disposal protocols in owned implementation: ${filename}`, () => {
    for (const code of runtimeAccesses) {
      assert.ok(lint({ code, filename }).some(message => message.messageId === 'protocolOnlyInTest'), code);
    }
  });
}

for (const filename of ['tests/tools/example.test.ts', 'tests/tools/example.test-support.ts']) {
  test(`allows disposal protocols in actual Node test modules: ${filename}`, () => {
    for (const code of runtimeAccesses) {
      assert.deepEqual(lint({ code, filename }), [], code);
    }
  });
}

for (const filename of ['tools/runtime.ts', 'tools/runtime.mjs', 'runtimes/example/host/runtime.js', 'runtimes/example/host/future.ts', 'tests/example.test.mjs', 'tests/example.test.tsx', 'tests/example.test-support.ts', 'tests/test-support/helper.ts']) {
  test(`blocks using declarations outside *.test.ts: ${filename}`, () => {
    for (const code of ['using resource = make();', 'async function run() { await using resource = make(); }']) {
      assert.ok(lint({ code, filename }).some(message => message.messageId === 'usingOnlyInTest'), code);
    }
  });
}

test('actual Node test filenames permit synchronous and asynchronous using', () => {
  assert.deepEqual(lint({ code: 'using resource = make(); async function run() { await using other = make(); }', filename: 'tests/resource.test.ts' }), []);
});

for (const filename of ['tests/example.test.ts', 'tests/example.test-support.ts', 'tools/runtime.ts', 'runtimes/example/host/runtime.mjs']) {
  test(`blocks resource polyfills even in test modules: ${filename}`, () => {
    for (const code of polyfills) {
      assert.ok(lint({ code, filename }).some(message => message.messageId === 'noPolyfill'), code);
    }
  });
}

test('lookalike support filenames do not grant runtime protocol permissions', () => {
  for (const filename of ['tests/example.test-support.tsx', 'tests/example.test-support.mjs', 'tests/test-support/helper.ts', 'tests/example.test-helpers.ts']) {
    assert.ok(lint({ code: 'Symbol.dispose;', filename }).some(message => message.messageId === 'protocolOnlyInTest'));
  }
});

test('ordinary symbols, cleanup methods and shadowed bindings remain valid', () => {
  for (const code of [
    'resource.dispose(); resource.asyncDispose(); resource.close();',
    'import "core-js/actual/array/to-sorted";',
    'Symbol.iterator; Symbol.asyncIterator; Symbol.for("application-key");',
    'globalThis.Symbol.iterator; globalThis.Symbol.asyncIterator;',
    'const S = globalThis.Symbol; S.for("application-key"); S.iterator;',
    'const { Symbol: S } = globalThis; S.asyncIterator;',
    'function run(Symbol: { dispose: number }) { return Symbol.dispose; }',
    'const Symbol = { dispose: 1 }; Symbol.dispose;',
    'const Object = { defineProperty() {} }; Object.defineProperty(Symbol, "dispose", {});',
    'const box = { DisposableStack: 1 }; box.DisposableStack;', 'class DisposableStack {} new DisposableStack();',
    'Object.defineProperty(Symbol, "applicationSymbol", { value: Symbol() });',
  ]) {
    assert.deepEqual(lint({ code }), [], code);
  }
});
