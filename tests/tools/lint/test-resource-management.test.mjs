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

const protocolAccesses = [
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
];

const nativeRuntimeAccesses = [
  'new DisposableStack();', 'new globalThis.AsyncDisposableStack();',
  'const C = SuppressedError; new C(first, second);',
  'const { DisposableStack: Stack } = globalThis; new Stack();',
  'Reflect.get(globalThis, "AsyncDisposableStack");',
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

for (const filename of ['tools/runtime.mjs', 'runtimes/example/host/runtime.js', 'runtimes/example/host/future.ts', 'runtimes/example/host/tools/future.ts', 'tools/../runtimes/example/host/future.ts', 'src/tools/future.ts']) {
  test(`blocks disposal protocols in owned implementation: ${filename}`, () => {
    for (const code of [...protocolAccesses, ...nativeRuntimeAccesses]) {
      assert.ok(lint({ code, filename }).some(message => message.messageId === 'protocolOnlyInNode'), code);
    }
  });
}

for (const filename of ['tests/tools/example.test.ts', 'tests/tools/example.test-support.ts', '.github/tests/example.test.ts', '.github/tests/example.test-support.ts']) {
  test(`allows disposal protocols in actual Node test modules: ${filename}`, () => {
    for (const code of [...protocolAccesses, ...nativeRuntimeAccesses]) {
      assert.deepEqual(lint({ code, filename }), [], code);
    }
  });
}

for (const filename of ['tools/runtime.mjs', 'tools/runtime.js', 'tools/runtime.cjs', 'tools/runtime.mts', 'tools/runtime.cts', 'runtimes/example/host/runtime.js', 'runtimes/example/host/future.ts', 'runtimes/example/host/tools/future.ts', 'src/tools/future.ts', 'tests/example.test.mjs', 'tests/example.test.tsx', 'tests/example.test-support.ts', 'tools/helper.test-support.ts', 'tests/test-support/helper.ts']) {
  test(`blocks using declarations outside Node tools and *.test.ts: ${filename}`, () => {
    for (const code of ['using resource = make();', 'async function run() { await using resource = make(); }']) {
      assert.ok(lint({ code, filename }).some(message => message.messageId === 'usingOnlyInNode'), code);
    }
  });
}

test('actual Node test filenames permit synchronous and asynchronous using', () => {
  assert.deepEqual(lint({ code: 'using resource = make(); async function run() { await using other = make(); }', filename: 'tests/resource.test.ts' }), []);
});

test('owned Node MJS tests can consume typed cleanup without defining protocols or using declarations', () => {
  for (const code of ['environment[Symbol.dispose]();', 'await environment[Symbol.asyncDispose]();']) {
    assert.deepEqual(lint({ code, filename: 'tests/tools/environment.test.mjs' }), [], code);
    assert.ok(lint({ code, filename: 'runtimes/example/host/environment.test.mjs' }).some(message => message.messageId === 'protocolOnlyInNode'), code);
  }
  for (const code of ['Symbol.dispose;', 'const key = Symbol.dispose;', 'const value = { [Symbol.dispose]() {} };']) {
    assert.ok(lint({ code, filename: 'tests/tools/environment.test.mjs' }).some(message => message.messageId === 'protocolOnlyInNode'), code);
  }
});

for (const filename of ['tools/runtime.ts', 'tools/shared/nested/runtime.ts', 'runtimes/llama-cpp/tools/runtime.ts', 'runtimes/stable-diffusion-cpp/tools/nested/runtime.ts']) {
  test(`Node tooling permits using and disposal symbols but not unavailable native runtime APIs: ${filename}`, () => {
    assert.deepEqual(lint({ code: 'using resource = make(); async function run() { await using other = make(); }', filename }), []);
    for (const code of protocolAccesses) {
      assert.deepEqual(lint({ code, filename }), [], code);
    }
    for (const code of nativeRuntimeAccesses) {
      assert.ok(lint({ code, filename }).some(message => message.messageId === 'unsupportedNodeRuntime'), code);
    }
  });
}

for (const filename of ['tests/example.test.ts', 'tests/example.test-support.ts', 'tools/runtime.ts', 'runtimes/llama-cpp/tools/runtime.ts', 'runtimes/example/host/runtime.mjs']) {
  test(`blocks resource polyfills even in test modules: ${filename}`, () => {
    for (const code of polyfills) {
      assert.ok(lint({ code, filename }).some(message => message.messageId === 'noPolyfill'), code);
    }
  });
}

test('lookalike support filenames do not grant runtime protocol permissions', () => {
  for (const filename of ['tests/example.test-support.tsx', 'tests/example.test-support.mjs', 'tests/test-support/helper.ts', 'tests/example.test-helpers.ts']) {
    assert.ok(lint({ code: 'Symbol.dispose;', filename }).some(message => message.messageId === 'protocolOnlyInNode'));
  }
});

test('ordinary symbols, cleanup methods and shadowed bindings remain valid', () => {
  const cases = [
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
  ];
  for (const filename of ['tools/runtime.ts', 'runtimes/example/host/future.ts']) {
    for (const code of cases) {
      assert.deepEqual(lint({ code, filename }), [], code);
    }
  }
});
