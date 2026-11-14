import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

function call({ module, name, args }: { module: object; name: string; args: readonly unknown[] }): unknown {
  const method: unknown = Reflect.get(module, name);
  assert.equal(typeof method, 'function');
  if (typeof method !== 'function') {
    throw new Error(`Missing generated method: ${name}`);
  }
  return Reflect.apply(method, module, args);
}

const [directory, pointer64, suspension] = process.argv.slice(2);
assert.ok(directory !== undefined);
assert.ok(pointer64 === '0' || pointer64 === '1');
assert.ok(suspension === 'direct' || suspension === 'asyncify' || suspension === 'jspi');
const imported: unknown = await import(pathToFileURL(join(directory, 'core.mjs')).href);
assert.ok(typeof imported === 'object' && imported !== null);
const factory: unknown = Reflect.get(imported, 'default');
assert.equal(typeof factory, 'function');
if (typeof factory !== 'function') {
  throw new Error('Missing generated module factory');
}
const module: unknown = await Reflect.apply(factory, undefined, [{ wasmBinary: await readFile(join(directory, 'core.wasm')) }]);
assert.ok(typeof module === 'object' && module !== null);
const pointer = call({ module, name: '_malloc', args: [32] });
assert.equal(typeof pointer, 'number');
assert.equal(call({ module, name: '_free', args: [pointer] }), undefined);
assert.equal(call({ module, name: '_converted_pointer', args: [513] }), 513);
assert.equal(call({ module, name: '_optional_pointer', args: [19] }), pointer64 === '1' ? 19n : 19);
if (pointer64 === '1') {
  assert.equal(call({ module, name: '_optional_pointer', args: [undefined] }), 0n);
}
const value = 0x123456789abcdefn;
assert.equal(call({ module, name: '_raw_i64', args: [value] }), value ^ 0x123456789abcdef0n);
assert.equal(call({ module, name: '_sync_value', args: [4] }), 7);
const selected = call({ module, name: '_async_value', args: [value] });
assert.equal(selected instanceof Promise, suspension === 'jspi');
assert.equal(await selected, value + 1n);
if (suspension === 'asyncify') {
  const rewound = call({ module, name: 'ccall', args: ['suspend_i64', 'bigint', ['bigint'], [value], { async: true }] });
  assert.ok(rewound instanceof Promise);
  assert.equal(await rewound, value + 1n);
}
process.stdout.write(JSON.stringify({ pointer64, suspension, malloc: 'number', free: 'synchronous', rawI64: 'bigint', rewind: suspension === 'asyncify' }) + '\n');
