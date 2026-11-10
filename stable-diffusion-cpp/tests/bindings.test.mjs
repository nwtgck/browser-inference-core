import test from 'node:test';
import assert from 'node:assert/strict';
import { attachCore } from '../examples/runtime/bindings.mjs';

function fixture() {
  const schema = { schemaSha256: 'a'.repeat(64), abiVersion: 1, functions: [
    { name: 'llama_echo', export: '_sdc_llama_echo', parameters: [{ kind: 'pointer' }], returnKind: 'pointer' },
  ], records: [{ name: 'params', id: 0, fields: [
    { name: 'count', id: 0, kind: 'unsigned' }, { name: 'ptr', id: 1, kind: 'pointer' },
    { name: 'scale', id: 2, kind: 'float' }, { name: 'values', id: 3, kind: 'array' },
  ] }], constants: ['TEST'] };
  let bump = 64;
  const module = {
    HEAPU8: new Uint8Array(1024),
    _sdc_schema_hash: () => 512n, _sdc_abi_version: () => 1, _sdc_pointer_bytes: () => 8,
    _sdc_sizeof_record: () => 32n, _sdc_alignof_record: () => 8n,
    _sdc_offsetof_field: (_, id) => [0n, 8n, 16n, 20n][id],
    _sdc_sizeof_field: (_, id) => [4n, 8n, 4n, 12n][id],
    _sdc_constant: () => 17n,
    _sdc_malloc: size => { const result = bump; bump += Number(size); return BigInt(result); },
    _sdc_free() {}, _sdc_llama_echo: pointer => pointer,
  };
  module.HEAPU8.set(new TextEncoder().encode(schema.schemaSha256), 512);
  return { module, schema, core: attachCore(module, schema) };
}

test('constants, ABI sizes and field addresses come from the core', () => {
  const { core } = fixture();
  assert.equal(core.constant('TEST'), 17);
  assert.equal(core.recordSize('params'), 32);
  assert.equal(core.recordAlignment('params'), 8);
  assert.equal(core.fieldAddress('params', 64n, 'ptr'), 72n);
});
test('32-bit integers, pointers and floats are accessed without handwritten offsets', () => {
  const { core } = fixture(); const p = core.allocRecord('params');
  core.setField('params', p, 'count', 4294967295);
  core.setField('params', p, 'ptr', 0x100000000n);
  core.setField('params', p, 'scale', 0.5);
  assert.equal(core.getField('params', p, 'count'), 4294967295);
  assert.equal(core.getField('params', p, 'ptr'), 0x100000000n);
  assert.equal(core.getField('params', p, 'scale'), 0.5);
  assert.throws(() => core.setField('params', p, 'count', 2 ** 32), RangeError);
  assert.throws(() => core.getField('params', p, 'values'), TypeError);
});
test('heap replacement is observed instead of keeping stale views', () => {
  const { module, core } = fixture(); const old = core.bytes(64n, 4);
  module.HEAPU8 = new Uint8Array(2048);
  core.bytes(64n, 4).fill(23);
  assert.deepEqual([...old], [0, 0, 0, 0]);
  assert.deepEqual([...core.bytes(64n, 4)], [23, 23, 23, 23]);
});
test('unsafe indexes and out-of-bounds accesses fail', () => {
  const { core } = fixture();
  assert.throws(() => core.bytes(1n << 54n, 1), RangeError);
  assert.throws(() => core.bytes(1020n, 5), RangeError);
  assert.throws(() => core.bytes(0n, -1), RangeError);
});
test('UTF-8 allocation is terminated and bounded', () => {
  const { core } = fixture(); const p = core.utf8('\u65e5\u672c\u8a9e');
  assert.equal(core.readUtf8(p), '\u65e5\u672c\u8a9e');
  assert.equal(core.readUtf8(0n), null);
  assert.throws(() => core.readUtf8(p, 2), RangeError);
});
test('calls validate the normalized bigint boundary', async () => {
  const { core } = fixture();
  assert.equal(await core.api.llama_echo(20n), 20n);
  await assert.rejects(core.api.llama_echo(20), TypeError);
  await assert.rejects(core.api.llama_echo(-1n), RangeError);
  await assert.rejects(core.api.llama_echo(1n << 64n), RangeError);
});

test('generation results use the schema signature and native pointer width on both Wasm widths', async () => {
  for (const pointerBytes of [4, 8]) {
    const { module, schema } = fixture();
    const calls = [];
    const resultSchema = { ...schema, abiVersion: 2, functions: [
      { name: 'generate_image', export: '_sdc_generate_image',
        parameters: [{ kind: 'pointer' }, { kind: 'pointer' }, { kind: 'pointer' }], returnKind: 'boolean' },
      { name: 'free_sd_image_generation_result', export: '_sdc_free_sd_image_generation_result',
        parameters: [{ kind: 'pointer' }], returnKind: 'void' },
    ], records: [{ name: 'sd_image_generation_result_t', id: 0, fields: [
      { name: 'items', id: 0, kind: 'pointer' }, { name: 'count', id: 1, kind: 'signed' },
    ] }] };
    module._sdc_abi_version = () => 2;
    module._sdc_pointer_bytes = () => pointerBytes;
    module._sdc_sizeof_record = () => BigInt(pointerBytes * 2);
    module._sdc_alignof_record = () => BigInt(pointerBytes);
    module._sdc_offsetof_field = (_, id) => id === 0 ? 0n : BigInt(pointerBytes);
    module._sdc_sizeof_field = (_, id) => id === 0 ? BigInt(pointerBytes) : 4n;
    module._sdc_generate_image = (...args) => { calls.push(args); return 1; };
    module._sdc_free_sd_image_generation_result = result => { calls.push([result]); };
    const core = attachCore(module, resultSchema);
    await assert.rejects(core.api.generate_image(1n, 2n, 3n, 4n), /expected 3 arguments/);
    assert.deepEqual(calls, []);
    assert.equal(await core.api.generate_image(1n, 2n, 3n), 1);
    assert.deepEqual(calls, [[1n, 2n, 3n]]);

    const result = core.allocRecord('sd_image_generation_result_t');
    const items = pointerBytes === 4 ? 0xfffffff0n : 0x100000010n;
    core.setField('sd_image_generation_result_t', result, 'items', items);
    core.setField('sd_image_generation_result_t', result, 'count', 2);
    assert.equal(core.getField('sd_image_generation_result_t', result, 'items'), items);
    assert.equal(core.getField('sd_image_generation_result_t', result, 'count'), 2);
    await core.api.free_sd_image_generation_result(result);
    assert.deepEqual(calls.at(-1), [result]);
  }
});

test('generation progress counters retain signed 64-bit precision', () => {
  const { module, schema } = fixture();
  const progressSchema = { ...schema, records: [{ name: 'sd_generation_progress_t', id: 0, fields: [
    { name: 'completed', id: 0, kind: 'i64' }, { name: 'total', id: 1, kind: 'i64' },
  ] }] };
  module._sdc_sizeof_record = () => 16n;
  module._sdc_offsetof_field = (_, id) => BigInt(id * 8);
  module._sdc_sizeof_field = () => 8n;
  const core = attachCore(module, progressSchema);
  const progress = core.allocRecord('sd_generation_progress_t');
  core.setField('sd_generation_progress_t', progress, 'completed', (1n << 53n) + 1n);
  core.setField('sd_generation_progress_t', progress, 'total', 0n);
  assert.equal(core.getField('sd_generation_progress_t', progress, 'completed'), (1n << 53n) + 1n);
  assert.equal(core.getField('sd_generation_progress_t', progress, 'total'), 0n);
  core.setField('sd_generation_progress_t', progress, 'completed', -1n);
  assert.equal(core.getField('sd_generation_progress_t', progress, 'completed'), -1n);
  assert.throws(() => core.setField('sd_generation_progress_t', progress, 'total', 1n << 63n), RangeError);
});

test('concurrent and reentrant native calls are rejected rather than queued', async () => {
  const { module, core } = fixture(); let finish;
  module._sdc_llama_echo = () => new Promise(resolve => { finish = resolve; });
  const first = core.api.llama_echo(1n);
  assert.equal(core.busy, true);
  await assert.rejects(core.api.llama_echo(2n), /serialize/);
  assert.throws(() => core.free(64n), /serialize/);
  finish(7n); assert.equal(await first, 7n); assert.equal(core.busy, false);
});
test('missing exports and ABI mismatch fail on attachment', () => {
  const { module, schema } = fixture();
  delete module._sdc_llama_echo;
  assert.throws(() => attachCore(module, schema), /Missing export/);
  module._sdc_abi_version = () => 2;
  assert.throws(() => attachCore(module, schema), /ABI versions/);
});

test('Asyncify retains the guard until ccall completes and preserves normalized bigint values', async () => {
  const { module, schema } = fixture();
  let finish;
  module._sdc_pointer_bytes = () => 4;
  module._sdc_llama_echo = () => { throw Error('Raw exports cannot represent Asyncify completion'); };
  module.ccall = (name, returnType, argTypes, args, options) => {
    assert.equal(name, 'sdc_llama_echo');
    assert.equal(returnType, 'bigint');
    assert.deepEqual(argTypes, ['bigint']);
    assert.deepEqual(args, [20n]);
    assert.deepEqual(options, { async: true });
    return new Promise(resolve => { finish = resolve; });
  };
  const core = attachCore(module, schema, { suspension: 'asyncify' });
  const pending = core.api.llama_echo(20n);
  assert.equal(core.busy, true);
  await assert.rejects(core.api.llama_echo(20n), /serialize/);
  assert.throws(() => core.free(64n), /serialize/);
  finish(23n);
  assert.equal(await pending, 23n);
  assert.equal(core.busy, false);
});

test('wasm32 JSPI awaits the export Promise and retains bigint arguments without ccall', async () => {
  const { module, schema } = fixture();
  let finish;
  module._sdc_pointer_bytes = () => 4;
  module._sdc_llama_echo = pointer => {
    assert.equal(pointer, 20n);
    return new Promise(resolve => { finish = resolve; });
  };
  module.ccall = () => { throw Error('JSPI uses the native Promise export'); };
  const core = attachCore(module, schema);
  assert.equal(core.pointerBytes, 4);
  await assert.rejects(core.api.llama_echo(20), /bigint/);
  const pending = core.api.llama_echo(20n);
  assert.equal(core.busy, true);
  await assert.rejects(core.api.llama_echo(20n), /serialize/);
  assert.throws(() => core.free(64n), /serialize/);
  finish(23n);
  assert.equal(await pending, 23n);
  assert.equal(core.busy, false);
});

test('Asyncify uses return storage for records and releases the guard after failure', async () => {
  const { module, schema } = fixture();
  const recordSchema = { ...schema, functions: [
    { name: 'llama_record', export: '_sdc_llama_record', parameters: [{ kind: 'unsigned' }], returnKind: 'record' },
  ] };
  module._sdc_llama_record = () => {};
  module.ccall = (name, returnType, argTypes, args, options) => {
    assert.equal(name, 'sdc_llama_record');
    assert.equal(returnType, null);
    assert.deepEqual(argTypes, ['bigint', 'number']);
    assert.deepEqual(args, [64n, 7]);
    assert.deepEqual(options, { async: true });
    return Promise.reject(new Error('Native operation failed'));
  };
  const core = attachCore(module, recordSchema, { suspension: 'asyncify' });
  await assert.rejects(core.api.llama_record(64n, 7), /Native operation failed/);
  assert.equal(core.busy, false);
});

test('Asyncify attachment requires its completion helper', () => {
  const { module, schema } = fixture();
  assert.throws(() => attachCore(module, schema, { suspension: 'asyncify' }), /ccall/);
  assert.throws(() => attachCore(module, schema, { suspension: 'unknown' }), /suspension/);
});

test('a schema from a different build is rejected', () => {
  const { module, schema } = fixture();
  assert.throws(() => attachCore(module, { ...schema, schemaSha256: 'b'.repeat(64) }), /do not match/);
});
