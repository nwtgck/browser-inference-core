import test from 'node:test';
import assert from 'node:assert/strict';
import { attachCore } from '../../../runtimes/llama-cpp/host/bindings.mjs';

function fixture(pointerBytes = 8) {
  const pointer = { kind: 'pointer', bits: pointerBytes * 8 };
  const schema = {
    schemaSha256: 'a'.repeat(64),
    abiVersion: 3,
    runtime: 'llama-cpp',
    pointerRepresentation: 'bigint_native',
    pointerBytes,
    functions: [{ name: 'llama_echo', export: '_llama_echo', bridge: false, parameters: [{ name: 'pointer', cType: 'void *', kind: 'pointer', abiType: pointer }], returnType: 'void *', returnKind: 'pointer', abiType: pointer }],
    records: [{
      name: 'params',
      cType: 'struct params',
      id: 0,
      size: 32,
      alignment: 8,
      fields: [
        { name: 'count', id: 0, kind: 'unsigned', abiType: { kind: 'integer', bits: 32, signed: false }, offset: 0, size: 4 },
        { name: 'ptr', id: 1, kind: 'pointer', abiType: pointer, offset: 8, size: pointerBytes },
        { name: 'scale', id: 2, kind: 'float', abiType: { kind: 'float', bits: 32 }, offset: 16, size: 4 },
        { name: 'values', id: 3, kind: 'array', abiType: { kind: 'array', dimensions: [3] }, offset: 20, size: 12 },
      ],
    }],
    constants: ['TEST'],
    constantValues: { TEST: '17' },
    callbackMetadata: { version: 2, getters: [] },
    suspendingExports: [],
  };
  let bump = 64;
  const module = {
    HEAPU8: new Uint8Array(1024),
    _bicore_schema_hash: () => pointerBytes === 4 ? 512 : 512n,
    _bicore_abi_version: () => 3,
    _bicore_pointer_bytes: () => pointerBytes,
    _malloc: size => {
      const result = bump;
      bump += size;
      return result;
    },
    _free() {},
    _llama_echo: pointer => pointer,
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
  const { core } = fixture();
  const p = core.allocRecord('params');
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
  const { module, core } = fixture();
  const old = core.bytes(64n, 4);
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
  const { core } = fixture();
  const p = core.utf8('\u65e5\u672c\u8a9e');
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
test('file-row metadata uses the public gate and native pointer widths', async () => {
  for (const pointerBytes of [4, 8]) {
    const { module, schema } = fixture(pointerBytes);
    const unsigned = { kind: 'integer', bits: 32, signed: false };
    const signed = { kind: 'integer', bits: 32, signed: true };
    const helpers = [
      { name: 'bicore_file_backed_rows_version', export: '_bicore_file_backed_rows_version', bridge: false, parameters: [], returnKind: 'unsigned', abiType: unsigned },
      { name: 'bicore_model_file_backed_shard', export: '_bicore_model_file_backed_shard', bridge: false, parameters: [{ name: 'model', kind: 'pointer', abiType: { kind: 'pointer', bits: pointerBytes * 8 } }], returnKind: 'signed', abiType: signed },
    ];
    const metadata = { ...schema, functions: [...schema.functions, ...helpers], callbackMetadata: { version: 2, getters: helpers.map(({ name, export: exported }) => ({ name, export: exported })) } };
    const pointer = pointerBytes === 4 ? 0xffff_ffffn : 0x1_0000_0000n;
    let queries = 0;
    module._bicore_file_backed_rows_version = () => pointerBytes === 8 ? 1 : 0;
    module._bicore_model_file_backed_shard = model => {
      assert.equal(model, pointerBytes === 4 ? Number(pointer) : pointer);
      queries++;
      return -1;
    };
    let finish;
    module._llama_echo = () => new Promise(resolve => {
      finish = resolve;
    });
    const core = attachCore(module, metadata);
    assert.equal(await core.api.bicore_file_backed_rows_version(), pointerBytes === 8 ? 1 : 0);
    assert.equal(await core.api.bicore_model_file_backed_shard(pointer), -1);
    await assert.rejects(core.api.bicore_model_file_backed_shard(Number(pointer)), TypeError);
    const pending = core.api.llama_echo(1n);
    await assert.rejects(core.api.bicore_file_backed_rows_version(), /serialize/u);
    await assert.rejects(core.api.bicore_model_file_backed_shard(pointer), /serialize/u);
    assert.throws(() => core.free(64n), /serialize/u);
    finish(pointerBytes === 4 ? 1 : 1n);
    assert.equal(await pending, 1n);
    assert.equal(await core.api.bicore_model_file_backed_shard(pointer), -1);
    assert.equal(queries, 2);
  }
});
test('concurrent and reentrant native calls are rejected rather than queued', async () => {
  const { module, schema } = fixture();
  let finish;
  module._llama_echo = () => new Promise(resolve => {
    finish = resolve;
  });
  const core = attachCore(module, schema);
  const first = core.api.llama_echo(1n);
  assert.equal(core.busy, true);
  await assert.rejects(core.api.llama_echo(2n), /serialize/);
  assert.throws(() => core.free(64n), /serialize/);
  finish(7n);
  assert.equal(await first, 7n);
  assert.equal(core.busy, false);
});
test('missing exports and ABI mismatch fail on attachment', () => {
  const { module, schema } = fixture();
  delete module._llama_echo;
  assert.throws(() => attachCore(module, schema), /Missing native export/);
  module._bicore_abi_version = () => 2;
  assert.throws(() => attachCore(module, schema), /ABI version/);
});

test('Asyncify retains the guard until ccall completes and preserves normalized bigint values', async () => {
  const { module, schema } = fixture(4);
  let finish;
  module._llama_echo = () => {
    throw Error('Raw exports cannot represent Asyncify completion');
  };
  module.ccall = (name, returnType, argTypes, args, options) => {
    assert.equal(name, 'llama_echo');
    assert.equal(returnType, 'number');
    assert.deepEqual(argTypes, ['number']);
    assert.deepEqual(args, [20]);
    assert.deepEqual(options, { async: true });
    return new Promise(resolve => {
      finish = resolve;
    });
  };
  const core = attachCore(module, schema, { suspension: 'asyncify' });
  const pending = core.api.llama_echo(20n);
  assert.equal(core.busy, true);
  await assert.rejects(core.api.llama_echo(20n), /serialize/);
  assert.throws(() => core.free(64n), /serialize/);
  finish(23);
  assert.equal(await pending, 23n);
  assert.equal(core.busy, false);
});

test('wasm32 JSPI awaits the export Promise and retains bigint arguments without ccall', async () => {
  const { module, schema } = fixture(4);
  let finish;
  module._llama_echo = pointer => {
    assert.equal(pointer, 20);
    return new Promise(resolve => {
      finish = resolve;
    });
  };
  module.ccall = () => {
    throw Error('JSPI uses the native Promise export');
  };
  const core = attachCore(module, schema);
  assert.equal(core.pointerBytes, 4);
  await assert.rejects(core.api.llama_echo(20), /pointer/);
  const pending = core.api.llama_echo(20n);
  assert.equal(core.busy, true);
  await assert.rejects(core.api.llama_echo(20n), /serialize/);
  assert.throws(() => core.free(64n), /serialize/);
  finish(23);
  assert.equal(await pending, 23n);
  assert.equal(core.busy, false);
});

test('Asyncify uses return storage for records and releases the guard after failure', async () => {
  const { module, schema } = fixture();
  const recordSchema = {
    ...schema,
    functions: [
      { name: 'llama_record', export: '_bicore_llama_record', bridge: true, parameters: [{ kind: 'unsigned', abiType: { kind: 'integer', bits: 32, signed: false } }], returnKind: 'record', abiType: { kind: 'record', record: 'params' } },
    ],
  };
  module._bicore_llama_record = () => {};
  module.ccall = (name, returnType, argTypes, args, options) => {
    assert.equal(name, 'bicore_llama_record');
    assert.equal(returnType, undefined);
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

test('Asyncify reuses compiled kind metadata across ordinary calls', async () => {
  const { module, schema } = fixture(4);
  let previous;
  let calls = 0;
  module.ccall = (_name, _returnType, kinds, args) => {
    if (previous) {
      assert.equal(kinds, previous);
    }
    previous = kinds;
    calls++;
    return args[0];
  };
  const core = attachCore(module, schema, { suspension: 'asyncify' });
  assert.equal(await core.api.llama_echo(20n), 20n);
  assert.equal(await core.api.llama_echo(21n), 21n);
  assert.equal(calls, 2);
});
