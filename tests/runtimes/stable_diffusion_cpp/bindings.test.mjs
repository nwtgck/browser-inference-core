import test from 'node:test';
import assert from 'node:assert/strict';
import { attachCore } from '../../../runtimes/stable-diffusion-cpp/host/bindings.mjs';

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

test('generation results use the schema signature and native pointer width on both Wasm widths', async () => {
  for (const pointerBytes of [4, 8]) {
    const { module, schema } = fixture(pointerBytes);
    const calls = [];
    const pointer = { kind: 'pointer', bits: pointerBytes * 8 };
    const resultSchema = {
      ...schema,
      functions: [
        {
          name: 'generate_image',
          export: '_generate_image',
          bridge: false,
          parameters: [0, 1, 2].map(() => ({ kind: 'pointer', abiType: pointer })),
          returnKind: 'boolean',
          abiType: { kind: 'integer', bits: 8, signed: false },
        },
        {
          name: 'free_sd_image_generation_result',
          export: '_free_sd_image_generation_result',
          bridge: false,
          parameters: [{ kind: 'pointer', abiType: pointer }],
          returnKind: 'void',
          abiType: { kind: 'void' },
        },
      ],
      records: [{
        name: 'sd_image_generation_result_t',
        id: 0,
        size: pointerBytes * 2,
        alignment: pointerBytes,
        fields: [
          { name: 'items', id: 0, kind: 'pointer', abiType: pointer, offset: 0, size: pointerBytes },
          { name: 'count', id: 1, kind: 'signed', abiType: { kind: 'integer', bits: 32, signed: true }, offset: pointerBytes, size: 4 },
        ],
      }],
    };
    module._generate_image = (...args) => {
      calls.push(args);
      return 1;
    };
    module._free_sd_image_generation_result = result => {
      calls.push([result]);
    };
    const core = attachCore(module, resultSchema);
    await assert.rejects(core.api.generate_image(1n, 2n, 3n, 4n), /argument count/);
    assert.deepEqual(calls, []);
    assert.equal(await core.api.generate_image(1n, 2n, 3n), 1);
    assert.deepEqual(calls, [pointerBytes === 4 ? [1, 2, 3] : [1n, 2n, 3n]]);

    const result = core.allocRecord('sd_image_generation_result_t');
    const items = pointerBytes === 4 ? 0xfffffff0n : 0x100000010n;
    core.setField('sd_image_generation_result_t', result, 'items', items);
    core.setField('sd_image_generation_result_t', result, 'count', 2);
    assert.equal(core.getField('sd_image_generation_result_t', result, 'items'), items);
    assert.equal(core.getField('sd_image_generation_result_t', result, 'count'), 2);
    await core.api.free_sd_image_generation_result(result);
    assert.deepEqual(calls.at(-1), [pointerBytes === 4 ? Number(result) : result]);
  }
});

test('generation progress counters retain signed 64-bit precision', () => {
  const { module, schema } = fixture();
  const progressSchema = {
    ...schema,
    records: [{
      name: 'sd_generation_progress_t',
      id: 0,
      size: 16,
      alignment: 8,
      fields: [
        { name: 'completed', id: 0, kind: 'i64', abiType: { kind: 'integer', bits: 64, signed: true }, offset: 0, size: 8 },
        { name: 'total', id: 1, kind: 'i64', abiType: { kind: 'integer', bits: 64, signed: true }, offset: 8, size: 8 },
      ],
    }],
  };
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
