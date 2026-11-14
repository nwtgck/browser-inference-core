import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createAbiCaller, attachAbiCore, fromNativePointer, type NativeModule } from '#runtimes/shared/host/abi.mjs';

const [directory, suspension] = process.argv.slice(2);
assert.ok(directory);
assert.ok(suspension === 'direct' || suspension === 'jspi' || suspension === 'asyncify');
const schema = JSON.parse(readFileSync(path.join(directory, 'generated/schema.json'), 'utf8'));
const factory = (await import(pathToFileURL(path.join(directory, 'fixture.mjs')).href)).default;
const { createLowLevelFunctions } = await import(pathToFileURL(path.join(directory, 'generated/functions.mjs')).href);
const module: NativeModule & {
  addFunction(callback: (value: number) => number, signature: string): number | bigint;
  removeFunction(pointer: number | bigint): void;
} = await factory();
let busy = false;
let enters = 0;
let leaves = 0;
const caller = createAbiCaller({
  module,
  schema,
  suspension,
  gate: {
    enter() {
      assert.equal(busy, false);
      busy = true;
      enters++;
    },
    leave() {
      assert.equal(busy, true);
      busy = false;
      leaves++;
    },
  },
});
const api = createLowLevelFunctions({ invoker: caller.invoker });
assert.equal(await api.ggml_fixture_i32(-2147483648), -2147483648);
assert.equal(await api.ggml_fixture_i32(2147483647), 2147483647);
assert.equal(await api.ggml_fixture_u32(4294967295), 4294967295);
assert.equal(await api.ggml_fixture_u8(255), 255);
assert.equal(await api.ggml_fixture_bool(1), 1);
assert.equal(await api.ggml_fixture_enum(-3), -3);
assert.equal(await api.ggml_fixture_u64(18446744073709551615n), 18446744073709551615n);
assert.equal(await api.ggml_fixture_i64(-9223372036854775808n), -9223372036854775808n);
assert.equal(await api.ggml_fixture_float(Infinity), Infinity);
assert.ok(Number.isNaN(await api.ggml_fixture_float(NaN)));
const highPointer = schema.pointerBytes === 4 ? 4294967295n : 18446744073709551615n;
assert.equal(await api.ggml_fixture_pointer(highPointer), highPointer);
assert.equal(await api.ggml_fixture_size(highPointer), highPointer);
await assert.rejects(api.ggml_fixture_i32(2147483648), /declared width/u);
await assert.rejects(api.ggml_fixture_u8(256), /declared width/u);
await assert.rejects(api.ggml_fixture_bool(2), /declared width/u);
await assert.rejects(api.ggml_fixture_u64(-1n), /declared width/u);
await assert.rejects(api.ggml_fixture_pointer(highPointer + 1n), /pointer/u);
assert.equal(busy, false);
assert.equal(enters, leaves);
assert.equal(caller.callSync({ name: 'ggml_fixture_getter', args: [] }), 11);
assert.throws(() => caller.callSync({ name: 'ggml_fixture_i32', args: [0] }), /synchronous leaf/u);

const callback = module.addFunction(value => {
  assert.equal(busy, true);
  const metadata = caller.callSync({ name: 'ggml_fixture_getter', args: [] });
  assert.equal(typeof metadata, 'number');
  assert.ok(typeof metadata === 'number');
  return value + metadata;
}, 'ii');
try {
  assert.equal(await api.ggml_fixture_callback_call(fromNativePointer({ value: callback, pointerBytes: schema.pointerBytes }), 5), 16);
} finally {
  module.removeFunction(callback);
}
const core = attachAbiCore({ module, schema, suspension, createApi: ({ invoker }) => createLowLevelFunctions({ invoker }) });
assert.equal(core.recordSize('ggml_bf16_t'), 2);
const bf16 = core.allocRecord('ggml_bf16_t');
try {
  core.setField('ggml_bf16_t', bf16, 'bits', 65535);
  assert.equal(core.getField('ggml_bf16_t', bf16, 'bits'), 65535);
} finally {
  core.free(bf16);
}
const record = core.allocRecord('ggml_fixture_record');
try {
  await core.api.ggml_fixture_record_default(record);
  assert.equal(core.getField('ggml_fixture_record', record, 'count'), 4);
  assert.equal(core.getField('ggml_fixture_record', record, 'active'), 1);
  assert.equal(core.getField('ggml_fixture_record', record, 'signed_value'), -2);
  assert.equal(await core.api.ggml_fixture_record_sum(record), 34);
  const nested = core.fieldAddress('ggml_fixture_record', record, 'nested');
  assert.equal(core.getField('ggml_fixture_record.nested', nested, 'payload'), 23);
  const inner = core.fieldAddress('ggml_fixture_record.nested', nested, 'inner');
  assert.equal(core.getField('ggml_fixture_record.nested.inner', inner, 'code'), -3);
  core.setField('ggml_fixture_record.nested', nested, 'payload', 30);
  assert.equal(await core.api.ggml_fixture_record_sum(record), 41);
  core.setField('ggml_fixture_record.nested', nested, 'payload', 23);
  const sample = core.fieldAddress('ggml_fixture_record', record, 'samples') + BigInt(core.recordSize('ggml_fixture_record.samples'));
  assert.equal(core.getField('ggml_fixture_record.samples', sample, 'value'), -7);
  core.setField('ggml_fixture_record', record, 'count', 10);
  assert.equal(await core.api.ggml_fixture_record_sum(record), 40);
  await assert.rejects(core.api.ggml_fixture_record_sum(record + 1n), /record storage/u);
  await assert.rejects(core.api.ggml_fixture_record_sum(0n), /record storage/u);
  assert.equal(core.constant('GGML_FIXTURE_WIDTH'), schema.pointerBytes);
} finally {
  core.free(record);
}
if (suspension !== 'direct') {
  const pending = api.ggml_fixture_wait(41);
  assert.equal(busy, true);
  assert.equal(caller.callSync({ name: 'ggml_fixture_getter', args: [] }), 11);
  await assert.rejects(api.ggml_fixture_i32(1));
  assert.equal(await pending, 42);
  assert.equal(busy, false);
}
assert.equal(enters, leaves);
writeFileSync(1, JSON.stringify({ pointerBytes: schema.pointerBytes, suspension, functions: schema.functions.length, schemaSha256: schema.schemaSha256 }) + '\n');
