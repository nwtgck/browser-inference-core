import assert from 'node:assert/strict';
import test from 'node:test';
import { copyFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import ts from 'typescript';
import { decodeAbiSchema } from '#dto/shared';
import { readJson } from '#tools/shared/json';
import { READ_ENV_BICORE_SPEECH_ABI_SCHEMA } from '#tools/shared/node-env';
import { abiSchemaFingerprint } from '#tools/abi/schema';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

const schemaPath = READ_ENV_BICORE_SPEECH_ABI_SCHEMA();

test('actual speech metadata preserves opaque handles, Haru dialogue callbacks and cancellation', { skip: schemaPath === undefined }, async () => {
  assert.ok(schemaPath);
  const schema = await readJson({ path: schemaPath, decode: decodeAbiSchema });
  assert.equal(schema.runtime, 'speech-cpp');
  assert.equal(schema.schemaSha256, abiSchemaFingerprint({ schema }));
  const functions = new Map(schema.functions.map(fn => [fn.name, fn]));
  for (const name of ['speech_api_version_major', 'speech_api_version_minor', 'speech_load_params_new', 'speech_model_info_open', 'speech_model_load', 'speech_request_new', 'speech_synthesize', 'speech_transcribe', 'speech_dialogue_start', 'speech_dialogue_push', 'speech_dialogue_end', 'speech_dialogue_free']) {
    const fn = functions.get(name);
    assert.ok(fn, name);
    assert.ok(schema.exports.includes(fn.export), name);
    assert.equal(fn.bridge, false, name);
  }
  assert.ok(schema.functions.every(fn => fn.name.startsWith('speech_')));
  const dialogue = functions.get('speech_dialogue_start');
  assert.equal(dialogue?.parameters[0]?.kind, 'pointer');
  assert.equal(dialogue?.parameters[2]?.abiType.callback, 'speech_dialogue_callback');
  assert.equal(dialogue?.parameters[4]?.kind, 'pointer');
  const callback = schema.callbacks.find(value => value.name === 'speech_dialogue_callback');
  assert.ok(callback);
  assert.deepEqual(callback.parameters.map(value => value.kind), ['pointer', 'u64', 'pointer', 'pointer']);
  assert.equal(callback.returnKind, 'signed');
  assert.ok(schema.suspendingExports.includes('speech_dialogue_push'));
  for (const name of ['speech_request_cancel', 'speech_last_error', 'speech_last_error_option']) {
    assert.ok(schema.callbackMetadata.getters.some(fn => fn.name === name));
    assert.ok(!schema.suspendingExports.includes(name));
  }
  for (const name of ['SPEECH_OK', 'SPEECH_TASK_DIALOGUE', 'SPEECH_ERROR_INVALID_ARGUMENT', 'SPEECH_API_VERSION_MAJOR', 'SPEECH_API_VERSION_MINOR']) {
    assert.ok(schema.constants.includes(name), name);
  }
  assert.equal(schema.capabilities?.['ggufFileOffsetBits'], 64);
  assert.equal(schema.capabilities?.['callerOwnedRandomAccess'], 1);
});

test('a consumer compiles against the actual generated speech C API declarations', { skip: schemaPath === undefined }, async () => {
  assert.ok(schemaPath);
  await using temporary = await temporaryDirectory();
  for (const name of ['functions.d.ts', 'core.d.ts', 'module-options.d.ts']) {
    await copyFile(join(dirname(schemaPath), name), join(temporary.root, name));
  }
  await copyFile(join(process.cwd(), 'runtimes/shared/host/abi.d.ts'), join(temporary.root, 'host-abi.d.ts'));
  await writeFile(join(temporary.root, 'package.json'), '{"type":"module"}');
  const consumer = join(temporary.root, 'consumer.ts');
  await writeFile(consumer, `\
import type { LowLevelFunctions } from './functions.js';
import type { SpeechModule } from './core.js';
import { createAbiCaller } from './host-abi.js';
import type { AbiSchema } from './host-abi.js';
declare const api: LowLevelFunctions;
declare const module: SpeechModule;
declare const schema: AbiSchema;
const start: Promise<number> = api.speech_dialogue_start(1n, 24000, 2n, 3n, 4n);
const push: Promise<number> = api.speech_dialogue_push(1n, 2n, 3200n);
const version: Promise<bigint> = api.speech_version();
const cancel: Promise<void> = api.speech_request_cancel(1n);
const seed: bigint | Promise<bigint> = module._speech_result_seed(1n);
const frameCallback = module.addFunction((samples: bigint, nSamples: bigint, text: bigint, userData: bigint): number => {
  void [samples, nSamples, text, userData];
  return 0;
}, 'ipppp');
const rawStart: number | Promise<number> = module._speech_dialogue_start(1n, 24000, BigInt(frameCallback), 3n, 4n);
const logCallback = module.addFunction((level: number, text: bigint, userData: bigint): void => {
  void [level, text, userData];
}, 'vipp');
const rawCancel: void = module._speech_request_cancel(1n);
const caller = createAbiCaller({ module, schema, suspension: 'jspi', gate: undefined });
const syncCancel: number | bigint | void = caller.callSync({ name: 'speech_request_cancel', args: [1n] });
// @ts-expect-error C pointer arguments retain bigint at the public boundary.
api.speech_dialogue_push(1, 2n, 3200n);
// @ts-expect-error Callback parameters are native scalars, not JavaScript objects.
module.addFunction((_value: { sample: bigint }) => 0, 'ip');
// @ts-expect-error Native callbacks cannot return a Promise.
module.addFunction(() => Promise.resolve(0), 'i');
void [start, push, version, cancel, seed, frameCallback, rawStart, logCallback, rawCancel, syncCancel];
`);
  const program = ts.createProgram([consumer], { target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.NodeNext, moduleResolution: ts.ModuleResolutionKind.NodeNext, strict: true, noEmit: true, skipLibCheck: false, types: [], lib: ['lib.es2023.d.ts', 'lib.dom.d.ts'] });
  const diagnostics = ts.getPreEmitDiagnostics(program);
  assert.equal(diagnostics.length, 0, ts.formatDiagnosticsWithColorAndContext(diagnostics, { getCanonicalFileName: file => file, getCurrentDirectory: () => temporary.root, getNewLine: () => '\n' }));
});
