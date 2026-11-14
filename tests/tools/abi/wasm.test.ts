import assert from 'node:assert/strict';
import { cp, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { generateBindings } from '#tools/abi/generate';
import { runProcess } from '#tools/shared/process';

const checkout = path.resolve(path.dirname(createRequire(import.meta.url).resolve('#source/toolchain.json')), '..');
const fixtures = path.join(checkout, 'tests/tools/abi/fixtures');
const sdk = process.env.BICORE_EMSDK_ROOT;
const sdkNode = process.env.BICORE_EMSDK_NODE;

for (const variant of [
  { pointer64: 0, suspension: 'asyncify' },
  { pointer64: 1, suspension: 'direct' },
  { pointer64: 0, suspension: 'jspi' },
]) {
  test(`actual C declarations through Wasm and host types: memory64=${variant.pointer64}, ${variant.suspension}`, { skip: !sdk || !sdkNode }, async () => {
    assert.ok(sdk);
    assert.ok(sdkNode);
    using compilerEnvironment = {
      previous: process.env.EM_CONFIG,
      [Symbol.dispose]() {
        if (this.previous === undefined) {
          delete process.env.EM_CONFIG;
        } else {
          process.env.EM_CONFIG = this.previous;
        }
      },
    };
    process.env.EM_CONFIG = path.join(sdk, '.emscripten');
    const work = await mkdtemp(path.join(os.tmpdir(), 'bicore-abi-wasm-'));
    try {
      const compileRoot = path.join(work, 'compile');
      const native = path.join(compileRoot, 'runtimes/llama-cpp/native/abi');
      const build = path.join(work, 'build');
      await mkdir(native, { recursive: true });
      for (const name of ['api.h', 'metadata.cpp', 'bridge.cpp']) {
        await cp(path.join(fixtures, name), path.join(native, name));
      }
      const env = { ...process.env, EM_CONFIG: path.join(sdk, '.emscripten'), EMSDK_NODE: sdkNode };
      await runProcess({ command: 'cmake', args: ['-S', fixtures, '-B', build, '-G', 'Ninja', `-DCMAKE_TOOLCHAIN_FILE=${path.join(sdk, 'upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake')}`, `-DCOMPILE_ROOT=${compileRoot}`, `-DPOINTER64=${variant.pointer64}`, `-DSUSPENSION=${variant.suspension}`, `-DCMAKE_CROSSCOMPILING_EMULATOR=${sdkNode}`], cwd: checkout, env });
      const schema = await generateBindings({
        policy: { runtime: 'llama-cpp', astScope: undefined, publicFunction: ({ entry: { node, namespaces } }) => namespaces.length === 0 && node.kind === 'FunctionDecl' && node.name?.startsWith('ggml_fixture_') === true, exclusions: {}, acceptedDeprecated: [], synchronous: ['ggml_fixture_getter'] },
        compileRoot,
        compilationDatabase: path.join(build, 'compile_commands.json'),
        translationUnit: path.join(native, 'metadata.cpp'),
        bridgeUnit: path.join(native, 'bridge.cpp'),
        output: path.join(build, 'generated'),
        runtimeDirectory: undefined,
      });
      assert.equal(schema.pointerBytes, variant.pointer64 ? 8 : 4);
      assert.equal(schema.functions.filter(fn => fn.bridge).length, 2);
      assert.ok(!schema.functions.some(fn => fn.name === 'ggml_fixture_header_local'));
      assert.ok(schema.excluded.some(entry => entry.name === 'ggml_fixture_header_local' && entry.reason.includes('header-local')));
      assert.ok(schema.callbacks.some(callback => callback.name === 'ggml_fixture_callback'));
      assert.ok(!schema.suspendingExports.includes('ggml_fixture_getter'));
      const record = schema.records.find(value => value.name === 'ggml_fixture_record');
      assert.ok(record);
      assert.ok(record.fields.some(field => field.name === 'signed_value'));
      assert.deepEqual(record.fields.find(field => field.name === 'values')?.abiType.dimensions, [3]);
      assert.equal(record.fields.find(field => field.name === 'values')?.abiType.element?.kind, 'unsigned');
      assert.equal(record.fields.find(field => field.name === 'flags')?.abiType.element?.kind, 'boolean');
      assert.equal(record.fields.find(field => field.name === 'samples')?.abiType.element?.abiType.record, 'ggml_fixture_record.samples');
      await runProcess({ command: 'cmake', args: [`-DBICORE_FINGERPRINT_JSON=${path.join(build, 'generated/abi-fingerprint.json')}`, `-DBICORE_FINGERPRINT_TEMPLATE=${path.join(checkout, 'tools/abi/fingerprint.h.in')}`, `-DBICORE_FINGERPRINT_HEADER=${path.join(build, 'generated/fingerprint.h')}`, '-P', path.join(checkout, 'tools/abi/ConfigureFingerprint.cmake')], cwd: checkout });
      await runProcess({ command: 'cmake', args: ['--build', build, '--target', 'fixture', '--parallel', '2'], cwd: checkout, env });
      const execution = await runProcess({ command: sdkNode, args: ['--experimental-wasm-jspi', fileURLToPath(new URL('./wasm.test-support.js', import.meta.url)), build, variant.suspension], cwd: checkout, env, capture: 'file' });
      assert.match(execution.stdout.toString(), /"schemaSha256":"[0-9a-f]{64}"/u);

      // The consumer imports the emitted public declaration and the real shared
      // caller contract. These negative calls must remain compile errors.
      await writeFile(path.join(build, 'consumer.mts'), `\
import { createLowLevelFunctions } from './generated/functions.mjs';
import { createAbiCaller } from './abi.mjs';
declare const module: Parameters<typeof createAbiCaller>[0]['module'];
const caller = createAbiCaller({ module, schema: {}, suspension: 'direct', gate: undefined });
const api = createLowLevelFunctions({ invoker: caller.invoker });
const token: Promise<number> = api.ggml_fixture_i32(1);
const pointer: Promise<bigint> = api.ggml_fixture_pointer(1n);
// @ts-expect-error unknown API cannot be called
api.missing_function();
// @ts-expect-error wasm pointers use bigint in the public API
api.ggml_fixture_pointer(1);
// @ts-expect-error scalar token result is not a bigint
const invalid: Promise<bigint> = api.ggml_fixture_i32(1);
void [token, pointer, invalid];
`);
      // Native ESM uses .mjs; TypeScript resolves its declaration as .d.mts.
      await cp(path.join(build, 'generated/functions.d.ts'), path.join(build, 'generated/functions.d.mts'));
      await cp(path.join(checkout, 'runtimes/shared/host/abi.d.ts'), path.join(build, 'abi.d.mts'));
      await runProcess({ command: process.execPath, args: [path.join(checkout, 'node_modules/typescript/bin/tsc'), '--noEmit', '--strict', '--target', 'ES2023', '--module', 'NodeNext', '--moduleResolution', 'NodeNext', '--lib', 'ES2023,DOM', path.join(build, 'consumer.mts')], cwd: checkout, capture: 'file' });
      if (process.env.BICORE_ABI_KEEP_OUTPUT) {
        process.stderr.write(`ABI fixture retained at ${build}\n`);
      }
    } finally {
      if (!process.env.BICORE_ABI_KEEP_OUTPUT) {
        await rm(work, { recursive: true, force: true });
      }
    }
  });
}
