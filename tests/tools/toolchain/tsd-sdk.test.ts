import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import ts from 'typescript';
import { loadToolchainConfiguration } from '#tools/toolchain/config';
import { verifyAsyncifyBigIntPatch, verifyTypeScriptPatch } from '#tools/toolchain/prepare';
import { READ_ENV_BICORE_EMSDK_NODE, READ_ENV_BICORE_EMSDK_ROOT } from '#tools/shared/node-env';
import { runProcess, type InheritedChildEnvironment } from '#tools/shared/process';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

const sdk = READ_ENV_BICORE_EMSDK_ROOT();
const sdkNode = READ_ENV_BICORE_EMSDK_NODE();

for (const variant of [{ pointer64: 0, suspension: 'asyncify' }, { pointer64: 1, suspension: 'direct' }, { pointer64: 1, suspension: 'jspi' }] as const) {
  test(`actual SDK declarations match JS conversion: memory64=${variant.pointer64}, ${variant.suspension}`, { skip: !sdk || !sdkNode }, async () => {
    assert.ok(sdk !== undefined && sdkNode !== undefined);
    await using temporary = await temporaryDirectory();
    const checkoutRoot = process.cwd();
    const emscriptenRoot = join(sdk, 'upstream/emscripten');
    const toolchain = await loadToolchainConfiguration({ checkoutRoot });
    await verifyTypeScriptPatch({ checkoutRoot, emscriptenRoot, patch: toolchain.emscriptenTypeScriptPatch });
    if (variant.suspension === 'asyncify') {
      await verifyAsyncifyBigIntPatch({ checkoutRoot, emscriptenRoot, patch: toolchain.emscriptenAsyncifyBigIntPatch });
    }
    const env: InheritedChildEnvironment = { kind: 'inherit', removePrefixes: [], overrides: { EM_CONFIG: join(sdk, '.emscripten'), EM_CACHE: join(emscriptenRoot, 'cache'), EMSDK_NODE: sdkNode } };
    const exports = ['_malloc', '_free', '_raw_i64', '_converted_pointer', '_optional_pointer', '_async_value', '_sync_value'];
    const args = [resolve('tests/tools/toolchain/fixtures/tsd-signatures.c'), '--no-entry', '--bind', '-sDEFAULT_TO_CXX=1', '-O0', '-sMODULARIZE=1', '-sEXPORT_ES6=1', '-sENVIRONMENT=node', '-sINCOMING_MODULE_JS_API=wasmBinary', '-sWASM_BIGINT=1', `-sMEMORY64=${variant.pointer64}`, `-sSIGNATURE_CONVERSIONS=${JSON.stringify(['converted_pointer:pp', 'optional_pointer:_P'])}`, '-sEXPORTED_RUNTIME_METHODS=HEAPU8,ccall', '--emit-tsd', join(temporary.root, 'core.d.ts'), '-o', join(temporary.root, 'core.mjs')];
    if (variant.suspension === 'jspi') {
      args.push('-sJSPI=1', '-sJSPI_EXPORTS=async_value');
    }
    if (variant.suspension === 'asyncify') {
      args.push('-sASYNCIFY=1', '-DTEST_ASYNCIFY=1');
      exports.push('_suspend_i64');
    }
    args.push(`-sEXPORTED_FUNCTIONS=${JSON.stringify(exports)}`);
    await runProcess({ command: join(emscriptenRoot, 'emcc'), args, cwd: temporary.root, env, capture: 'file' });
    const consumer = join(temporary.root, 'consumer.mts');
    const undefinedCall = variant.pointer64 ? '' : `\
// @ts-expect-error optional pointer conversion is only installed for memory64
core._optional_pointer(undefined);
`;
    await writeFile(consumer, `\
import type { MainModule } from './core.js';
declare const core: MainModule;
const pointer: number = core._malloc(32);
const released: void = core._free(pointer);
const raw: (value: bigint) => bigint = core._raw_i64;
const selected: (value: bigint) => ${variant.suspension === 'jspi' ? 'Promise<bigint>' : 'bigint'} = core._async_value;
const synchronous: number = core._sync_value(4);
const converted: number = core._converted_pointer(513);
const optional: ${variant.pointer64 ? 'bigint' : 'number'} = core._optional_pointer(${variant.pointer64 ? 'undefined' : '19'});
// @ts-expect-error the SDK's converted allocation API accepts Number sizes
core._malloc(1n);
// @ts-expect-error the SDK's converted allocation API returns Number pointers
const wrongPointer: bigint = core._malloc(32);
declare const boxed: BigInt;
// @ts-expect-error genuine i64 exports use primitive bigint
core._raw_i64(boxed);
${undefinedCall}
void [pointer, released, raw, selected, synchronous, converted, optional, wrongPointer];
`);
    const program = ts.createProgram({ rootNames: [consumer], options: { target: ts.ScriptTarget.ES2023, lib: ['lib.es2023.d.ts', 'lib.dom.d.ts'], module: ts.ModuleKind.NodeNext, moduleResolution: ts.ModuleResolutionKind.NodeNext, strict: true, exactOptionalPropertyTypes: true, types: [], noEmit: true } });
    assert.deepEqual(ts.getPreEmitDiagnostics(program).map(diagnostic => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')), []);
    const execution = await runProcess({ command: sdkNode, args: ['--experimental-wasm-jspi', fileURLToPath(new URL('./tsd-runtime.test-support.js', import.meta.url)), temporary.root, String(variant.pointer64), variant.suspension], cwd: temporary.root, env, capture: 'file' });
    assert.match(execution.stdout.toString('utf8'), /"malloc":"number","free":"synchronous","rawI64":"bigint"/u);
  });
}
