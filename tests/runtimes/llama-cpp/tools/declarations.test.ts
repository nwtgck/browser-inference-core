import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import ts from 'typescript';
import { prepareModuleDeclarations } from '#runtimes/llama-cpp/tools/declarations';

test('SDK preparation corrects only global BigInt type tokens in native export methods', () => {
  const source = `\
// BigInt remains unchanged in comments.
declare const value: typeof BigInt;
declare namespace Records { interface BigInt { readonly record: true } }
interface WasmModule {
  _native(BigInt: BigInt): BigInt;
  _generic<BigInt>(value: BigInt): BigInt;
  _record(value: Records.BigInt): Records.BigInt;
  label: "BigInt";
}
interface EmbindModule { keep(value: BigInt): BigInt }
export type MainModule = WasmModule & EmbindModule;
`;
  const prepared = prepareModuleDeclarations({ source });
  assert.equal(prepared.correctedBigIntReferences, 2);
  assert.equal(prepared.source, source.replace('_native(BigInt: BigInt): BigInt', '_native(BigInt: bigint): bigint'));
  assert.deepEqual(prepareModuleDeclarations({ source: prepared.source }), { source: prepared.source, correctedBigIntReferences: 0 });
  const namedRecord = `\
interface BigInt { readonly record: true }
interface WasmModule { _record(value: BigInt): BigInt }
export type MainModule = WasmModule;
`;
  assert.deepEqual(prepareModuleDeclarations({ source: namedRecord }), { source: namedRecord, correctedBigIntReferences: 0 });
  const globalAugmentation = 'interface BigInt { readonly record: true } interface WasmModule { _record(): BigInt }';
  assert.deepEqual(prepareModuleDeclarations({ source: globalAugmentation }), { source: globalAugmentation, correctedBigIntReferences: 0 });
  assert.equal(prepareModuleDeclarations({ source: 'interface WasmModule { _first(): BigInt } interface WasmModule { _second(value: BigInt): void }' }).correctedBigIntReferences, 2);
  assert.throws(() => prepareModuleDeclarations({ source: 'interface Other {}' }), /WasmModule/u);
  assert.throws(() => prepareModuleDeclarations({ source: 'interface WasmModule { _broken(: BigInt }' }), /Invalid generated/u);
});

test('prepared SDK native declarations accept primitive consumers and reject boxed arguments', async () => {
  await using temporary = {
    path: await mkdtemp(join(tmpdir(), 'llama-sdk-declarations-')),
    async [Symbol.asyncDispose]() {
      await rm(this.path, { recursive: true, force: true });
    },
  };
  const source = `\
interface WasmModule { _last_error(): BigInt; _echo(value: BigInt): BigInt }
export type MainModule = WasmModule;
`;
  const declaration = join(temporary.path, 'module.d.ts');
  const consumer = join(temporary.path, 'consumer.ts');
  async function diagnostics({ declarationSource, consumerSource }: { declarationSource: string; consumerSource: string }): Promise<readonly ts.Diagnostic[]> {
    await writeFile(declaration, declarationSource);
    await writeFile(consumer, consumerSource);
    const program = ts.createProgram({ rootNames: [consumer], options: { target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.NodeNext, moduleResolution: ts.ModuleResolutionKind.NodeNext, strict: true, skipLibCheck: false, types: [], noEmit: true } });
    return ts.getPreEmitDiagnostics(program);
  }
  const positive = `\
import type { MainModule } from './module.js';
declare const module: MainModule;
const error: bigint = module._last_error();
const echo: (value: bigint) => bigint = module._echo;
module._echo(error);
void echo;
`;
  assert.ok((await diagnostics({ declarationSource: source, consumerSource: positive })).some(diagnostic => diagnostic.code === 2322));
  const prepared = prepareModuleDeclarations({ source }).source;
  assert.deepEqual(await diagnostics({ declarationSource: prepared, consumerSource: positive }), []);
  const negative = `\
import type { MainModule } from './module.js';
declare const module: MainModule;
declare const boxed: BigInt;
module._echo(boxed);
`;
  assert.ok((await diagnostics({ declarationSource: prepared, consumerSource: negative })).some(diagnostic => diagnostic.code === 2345));
});
