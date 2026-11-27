import test from 'node:test';
import assert from 'node:assert/strict';
import { cp, mkdir, access, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import ts from 'typescript';
import { checkPublicExports } from '#tools/package/public-exports';
import { packageFixture } from './fixtures.test-support.js';

test('generated package exports resolve in a real browser consumer and reject wrong profiles and signatures', async () => {
  await using fixture = await packageFixture();
  const consumer = join(fixture.root, 'consumer');
  await mkdir(join(consumer, 'node_modules'), { recursive: true });
  await cp(fixture.directory, join(consumer, 'node_modules/browser-inference-core'), { recursive: true });
  await assert.rejects(access(join(consumer, 'node_modules/browser-inference-core/runtimes/llama-cpp-browser-core/profiles/webgpu-wasm64-jspi-gpu-profile')));
  await writeFile(join(consumer, 'package.json'), '{"name":"package-type-consumer","type":"module","private":true}');
  function compile({ file }: { file: string }): readonly ts.Diagnostic[] {
    const program = ts.createProgram([file], { target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.NodeNext, moduleResolution: ts.ModuleResolutionKind.NodeNext, strict: true, noEmit: true, skipLibCheck: false, types: [], lib: ['lib.es2023.d.ts', 'lib.dom.d.ts', 'lib.dom.iterable.d.ts'] });
    return ts.getPreEmitDiagnostics(program);
  }
  const positive = join(consumer, 'positive.ts');
  await writeFile(positive, `\
import imageCatalog from 'browser-inference-core/stable-diffusion-cpp/api/schema.mjs';
import llamaCatalog from 'browser-inference-core/llama-cpp/api/schema.mjs';
import { createCore, type ModuleByProfile, type Profile } from 'browser-inference-core/llama-cpp/host';
import type { LowLevelFunctions as LlamaFunctions } from 'browser-inference-core/llama-cpp/api/functions';
import type { ModelGraphNativeExports, Q8DiagnosticFunctions } from 'browser-inference-core/llama-cpp/api/extensions';
import type { CoreModule as ImageModule, ModuleByProfile as ImageModules } from 'browser-inference-core/stable-diffusion-cpp/host';
import type { LowLevelFunctions as ImageFunctions } from 'browser-inference-core/stable-diffusion-cpp/api/functions';
import imageFactory from 'browser-inference-core/stable-diffusion-cpp/profiles/webgpu-wasm32-jspi/browser/core.mjs';
import speechFactory from 'browser-inference-core/speech-cpp/profiles/webgpu-wasm32-jspi/browser/core.mjs';
import { attachCore as attachSpeech, schema as speechCatalog, type CoreModule as SpeechModule, type ModuleByProfile as SpeechModules, type FactoryByProfile as SpeechFactories, type CoreFactory as SpeechFactory } from 'browser-inference-core/speech-cpp/host';
import type { LowLevelFunctions as SpeechFunctions } from 'browser-inference-core/speech-cpp/api/functions';
const profile: Profile = 'cpu-wasm32';
void createCore({ profile });
declare const native: ModuleByProfile['cpu-wasm32']['browser'];
const nativeResult: number = native._fixture_version();
const graph: ModelGraphNativeExports = native;
declare const diagnostic: Q8DiagnosticFunctions;
const diagnosticVersion: number = diagnostic.version(1n);
const diagnosticRun: Promise<number> = diagnostic.runV1(1n, 2n, 3n, 4n, 128);
declare const llama: LlamaFunctions;
const llamaResult: Promise<number> = llama.fixture_version();
declare const image: ImageModule;
declare const imageTarget: ImageModules['webgpu-wasm64-jspi']['browser'];
const target: ImageModule = imageTarget;
const imageNativeResult: number = image._sd_browser_bf16_retention_version();
declare const imageFunctions: ImageFunctions;
const imageResult: Promise<number> = imageFunctions.sd_browser_bf16_retention_version();
const catalogs: string[] = [imageCatalog.schemaSha256, llamaCatalog.schemaSha256];
void imageFactory({ locateFile: path => path, print: text => { void text; } });
const typedSpeechFactory: SpeechFactory = speechFactory;
declare const selectedFactory: SpeechFactories['webgpu-wasm32-jspi']['browser'];
const exactFactory: typeof speechFactory = selectedFactory;
void typedSpeechFactory({ locateFile: path => path });
void exactFactory({ printErr: text => { void text; } });
declare const speech: SpeechModule;
declare const speechTarget: SpeechModules['webgpu-wasm32-jspi']['browser'];
const speechModule: SpeechModule = speechTarget;
const speechNativeResult: number = speech._speech_api_version_major();
declare const speechFunctions: SpeechFunctions;
const speechResult: Promise<number> = speechFunctions.speech_api_version_major();
void attachSpeech(speech, speechCatalog, { suspension: 'jspi' });
void speechFactory({ locateFile: path => path });
void [nativeResult, graph, diagnosticVersion, diagnosticRun, llamaResult, target, imageNativeResult, imageResult, catalogs, speechModule, speechNativeResult, speechResult];
`);
  const diagnostics = compile({ file: positive });
  assert.equal(diagnostics.length, 0, ts.formatDiagnosticsWithColorAndContext(diagnostics, { getCanonicalFileName: file => file, getCurrentDirectory: () => consumer, getNewLine: () => '\n' }));
  const negative = join(consumer, 'negative.ts');
  await writeFile(negative, `\
import type { Profile } from 'browser-inference-core/llama-cpp/host';
import type { LowLevelFunctions } from 'browser-inference-core/llama-cpp/api/functions';
import type { ModelGraphNativeExports, Q8DiagnosticFunctions } from 'browser-inference-core/llama-cpp/api/extensions';
import imageFactory from 'browser-inference-core/stable-diffusion-cpp/profiles/webgpu-wasm32-jspi/browser/core.mjs';
const unavailable: Profile = 'not-packaged';
declare const api: LowLevelFunctions;
api.nonexistent_function();
api.fixture_version('wrong argument');
const wrongReturn: Promise<bigint> = api.fixture_version();
void imageFactory({ locateFile: 123 });
declare const graph: ModelGraphNativeExports;
graph.absent_extension_function();
declare const diagnostic: Q8DiagnosticFunctions;
diagnostic.version(1);
const wrongDiagnostic: number = diagnostic.runV1(1n, 2n, 3n, 4n, 128);
`);
  const failures = compile({ file: negative });
  assert(failures.every(item => item.file?.fileName === negative), ts.formatDiagnosticsWithColorAndContext(failures, { getCanonicalFileName: file => file, getCurrentDirectory: () => consumer, getNewLine: () => '\n' }));
  assert.deepEqual(failures.map(item => item.code).sort(), [2322, 2322, 2322, 2322, 2339, 2339, 2345, 2554].sort());
});

test('publication rejects missing common and profile-specific public declaration bindings', async () => {
  await using fixture = await packageFixture();
  const packageJson = join(fixture.directory, 'package.json');
  const broken = (await readFile(packageJson, 'utf8'))
    .replace('/host/abi.d.ts', '/host/missing.d.ts')
    .replace('*/model-graph-v1.functions.d.ts', '*/missing-graph.functions.d.ts');
  await writeFile(packageJson, broken);
  await assert.rejects(checkPublicExports({ directory: fixture.directory, runtimes: fixture.validated.runtimes }), (error: unknown) => {
    assert.ok(error instanceof Error);
    assert.match(error.message, /Could not find a declaration file for module 'browser-inference-core\/llama-cpp\/host\/abi'/u);
    assert.match(error.message, /Cannot find module 'browser-inference-core\/llama-cpp\/profiles\/cpu-wasm32\/browser\/model-graph-v1.functions'/u);
    return true;
  });
});

test('publication verifies exactly the runtime files exposed by Node import conditions', async () => {
  await using fixture = await packageFixture();
  const packageJson = join(fixture.directory, 'package.json');
  const original = await readFile(packageJson, 'utf8');
  const check = () => checkPublicExports({ directory: fixture.directory, runtimes: fixture.validated.runtimes });
  await check();

  const mutations = [
    {
      original: 'llama-cpp-browser-core/host/index.mjs',
      replacement: 'llama-cpp-browser-core/host/missing.mjs',
      error: /Public runtime import browser-inference-core\/llama-cpp\/host resolved to .*host\/missing\.mjs/u,
    },
    {
      original: 'stable-diffusion-cpp-browser-core/profiles/*/core.mjs',
      replacement: 'stable-diffusion-cpp-browser-core/profiles/webgpu-wasm64-jspi/browser/core.mjs',
      error: /Public runtime import browser-inference-core\/stable-diffusion-cpp\/profiles\/.*\/core\.mjs resolved to .*webgpu-wasm64-jspi\/browser\/core\.mjs/u,
    },
    {
      original: 'llama-cpp-browser-core/profiles/*/core.mjs',
      replacement: 'llama-cpp-browser-core/profiles/*/missing.mjs',
      error: /Public runtime import browser-inference-core\/llama-cpp\/profiles\/.*\/core\.mjs resolved to .*missing\.mjs/u,
    },
    {
      original: 'speech-cpp-browser-core/profiles/*/core.wasm',
      replacement: 'speech-cpp-browser-core/profiles/*/missing.wasm',
      error: /Public runtime import browser-inference-core\/speech-cpp\/profiles\/.*\/core\.wasm resolved to .*missing\.wasm/u,
    },
  ];
  for (const mutation of mutations) {
    await writeFile(packageJson, original.replace(mutation.original, mutation.replacement));
    await assert.rejects(check(), mutation.error);
  }

  await writeFile(packageJson, original);
  await rm(join(fixture.directory, 'runtimes/llama-cpp-browser-core/profiles/cpu-wasm32/browser/core.wasm'));
  await assert.rejects(check(), /Public runtime import browser-inference-core\/llama-cpp\/profiles\/cpu-wasm32\/browser\/core\.wasm is missing its packaged file/u);
});
