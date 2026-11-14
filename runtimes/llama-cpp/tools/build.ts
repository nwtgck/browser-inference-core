import { readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { LlamaProvenance } from '#dto/llama-cpp';
import { decodeCompilationDatabase } from '#dto/shared';
import { directory, fileIdentities, fileIdentity, optionalStat, treeIdentity } from '#tools/shared/files';
import { parseJson } from '#tools/shared/json';
import { compileArguments, recordCompilerInputs, type CompileEntry } from '#tools/shared/compiler-inputs';
import { withOutputDirectories } from '#tools/shared/output';
import { ProcessError, runProcess } from '#tools/shared/process';
import { exportCommit, gitEnvironment, gitRead, observeSource, snapshotCompileInputs, type SourceMode, type SourceSnapshot, type SourceState } from '#tools/source/capture';
import { stageProfile } from '#tools/package/stage';
import { verifyAsyncifyBigIntPatch } from '#tools/toolchain/prepare';
import { loadConfiguration, runtimePath, selectConfiguration, sourceCaptureSelection, upstreamCommit, upstreamPath, visionBf16StorageMetadata, type Configuration, type Profile, type Variant } from '#runtimes/llama-cpp/tools/config';
import { prepareSources, verifyPreparedSources, type CompileInputs, type PreparedSources } from '#runtimes/llama-cpp/tools/prepare';
import { prepareModuleDeclarations } from '#runtimes/llama-cpp/tools/declarations';

function definitions({ entry }: { entry: CompileEntry }): string[] {
  const arguments_ = compileArguments({ entry });
  const result: string[] = [];
  for (let index = 0; index < arguments_.length; index++) {
    const argument = arguments_[index];
    if (argument === '-D' || argument === '-U') {
      const value = arguments_[++index];
      if (value === undefined) {
        throw new Error('Missing compile definition');
      }
      result.push(argument + value);
    } else if (argument?.startsWith('-D') || argument?.startsWith('-U')) {
      result.push(argument);
    }
  }
  return result;
}

function definitionValues({ entry, name }: { entry: CompileEntry; name: string }): string[] {
  return definitions({ entry }).filter(value => ['-D', '-U'].some(prefix => value === `${prefix}${name}` || value.startsWith(`${prefix}${name}=`)));
}

export function verifyCompileConfiguration({ entries, profile }: { entries: readonly CompileEntry[]; profile: Profile }): void {
  const enabled = profile.gpuProfile === true;
  if (enabled && !profile.webgpu) {
    throw new Error('GPU profiling requires WebGPU');
  }
  const verify = ({ filename, expectedCount, definition, expected }: { filename: string; expectedCount: number; definition: string; expected: readonly string[] }): void => {
    const selected = entries.filter(entry => basename(entry.file) === filename);
    if (selected.length !== expectedCount) {
      throw new Error(`Unexpected translation unit count for ${filename}: ${selected.length}`);
    }
    for (const entry of selected) {
      const actual = definitionValues({ entry, name: definition });
      if (JSON.stringify(actual) !== JSON.stringify(expected)) {
        throw new Error(`Compile definition mismatch for ${filename}: ${actual.join(' ')}`);
      }
    }
  };
  verify({ filename: 'ggml-webgpu.cpp', expectedCount: profile.webgpu ? 1 : 0, definition: 'GGML_WEBGPU_GPU_PROFILE', expected: enabled ? ['-DGGML_WEBGPU_GPU_PROFILE=1'] : [] });
  for (const entry of entries.filter(value => basename(value.file) === 'ggml-webgpu.cpp')) {
    const uploads = definitionValues({ entry, name: 'GGML_WEBGPU_BATCH_PARAM_UPLOADS' });
    if (uploads.length !== 1 || !['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS', '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS=1'].includes(uploads[0] ?? '')) {
      throw new Error('Batch parameter uploads must be enabled unambiguously');
    }
  }
  for (const filename of ['webgpu-q8-diagnostic.cpp']) {
    verify({ filename, expectedCount: enabled ? 1 : 0, definition: 'LCB_WEBGPU_Q8_DIAGNOSTIC', expected: ['-DLCB_WEBGPU_Q8_DIAGNOSTIC=1'] });
  }
  for (const filename of ['mtmd-model.cpp', 'mtmd-subcontext.cpp', 'mtmd.cpp']) {
    verify({ filename, expectedCount: 1, definition: 'LCB_MTMD_WEBGPU_BF16_RUNTIME', expected: [`-DLCB_MTMD_WEBGPU_BF16_RUNTIME=${profile.webgpu ? 1 : 0}`] });
    const entry = entries.find(value => basename(value.file) === filename);
    if (entry !== undefined && definitions({ entry }).some(value => /^-[DU]LCB_MTMD_RETAIN_BF16(?:=|$)/u.test(value))) {
      throw new Error('Obsolete build-time vision BF16 definition');
    }
  }
}

export function annotateBrowserModule({ source }: { source: string }): string {
  const matches = [...source.matchAll(/\bnew\s+URL\s*\(\s*(?<annotation>\/\* @vite-ignore \*\/\s*)?(?<argument>(?<quote>['"])core\.wasm\k<quote>)\s*,\s*import\s*\.\s*meta\s*\.\s*url\s*,?\s*\)/gu)];
  const allWasm = [...source.matchAll(/\bnew\s+URL\s*\(\s*(?:\/\*[\s\S]*?\*\/\s*)*['"`][^'"`\r\n]*\.wasm[^'"`\r\n]*['"`]/gu)];
  const match = matches[0];
  if (matches.length !== 1 || allWasm.length !== 1 || match?.groups?.['argument'] === undefined) {
    throw new Error('Unknown generated browser Wasm URL boundary');
  }
  if (match.groups['annotation'] !== undefined) {
    return source;
  }
  const argumentIndex = source.indexOf(match.groups['argument'], match.index);
  return `${source.slice(0, argumentIndex)}/* @vite-ignore */${source.slice(argumentIndex)}`;
}

export type BuildContext = { checkoutRoot: string; toolingRoot: string; outputRoot: string; toolchainRoot: string };
export type BuildSelection = { profile: string; variant: string; kind: 'native' | 'wasm'; jobs: number; fresh: boolean; includeToolchainNotices: boolean; configureOnly: boolean; python: string };
export type CapturedSources = { snapshot: SourceSnapshot; inputs: CompileInputs; llamaCommit: string; buildInfo: { commit: string; number: number } };
export type BuiltProfile = { kind: 'built'; buildDirectory: string; runtimeDirectory: string; generatedDirectory: string; provenance: LlamaProvenance };
export type ConfiguredProfile = { kind: 'configured'; buildDirectory: string; compilationDatabase: string };

/** Stabilize only compiler inputs; the current checkout continues to own Node tools. */
export async function prepareBuildInputs({ context, source, selection, additionalInputs }: { context: BuildContext; source: SourceState; selection: { profile: string; variant: string; kind: 'native' | 'wasm' }; additionalInputs: readonly string[] }): Promise<CapturedSources> {
  let revision: string | undefined;
  switch (source.mode) {
  case 'release': revision = source.head;
    break;
  case 'local': revision = undefined;
    break;
  default: {
    const _ex: never = source;
    throw new Error(`Unknown source mode: ${String(_ex)}`);
  }
  }
  const llamaCommit = await upstreamCommit({ checkoutRoot: context.checkoutRoot, revision });
  const destination = resolve(context.outputRoot, 'source-inputs', selection.kind, selection.profile, selection.variant);
  let snapshot: SourceSnapshot | undefined;
  await withOutputDirectories({
    destinations: [destination],
    replace: true,
    preserveUnchanged: true,
    prepare: async ({ paths }) => {
      const work = paths[0];
      if (work === undefined) {
        throw new Error('Missing source input output');
      }
      const { include, exclude } = sourceCaptureSelection({ kind: selection.kind });
      snapshot = await snapshotCompileInputs({ source, destination: join(work, 'compile'), include: [...include, ...additionalInputs], exclude, replace: false });
      await exportCommit({ sourceRoot: join(context.checkoutRoot, upstreamPath), revision: llamaCommit, destination: join(work, 'upstream/llama.cpp') });
    },
  });
  if (snapshot === undefined) {
    throw new Error('Source input snapshot was not created');
  }
  snapshot = { ...snapshot, compileRoot: join(destination, 'compile') };
  const upstream = join(destination, 'upstream/llama.cpp');
  const number = Number((await gitRead({ sourceRoot: join(context.checkoutRoot, upstreamPath), args: ['rev-list', '--count', llamaCommit] })).toString('utf8').trim());
  if (!Number.isSafeInteger(number) || number < 0) {
    throw new Error('Invalid upstream build number');
  }
  const commit = (await gitRead({ sourceRoot: join(context.checkoutRoot, upstreamPath), args: ['rev-parse', '--short', llamaCommit] })).toString('utf8').trim();
  return { snapshot, inputs: { compileRoot: snapshot.compileRoot, upstreamRoots: { llama: upstream } }, llamaCommit, buildInfo: { commit, number } };
}

function buildDirectory({ context, selection }: { context: BuildContext; selection: BuildSelection }): string {
  return resolve(context.outputRoot, 'build', selection.kind, selection.profile, selection.variant);
}

async function runBuildCommand({ command, args, cwd, env, log }: { command: string; args: readonly string[]; cwd: string; env: NodeJS.ProcessEnv; log: string }): Promise<void> {
  const invocation = Buffer.from(`${JSON.stringify([command, ...args])}\n`);
  try {
    const result = await runProcess({ command, args, cwd, env });
    await writeFile(log, Buffer.concat([invocation, result.stdout, result.stderr]));
  } catch (error) {
    const output = error instanceof ProcessError ? Buffer.concat([error.stdout, error.stderr]) : Buffer.from(String(error));
    await writeFile(log, Buffer.concat([invocation, output]));
    throw error;
  }
}

/** --emit-tsd uses the SDK's npm compiler, independently of project tooling. */
export async function checkSdkDeclarationCompiler({ emscriptenRoot, buildRoot, env }: { emscriptenRoot: string; buildRoot: string; env: NodeJS.ProcessEnv }): Promise<void> {
  // emcmake records config.NODE_JS[0] here; reuse its selected executable rather
  // than guessing an SDK Node version or using a global TypeScript compiler.
  const cache = await readFile(join(buildRoot, 'CMakeCache.txt'), 'utf8');
  const node = /^CMAKE_CROSSCOMPILING_EMULATOR:[^=]+=([^\r\n]+)$/mu.exec(cache)?.[1];
  if (node === undefined) {
    throw new Error('Configured Emscripten Node executable is missing from CMakeCache.txt');
  }
  try {
    // The package entry is JavaScript; npm's .bin entry can be a shell shim.
    await runBuildCommand({ command: node, args: [join(emscriptenRoot, 'node_modules/typescript/bin/tsc'), '--version'], cwd: buildRoot, env, log: join(buildRoot, 'declaration-compiler.log') });
  } catch (error) {
    throw new Error(`Emscripten --emit-tsd requires its SDK npm dependencies. Run npm ci --prefix "${emscriptenRoot}" --include=dev --ignore-scripts --no-audit --no-fund before building.\n${error instanceof Error ? error.message : String(error)}`);
  }
}

export async function buildPrepared({ context, source, prepared, selection, configuration, chosen }: { context: BuildContext; source: CapturedSources; prepared: PreparedSources; selection: BuildSelection; configuration: Configuration; chosen: { profile: Profile; variant: Variant } }): Promise<BuiltProfile | ConfiguredProfile> {
  const buildRoot = buildDirectory({ context, selection });
  await directory({ path: buildRoot, create: true });
  await rm(join(buildRoot, 'provenance.json'), { force: true });
  const { memory64, webgpu, jspi, asyncify, pthreads, maximumMemory, gpuProfile, ...unhandledProfile } = chosen.profile;
  unhandledProfile satisfies Record<PropertyKey, never>;
  if (pthreads) {
    throw new Error('Threaded runtime profiles require a pthread build integration');
  }
  const emscriptenRoot = join(context.toolchainRoot, 'emsdk/upstream/emscripten');
  const dawnPackage = join(context.toolchainRoot, 'emdawnwebgpu_pkg');
  let emccVersion = 'native';
  const cache = join(buildRoot, 'CMakeCache.txt');
  if (await optionalStat({ path: cache })) {
    const home = /^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$/mu.exec(await readFile(cache, 'utf8'))?.[1];
    if (home !== undefined && resolve(home) !== resolve(source.inputs.compileRoot, runtimePath)) {
      await rm(cache);
      await rm(join(buildRoot, 'CMakeFiles'), { recursive: true, force: true });
    }
  }
  const onOff = ({ value }: { value: boolean }): string => value ? 'ON' : 'OFF';
  const cmakeArgs = ['-S', join(source.inputs.compileRoot, runtimePath), '-B', buildRoot, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', `-DLCB_VARIANT=${selection.variant}`, `-DLCB_CHECKOUT_ROOT=${context.checkoutRoot}`, `-DLCB_COMPILE_ROOT=${source.inputs.compileRoot}`, `-DLCB_TOOLING_ROOT=${context.toolingRoot}`, `-DLCB_NODE=${process.execPath}`,
    `-DLCB_LLAMA_SOURCE=${prepared.root}`, `-DLCB_PREPARED_SOURCE=${prepared.root}`, `-DPython3_EXECUTABLE=${selection.python}`,
    `-DLCB_NATIVE_CHECK=${onOff({ value: selection.kind === 'native' })}`,
    `-DLLAMA_BUILD_COMMIT=${source.buildInfo.commit}`, `-DLLAMA_BUILD_NUMBER=${source.buildInfo.number}`, '-DGIT_EXE:FILEPATH=',
    `-DLCB_MEMORY64=${onOff({ value: memory64 })}`, `-DLCB_FILE_BACKED_PLE=${onOff({ value: prepared.receipt.options.fileRows })}`, `-DLCB_QSA_ALL_POOLS=${onOff({ value: prepared.receipt.options.qsa })}`,
    `-DLCB_WEBGPU=${onOff({ value: webgpu })}`, `-DGGML_WEBGPU_GPU_PROFILE=${onOff({ value: gpuProfile === true })}`,
    `-DLCB_WEBGPU_BF16_PROJECTOR=${onOff({ value: webgpu })}`, `-DLCB_JSPI=${onOff({ value: jspi })}`, `-DLCB_ASYNCIFY=${onOff({ value: asyncify })}`,
    `-DLCB_MAXIMUM_MEMORY=${maximumMemory}`];
  let command: string;
  let args: string[];
  switch (selection.kind) {
  case 'wasm': {
    const version = (await runProcess({ command: join(emscriptenRoot, 'emcc'), args: ['--version'], cwd: buildRoot })).stdout.toString('utf8').split('\n')[0];
    if (version === undefined || !version.includes(configuration.toolchain.emsdkVersion)) {
      throw new Error(`Expected Emscripten ${configuration.toolchain.emsdkVersion}; found ${version}`);
    }
    emccVersion = version;
    if (asyncify) {
      await verifyAsyncifyBigIntPatch({ emscriptenRoot, patchConfig: configuration.toolchain.emscriptenAsyncifyBigIntPatch });
    }
    if (webgpu) {
      for (const name of ['emdawnwebgpu.port.py', 'webgpu/include/webgpu/webgpu.h', 'webgpu_cpp/include/webgpu/webgpu_cpp.h']) {
        await fileIdentity({ path: join(dawnPackage, name) });
      }
    }
    if (webgpu) {
      cmakeArgs.push(`-DEMDAWNWEBGPU_DIR=${dawnPackage}`);
    }
    command = selection.python;
    args = [join(emscriptenRoot, 'emcmake.py'), 'cmake', ...cmakeArgs];
    break;
  }
  case 'native':
    // ABI declarations use the actual target compiler's Clang AST.
    cmakeArgs.push(`-DCMAKE_C_COMPILER=${process.env['CC'] ?? 'clang'}`, `-DCMAKE_CXX_COMPILER=${process.env['CXX'] ?? 'clang++'}`);
    command = 'cmake';
    args = cmakeArgs;
    break;
  default: {
    const _ex: never = selection.kind;
    throw new Error(`Unknown build kind: ${String(_ex)}`);
  }
  }
  const env = { ...gitEnvironment({ ceiling: source.inputs.compileRoot }), PYTHONDONTWRITEBYTECODE: '1', PATH: `${emscriptenRoot}:${process.env['PATH'] ?? ''}` };
  await runBuildCommand({ command, args, cwd: buildRoot, env, log: join(buildRoot, 'configure.log') });
  const compilationDatabase = join(buildRoot, 'compile_commands.json');
  const entries = parseJson({ text: await readFile(compilationDatabase, 'utf8'), decode: decodeCompilationDatabase });
  verifyCompileConfiguration({ entries, profile: chosen.profile });
  if (selection.configureOnly) {
    return { kind: 'configured', buildDirectory: buildRoot, compilationDatabase };
  }
  switch (selection.kind) {
  case 'wasm': await checkSdkDeclarationCompiler({ emscriptenRoot, buildRoot, env });
    break;
  case 'native': break;
  default: {
    const _ex: never = selection.kind;
    throw new Error(`Unknown build kind: ${String(_ex)}`);
  }
  }
  await runBuildCommand({ command: 'cmake', args: ['--build', buildRoot, '--target', 'core', '-j', String(selection.jobs)], cwd: buildRoot, env, log: join(buildRoot, 'compile.log') });
  const runtimeDirectory = join(buildRoot, 'runtime');
  let moduleDeclarationPreparation: LlamaProvenance['moduleDeclarationPreparation'];
  switch (selection.kind) {
  case 'wasm': {
    const declarations = join(runtimeDirectory, 'core.d.ts');
    const input = await fileIdentity({ path: declarations });
    const original = await readFile(declarations, 'utf8');
    const preparedDeclarations = prepareModuleDeclarations({ source: original });
    if (preparedDeclarations.source !== original) {
      await writeFile(declarations, preparedDeclarations.source);
    }
    moduleDeclarationPreparation = { input, output: await fileIdentity({ path: declarations }), correctedBigIntReferences: preparedDeclarations.correctedBigIntReferences };
    break;
  }
  case 'native': break;
  default: {
    const _ex: never = selection.kind;
    throw new Error(`Unknown build kind: ${String(_ex)}`);
  }
  }
  let browserModulePreparation: LlamaProvenance['browserModulePreparation'];
  if (selection.kind === 'wasm' && selection.variant === 'browser') {
    const module = join(runtimeDirectory, 'core.mjs');
    const input = await fileIdentity({ path: module });
    const sourceText = await readFile(module, 'utf8');
    const result = annotateBrowserModule({ source: sourceText });
    if (sourceText !== result) {
      await writeFile(module, result);
    }
    browserModulePreparation = { input, output: await fileIdentity({ path: module }), alreadyAnnotated: sourceText === result };
  }
  await verifyPreparedSources({ prepared });
  const upstream = source.inputs.upstreamRoots['llama'];
  if (upstream === undefined) {
    throw new Error('Missing captured upstream');
  }
  const upstreamIdentity = treeIdentity({ files: await fileIdentities({ root: upstream }) });
  if (upstreamIdentity.files !== prepared.receipt.upstream.files || upstreamIdentity.sha256 !== prepared.receipt.upstream.sha256) {
    throw new Error('Captured upstream inputs changed during build');
  }
  if (JSON.stringify(await fileIdentities({ root: source.inputs.compileRoot })) !== JSON.stringify(source.snapshot.files)) {
    throw new Error('Captured compiler inputs changed during build');
  }
  const state = source.snapshot.source;
  switch (state.mode) {
  case 'release': await observeSource({ checkoutRoot: context.checkoutRoot, mode: { kind: 'release', expectedCommit: state.head } });
    break;
  case 'local': break;
  default: {
    const _ex: never = state;
    throw new Error(`Unknown source mode: ${String(_ex)}`);
  }
  }
  const compiledInputs = await recordCompilerInputs({ buildRoot, roots: { prepared: prepared.root, compile: source.inputs.compileRoot, build: buildRoot, emscripten: emscriptenRoot } });
  const publishable = state.publishable && selection.kind === 'wasm';
  const provenance: LlamaProvenance = {
    runtime: 'llama-cpp',
    profile: selection.profile,
    variant: selection.variant,
    configuration: chosen.profile,
    variantConfiguration: chosen.variant,
    sourceCommit: state.head,
    sourceDirty: !publishable,
    llamaCommit: source.llamaCommit,
    toolchain: configuration.toolchain,
    emccVersion,
    nodeVersion: process.version,
    sourceInputs: { mode: state.mode, publishable, sourceCommit: state.head, files: source.snapshot.files },
    upstreamBuildInfo: source.buildInfo,
    preparedSources: prepared.receipt,
    compiledInputs,
    cmakeCommand: [command, ...args],
    builtAtUnix: Math.floor(Date.now() / 1000),
    buildCache: { ...(process.env['BIC_CACHE_ID'] === undefined ? {} : { identity: process.env['BIC_CACHE_ID'] }), ...(process.env['EM_COMPILER_WRAPPER'] === undefined ? {} : { compilerWrapper: process.env['EM_COMPILER_WRAPPER'] }) },
    visionBf16Storage: visionBf16StorageMetadata(),
    ...(moduleDeclarationPreparation === undefined ? {} : { moduleDeclarationPreparation }),
    ...(browserModulePreparation === undefined ? {} : { browserModulePreparation }),
    validation: { compiled: true, realModelInference: false, browserSmoke: false },
  };
  await writeFile(join(buildRoot, 'provenance.json'), `${JSON.stringify(provenance, undefined, 2)}\n`);
  return { kind: 'built', buildDirectory: buildRoot, runtimeDirectory, generatedDirectory: join(buildRoot, 'generated'), provenance };
}

export async function buildRuntime({ context, selection, mode }: { context: BuildContext; selection: BuildSelection; mode: SourceMode }): Promise<BuiltProfile | ConfiguredProfile> {
  if (!Number.isSafeInteger(selection.jobs) || selection.jobs <= 0) {
    throw new Error('Build jobs must be a positive integer');
  }
  const bindingsEntry = fileURLToPath(import.meta.resolve('#runtimes/llama-cpp/tools/bindings'));
  if (resolve(context.toolingRoot, runtimePath, 'tools/bindings.js') !== resolve(bindingsEntry)) {
    throw new Error('Tooling root must belong to the executing checkout');
  }
  const configuration = await loadConfiguration({ checkoutRoot: context.checkoutRoot });
  const chosen = selectConfiguration({ configuration, profile: selection.profile, variant: selection.variant });
  const state = await observeSource({ checkoutRoot: context.checkoutRoot, mode });
  const source = await prepareBuildInputs({ context, source: state, selection, additionalInputs: [] });
  const buildRoot = buildDirectory({ context, selection });
  if (selection.fresh) {
    await rm(buildRoot, { recursive: true, force: true });
  }
  const prepared = await prepareSources({
    inputs: source.inputs,
    output: join(buildRoot, 'prepared'),
    python: selection.python,
    options: { target: selection.kind, webgpu: chosen.profile.webgpu, gpuProfile: chosen.profile.gpuProfile === true, fileRows: chosen.profile.memory64, qsa: true, mtmdMode: chosen.profile.webgpu ? 'webgpu' : 'cpu' },
  });
  const result = await buildPrepared({ context, source, prepared, selection, configuration, chosen });
  switch (result.kind) {
  case 'configured': return result;
  case 'built':
    switch (selection.kind) {
    case 'native': break;
    case 'wasm': {
      const runtimeFiles = (await readdir(result.runtimeDirectory)).sort().filter(name => /[.]mjs$|[.]wasm$|[.]d[.]ts$|[.]json$/u.test(name)).map(name => ({ source: join(result.runtimeDirectory, name), name }));
      await stageProfile({
        buildDirectory: join(context.outputRoot, 'stage', selection.profile, selection.variant),
        generatedDirectory: result.generatedDirectory,
        runtimeFiles,
        provenance: result.provenance,
        noticeRoots: selection.includeToolchainNotices ? [join(context.toolchainRoot, 'emsdk'), join(context.toolchainRoot, 'emdawnwebgpu_pkg')] : [],
        noticeDestination: selection.includeToolchainNotices ? join(context.outputRoot, 'toolchain-licenses') : undefined,
      });
      break;
    }
    default: {
      const _ex: never = selection.kind;
      throw new Error(`Unknown build kind: ${String(_ex)}`);
    }
    }
    break;
  default: {
    const _ex: never = result;
    throw new Error(`Unknown build result: ${String(_ex)}`);
  }
  }
  return result;
}
