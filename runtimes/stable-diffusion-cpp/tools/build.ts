import { readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import type { StableDiffusionProvenance } from '#dto/stable-diffusion-cpp';
import { directory, fileIdentities, regularFile } from '#tools/shared/files';
import { runProcess } from '#tools/shared/process';
import { READ_ENV_BIC_CACHE_ID, READ_ENV_EM_COMPILER_WRAPPER } from '#tools/shared/node-env';
import { observeSource, snapshotCompileInputs, type SourceMode, type SourceSnapshot } from '#tools/source/capture';
import { stageProfile } from '#tools/package/stage';
import { recordCompilerInputs } from '#tools/shared/compiler-inputs';
import { loadToolchainConfiguration } from '#tools/toolchain/config';
import { verifyAsyncifyBigIntPatch } from '#tools/toolchain/prepare';
import { loadConfiguration, runtimeDirectory, selectConfiguration, type Profile, type UpstreamDeclarations, type Variant } from './config.js';
import { prepareRuntimeSources, verifyPreparedSources, type CompileInputs, type PreparedRuntime } from './prepare.js';

export type BuildContext = { checkoutRoot: string; toolingRoot: string; outputRoot: string; toolchainRoot: string };
export type BuildSelection = {
  profile: string;
  variant: string;
  kind: 'native' | 'wasm';
  jobs: number;
  fresh: boolean;
  includeToolchainNotices: boolean;
};
export type BuildResult = { buildRoot: string; provenance: string; stagedRoot?: string };
export type BuiltProfile = { buildRoot: string; runtimeRoot: string; generatedRoot: string; provenance: StableDiffusionProvenance };

const compileInputDirectories = [
  `${runtimeDirectory}/CMakeLists.txt`, `${runtimeDirectory}/native`, `${runtimeDirectory}/bindings`,
  `${runtimeDirectory}/config/variants.json`, `${runtimeDirectory}/upstream-patches`,
  'tools/abi/Bindings.cmake', 'tools/abi/ConfigureFingerprint.cmake', 'tools/abi/fingerprint.h.in',
  'tools/shared/TargetSources.cmake',
  'tests/runtimes/stable_diffusion_cpp', 'tests/runtimes/stable-diffusion-cpp/native',
  'tests/fixtures/dawn-native-api',
] as const;

/** Compiler inputs only: executable tooling/configuration remain in the checkout. */
export function sourceCaptureSelection(): { include: readonly string[]; exclude: readonly string[] } {
  return { include: compileInputDirectories, exclude: [] };
}

export async function prepareBuildInputs({ checkoutRoot, snapshotRoot, preparedRoot, mode, declarations }: { checkoutRoot: string; snapshotRoot: string; preparedRoot: string; mode: SourceMode; declarations: UpstreamDeclarations }): Promise<{ snapshot: SourceSnapshot; prepared: PreparedRuntime }> {
  const source = await observeSource({ checkoutRoot, mode });
  const snapshot = await snapshotCompileInputs({ source, destination: snapshotRoot, ...sourceCaptureSelection(), replace: true });
  const prepared = await prepareRuntimeSources({ snapshot, declarations, output: preparedRoot, replace: true });
  return { snapshot, prepared };
}

export async function buildRuntime({ context, selection, mode }: { context: BuildContext; selection: BuildSelection; mode: SourceMode }): Promise<BuildResult> {
  const configuration = await loadConfiguration({ checkoutRoot: context.checkoutRoot });
  const bindingsEntry = fileURLToPath(import.meta.resolve('#runtimes/stable-diffusion-cpp/tools/bindings'));
  if (path.resolve(bindingsEntry) !== path.resolve(context.toolingRoot, runtimeDirectory, 'tools/bindings.js')) {
    throw new Error('Image tooling root does not belong to the running checkout');
  }
  const selected = selectConfiguration({ configuration, profile: selection.profile, variant: selection.variant });
  if (!Number.isSafeInteger(selection.jobs) || selection.jobs < 1) {
    throw new Error('Jobs must be a positive integer');
  }
  const buildRoot = path.join(context.outputRoot, 'build', selection.kind, selection.profile, selection.variant);
  if (selection.fresh) {
    await rm(buildRoot, { recursive: true, force: true });
  }
  await directory({ path: buildRoot, create: true });
  await rm(path.join(buildRoot, 'provenance.json'), { force: true });
  const { snapshot, prepared } = await prepareBuildInputs({
    checkoutRoot: context.checkoutRoot,
    snapshotRoot: path.join(context.outputRoot, 'source-inputs', selection.kind, selection.profile, selection.variant),
    preparedRoot: path.join(buildRoot, 'prepared'),
    mode,
    declarations: configuration.upstreams,
  });
  const built = await buildPrepared({ context, selection, selected, snapshot, prepared, buildRoot });
  const after = await observeSource({ checkoutRoot: context.checkoutRoot, mode });
  if (after.head !== snapshot.source.head) {
    throw new Error('Checkout HEAD changed during image-runtime build');
  }
  const provenance = path.join(buildRoot, 'provenance.json');
  await writeFile(provenance, `${JSON.stringify(built.provenance, undefined, 2)}\n`);
  if (!snapshot.source.publishable || selection.kind === 'native') {
    return { buildRoot, provenance };
  }
  const stagedRoot = path.join(context.outputRoot, 'stage', selection.profile, selection.variant);
  const dawnRoot = path.join(context.toolchainRoot, 'emdawnwebgpu_pkg');
  const emscriptenRoot = path.join(context.toolchainRoot, 'emsdk/upstream/emscripten');
  await stageProfile({
    buildDirectory: stagedRoot,
    generatedDirectory: built.generatedRoot,
    runtimeFiles: ['core.mjs', 'core.wasm', 'core.d.ts', 'module-options.d.ts'].map((name) => ({ source: path.join(built.runtimeRoot, name), name })),
    provenance: built.provenance,
    noticeRoots: selection.includeToolchainNotices ? [emscriptenRoot, dawnRoot] : [],
    noticeDestination: selection.includeToolchainNotices ? path.join(context.outputRoot, 'toolchain-licenses') : undefined,
  });
  return { buildRoot, provenance, stagedRoot };
}

export function cmakeArguments({ context, inputs, buildRoot, preparedRoot, profile, variant, nativeCheck, dawnRoot }: { context: BuildContext; inputs: CompileInputs; buildRoot: string; preparedRoot: string; profile: Profile; variant: string; nativeCheck: boolean; dawnRoot: string }): string[] {
  const { memory64, webgpu, jspi, asyncify: _asyncify, pthreads: _pthreads, maximumMemory, ...unhandledProfile } = profile;
  unhandledProfile satisfies Record<PropertyKey, never>;
  return ['cmake', '-S', path.join(inputs.compileRoot, runtimeDirectory), '-B', buildRoot, '-G', 'Ninja',
    // Native ABI analysis consumes this target's actual Clang compile command.
    ...(nativeCheck ? ['-DCMAKE_C_COMPILER=clang', '-DCMAKE_CXX_COMPILER=clang++'] : []),
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', `-DSDCB_VARIANT=${variant}`,
    `-DSDCB_SOURCE=${inputs.upstreamRoots['stable-diffusion']}`, `-DSDCB_GGML_SOURCE=${inputs.upstreamRoots.ggml}`,
    `-DSDCB_PROVENANCE=${path.join(preparedRoot, 'patch-provenance.json')}`,
    `-DSDCB_CHECKOUT_ROOT=${context.checkoutRoot}`, `-DSDCB_COMPILE_ROOT=${inputs.compileRoot}`,
    `-DSDCB_TOOLING_ROOT=${context.toolingRoot}`, `-DSDCB_NODE=${process.execPath}`,
    `-DSDCB_NATIVE_CHECK=${nativeCheck ? 'ON' : 'OFF'}`, `-DSDCB_WEBGPU=${webgpu ? 'ON' : 'OFF'}`,
    `-DSDCB_JSPI=${jspi ? 'ON' : 'OFF'}`, `-DSDCB_MEMORY64=${memory64 ? 'ON' : 'OFF'}`,
    `-DSDCB_MAXIMUM_MEMORY=${String(maximumMemory)}`, `-DEMDAWNWEBGPU_DIR=${dawnRoot}`, `-DSDCB_DAWN_PACKAGE=${dawnRoot}`];
}

export async function resetCmakeSource({ buildRoot, compileRoot }: { buildRoot: string; compileRoot: string }): Promise<void> {
  let cache: string;
  try {
    cache = await readFile(path.join(buildRoot, 'CMakeCache.txt'), 'utf8');
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      return;
    }
    throw error;
  }
  const selected = /^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$/m.exec(cache)?.[1];
  if (selected === undefined || path.resolve(selected) === path.resolve(compileRoot, runtimeDirectory)) {
    return;
  }
  await rm(path.join(buildRoot, 'CMakeCache.txt'));
  await rm(path.join(buildRoot, 'CMakeFiles'), { recursive: true, force: true });
}

export async function buildPrepared({ context, selection, selected, snapshot, prepared, buildRoot }: { context: BuildContext; selection: BuildSelection; selected: { profile: Profile; variant: Variant }; snapshot: SourceSnapshot; prepared: PreparedRuntime; buildRoot: string }): Promise<BuiltProfile> {
  const toolchain = await loadToolchainConfiguration({ checkoutRoot: context.checkoutRoot });
  const dawnRoot = path.join(context.toolchainRoot, 'emdawnwebgpu_pkg');
  const emscriptenRoot = path.join(context.toolchainRoot, 'emsdk/upstream/emscripten');
  const nativeCheck = selection.kind === 'native';
  let emccVersion: string | undefined;
  if (!nativeCheck) {
    await regularFile({ path: path.join(dawnRoot, 'emdawnwebgpu.port.py') });
    emccVersion = (await runProcess({ command: path.join(emscriptenRoot, 'emcc'), args: ['--version'], cwd: buildRoot })).stdout.toString('utf8').split('\n')[0];
    if (emccVersion === undefined || !emccVersion.split(/[^\d.]+/u).includes(toolchain.emsdkVersion)) {
      throw new Error('Wrong Emscripten version');
    }
    if (selected.profile.asyncify) {
      await verifyAsyncifyBigIntPatch({ checkoutRoot: context.checkoutRoot, emscriptenRoot, patch: toolchain.emscriptenAsyncifyBigIntPatch });
    }
  }
  const command = cmakeArguments({ context, inputs: prepared.inputs, buildRoot, preparedRoot: prepared.root, profile: selected.profile, variant: selection.variant, nativeCheck, dawnRoot });
  await resetCmakeSource({ buildRoot, compileRoot: snapshot.compileRoot });
  await runProcess({ command: nativeCheck ? 'cmake' : path.join(emscriptenRoot, 'emcmake'), args: nativeCheck ? command.slice(1) : command, cwd: buildRoot, capture: 'tee' });
  await runProcess({ command: 'cmake', args: ['--build', buildRoot, '--parallel', String(selection.jobs), ...(nativeCheck ? [] : ['--target', 'core'])], cwd: buildRoot, capture: 'tee' });
  await verifyPreparedSources({ preparedRoot: prepared.root, facts: prepared.facts });
  if (!isDeepStrictEqual(snapshot.files, await fileIdentities({ root: snapshot.compileRoot }))) {
    throw new Error('Image-runtime compiler inputs changed during build');
  }
  const compiledInputs = await recordCompilerInputs({ buildRoot, roots: { compile: snapshot.compileRoot, prepared: prepared.root, build: buildRoot, tooling: context.toolingRoot, emscripten: emscriptenRoot } });
  const cacheIdentity = READ_ENV_BIC_CACHE_ID();
  const compilerWrapper = READ_ENV_EM_COMPILER_WRAPPER();
  const provenance: StableDiffusionProvenance = {
    runtime: 'stable-diffusion-cpp',
    profile: selection.profile,
    variant: selection.variant,
    sourceCommit: snapshot.source.head,
    sourceDirty: !snapshot.source.publishable,
    upstreams: prepared.facts.receipt.upstreams,
    configuration: selected.profile,
    variantConfiguration: selected.variant,
    cmakeCommand: command,
    patches: prepared.facts.receipt,
    sourceInputs: { mode: snapshot.source.mode, publishable: snapshot.source.publishable, sourceCommit: snapshot.source.head, files: snapshot.files },
    preparedSources: prepared.facts.identity,
    compiledInputs,
    nativeCheck,
    toolchain,
    ...(emccVersion === undefined ? {} : { emccVersion }),
    buildCache: { ...(cacheIdentity === undefined ? {} : { identity: cacheIdentity }), ...(compilerWrapper === undefined ? {} : { compilerWrapper }) },
    validation: { compiled: true, browserSmoke: false, realModelInference: false },
  };
  return { buildRoot, runtimeRoot: path.join(buildRoot, 'runtime'), generatedRoot: path.join(buildRoot, 'generated'), provenance };
}
