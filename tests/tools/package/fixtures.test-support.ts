import { mkdtemp, mkdir, writeFile, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { artifactDirectories } from '#tools/shared/contracts';
import { decodeAbiSchema, type RuntimeName, type AbiSchema, type AbiExtension, type PatchReceipt, type FileEntry } from '#dto/shared';
import { decodeLlamaManifest, decodeLlamaProvenance, type LlamaProvenance, type LlamaManifest } from '#dto/llama-cpp';
import { decodeStableDiffusionManifest, decodeStableDiffusionProvenance, type StableDiffusionProvenance, type StableDiffusionManifest, type Upstreams } from '#dto/stable-diffusion-cpp';
import type { RuntimeManifest } from '#tools/package/runtime';
import { fileIdentity } from '#tools/shared/files';
import { abiSchemaFingerprint, assembleAbiCatalog, renderFunctionTypes, renderFunctionModule, renderSchemaModule } from '#tools/abi/schema';
import { renderModuleTypes, renderExtensionTypes } from '#tools/abi/generate';
import { assembleRuntimePackage } from '#tools/package/assemble-runtime';
import { embeddedNotices } from '#tools/package/licenses';
import { verifyRuntimeInput as verifyLlama } from '#runtimes/llama-cpp/tools/verify';
import { verifyRuntimeInput as verifyImage, imageManifestCapabilities } from '#runtimes/stable-diffusion-cpp/tools/verify';
import { loadConfiguration as llamaConfiguration } from '#runtimes/llama-cpp/tools/config';
import { loadConfiguration as imageConfiguration } from '#runtimes/stable-diffusion-cpp/tools/config';
import { patches } from '#runtimes/stable-diffusion-cpp/tools/recipe';
import { assemblePackage } from '#tools/package/aggregate';
import { packageFiles } from '#tools/package/files';

export const sourceRoot = resolve('.');
export const fixtureSource = 'a'.repeat(40);

export async function write({ path, data }: { path: string; data: string | Uint8Array }): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, data);
}

function leb({ value }: { value: number }): number[] {
  const result = [];
  do {
    let byte = value & 127;
    value >>>= 7;
    if (value) {
      byte |= 128;
    }
    result.push(byte);
  } while (value);
  return result;
}

/** A valid tiny Wasm function, used only as a packaging fixture. */
export function fixtureWasm({ names }: { names: readonly string[] }): Buffer {
  const functions = [...leb({ value: names.length }), ...names.map(() => 0)];
  const exports = [...leb({ value: names.length }), ...names.flatMap((name, index) => {
    const bytes = Buffer.from(name);
    return [...leb({ value: bytes.length }), ...bytes, 0, ...leb({ value: index })];
  })];
  const code = [...leb({ value: names.length }), ...names.flatMap(() => [4, 0, 65, 1, 11])];
  return Buffer.from([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 127, 3, ...leb({ value: functions.length }), ...functions, 7, ...leb({ value: exports.length }), ...exports, 10, ...leb({ value: code.length }), ...code]);
}

export function fixtureAbi({ runtime, memory64, gpuProfile }: { runtime: RuntimeName; memory64: boolean; gpuProfile: boolean }): AbiSchema {
  const name = runtime === 'llama-cpp' ? 'fixture_version' : 'sd_browser_bf16_retention_version';
  const schema: AbiSchema = { abiVersion: 3, runtime, pointerRepresentation: 'bigint_native', pointerBytes: memory64 ? 8 : 4, schemaSha256: '0'.repeat(64), functions: [{ name, export: '_' + name, returnType: 'uint32_t', returnKind: 'unsigned', abiType: { kind: 'integer', bits: 32, signed: false }, parameters: [], bridge: false }], records: [], callbacks: [], constants: runtime === 'llama-cpp' ? [] : ['SD_TYPE_BF16'], constantValues: runtime === 'llama-cpp' ? {} : { SD_TYPE_BF16: '30' }, excluded: [], exports: ['_' + name], suspendingExports: [], callbackMetadata: { version: 2, getters: [] }, ...(runtime === 'stable-diffusion-cpp' ? { capabilities: { ggufFileOffsetBits: 64, safetensorsFileOffsetBits: 64, callerOwnedRandomAccess: 1, upstreamApi: 1, ggufShards: 1, ggufOriginalShapes: 1, bf16WeightRetention: 1, modelIoMask: 7 } } : {}) };
  if (runtime === 'llama-cpp') {
    const extensions: Record<string, AbiExtension> = {};
    for (const extensionId of ['model-graph-v1', ...(gpuProfile ? ['webgpu-q8-diagnostic-v1'] : [])]) {
      extensions[extensionId] = { extensionId, version: 1, abiVersion: 1, baseAbiVersion: 3, pointerRepresentation: 'bigint_u64', suspension: 'none', versionExport: '_' + name, schemaHashExport: '_' + name, functions: schema.functions, records: [], constants: [], constantValues: {} };
      if (extensionId === 'webgpu-q8-diagnostic-v1') {
        const run = { ...schema.functions[0]!, name: 'fixture_run_v1', export: '_fixture_run_v1' };
        extensions[extensionId]!.functions = [...schema.functions, run];
        schema.exports.push(run.export);
      }
    }
    schema.extensions = extensions;
  }
  schema.schemaSha256 = abiSchemaFingerprint({ schema });
  return schema;
}

export async function makeRuntime({ root, runtime, selection }: { root: string; runtime: RuntimeName; selection: 'minimal' | 'configured' }): Promise<void> {
  const configuration = await (runtime === 'llama-cpp' ? llamaConfiguration : imageConfiguration)({ checkoutRoot: sourceRoot });
  let profiles: string[];
  switch (selection) {
  case 'minimal': profiles = runtime === 'llama-cpp' ? ['cpu-wasm32'] : Object.keys(configuration.profiles);
    break;
  case 'configured': profiles = Object.keys(configuration.profiles);
    break;
  default: {
    const exhaustive: never = selection;
    throw new Error(String(exhaustive));
  }
  }
  const buildRoot = join(root, 'build-inputs', runtime);
  const upstreams: Upstreams = { 'stable-diffusion': { path: 'vendor/stable-diffusion.cpp', repository: 'https://example.invalid/sd', commit: 'b'.repeat(40) }, ggml: { path: 'vendor/ggml-webgpu-source', repository: 'https://example.invalid/ggml', sourceSubdirectory: 'ggml', commit: 'c'.repeat(40) } };
  const patchReceipts: PatchReceipt[] = [];
  for (const patch of patches) {
    patchReceipts.push({ file: patch.file, target: patch.target, mode: 'apply', reason: patch.reason, files: [], sha256: (await fileIdentity({ path: join(sourceRoot, 'runtimes/stable-diffusion-cpp/upstream-patches', patch.file) })).sha256 });
  }
  for (const profile of profiles) {
    for (const variant of Object.keys(configuration.variants)) {
      const config = configuration.profiles[profile];
      if (!config) {
        throw new Error('Missing fixture configuration');
      }
      const build = join(buildRoot, profile, variant);
      const schema = fixtureAbi({ runtime, memory64: config.memory64, gpuProfile: 'gpuProfile' in config && config.gpuProfile === true });
      await write({ path: join(build, 'generated/schema.json'), data: JSON.stringify(schema, undefined, 2) + '\n' });
      await write({ path: join(build, 'generated/schema.mjs'), data: renderSchemaModule({ schema }) });
      await write({ path: join(build, 'generated/functions.d.ts'), data: renderFunctionTypes({ schema }) });
      await write({ path: join(build, 'generated/functions.mjs'), data: renderFunctionModule({ schema }) });
      await write({ path: join(build, 'generated/exports.json'), data: JSON.stringify(schema.exports) });
      await write({
        path: join(build, 'runtime/core.mjs'),
        data: `\
/** Tiny packaging fixture; no model inference implementation. */
export default async function fixture({ wasmBinary } = {}) {
  if (!wasmBinary) throw Error('Packaging fixture requires its explicit tiny Wasm');
  const { instance } = await WebAssembly.instantiate(wasmBinary, {});
  const module = {};
${schema.exports.map(name => `  module[${JSON.stringify(name)}] = instance.exports[${JSON.stringify(name.slice(1))}];`).join('\n')}
  return module;
}
`,
      });
      await write({ path: join(build, 'runtime/core.d.ts'), data: runtime === 'stable-diffusion-cpp' ? renderModuleTypes({ schema }) : `export interface MainModule { HEAPU8: Uint8Array; FS: object; addFunction<Args extends (number | bigint)[]>(callback: (...args: Args) => number | bigint | void, signature: string): number | bigint; removeFunction(pointer: number | bigint): void; _fixture_version(): number; }\nexport default function(options?: object): Promise<MainModule>;\n` });
      if (runtime === 'stable-diffusion-cpp') {
        await write({ path: join(build, 'runtime/module-options.d.ts'), data: await readFile(join(sourceRoot, 'runtimes/stable-diffusion-cpp/bindings/module-options.d.ts')) });
      }
      await write({ path: join(build, 'runtime/core.wasm'), data: fixtureWasm({ names: schema.exports.map(name => name.slice(1)) }) });
      for (const extension of Object.values(schema.extensions ?? {})) {
        const name = extension.extensionId === 'webgpu-q8-diagnostic-v1' ? 'q8-diagnostic-v1' : extension.extensionId;
        await write({ path: join(build, 'runtime', name + '.schema.json'), data: JSON.stringify({ ...extension, schemaSha256: schema.schemaSha256 }) });
        await write({ path: join(build, 'runtime', name + '.functions.d.ts'), data: renderExtensionTypes({ extension }) });
      }
      const variantConfiguration = configuration.variants[variant];
      if (!variantConfiguration) {
        throw new Error('Missing fixture variant');
      }
      const common = { profile, variant, sourceCommit: fixtureSource, sourceDirty: false, configuration: config, variantConfiguration, sourceInputs: { mode: 'release' as const, publishable: true, sourceCommit: fixtureSource, files: {} }, compiledInputs: { translationUnits: {}, targetSources: {} }, toolchain: {}, cmakeCommand: [], buildCache: {}, validation: { compiled: true, browserSmoke: false, realModelInference: false } };
      const record: LlamaProvenance | StableDiffusionProvenance = runtime === 'llama-cpp'
        ? { ...common, runtime, llamaCommit: 'b'.repeat(40), emccVersion: 'fixture', nodeVersion: 'fixture', upstreamBuildInfo: { commit: 'fixture', number: 0 }, visionBf16Storage: { version: 1, modes: ['legacy_expand', 'retain_checked'], default: 'legacy_expand' }, preparedSources: { formatVersion: 1, options: { target: 'wasm', webgpu: config.webgpu, gpuProfile: 'gpuProfile' in config && config.gpuProfile === true, fileRows: config.memory64, qsa: true, mtmdMode: config.webgpu ? 'webgpu' : 'cpu' }, upstream: { files: 0, sha256: '0'.repeat(64) }, inputs: {}, runtimeInputs: {}, patches: [], selections: [], shaders: {}, outputs: {} } }
        : { ...common, runtime, upstreams, patches: { runtime, upstreams, patches: patchReceipts, vendorCheckoutModified: false }, preparedSources: { files: 0, sha256: '0'.repeat(64) }, nativeCheck: false };
      await write({ path: join(build, 'provenance.json'), data: JSON.stringify(record) });
    }
  }
  const options = {
    sourceRoot,
    runtime,
    buildRoot,
    destination: join(root, 'package-inputs', runtime),
    profiles,
    variants: Object.keys(configuration.variants),
    extraFiles: [{ source: join(sourceRoot, 'LICENSE'), name: 'LICENSE' }, { source: join(sourceRoot, 'tools/package/runtime-readme.md'), name: 'README.md' }],
    licenseRoots: runtime === 'llama-cpp' ? ['llama.cpp'] : [],
    copyLicenses: async ({ output }: { output: string }) => {
      const names = runtime === 'llama-cpp' ? ['licenses/0/LICENSE'] : ['licenses/stable-diffusion/LICENSE', 'licenses/ggml/LICENSE', 'licenses/toolchain/emscripten/LICENSE', 'licenses/toolchain/emdawnwebgpu_pkg/LICENSE'];
      for (const name of names) {
        await write({ path: join(output, name), data: 'Copyright packaging fixture authors; synthetic notice.\n' });
      }
      for (const name of Object.keys(embeddedNotices[runtime])) {
        await write({ path: join(output, 'licenses/embedded', name + '.txt'), data: '/* Copyright all fixture contributors; synthetic notice. */\nint ignored_fixture_body;\n' });
      }
    },
  };
  if (runtime === 'llama-cpp') {
    await assembleRuntimePackage({ ...options, decodeProvenance: decodeLlamaProvenance, createManifest: ({ sourceCommit, first, profiles, files, catalog }): LlamaManifest => ({ runtime, formatVersion: 2, abiVersion: 3, sourceCommit, llamaCommit: first.llamaCommit, schemaSha256: catalog.schemaSha256, licenseRoots: ['llama.cpp'], profiles, files }), verify: async ({ input }) => verifyLlama({ sourceRoot, input, requireClean: true, checkNpmPack: false }) });
  } else {
    await assembleRuntimePackage({ ...options, decodeProvenance: decodeStableDiffusionProvenance, createManifest: ({ sourceCommit, first, profiles, files, catalog }): StableDiffusionManifest => ({ runtime, formatVersion: 2, abiVersion: 3, sourceCommit, upstreams: first.upstreams, experimental: true, capabilities: imageManifestCapabilities({ schema: catalog }), schemaSha256: catalog.schemaSha256, profiles, files }), verify: async ({ input }) => verifyImage({ sourceRoot, input, requireClean: true, checkNpmPack: false }) });
  }
}

export async function packageFixture() {
  const root = await mkdtemp(join(tmpdir(), 'bicore-package-'));
  for (const runtime of ['llama-cpp', 'stable-diffusion-cpp'] as const) {
    await makeRuntime({ root, runtime, selection: 'minimal' });
  }
  const directory = join(root, 'package');
  const validated = await assemblePackage({ sourceRoot, inputs: join(root, 'package-inputs'), destination: directory, checkNpmPack: false });
  return {
    root,
    directory,
    validated,
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
}

export async function rehashRuntime({ directory, runtime, change }: { directory: string; runtime: RuntimeName; change: (manifest: RuntimeManifest) => void }): Promise<void> {
  const root = join(directory, artifactDirectories[runtime]);
  const raw: unknown = JSON.parse(await readFile(join(root, 'manifest.json'), 'utf8'));
  const manifest: RuntimeManifest = runtime === 'llama-cpp' ? decodeLlamaManifest({ input: raw }) : decodeStableDiffusionManifest({ input: raw });
  manifest.files = [...(await packageFiles({ root, exclude: ['manifest.json'] })).values()];
  change(manifest);
  await writeFile(join(root, 'manifest.json'), JSON.stringify(manifest));
  const parent = JSON.parse(await readFile(join(directory, 'manifest.json'), 'utf8')) as { files: { path: string; bytes: number; sha256: string }[] };
  const entry = parent.files.find(file => file.path === `${artifactDirectories[runtime]}/manifest.json`);
  if (!entry) {
    throw new Error('Missing fixture root binding');
  }
  Object.assign(entry, await fileIdentity({ path: join(root, 'manifest.json') }));
  await writeFile(join(directory, 'manifest.json'), JSON.stringify(parent));
}

/** Deliberately rebuild every byte binding so negative tests reach semantic guards. */
export async function rewriteLlamaTargetMetadata({ directory, transform }: { directory: string; transform: ({ profile, variant, schema }: { profile: string; variant: string; schema: AbiSchema }) => void }): Promise<void> {
  const root = join(directory, artifactDirectories['llama-cpp']);
  const manifest = decodeLlamaManifest({ input: JSON.parse(await readFile(join(root, 'manifest.json'), 'utf8')) });
  const schemas: AbiSchema[] = [];
  for (const [profile, information] of Object.entries(manifest.profiles)) {
    for (const variant of Object.keys(information.variants)) {
      const targetRoot = join(root, 'profiles', profile, variant);
      const schema = decodeAbiSchema({ input: JSON.parse(await readFile(join(targetRoot, 'abi.json'), 'utf8')) });
      transform({ profile, variant, schema });
      schema.schemaSha256 = abiSchemaFingerprint({ schema });
      schemas.push(schema);
      await writeFile(join(targetRoot, 'abi.json'), JSON.stringify(schema));
      for (const id of ['model-graph-v1', 'webgpu-q8-diagnostic-v1']) {
        const name = id === 'webgpu-q8-diagnostic-v1' ? 'q8-diagnostic-v1' : id;
        const extension = schema.extensions?.[id];
        for (const suffix of ['.schema.json', '.functions.d.ts']) {
          await rm(join(targetRoot, name + suffix), { force: true });
        }
        if (extension) {
          await writeFile(join(targetRoot, name + '.schema.json'), JSON.stringify({ ...extension, schemaSha256: schema.schemaSha256 }));
          await writeFile(join(targetRoot, name + '.functions.d.ts'), renderExtensionTypes({ extension }));
        }
      }
    }
  }
  const catalog = assembleAbiCatalog({ schemas });
  await writeFile(join(root, 'api/schema.json'), JSON.stringify(catalog));
  await writeFile(join(root, 'api/schema.mjs'), renderSchemaModule({ schema: catalog }));
  await rehashRuntime({
    directory,
    runtime: 'llama-cpp',
    change: manifest => {
      manifest.schemaSha256 = catalog.schemaSha256;
    },
  });
}
