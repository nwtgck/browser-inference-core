import { readFile, open } from 'node:fs/promises';
import { join } from 'node:path';
import { decodeAbiCatalog, decodeAbiSchema, decodeAbiExtensionSidecar, decodeLicenseIndex, decodePackageMetadata, decodeStringArray, type RuntimeName, type FileEntry, type FileIdentity, type PackageMetadata, type LicenseIndex, type AbiCatalog, type AbiSchema, type AbiExtensionSidecar } from '#dto/shared';
import type { LlamaManifest, LlamaProvenance } from '#dto/llama-cpp';
import type { StableDiffusionManifest, StableDiffusionProvenance } from '#dto/stable-diffusion-cpp';
import { inventory } from '#tools/shared/contracts';
import { parseJson } from '#tools/shared/json';
import { regularFile, containedPath } from '#tools/shared/files';
import { readMetadata, packageFiles, assertSame, checkNpmPack, requireFiles, readMetadataBytes, bytesIdentity } from './files.js';
import { assertWasmExports } from './wasm-exports.js';
import { runtimeExports } from './layout.js';
import { renderExtensionTypes, renderExtensionDeclarations, renderModuleTypes } from '#tools/abi/generate';
import { abiSchemaFingerprint, assembleAbiCatalog, renderSchemaModule, renderFunctionTypes, renderFunctionModule } from '#tools/abi/schema';

export type RuntimeProfiles = Record<string, { variants: Record<string, LlamaProvenance | StableDiffusionProvenance> }>;
export type RuntimeManifest = LlamaManifest | StableDiffusionManifest;
export type RuntimePackageInput<M extends RuntimeManifest = RuntimeManifest> = {
  directory: string;
  manifest: M;
  manifestIdentity: FileIdentity;
  metadata: PackageMetadata;
  catalog: AbiCatalog;
  targets: Map<string, AbiSchema>;
  sidecars: Map<string, AbiExtensionSidecar>;
  exports: string[];
  licenseIndex: LicenseIndex;
};
export type RuntimeValidation<M extends RuntimeManifest = RuntimeManifest> = { input: RuntimePackageInput<M>; manifest: M; files: Map<string, FileEntry> };

export const apiFiles = ['schema.json', 'schema.mjs', 'functions.d.ts', 'functions.mjs', 'exports.json'] as const;
export const hostFiles = ['index.mjs', 'index.d.ts', 'bindings.mjs', 'read-only-file.mjs', 'README.md'] as const;
const forbiddenPackageFields = ['scripts', 'dependencies', 'devDependencies', 'optionalDependencies', 'workspaces', 'peerDependencies', 'peerDependenciesMeta', 'bundleDependencies', 'bundledDependencies'] as const;

export function validatePackageMetadata({ metadata, name, exports }: { metadata: PackageMetadata; name: string; exports: unknown }): void {
  if (metadata.name !== name || metadata.type !== 'module') {
    throw new Error('Wrong runtime package identity');
  }
  for (const field of forbiddenPackageFields) {
    if (field in metadata) {
      throw new Error(`Unexpected install-time input: ${field}`);
    }
  }
  assertSame({ actual: metadata.exports, expected: exports, message: 'Unexpected public package exports' });
}

/** Entry for a saved runtime artifact, including older or incomplete user inputs. */
export async function readRuntimeArtifact<M extends RuntimeManifest>({ directory, decodeManifest }: { directory: string; decodeManifest: ({ input }: { input: unknown }) => M }): Promise<RuntimePackageInput<M>> {
  const raw = await readMetadataBytes({ path: join(directory, 'manifest.json') });
  const manifest = parseJson({ text: raw.toString('utf8'), decode: decodeManifest });
  return readRuntimePackage({ directory, manifest, manifestIdentity: bytesIdentity({ data: raw }) });
}

/** Read an existing artifact once. Internal assembly carries these typed values directly. */
export async function readRuntimePackage<M extends RuntimeManifest>({ directory, manifest, manifestIdentity }: { directory: string; manifest: M; manifestIdentity: FileIdentity }): Promise<RuntimePackageInput<M>> {
  const metadata = await readMetadata({ path: join(directory, 'package.json'), decode: decodePackageMetadata });
  const catalog = await readMetadata({ path: join(directory, 'api/schema.json'), decode: decodeAbiCatalog });
  const targets = new Map<string, AbiSchema>();
  const sidecars = new Map<string, AbiExtensionSidecar>();
  for (const [profile, information] of Object.entries(manifest.profiles)) {
    for (const variant of Object.keys(information.variants)) {
      const key = `${profile}/${variant}`;
      const target = await readMetadata({ path: containedPath({ root: directory, relative: `profiles/${key}/abi.json` }), decode: decodeAbiSchema });
      targets.set(key, target);
      for (const extensionId of Object.keys(target.extensions ?? {})) {
        const prefix = `profiles/${key}/${extensionId === 'webgpu-q8-diagnostic-v1' ? 'q8-diagnostic-v1' : extensionId}`;
        sidecars.set(prefix, await readMetadata({ path: containedPath({ root: directory, relative: prefix + '.schema.json' }), decode: decodeAbiExtensionSidecar }));
      }
    }
  }
  const exports = await readMetadata({ path: join(directory, 'api/exports.json'), decode: decodeStringArray });
  const licenseIndex = await readMetadata({ path: join(directory, 'native-license-index.json'), decode: decodeLicenseIndex });
  return { directory, manifest, manifestIdentity, metadata, catalog, targets, sidecars, exports, licenseIndex };
}

export function hostProfiles({ runtime, profiles, variants }: { runtime: RuntimeName; profiles: readonly string[]; variants: readonly string[] }): string {
  const sortedProfiles = [...profiles].sort();
  const sortedVariants = [...variants].sort();
  if (!sortedProfiles.length || !sortedVariants.includes('browser') || new Set(profiles).size !== profiles.length || new Set(variants).size !== variants.length || [...profiles, ...variants].some(name => !/^[a-z0-9]+(?:-[a-z0-9]+)*$/u.test(name))) {
    throw new Error('Invalid package profile or variant selection');
  }
  let module: string;
  let coreModuleAlias: string | undefined;
  switch (runtime) {
  case 'llama-cpp': module = 'MainModule';
    coreModuleAlias = undefined;
    break;
  case 'stable-diffusion-cpp': module = 'StableDiffusionModule';
    coreModuleAlias = 'export type CoreModule = ModuleByProfile[Profile][Variant];';
    break;
  default: {
    const exhaustive: never = runtime;
    throw new Error(String(exhaustive));
  }
  }
  const lines = ["/** Generated from this package's profile/variant set. */"];
  for (const [i, profile] of sortedProfiles.entries()) {
    for (const [j, variant] of sortedVariants.entries()) {
      lines.push(`import type { ${module} as Module_${i}_${j} } from '../profiles/${profile}/${variant}/core.js';`);
    }
  }
  lines.push('export interface ModuleByProfile {');
  for (const [i, profile] of sortedProfiles.entries()) {
    lines.push(`  '${profile}': {`);
    for (const [j, variant] of sortedVariants.entries()) {
      lines.push(`    '${variant}': Module_${i}_${j};`);
    }
    lines.push('  };');
  }
  lines.push('}', 'export type Profile = keyof ModuleByProfile;', 'export type Variant = keyof ModuleByProfile[Profile];');
  if (coreModuleAlias !== undefined) {
    lines.push(coreModuleAlias);
  }
  return lines.join('\n') + '\n';
}

export async function validateRuntimePayload<M extends RuntimeManifest>({ input, runtime, requireClean }: { input: RuntimePackageInput<M>; runtime: RuntimeName; requireClean: boolean }): Promise<RuntimeValidation<M>> {
  const { directory, manifest, metadata } = input;
  validatePackageMetadata({ metadata, name: `${runtime}-browser-core`, exports: runtimeExports({ runtime, prefix: '.' }) });
  if (manifest.runtime !== runtime) {
    throw new Error('Manifest belongs to another runtime');
  }
  const expected = inventory({ entries: manifest.files });
  if (expected.has('manifest.json')) {
    throw new Error('Runtime inventory cannot include its own manifest');
  }
  const files = await packageFiles({ root: directory, exclude: [] });
  const payload = new Map(files);
  payload.delete('manifest.json');
  assertSame({ actual: payload, expected, message: 'Manifest does not exactly cover runtime payload bytes' });
  const ownIdentity = files.get('manifest.json');
  assertSame({ actual: ownIdentity && { bytes: ownIdentity.bytes, sha256: ownIdentity.sha256 }, expected: input.manifestIdentity, message: 'Runtime manifest changed after reading' });
  if (files.has('.gitmodules') || files.has('binding.gyp')) {
    throw new Error('Source/build input in runtime package');
  }
  await requireFiles({ files, names: ['package.json', 'LICENSE', 'README.md', 'native-license-index.json', 'api/schema.d.ts', ...apiFiles.map(name => 'api/' + name), ...[...hostFiles, 'profiles.d.ts', 'abi.mjs', 'abi.d.ts'].map(name => 'host/' + name)] });
  if (!Object.keys(manifest.profiles).length) {
    throw new Error('No runtime profiles');
  }
  const profiles: RuntimeProfiles = manifest.profiles;
  for (const [profile, info] of Object.entries(profiles)) {
    for (const [variant, provenance] of Object.entries(info.variants)) {
      if (provenance.profile !== profile || provenance.variant !== variant || provenance.sourceCommit !== manifest.sourceCommit) {
        throw new Error(`Profile/variant provenance or source mismatch: ${profile}/${variant}`);
      }
      if (!provenance.validation.compiled) {
        throw new Error('Cannot package an uncompiled runtime');
      }
      if (requireClean && provenance.sourceDirty) {
        throw new Error(`Cannot publish dirty source build ${profile}/${variant}: ${JSON.stringify({ sourceStatusBeforeBuild: provenance['sourceStatusBeforeBuild'], sourceStatusAfterBuild: provenance['sourceStatusAfterBuild'] })}`);
      }
      await requireFiles({ files, names: ['mjs', 'wasm', 'd.ts', 'json'].map(extension => extension === 'json' ? `profiles/${profile}/${variant}/abi.json` : `profiles/${profile}/${variant}/core.${extension}`) });
      switch (runtime) {
      case 'llama-cpp': break;
      case 'stable-diffusion-cpp':
        await requireFiles({ files, names: [`profiles/${profile}/${variant}/module-options.d.ts`] });
        break;
      default: {
        const exhaustive: never = runtime;
        throw new Error(String(exhaustive));
      }
      }
    }
  }
  return { input, manifest, files };
}

/** Decode only the Wasm export section. This does not instantiate or execute an artifact. */
export function wasmExports({ data }: { data: Uint8Array }): Set<string> {
  if (!Buffer.from(data.subarray(0, 8)).equals(Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]))) {
    throw new Error('Not a WebAssembly module');
  }
  let cursor = 8;
  function leb({ limit }: { limit: number }): number {
    let value = 0;
    let shift = 0;
    for (let i = 0; i < 5; i++) {
      const byte = data[cursor++];
      if (byte === undefined || cursor > limit) {
        throw new Error('Truncated WebAssembly integer');
      }
      value += (byte & 127) * 2 ** shift;
      if (!(byte & 128)) {
        if (value > 0xffffffff) {
          throw new Error('Oversized WebAssembly integer');
        }
        return value;
      }
      shift += 7;
    }
    throw new Error('Invalid WebAssembly integer');
  }
  const exports = new Set<string>();
  let found = false;
  while (cursor < data.length) {
    const section = data[cursor++];
    const size = leb({ limit: data.length });
    const end = cursor + size;
    if (end > data.length) {
      throw new Error('Truncated WebAssembly section');
    }
    if (section === 7) {
      if (found) {
        throw new Error('Duplicate WebAssembly export section');
      }
      found = true;
      const count = leb({ limit: end });
      for (let i = 0; i < count; i++) {
        const length = leb({ limit: end });
        const stop = cursor + length;
        if (stop > end) {
          throw new Error('Truncated WebAssembly export name');
        }
        const name = new TextDecoder('utf-8', { fatal: true }).decode(data.subarray(cursor, stop));
        cursor = stop;
        const kind = data[cursor++];
        if (kind === undefined || kind > 4) {
          throw new Error('Invalid WebAssembly export kind');
        }
        leb({ limit: end });
        if (exports.has(name)) {
          throw new Error('Duplicate WebAssembly export name');
        }
        if (kind === 0) {
          exports.add(name);
        }
      }
      if (cursor !== end) {
        throw new Error('Unexpected WebAssembly export bytes');
      }
    }
    cursor = end;
  }
  return exports;
}

export async function validateApi({ input, files }: { input: RuntimePackageInput; files: ReadonlyMap<string, FileEntry> }): Promise<AbiCatalog> {
  const { directory, manifest, catalog: schema } = input;
  const runtime = manifest.runtime;
  if (schema.runtime !== runtime || schema.schemaSha256 !== abiSchemaFingerprint({ schema })) {
    throw new Error('Invalid runtime ABI catalog fingerprint');
  }
  const entry = files.get('api/schema.json');
  if (!entry) {
    throw new Error('Missing runtime schema');
  }
  if (manifest['abiVersion'] !== 3 || manifest['schemaSha256'] !== schema.schemaSha256) {
    throw new Error('Schema file fingerprint differs from manifest');
  }
  if (await readFile(join(directory, 'api/schema.mjs'), 'utf8') !== renderSchemaModule({ schema })) {
    throw new Error('Schema module differs from catalog');
  }
  if (await readFile(join(directory, 'api/functions.d.ts'), 'utf8') !== renderFunctionTypes({ schema })) {
    throw new Error('Public TypeScript declarations differ from schema');
  }
  if (await readFile(join(directory, 'api/functions.mjs'), 'utf8') !== renderFunctionModule({ schema })) {
    throw new Error('Public JavaScript facade differs from schema');
  }
  let extensionTypes: string | undefined;
  switch (runtime) {
  case 'llama-cpp':
    extensionTypes = renderExtensionDeclarations({ declarations: schema.extensionDeclarations ?? [] });
    await requireFiles({ files, names: ['api/extensions.d.ts'] });
    if (await readFile(join(directory, 'api/extensions.d.ts'), 'utf8') !== extensionTypes) {
      throw new Error('Public extension types differ from compiler declarations');
    }
    break;
  case 'stable-diffusion-cpp': break;
  default: {
    const exhaustive: never = runtime;
    throw new Error(String(exhaustive));
  }
  }
  const targets: AbiSchema[] = [];
  for (const [profile, information] of Object.entries(manifest.profiles)) {
    for (const variant of Object.keys(information.variants)) {
      const target = input.targets.get(`${profile}/${variant}`);
      if (!target) {
        throw new Error('Missing compiled target ABI');
      }
      if (target.runtime !== runtime || abiSchemaFingerprint({ schema: target }) !== target.schemaSha256) {
        throw new Error('Invalid compiled target ABI fingerprint');
      }
      targets.push(target);
      if (extensionTypes !== undefined) {
        const expected = renderExtensionDeclarations({ declarations: target.extensionDeclarations ?? [] });
        const relative = `profiles/${profile}/${variant}/extensions.d.ts`;
        await requireFiles({ files, names: [relative] });
        if (expected !== extensionTypes || await readFile(join(directory, relative), 'utf8') !== expected) {
          throw new Error('Target extension types differ from public compiler declarations');
        }
      }
      for (const extensionId of ['model-graph-v1', 'webgpu-q8-diagnostic-v1']) {
        const prefix = `profiles/${profile}/${variant}/${extensionId === 'webgpu-q8-diagnostic-v1' ? 'q8-diagnostic-v1' : extensionId}`;
        const extension = target.extensions?.[extensionId];
        const present = files.has(prefix + '.schema.json') || files.has(prefix + '.functions.d.ts');
        if (!extension) {
          if (present) {
            throw new Error('Optional boundary sidecar has no compiled declaration');
          }
          continue;
        }
        if ([extension.versionExport, extension.schemaHashExport, ...extension.functions.map(fn => fn.export)].some(name => !target.exports.includes(name))) {
          throw new Error('Optional boundary declaration names an uncompiled export');
        }
        const declaration = target.extensionDeclarations?.find(value => value.extensionId === extensionId);
        assertSame({ actual: declaration, expected: { extensionId, functions: extension.functions }, message: 'Implemented extension differs from its public compiler declaration' });
        await requireFiles({ files, names: [prefix + '.schema.json', prefix + '.functions.d.ts'] });
        assertSame({ actual: input.sidecars.get(prefix), expected: { ...extension, schemaSha256: target.schemaSha256 }, message: 'Optional boundary sidecar differs from compiled declaration' });
        if (await readFile(join(directory, prefix + '.functions.d.ts'), 'utf8') !== renderExtensionTypes({ extension })) {
          throw new Error('Optional boundary types differ from compiled declaration');
        }
      }
      if (runtime === 'stable-diffusion-cpp' && await readFile(join(directory, 'profiles', profile, variant, 'core.d.ts'), 'utf8') !== renderModuleTypes({ schema: target })) {
        throw new Error('Native module TypeScript declarations differ from compiled ABI');
      }
      await validateProfileWasm({ directory, files, relative: `profiles/${profile}/${variant}/core.wasm`, exports: target.exports });
    }
  }
  assertSame({ actual: schema, expected: assembleAbiCatalog({ schemas: targets }), message: 'ABI catalog does not exactly cover compiled profile targets' });
  const exports = input.exports;
  const common = targets[0]?.exports.filter(name => targets.every(target => target.exports.includes(name))) ?? [];
  assertSame({ actual: exports, expected: common, message: 'Public exports differ from compiled target catalog' });
  return schema;
}

/** Inspect the export section while seeking over code/data sections. */
export async function readWasmExports({ path }: { path: string }): Promise<Set<string>> {
  await using file = await open(await regularFile({ path }), 'r');
  const size = (await file.stat()).size;
  let position = 0;
  async function take({ bytes }: { bytes: number }): Promise<Buffer> {
    if (position + bytes > size) {
      throw new Error('Truncated WebAssembly section');
    }
    const data = Buffer.alloc(bytes);
    let offset = 0;
    while (offset < bytes) {
      const read = await file.read(data, offset, bytes - offset, position);
      if (!read.bytesRead) {
        throw new Error('Truncated WebAssembly file');
      }
      offset += read.bytesRead;
      position += read.bytesRead;
    }
    return data;
  }
  async function leb({ limit }: { limit: number }): Promise<number> {
    let value = 0;
    let shift = 0;
    for (let i = 0; i < 5; i++) {
      if (position >= limit) {
        throw new Error('Truncated WebAssembly integer');
      }
      const byte = (await take({ bytes: 1 }))[0];
      if (byte === undefined) {
        throw new Error('Truncated WebAssembly integer');
      }
      value += (byte & 127) * 2 ** shift;
      if (!(byte & 128)) {
        if (value > 0xffffffff) {
          throw new Error('Oversized WebAssembly integer');
        }
        return value;
      }
      shift += 7;
    }
    throw new Error('Invalid WebAssembly integer');
  }
  if (!(await take({ bytes: 8 })).equals(Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]))) {
    throw new Error('Not a WebAssembly module');
  }
  const names = new Set<string>();
  const all = new Set<string>();
  let found = false;
  while (position < size) {
    const section = (await take({ bytes: 1 }))[0];
    const length = await leb({ limit: size });
    const end = position + length;
    if (end > size) {
      throw new Error('Truncated WebAssembly section');
    }
    if (section === 7) {
      if (found) {
        throw new Error('Duplicate WebAssembly export section');
      }
      found = true;
      const count = await leb({ limit: end });
      for (let i = 0; i < count; i++) {
        const bytes = await leb({ limit: end });
        if (position + bytes >= end) {
          throw new Error('Truncated WebAssembly export');
        }
        const name = new TextDecoder('utf-8', { fatal: true }).decode(await take({ bytes }));
        const kind = (await take({ bytes: 1 }))[0];
        if (kind === undefined || kind > 4 || all.has(name)) {
          throw new Error('Invalid or duplicate WebAssembly export');
        }
        all.add(name);
        await leb({ limit: end });
        if (kind === 0) {
          names.add(name);
        }
      }
      if (position !== end) {
        throw new Error('Unexpected WebAssembly export bytes');
      }
    }
    position = end;
  }
  return names;
}

export async function validateProfileWasm({ directory, files, relative, exports }: { directory: string; files: ReadonlyMap<string, FileEntry>; relative: string; exports: readonly string[] }): Promise<void> {
  const entry = files.get(relative);
  if (!entry) {
    throw new Error(`Missing Wasm: ${relative}`);
  }
  // Payload hashes were already checked once. Read for a distinct ABI inspection.
  const names = await readWasmExports({ path: join(directory, relative) });
  assertWasmExports({ source: await readFile(join(directory, relative.replace(/\.wasm$/u, '.mjs')), 'utf8'), wasmNames: names, expected: exports });
}

export async function finishRuntimeValidation({ directory, files, checkNpm }: { directory: string; files: ReadonlyMap<string, FileEntry>; checkNpm: boolean }): Promise<void> {
  if (checkNpm) {
    await checkNpmPack({ directory, files });
  }
}
