import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { decodeRootManifest, decodePackageMetadata, type RootManifest, type FileIdentity, type FileEntry, type PackageMetadata } from '#dto/shared';
import { decodeLlamaManifest, type LlamaManifest } from '#dto/llama-cpp';
import { decodeStableDiffusionManifest, type StableDiffusionManifest } from '#dto/stable-diffusion-cpp';
import { decodeSpeechManifest, type SpeechManifest } from '#dto/speech-cpp';
import { verifyRuntimePackage as readSpeech, verifyRuntimeInput as verifySpeech } from '#runtimes/speech-cpp/tools/verify';
import { artifactDirectories, runtimeNames, inventory, packageName } from '#tools/shared/contracts';
import { fileIdentity } from '#tools/shared/files';
import { withOutputDirectories } from '#tools/shared/output';
import { parseJson } from '#tools/shared/json';
import { packageExports } from './layout.js';
import { assertSame, copyRegular, copyBoundTree, readBoundFile, readMetadataBytes, bytesIdentity, listPackagePaths, checkNpmPack } from './files.js';
import { readRuntimePackage, validatePackageMetadata, type RuntimeValidation } from './runtime.js';
import { verifyRuntimePackage as readLlama, verifyRuntimeInput as verifyLlama } from '#runtimes/llama-cpp/tools/verify';
import { verifyRuntimePackage as readImage, verifyRuntimeInput as verifyImage } from '#runtimes/stable-diffusion-cpp/tools/verify';
import { artifactIntegrityPath, readArtifactIntegrity, type ValidatedIntegrity } from './integrity.js';

const rootFiles = ['LICENSE', 'README.md', 'package.json'] as const;
export type RuntimeManifests = { 'llama-cpp': LlamaManifest; 'stable-diffusion-cpp': StableDiffusionManifest; 'speech-cpp': SpeechManifest };
export type ValidatedPackage = {
  manifest: RootManifest; manifestIdentity: FileIdentity; metadata: PackageMetadata; runtimes: RuntimeManifests;
  integrity: ValidatedIntegrity | undefined;
  /** Transient closure for Git/npm boundaries. Never another serialized inventory. */
  files: Map<string, FileEntry>;
};
type RuntimeValidations = { 'llama-cpp': RuntimeValidation<LlamaManifest>; 'stable-diffusion-cpp': RuntimeValidation<StableDiffusionManifest>; 'speech-cpp': RuntimeValidation<SpeechManifest> };

async function validateAggregateRoot({ directory, manifest, manifestBytes, manifestIdentity, metadata, runtimes, checkNpm }: { directory: string; manifest: RootManifest; manifestBytes: Buffer; manifestIdentity: FileIdentity; metadata: PackageMetadata; runtimes: RuntimeValidations; checkNpm: boolean }): Promise<ValidatedPackage> {
  validatePackageMetadata({ metadata, name: packageName, exports: packageExports() });
  const expected = inventory({ entries: manifest.files });
  const owned = [...rootFiles, ...runtimeNames.map(runtime => `${artifactDirectories[runtime]}/manifest.json`)].sort();
  assertSame({ actual: [...expected.keys()].sort(), expected: owned, message: 'Root inventory must own only root files and runtime manifests' });
  const closure = new Map(expected);
  for (const runtime of runtimeNames) {
    const binding = manifest.runtimes[runtime];
    const relative = artifactDirectories[runtime];
    if (binding.manifest !== relative + '/manifest.json') {
      throw new Error('Wrong runtime manifest binding');
    }
    const inner = runtimes[runtime];
    if (inner.manifest.sourceCommit !== manifest.sourceCommit) {
      throw new Error('Mixed source commits');
    }
    const entry = expected.get(binding.manifest);
    if (!entry) {
      throw new Error('Missing runtime manifest binding');
    }
    assertSame({ actual: { bytes: entry.bytes, sha256: entry.sha256 }, expected: inner.input.manifestIdentity, message: 'Root runtime binding differs from validated manifest' });
    for (const file of inner.manifest.files) {
      const path = relative + '/' + file.path;
      if (closure.has(path)) {
        throw new Error(`Duplicate artifact closure path: ${path}`);
      }
      closure.set(path, { path, bytes: file.bytes, sha256: file.sha256 });
    }
  }
  for (const entry of expected.values()) {
    await readBoundFile({ root: directory, entry });
  }
  closure.set('manifest.json', { path: 'manifest.json', ...manifestIdentity });
  assertSame({ actual: await fileIdentity({ path: join(directory, 'manifest.json') }), expected: manifestIdentity, message: 'Aggregate manifest changed after reading' });
  // The enclosing, attested YAML cannot appear in its own embedded JSON inventory.
  // Permit only this one separately decoded and byte-bound metadata file.
  const integrity = await readArtifactIntegrity({ directory, manifest: manifestBytes, sourceCommit: manifest.sourceCommit });
  if (integrity !== undefined) {
    closure.set(artifactIntegrityPath, { path: artifactIntegrityPath, ...integrity.identity });
  }
  assertSame({ actual: [...await listPackagePaths({ root: directory, exclude: [] })].sort(), expected: [...closure.keys()].sort(), message: 'Manifest closure does not cover the complete package tree' });
  if (checkNpm) {
    await checkNpmPack({ directory, files: closure });
  }
  return { manifest, manifestIdentity, metadata, integrity, runtimes: { 'llama-cpp': runtimes['llama-cpp'].manifest, 'stable-diffusion-cpp': runtimes['stable-diffusion-cpp'].manifest, 'speech-cpp': runtimes['speech-cpp'].manifest }, files: closure };
}

/** External artifact entry: decode each manifest once, then share complete typed values. */
export async function validatePackage({ sourceRoot, directory, requireClean, checkNpmPack: checkNpm }: { sourceRoot: string; directory: string; requireClean: boolean; checkNpmPack: boolean }): Promise<ValidatedPackage> {
  const raw = await readMetadataBytes({ path: join(directory, 'manifest.json') });
  const manifest = parseJson({ text: raw.toString('utf8'), decode: decodeRootManifest });
  const expected = inventory({ entries: manifest.files });
  async function readOwned({ relative }: { relative: string }): Promise<Buffer> {
    const entry = expected.get(relative);
    if (!entry) {
      throw new Error(`Missing root artifact: ${relative}`);
    }
    return readBoundFile({ root: directory, entry });
  }
  const metadata = parseJson({ text: (await readOwned({ relative: 'package.json' })).toString('utf8'), decode: decodePackageMetadata });
  const llamaPath = `${artifactDirectories['llama-cpp']}/manifest.json`;
  const imagePath = `${artifactDirectories['stable-diffusion-cpp']}/manifest.json`;
  const speechPath = `${artifactDirectories['speech-cpp']}/manifest.json`;
  // Fixed layout is checked before using any binding as a filesystem path.
  if (manifest.runtimes['llama-cpp'].manifest !== llamaPath || manifest.runtimes['stable-diffusion-cpp'].manifest !== imagePath || manifest.runtimes['speech-cpp'].manifest !== speechPath) {
    throw new Error('Wrong runtime manifest binding');
  }
  const llamaRaw = await readOwned({ relative: llamaPath });
  const imageRaw = await readOwned({ relative: imagePath });
  const speechRaw = await readOwned({ relative: speechPath });
  const llamaInput = await readRuntimePackage({ directory: join(directory, artifactDirectories['llama-cpp']), manifest: parseJson({ text: llamaRaw.toString('utf8'), decode: decodeLlamaManifest }), manifestIdentity: bytesIdentity({ data: llamaRaw }) });
  const imageInput = await readRuntimePackage({ directory: join(directory, artifactDirectories['stable-diffusion-cpp']), manifest: parseJson({ text: imageRaw.toString('utf8'), decode: decodeStableDiffusionManifest }), manifestIdentity: bytesIdentity({ data: imageRaw }) });
  const speechInput = await readRuntimePackage({ directory: join(directory, artifactDirectories['speech-cpp']), manifest: parseJson({ text: speechRaw.toString('utf8'), decode: decodeSpeechManifest }), manifestIdentity: bytesIdentity({ data: speechRaw }) });
  const llama = await verifyLlama({ sourceRoot, input: llamaInput, requireClean, checkNpmPack: checkNpm });
  const image = await verifyImage({ sourceRoot, input: imageInput, requireClean, checkNpmPack: checkNpm });
  const speech = await verifySpeech({ sourceRoot, input: speechInput, requireClean, checkNpmPack: checkNpm });
  return validateAggregateRoot({ directory, manifest, manifestBytes: raw, manifestIdentity: bytesIdentity({ data: raw }), metadata, runtimes: { 'llama-cpp': llama, 'stable-diffusion-cpp': image, 'speech-cpp': speech }, checkNpm });
}

export async function assemblePackage({ sourceRoot, inputs, destination, checkNpmPack: checkNpm }: { sourceRoot: string; inputs: string; destination: string; checkNpmPack: boolean }): Promise<ValidatedPackage> {
  const llama = await readLlama({ sourceRoot, directory: join(inputs, 'llama-cpp'), requireClean: true, checkNpmPack: checkNpm });
  const image = await readImage({ sourceRoot, directory: join(inputs, 'stable-diffusion-cpp'), requireClean: true, checkNpmPack: checkNpm });
  const speech = await readSpeech({ sourceRoot, directory: join(inputs, 'speech-cpp'), requireClean: true, checkNpmPack: checkNpm });
  return assembleValidatedPackage({ sourceRoot, runtimes: { 'llama-cpp': llama, 'stable-diffusion-cpp': image, 'speech-cpp': speech }, destination, checkNpm });
}

/** Assemble actual selected local builds. Normal publication rejects these source records. */
export async function assembleLocalPackage({ sourceRoot, runtimes, destination }: { sourceRoot: string; runtimes: RuntimeValidations; destination: string }): Promise<ValidatedPackage> {
  for (const runtime of runtimeNames) {
    for (const profile of Object.values(runtimes[runtime].manifest.profiles)) {
      const variants: Record<string, { sourceDirty: boolean; sourceInputs: { mode: string; publishable: boolean } }> = profile.variants;
      for (const provenance of Object.values(variants)) {
        if (!provenance.sourceDirty || provenance.sourceInputs.mode !== 'local' || provenance.sourceInputs.publishable) {
          throw new Error('A local aggregate requires nonpublishable local build records');
        }
      }
    }
  }
  return assembleValidatedPackage({ sourceRoot, runtimes, destination, checkNpm: false });
}

async function assembleValidatedPackage({ sourceRoot, runtimes, destination, checkNpm }: { sourceRoot: string; runtimes: RuntimeValidations; destination: string; checkNpm: boolean }): Promise<ValidatedPackage> {
  return withOutputDirectories({
    destinations: [destination],
    replace: true,
    prepare: async ({ paths }) => {
      const output = paths[0];
      if (!output) {
        throw new Error('Missing aggregate output');
      }
      const llama = runtimes['llama-cpp'];
      const image = runtimes['stable-diffusion-cpp'];
      const speech = runtimes['speech-cpp'];
      if (llama.manifest.sourceCommit !== image.manifest.sourceCommit || llama.manifest.sourceCommit !== speech.manifest.sourceCommit) {
        throw new Error('Cannot mix source revisions');
      }
      for (const runtime of runtimeNames) {
        await copyBoundTree({ source: runtimes[runtime].input.directory, destination: join(output, artifactDirectories[runtime]), files: runtimes[runtime].files });
      }
      await copyRegular({ source: join(sourceRoot, 'LICENSE'), destination: join(output, 'LICENSE') });
      await copyRegular({ source: join(sourceRoot, 'tools/package/runtime-readme.md'), destination: join(output, 'README.md') });
      const metadata: PackageMetadata & { private: boolean; files: string[] } = { name: packageName, version: '0.1.0', private: true, type: 'module', license: 'MIT', files: ['runtimes/', 'manifest.json', artifactIntegrityPath, 'README.md', 'LICENSE'], exports: packageExports() };
      await writeFile(join(output, 'package.json'), JSON.stringify(metadata, undefined, 2) + '\n');
      const entries: FileEntry[] = [];
      for (const path of [...rootFiles, ...runtimeNames.map(runtime => artifactDirectories[runtime] + '/manifest.json')].sort()) {
        entries.push({ path, ...await fileIdentity({ path: join(output, path) }) });
      }
      const manifest: RootManifest = { formatVersion: 6, sourceCommit: llama.manifest.sourceCommit, runtimes: { 'llama-cpp': { manifest: `${artifactDirectories['llama-cpp']}/manifest.json`, manifestFormatVersion: 2 }, 'stable-diffusion-cpp': { manifest: `${artifactDirectories['stable-diffusion-cpp']}/manifest.json`, manifestFormatVersion: 2 }, 'speech-cpp': { manifest: `${artifactDirectories['speech-cpp']}/manifest.json`, manifestFormatVersion: 2 } }, files: entries };
      const raw = Buffer.from(JSON.stringify(manifest, undefined, 2) + '\n');
      await writeFile(join(output, 'manifest.json'), raw);
      return validateAggregateRoot({ directory: output, manifest, manifestBytes: raw, manifestIdentity: bytesIdentity({ data: raw }), metadata, runtimes: { 'llama-cpp': llama, 'stable-diffusion-cpp': image, 'speech-cpp': speech }, checkNpm });
    },
  });
}
