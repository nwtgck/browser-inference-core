import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { type RuntimeValidation } from '#tools/package/assemble-runtime';
import { assertSame, requireFiles } from '#tools/package/files';
import { fileIdentity } from '#tools/shared/files';
import { validateRuntimePayload, validateApi, finishRuntimeValidation, hostProfiles, readRuntimeArtifact, type RuntimePackageInput } from '#tools/package/runtime';
import { validateLicenseIndex } from '#tools/package/licenses';
import { loadConfiguration, runtimeDirectory, upstreamFacts } from './config.js';
import { type AbiSchema } from '#dto/shared';
import { decodeSpeechManifest, type SpeechManifest, type SpeechCapabilities } from '#dto/speech-cpp';
import { patches } from './recipe.js';

/** Public capability view of the implementation-owned compiled enum metadata. */
export function speechManifestCapabilities({ schema }: { schema: AbiSchema }): SpeechCapabilities {
  const capabilities = schema.capabilities;
  if (!capabilities) {
    throw new Error('Missing compiled speech capability metadata');
  }
  const ggufFileOffsetBits = capabilities['ggufFileOffsetBits'];
  if (ggufFileOffsetBits === undefined) {
    throw new Error('Missing speech file-offset capability metadata');
  }
  return {
    ggufFileOffsetBits,
    callerOwnedRandomAccess: capabilities['callerOwnedRandomAccess'] === 1,
    upstreamApi: capabilities['upstreamApi'] === 1,
  };
}

export async function verifyRuntimePackage({ sourceRoot, directory, requireClean, checkNpmPack }: { sourceRoot: string; directory: string; requireClean: boolean; checkNpmPack: boolean }): Promise<RuntimeValidation<SpeechManifest>> {
  const input = await readRuntimeArtifact({ directory, decodeManifest: decodeSpeechManifest });
  return verifyRuntimeInput({ sourceRoot, input, requireClean, checkNpmPack });
}

export async function verifyRuntimeInput({ sourceRoot, input, requireClean, checkNpmPack }: { sourceRoot: string; input: RuntimePackageInput<SpeechManifest>; requireClean: boolean; checkNpmPack: boolean }): Promise<RuntimeValidation<SpeechManifest>> {
  return verifyInput({ sourceRoot, input, requireClean, checkNpmPack, scope: 'publication' });
}

/** Actual selected assets for local type/ABI proof; never a publishable package. */
export async function verifyLocalRuntimeInput({ sourceRoot, input, checkNpmPack }: { sourceRoot: string; input: RuntimePackageInput<SpeechManifest>; checkNpmPack: boolean }): Promise<RuntimeValidation<SpeechManifest>> {
  for (const profile of Object.values(input.manifest.profiles)) {
    for (const provenance of Object.values(profile.variants)) {
      if (!provenance.sourceDirty || provenance.sourceInputs.mode !== 'local' || provenance.sourceInputs.publishable) {
        throw new Error('A local proof package requires nonpublishable local build records');
      }
    }
  }
  return verifyInput({ sourceRoot, input, requireClean: false, checkNpmPack, scope: 'local_proof' });
}

async function verifyInput({ sourceRoot, input, requireClean, checkNpmPack, scope }: { sourceRoot: string; input: RuntimePackageInput<SpeechManifest>; requireClean: boolean; checkNpmPack: boolean; scope: 'publication' | 'local_proof' }): Promise<RuntimeValidation<SpeechManifest>> {
  const directory = input.directory;
  const configuration = await loadConfiguration({ checkoutRoot: sourceRoot });
  const result = await validateRuntimePayload({ input, runtime: 'speech-cpp', requireClean });
  const { manifest, files } = result;
  if (manifest.runtime !== 'speech-cpp') {
    throw new Error('Missing speech runtime identity');
  }
  switch (scope) {
  case 'publication':
    assertSame({ actual: Object.keys(manifest.profiles).sort(), expected: Object.keys(configuration.profiles).sort(), message: 'Incomplete speech profiles' });
    break;
  case 'local_proof':
    if (Object.keys(manifest.profiles).some(name => !Object.hasOwn(configuration.profiles, name))) {
      throw new Error('Unknown local proof profile');
    }
    break;
  default: {
    const exhaustive: never = scope;
    throw new Error(String(exhaustive));
  }
  }
  const profileConfigurations = new Map(Object.entries(configuration.profiles));
  const variantConfigurations = new Map(Object.entries(configuration.variants));
  const expectedPatches = [];
  for (const patch of patches) {
    expectedPatches.push({ file: patch.file, target: patch.target, sha256: (await fileIdentity({ path: join(sourceRoot, runtimeDirectory, 'upstream-patches', patch.file) })).sha256 });
  }
  for (const [profile, info] of Object.entries(manifest.profiles)) {
    switch (scope) {
    case 'publication':
      assertSame({ actual: Object.keys(info.variants).sort(), expected: Object.keys(configuration.variants).sort(), message: 'Incomplete speech variants' });
      break;
    case 'local_proof':
      if (!Object.keys(info.variants).length || Object.keys(info.variants).some(name => !Object.hasOwn(configuration.variants, name))) {
        throw new Error('Missing or unknown local proof variants');
      }
      break;
    default: {
      const exhaustive: never = scope;
      throw new Error(String(exhaustive));
    }
    }
    for (const [variant, provenance] of Object.entries(info.variants)) {
      assertSame({ actual: provenance.configuration, expected: profileConfigurations.get(profile), message: 'Speech profile configuration mismatch' });
      assertSame({ actual: provenance.variantConfiguration, expected: variantConfigurations.get(variant), message: 'Speech variant configuration mismatch' });
      assertSame({ actual: upstreamFacts({ upstreams: provenance.upstreams }), expected: upstreamFacts({ upstreams: manifest.upstreams }), message: 'Mixed speech upstream identities' });
      const identities = provenance.patches.patches.map(({ file, target, sha256 }) => ({ file, target, sha256 }));
      assertSame({ actual: identities, expected: expectedPatches, message: 'Ordered speech patch provenance differs from selected source' });
    }
  }
  const catalog = await validateApi({ input, files });
  const capabilities = speechManifestCapabilities({ schema: catalog });
  const advertisedCapabilities = Object.fromEntries(Object.keys(capabilities).map(name => [name, manifest.capabilities[name]]));
  assertSame({ actual: advertisedCapabilities, expected: capabilities, message: 'Speech capabilities differ from compiled metadata' });
  for (const target of Object.values(catalog.targets)) {
    assertSame({ actual: target.capabilities, expected: catalog.capabilities, message: 'Mixed compiled speech capabilities' });
  }
  if (capabilities.ggufFileOffsetBits !== 64 || !capabilities.callerOwnedRandomAccess || !capabilities.upstreamApi) {
    throw new Error('Missing speech implementation capabilities');
  }
  for (const [profile, info] of Object.entries(manifest.profiles)) {
    for (const variant of Object.keys(info.variants)) {
      const target = input.targets.get(`${profile}/${variant}`);
      if (target?.pointerBytes !== (profileConfigurations.get(profile)?.memory64 ? 8 : 4)) {
        throw new Error('Speech profile address width differs from compiled ABI');
      }
    }
  }
  await requireFiles({ files, names: ['licenses/speech/LICENSE', 'licenses/ggml/LICENSE'] });
  for (const root of ['emscripten', 'emdawnwebgpu_pkg']) {
    if (![...files.keys()].some(name => name.startsWith(`licenses/toolchain/${root}/`))) {
      throw new Error(`Missing speech toolchain notices: ${root}`);
    }
  }
  const packagedVariants = Object.keys(Object.values(manifest.profiles)[0]?.variants ?? {});
  for (const profile of Object.values(manifest.profiles)) {
    assertSame({ actual: Object.keys(profile.variants).sort(), expected: [...packagedVariants].sort(), message: 'Profile variant sets differ' });
  }
  if (await readFile(join(directory, 'host/profiles.d.ts'), 'utf8') !== hostProfiles({ runtime: 'speech-cpp', profiles: Object.keys(manifest.profiles), variants: packagedVariants })) {
    throw new Error('Speech host profile declarations differ from packaged targets');
  }
  await validateLicenseIndex({ directory, runtime: 'speech-cpp', licenseRoots: [], metadata: input.metadata, index: input.licenseIndex });
  await finishRuntimeValidation({ directory, files, checkNpm: checkNpmPack });
  return result;
}
