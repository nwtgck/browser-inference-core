import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { decodeLlamaManifest, type LlamaManifest } from '#dto/llama-cpp';
import { assertSame } from '#tools/package/files';
import { requireFiles } from '#tools/package/files';
import { validateRuntimePayload, validateApi, finishRuntimeValidation, hostProfiles, readRuntimeArtifact, type RuntimePackageInput } from '#tools/package/runtime';
import { validateLicenseIndex, embeddedNotices } from '#tools/package/licenses';
import { loadConfiguration, visionBf16StorageMetadata } from './config.js';
import { type RuntimeValidation } from '#tools/package/assemble-runtime';

export async function verifyRuntimePackage({ sourceRoot, directory, requireClean, checkNpmPack }: { sourceRoot: string; directory: string; requireClean: boolean; checkNpmPack: boolean }): Promise<RuntimeValidation<LlamaManifest>> {
  const input = await readRuntimeArtifact({ directory, decodeManifest: decodeLlamaManifest });
  return verifyRuntimeInput({ sourceRoot, input, requireClean, checkNpmPack });
}

export async function verifyRuntimeInput({ sourceRoot, input, requireClean, checkNpmPack }: { sourceRoot: string; input: RuntimePackageInput<LlamaManifest>; requireClean: boolean; checkNpmPack: boolean }): Promise<RuntimeValidation<LlamaManifest>> {
  return verifyInput({ sourceRoot, input, requireClean, checkNpmPack, scope: 'publication' });
}

/** Actual selected assets for local type/ABI proof; never a publishable package. */
export async function verifyLocalRuntimeInput({ sourceRoot, input, checkNpmPack }: { sourceRoot: string; input: RuntimePackageInput<LlamaManifest>; checkNpmPack: boolean }): Promise<RuntimeValidation<LlamaManifest>> {
  for (const profile of Object.values(input.manifest.profiles)) {
    for (const provenance of Object.values(profile.variants)) {
      if (!provenance.sourceDirty || provenance.sourceInputs.mode !== 'local' || provenance.sourceInputs.publishable) {
        throw new Error('A local proof package requires nonpublishable local build records');
      }
    }
  }
  return verifyInput({ sourceRoot, input, requireClean: false, checkNpmPack, scope: 'local_proof' });
}

async function verifyInput({ sourceRoot, input, requireClean, checkNpmPack, scope }: { sourceRoot: string; input: RuntimePackageInput<LlamaManifest>; requireClean: boolean; checkNpmPack: boolean; scope: 'publication' | 'local_proof' }): Promise<RuntimeValidation<LlamaManifest>> {
  const directory = input.directory;
  const configuration = await loadConfiguration({ checkoutRoot: sourceRoot });
  const result = await validateRuntimePayload({ input, runtime: 'llama-cpp', requireClean });
  const { manifest, files } = result;
  const profileConfigurations = new Map(Object.entries(configuration.profiles));
  const variantConfigurations = new Map(Object.entries(configuration.variants));
  const llamaCommit = manifest.llamaCommit;
  const roots = manifest.licenseRoots;
  await requireFiles({ files, names: Object.keys(embeddedNotices['llama-cpp']).map(name => `licenses/embedded/${name}.txt`) });
  for (const [profile, info] of Object.entries(manifest.profiles)) {
    const expected = profileConfigurations.get(profile);
    if (!expected) {
      throw new Error(`Unknown llama profile: ${profile}`);
    }
    switch (scope) {
    case 'publication':
      assertSame({ actual: Object.keys(info.variants).sort(), expected: Object.keys(configuration.variants).sort(), message: 'Incomplete llama variants' });
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
      assertSame({ actual: provenance.variantConfiguration, expected: variantConfigurations.get(variant), message: 'Variant configuration mismatch' });
      if (provenance.configuration !== undefined) {
        assertSame({ actual: provenance.configuration, expected, message: 'Profile configuration mismatch' });
      }
      if (provenance['llamaCommit'] !== llamaCommit) {
        throw new Error('Mixed upstream commits');
      }
      if (provenance['visionBf16Storage'] !== undefined) {
        const advertisement = provenance.visionBf16Storage;
        assertSame({ actual: { version: advertisement.version, modes: advertisement.modes, default: advertisement.default }, expected: visionBf16StorageMetadata(), message: 'Unknown vision BF16 capability' });
      }
      const command = provenance.cmakeCommand;
      if (provenance['experimentalVisionBf16'] || command?.some(option => /^-DLCB_WEBGPU_BF16_STORAGE[=:]/u.test(option))) {
        throw new Error('Obsolete build-time vision BF16 selection');
      }
      const preparation = provenance.browserModulePreparation;
      if (preparation) {
        const entry = files.get(`profiles/${profile}/${variant}/core.mjs`);
        if (variant !== 'browser' || !entry) {
          throw new Error('Browser module preparation on an unsupported variant');
        }
        assertSame({ actual: { bytes: preparation.output.bytes, sha256: preparation.output.sha256 }, expected: { bytes: entry.bytes, sha256: entry.sha256 }, message: 'Browser module preparation output mismatch' });
      }
      const declarationPreparation = provenance.moduleDeclarationPreparation;
      if (declarationPreparation) {
        const entry = files.get(`profiles/${profile}/${variant}/core.d.ts`);
        if (!entry) {
          throw new Error('Missing prepared module declarations');
        }
        assertSame({ actual: { bytes: declarationPreparation.output.bytes, sha256: declarationPreparation.output.sha256 }, expected: { bytes: entry.bytes, sha256: entry.sha256 }, message: 'Module declaration preparation output mismatch' });
      }
    }
  }
  const catalog = await validateApi({ input, files });
  for (const [profile, info] of Object.entries(manifest.profiles)) {
    for (const variant of Object.keys(info.variants)) {
      const target = input.targets.get(`${profile}/${variant}`);
      if (target?.pointerBytes !== (profileConfigurations.get(profile)?.memory64 ? 8 : 4)) {
        throw new Error('Profile address width differs from compiled ABI');
      }
      if (!target.extensions?.['model-graph-v1']) {
        throw new Error('Missing compiled model graph boundary');
      }
      const expectedProfile = profileConfigurations.get(profile);
      const q8Expected = expectedProfile !== undefined && expectedProfile.webgpu && 'gpuProfile' in expectedProfile && expectedProfile.gpuProfile === true;
      if (Boolean(target.extensions['webgpu-q8-diagnostic-v1']) !== q8Expected) {
        throw new Error('Q8 diagnostic boundary differs from configured profile');
      }
    }
  }
  const packagedVariants = Object.keys(Object.values(manifest.profiles)[0]?.variants ?? {});
  for (const profile of Object.values(manifest.profiles)) {
    assertSame({ actual: Object.keys(profile.variants).sort(), expected: [...packagedVariants].sort(), message: 'Profile variant sets differ' });
  }
  if (await readFile(join(directory, 'host/profiles.d.ts'), 'utf8') !== hostProfiles({ runtime: 'llama-cpp', profiles: Object.keys(manifest.profiles), variants: packagedVariants })) {
    throw new Error('Host profile declarations differ from packaged targets');
  }
  await validateLicenseIndex({ directory, runtime: 'llama-cpp', licenseRoots: roots, metadata: input.metadata, index: input.licenseIndex });
  await finishRuntimeValidation({ directory, files, checkNpm: checkNpmPack });
  return result;
}
