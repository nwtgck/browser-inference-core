import { basename, join } from 'node:path';
import { assembleRuntimePackage } from '#tools/package/assemble-runtime';
import { collectNotices } from '#tools/package/licenses';
import { copyRegular } from '#tools/package/files';
import { loadConfiguration, runtimeDirectory } from './config.js';
import { decodeSpeechProvenance, type SpeechManifest } from '#dto/speech-cpp';
import { verifyRuntimeInput, verifyLocalRuntimeInput, speechManifestCapabilities } from './verify.js';

export async function buildPackage({ sourceRoot, buildRoot, destination, profiles, licenseRoots, checkNpmPack }: { sourceRoot: string; buildRoot: string; destination: string; profiles: readonly string[] | undefined; licenseRoots: readonly string[]; checkNpmPack: boolean }) {
  return buildSelectedPackage({ sourceRoot, buildRoot, destination, profiles, variants: undefined, licenseRoots, checkNpmPack, scope: 'publication' });
}

export async function buildLocalPackage({ sourceRoot, buildRoot, destination, profile, variant, licenseRoots }: { sourceRoot: string; buildRoot: string; destination: string; profile: string; variant: string; licenseRoots: readonly string[] }) {
  return buildSelectedPackage({ sourceRoot, buildRoot, destination, profiles: [profile], variants: [variant], licenseRoots, checkNpmPack: false, scope: 'local_proof' });
}

async function buildSelectedPackage({ sourceRoot, buildRoot, destination, profiles, variants, licenseRoots, checkNpmPack, scope }: { sourceRoot: string; buildRoot: string; destination: string; profiles: readonly string[] | undefined; variants: readonly string[] | undefined; licenseRoots: readonly string[]; checkNpmPack: boolean; scope: 'publication' | 'local_proof' }) {
  const configuration = await loadConfiguration({ checkoutRoot: sourceRoot });
  const root = join(sourceRoot, runtimeDirectory);
  if (scope === 'publication' && profiles !== undefined && profiles.slice().sort().join(',') !== Object.keys(configuration.profiles).sort().join(',')) {
    throw new Error('Speech package requires every configured profile');
  }
  if (licenseRoots.map(value => basename(value)).sort().join(',') !== 'emdawnwebgpu_pkg,emscripten') {
    throw new Error('Provide exactly Emscripten and Dawn notice roots');
  }
  return assembleRuntimePackage({
    sourceRoot,
    runtime: 'speech-cpp',
    buildRoot,
    destination,
    profiles: profiles ?? Object.keys(configuration.profiles),
    variants: variants ?? Object.keys(configuration.variants),
    extraFiles: [{ source: join(root, 'LICENSE'), name: 'LICENSE' }, { source: join(root, 'README.md'), name: 'README.md' }],
    licenseRoots: [],
    decodeProvenance: decodeSpeechProvenance,
    createManifest: ({ sourceCommit, first, profiles, files, catalog }): SpeechManifest => ({ formatVersion: 2, runtime: 'speech-cpp', abiVersion: 3, sourceCommit, upstreams: first.upstreams, experimental: true, capabilities: speechManifestCapabilities({ schema: catalog }), schemaSha256: catalog.schemaSha256, profiles, files }),
    copyLicenses: async ({ output }) => {
      const speech = join(root, 'vendor/speech.cpp');
      const ggml = join(root, 'vendor/ggml');
      // speech-static uses the owned library sources covered by this license.
      // Its CLI dependencies and nested ggml checkout are outside this build.
      await copyRegular({ source: join(speech, 'LICENSE'), destination: join(output, 'licenses/speech/LICENSE') });
      await collectNotices({ source: ggml, destination: join(output, 'licenses/ggml'), required: true });
      for (const source of licenseRoots) {
        await collectNotices({ source, destination: join(output, 'licenses/toolchain', basename(source)), required: true });
      }
    },
    verify: async ({ input }) => {
      switch (scope) {
      case 'publication': return verifyRuntimeInput({ sourceRoot, input, requireClean: false, checkNpmPack });
      case 'local_proof': return verifyLocalRuntimeInput({ sourceRoot, input, checkNpmPack });
      default: {
        const exhaustive: never = scope;
        throw new Error(String(exhaustive));
      }
      }
    },
  });
}
