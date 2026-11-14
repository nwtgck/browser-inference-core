import { basename, join } from 'node:path';
import { assembleRuntimePackage } from '#tools/package/assemble-runtime';
import { collectNotices, embeddedNotices } from '#tools/package/licenses';
import { copyRegular } from '#tools/package/files';
import { loadConfiguration, runtimeDirectory } from './config.js';
import { decodeStableDiffusionProvenance, type StableDiffusionManifest } from '#dto/stable-diffusion-cpp';
import { verifyRuntimeInput, verifyLocalRuntimeInput, imageManifestCapabilities } from './verify.js';

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
    throw new Error('Image package requires every configured profile');
  }
  if (licenseRoots.map(value => basename(value)).sort().join(',') !== 'emdawnwebgpu_pkg,emscripten') {
    throw new Error('Provide exactly Emscripten and Dawn notice roots');
  }
  return assembleRuntimePackage({
    sourceRoot,
    runtime: 'stable-diffusion-cpp',
    buildRoot,
    destination,
    profiles: profiles ?? Object.keys(configuration.profiles),
    variants: variants ?? Object.keys(configuration.variants),
    extraFiles: [{ source: join(root, 'LICENSE'), name: 'LICENSE' }, { source: join(root, 'README.md'), name: 'README.md' }],
    licenseRoots: [],
    decodeProvenance: decodeStableDiffusionProvenance,
    createManifest: ({ sourceCommit, first, profiles, files, catalog }): StableDiffusionManifest => ({ formatVersion: 2, runtime: 'stable-diffusion-cpp', abiVersion: 3, sourceCommit, upstreams: first.upstreams, experimental: true, capabilities: imageManifestCapabilities({ schema: catalog }), schemaSha256: catalog.schemaSha256, profiles, files }),
    copyLicenses: async ({ output }) => {
      const sd = join(root, 'vendor/stable-diffusion.cpp');
      const ggml = join(root, 'vendor/ggml-webgpu-source');
      await collectNotices({ source: sd, destination: join(output, 'licenses/stable-diffusion'), required: true });
      await copyRegular({ source: join(ggml, 'LICENSE'), destination: join(output, 'licenses/ggml/LICENSE') });
      await collectNotices({ source: join(ggml, 'ggml'), destination: join(output, 'licenses/ggml/ggml'), required: false });
      for (const name of Object.keys(embeddedNotices['stable-diffusion-cpp'])) {
        await copyRegular({ source: join(sd, 'thirdparty', name), destination: join(output, 'licenses/embedded', name + '.txt') });
      }
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
