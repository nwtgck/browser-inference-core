import { basename, join } from 'node:path';
import { assembleRuntimePackage } from '#tools/package/assemble-runtime';
import { collectNotices, embeddedNotices } from '#tools/package/licenses';
import { copyRegular } from '#tools/package/files';
import { loadConfiguration, runtimePath } from './config.js';
import { decodeLlamaProvenance, type LlamaManifest } from '#dto/llama-cpp';
import { verifyRuntimeInput, verifyLocalRuntimeInput } from './verify.js';

export async function buildPackage({ sourceRoot, buildRoot, destination, profiles, licenseRoots, checkNpmPack }: { sourceRoot: string; buildRoot: string; destination: string; profiles: readonly string[] | undefined; licenseRoots: readonly string[]; checkNpmPack: boolean }) {
  return buildSelectedPackage({ sourceRoot, buildRoot, destination, profiles, variants: undefined, licenseRoots, checkNpmPack, scope: 'publication' });
}

export async function buildLocalPackage({ sourceRoot, buildRoot, destination, profile, variant, licenseRoots }: { sourceRoot: string; buildRoot: string; destination: string; profile: string; variant: string; licenseRoots: readonly string[] }) {
  return buildSelectedPackage({ sourceRoot, buildRoot, destination, profiles: [profile], variants: [variant], licenseRoots, checkNpmPack: false, scope: 'local_proof' });
}

async function buildSelectedPackage({ sourceRoot, buildRoot, destination, profiles, variants, licenseRoots, checkNpmPack, scope }: { sourceRoot: string; buildRoot: string; destination: string; profiles: readonly string[] | undefined; variants: readonly string[] | undefined; licenseRoots: readonly string[]; checkNpmPack: boolean; scope: 'publication' | 'local_proof' }) {
  const configuration = await loadConfiguration({ checkoutRoot: sourceRoot });
  const root = join(sourceRoot, runtimePath);
  return assembleRuntimePackage({
    sourceRoot,
    runtime: 'llama-cpp',
    buildRoot,
    destination,
    profiles: profiles ?? Object.keys(configuration.profiles),
    variants: variants ?? Object.keys(configuration.variants),
    extraFiles: [{ source: join(root, 'LICENSE'), name: 'LICENSE' }, { source: join(root, 'packaging/README.md'), name: 'README.md' }, ...['chat-and-multimodal.md', 'browser-host.md'].map(name => ({ source: join(root, 'docs', name), name }))],
    licenseRoots: licenseRoots.map(value => basename(value)),
    decodeProvenance: decodeLlamaProvenance,
    createManifest: ({ sourceCommit, first, profiles, files, catalog }): LlamaManifest => ({ formatVersion: 2, runtime: 'llama-cpp', abiVersion: 3, sourceCommit, llamaCommit: first.llamaCommit, licenseRoots: licenseRoots.map(value => basename(value)), schemaSha256: catalog.schemaSha256, profiles, files }),
    copyLicenses: async ({ output }) => {
      let count = 0;
      for (const [index, source] of licenseRoots.entries()) {
        count += await collectNotices({ source, destination: join(output, 'licenses', String(index)), required: true });
      }
      if (!count) {
        throw new Error('No third-party notices collected');
      }
      for (const name of Object.keys(embeddedNotices['llama-cpp'])) {
        await copyRegular({ source: join(root, 'vendor/llama.cpp', name), destination: join(output, 'licenses/embedded', name + '.txt') });
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
