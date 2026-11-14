import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { decodeAbiSchema, decodeAbiExtensionSidecar, type AbiSchema, type AbiCatalog, type AbiExtensionSidecar, type FileEntry, type PackageMetadata, type RuntimeName } from '#dto/shared';
import type { LlamaProvenance } from '#dto/llama-cpp';
import type { StableDiffusionProvenance } from '#dto/stable-diffusion-cpp';
import { assembleAbiCatalog, renderSchemaModule, renderFunctionTypes, renderFunctionModule } from '#tools/abi/schema';
import { renderExtensionDeclarations } from '#tools/abi/generate';
import { withOutputDirectories } from '#tools/shared/output';
import { containedPath } from '#tools/shared/files';
import { copyRegular, copyRegularTree, packageFiles, readMetadata, bytesIdentity } from './files.js';
import { hostFiles, hostProfiles, type RuntimeManifest, type RuntimePackageInput, type RuntimeValidation } from './runtime.js';
import { runtimeExports } from './layout.js';
import { writeLicenseIndex } from './licenses.js';

export type { RuntimeValidation } from './runtime.js';
type Provenance = LlamaProvenance | StableDiffusionProvenance;
export type PackageProfiles<P extends Provenance> = Record<string, { variants: Record<string, P> }>;

export async function assembleRuntimePackage<P extends Provenance, M extends RuntimeManifest>({ sourceRoot, runtime, buildRoot, destination, profiles, variants, extraFiles, licenseRoots, copyLicenses, decodeProvenance, createManifest, verify }: {
  sourceRoot: string; runtime: RuntimeName; buildRoot: string; destination: string; profiles: readonly string[]; variants: readonly string[];
  extraFiles: readonly { source: string; name: string }[]; licenseRoots: readonly string[];
  copyLicenses: ({ output }: { output: string }) => Promise<void>;
  decodeProvenance: ({ input }: { input: unknown }) => P;
  createManifest: ({ sourceCommit, first, profiles, files, catalog }: { sourceCommit: string; first: P; profiles: PackageProfiles<P>; files: FileEntry[]; catalog: AbiCatalog }) => M;
  verify: ({ input }: { input: RuntimePackageInput<M> }) => Promise<RuntimeValidation<M>>;
}): Promise<RuntimeValidation<M>> {
  hostProfiles({ runtime, profiles, variants });
  return withOutputDirectories({
    destinations: [destination],
    replace: true,
    prepare: async ({ paths }) => {
      const output = paths[0];
      if (!output) {
        throw new Error('Missing runtime package output');
      }
      const provenance: PackageProfiles<P> = {};
      const targets = new Map<string, AbiSchema>();
      const sidecars = new Map<string, AbiExtensionSidecar>();
      let first: P | undefined;
      for (const profile of profiles) {
        const info: { variants: Record<string, P> } = { variants: {} };
        provenance[profile] = info;
        for (const variant of variants) {
          const build = join(buildRoot, profile, variant);
          const data = await readMetadata({ path: join(build, 'provenance.json'), decode: decodeProvenance });
          if (data.profile !== profile || data.variant !== variant) {
            throw new Error('Build profile/variant mismatch');
          }
          if (first !== undefined && first.sourceCommit !== data.sourceCommit) {
            throw new Error('Mixed source revisions');
          }
          first ??= data;
          info.variants[variant] = data;
          const schema = await readMetadata({ path: join(build, 'generated/schema.json'), decode: decodeAbiSchema });
          const key = `${profile}/${variant}`;
          targets.set(key, schema);
          for (const extensionId of Object.keys(schema.extensions ?? {})) {
            const name = extensionId === 'webgpu-q8-diagnostic-v1' ? 'q8-diagnostic-v1' : extensionId;
            sidecars.set(`profiles/${key}/${name}`, await readMetadata({ path: containedPath({ root: join(build, 'runtime'), relative: name + '.schema.json' }), decode: decodeAbiExtensionSidecar }));
          }
          const target = join(output, 'profiles', profile, variant);
          await copyRegularTree({ source: join(build, 'runtime'), destination: target });
          await copyRegular({ source: join(build, 'generated/schema.json'), destination: join(target, 'abi.json') });
        }
      }
      if (!first) {
        throw new Error('Missing runtime source commit');
      }
      const catalog = assembleAbiCatalog({ schemas: [...targets.values()] });
      await mkdir(join(output, 'api'));
      await mkdir(join(output, 'host'));
      await writeFile(join(output, 'api/schema.json'), JSON.stringify(catalog, undefined, 2) + '\n');
      await writeFile(join(output, 'api/schema.mjs'), renderSchemaModule({ schema: catalog }));
      await copyRegular({ source: join(sourceRoot, 'runtimes/shared/api/schema.d.ts'), destination: join(output, 'api/schema.d.ts') });
      await writeFile(join(output, 'api/functions.d.ts'), renderFunctionTypes({ schema: catalog }));
      await writeFile(join(output, 'api/functions.mjs'), renderFunctionModule({ schema: catalog }));
      switch (runtime) {
      case 'llama-cpp': await writeFile(join(output, 'api/extensions.d.ts'), renderExtensionDeclarations({ declarations: catalog.extensionDeclarations ?? [] }));
        break;
      case 'stable-diffusion-cpp': break;
      default: {
        const exhaustive: never = runtime;
        throw new Error(String(exhaustive));
      }
      }
      const exports = [...targets.values()][0]?.exports.filter(name => [...targets.values()].every(schema => schema.exports.includes(name))) ?? [];
      await writeFile(join(output, 'api/exports.json'), JSON.stringify(exports, undefined, 2) + '\n');
      for (const name of hostFiles) {
        await copyRegular({ source: join(sourceRoot, 'runtimes', runtime, 'host', name), destination: join(output, 'host', name) });
      }
      for (const name of ['abi.mjs', 'abi.d.ts']) {
        await copyRegular({ source: join(sourceRoot, 'runtimes/shared/host', name), destination: join(output, 'host', name) });
      }
      await writeFile(join(output, 'host/profiles.d.ts'), hostProfiles({ runtime, profiles, variants }));
      for (const file of extraFiles) {
        await copyRegular({ source: file.source, destination: join(output, file.name) });
      }
      await copyLicenses({ output });
      const guideFiles = extraFiles.map(file => file.name).filter(name => !['LICENSE', 'README.md'].includes(name));
      const metadata: PackageMetadata & { private: boolean; files: string[] } = { name: `${runtime}-browser-core`, version: '0.1.0', private: true, type: 'module', license: 'MIT', files: ['host/', 'profiles/', 'api/', 'manifest.json', 'native-license-index.json', 'licenses/', 'README.md', ...guideFiles, 'LICENSE'], exports: runtimeExports({ runtime, prefix: '.' }) };
      await writeFile(join(output, 'package.json'), JSON.stringify(metadata, undefined, 2) + '\n');
      const licenseIndex = await writeLicenseIndex({ directory: output, runtime, licenseRoots, metadata });
      const manifest = createManifest({ sourceCommit: first.sourceCommit, first, catalog, profiles: provenance, files: [...(await packageFiles({ root: output, exclude: [] })).values()] });
      const raw = Buffer.from(JSON.stringify(manifest, undefined, 2) + '\n');
      await writeFile(join(output, 'manifest.json'), raw);
      const validated = await verify({ input: { directory: output, manifest, manifestIdentity: bytesIdentity({ data: raw }), metadata, catalog, targets, sidecars, exports, licenseIndex } });
      return { ...validated, input: { ...validated.input, directory: destination } };
    },
  });
}
