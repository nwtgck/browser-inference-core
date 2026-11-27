import { link, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { decodeFrozenPublication, type SourceFacts, type ReportData, type FrozenPublication, type JsonObject, type Validation } from '#dto/shared';
import { artifactDirectories, runtimeNames } from '#tools/shared/contracts';
import { isContained, optionalStat, directory } from '#tools/shared/files';
import { withOutputDirectories } from '#tools/shared/output';
import { temporaryDirectory } from '#tools/shared/temporary-directory';
import { assembleValidatedPackage, readRuntimePackages, validatePackage, type ValidatedPackage, type RuntimeManifests } from '#tools/package/aggregate';
import { assertSame, bytesIdentity, copyBoundTree, readMetadata, readBoundFile, checkNpmPack } from '#tools/package/files';
import { artifactIntegrityPath, encodeArtifactIntegrity, checkTreeFingerprint } from '#tools/package/integrity';
import { checkPublicExports } from '#tools/package/public-exports';
import { canonicalJson } from '#tools/abi/schema';
import { gitBlob, gitObjectId, localGit, verifyGitTree } from './git.js';
import { unmeasuredPublication, type MeasurePublication } from './metrics.js';

export type VerifySource = ({ sourceCommit, manifests }: { sourceCommit: string; manifests: RuntimeManifests }) => Promise<SourceFacts>;
export type ConfirmedPublication = { commit: string; frozen: FrozenPublication };

export function checkFrozen({ frozen }: { frozen: FrozenPublication }): void {
  if (frozen.schemaVersion !== 3) {
    throw new Error('Unsupported private frozen state');
  }
  if (frozen.branch !== 'artifacts' && !frozen.branch.startsWith('artifacts/')) {
    throw new Error('Refusing to publish over a source branch');
  }
  if (frozen.sourceCommit !== frozen.report.data.sourceCommit || frozen.sourceCommit !== frozen.report.data.source.sourceCommit) {
    throw new Error('Frozen report source mismatch');
  }
  checkTreeFingerprint({ fingerprint: frozen.report.data.source.treeFingerprint });
  if (bytesIdentity({ data: Buffer.from(canonicalJson({ value: frozen.report.data })) }).sha256 !== frozen.report.sha256) {
    throw new Error('Frozen report data changed');
  }
}

export async function createReportData({ validated, verifySource }: { validated: ValidatedPackage; verifySource: VerifySource }): Promise<ReportData> {
  const facts = await verifySource({ sourceCommit: validated.manifest.sourceCommit, manifests: validated.runtimes });
  if (facts.sourceCommit !== validated.manifest.sourceCommit) {
    throw new Error('Verified source differs from package source');
  }
  function profiles<V extends Validation>({ values }: { values: Record<string, { variants: Record<string, { configuration: JsonObject; variantConfiguration: JsonObject; validation: V }> }> }): Record<string, { variants: Record<string, { configuration: JsonObject; variantConfiguration: JsonObject; validation: V }> }> {
    return Object.fromEntries(Object.entries(values).map(([name, profile]) => [name, { variants: Object.fromEntries(Object.entries(profile.variants).map(([variant, value]) => [variant, { configuration: value.configuration, variantConfiguration: value.variantConfiguration, validation: value.validation }])) }]));
  }
  const llama = validated.runtimes['llama-cpp'];
  const image = validated.runtimes['stable-diffusion-cpp'];
  const speech = validated.runtimes['speech-cpp'];
  return {
    sourceCommit: validated.manifest.sourceCommit,
    package: { name: 'browser-inference-core', version: validated.metadata.version },
    source: facts,
    runtimes: {
      'llama-cpp': { manifest: artifactDirectories['llama-cpp'] + '/manifest.json', abiVersion: llama.abiVersion, schemaSha256: llama.schemaSha256, upstreams: { llama: llama.llamaCommit }, profiles: profiles({ values: llama.profiles }) },
      'stable-diffusion-cpp': { manifest: artifactDirectories['stable-diffusion-cpp'] + '/manifest.json', abiVersion: image.abiVersion, schemaSha256: image.schemaSha256, upstreams: image.upstreams, profiles: profiles({ values: image.profiles }) },
      'speech-cpp': { manifest: artifactDirectories['speech-cpp'] + '/manifest.json', abiVersion: speech.abiVersion, schemaSha256: speech.schemaSha256, upstreams: speech.upstreams, profiles: profiles({ values: speech.profiles }) },
    },
  };
}

async function publicationPaths({ sourceRoot, input, work, receipt, branch }: { sourceRoot: string; input: string; work: string; receipt: string; branch: string }) {
  if (branch !== 'artifacts' && !branch.startsWith('artifacts/')) {
    throw new Error('Refusing to publish over a source branch');
  }
  await localGit({ cwd: sourceRoot, args: ['check-ref-format', '--branch', branch] });
  const snapshot = resolve(work);
  const receiptPath = resolve(receipt);
  if (isContained({ root: input, path: snapshot }) || isContained({ root: snapshot, path: input }) || isContained({ root: input, path: receiptPath }) || isContained({ root: receiptPath, path: input }) || isContained({ root: snapshot, path: receiptPath }) || isContained({ root: receiptPath, path: snapshot })) {
    throw new Error('Publication state must be outside the runtime package');
  }
  if (await optionalStat({ path: snapshot }) || await optionalStat({ path: receiptPath })) {
    throw new Error('Publication snapshot and receipt must be fresh');
  }
  await directory({ path: dirname(snapshot), create: true });
  await directory({ path: dirname(receiptPath), create: true });
  return { snapshot, receiptPath };
}

/** Local entry for an independently saved aggregate. Its files cross a copy boundary. */
export async function freezePackage({ sourceRoot, packageDirectory, work, receipt, branch, expectedSource, repository, verifySource, measure }: { sourceRoot: string; packageDirectory: string; work: string; receipt: string; branch: string; expectedSource: string; repository: string; verifySource: VerifySource; measure?: MeasurePublication }): Promise<FrozenPublication> {
  const observe = measure ?? unmeasuredPublication;
  const input = resolve(packageDirectory);
  const { snapshot, receiptPath } = await publicationPaths({ sourceRoot, input, work, receipt, branch });
  const validated = await observe({ phase: 'publication.freeze.validate-inputs', operation: () => validatePackage({ sourceRoot, directory: input, requireClean: true, checkNpmPack: false }) });
  if (validated.integrity !== undefined) {
    throw new Error('Freeze requires a newly assembled package without a previous integrity envelope');
  }
  if (validated.manifest.sourceCommit !== gitObjectId({ value: expectedSource })) {
    throw new Error('Package differs from selected source');
  }
  return installFrozenSnapshot({
    snapshot,
    receiptPath,
    prepare: async ({ staging }) => {
      await observe({ phase: 'publication.freeze.copy', operation: () => copyBoundTree({ source: input, destination: staging, files: validated.files }) });
      return freezeSnapshot({ snapshot: staging, validated, branch, repository, verifySource, measure: observe });
    },
  });
}

/** CI entry for downloaded runtime packages. Assemble directly into the tree to be staged. */
export async function assembleAndFreezePackage({ sourceRoot, inputs, work, receipt, branch, expectedSource, repository, verifySource, measure }: { sourceRoot: string; inputs: string; work: string; receipt: string; branch: string; expectedSource: string; repository: string; verifySource: VerifySource; measure?: MeasurePublication }): Promise<FrozenPublication> {
  const observe = measure ?? unmeasuredPublication;
  const input = resolve(inputs);
  const { snapshot, receiptPath } = await publicationPaths({ sourceRoot, input, work, receipt, branch });
  const sourceCommit = gitObjectId({ value: expectedSource });
  const runtimes = await observe({ phase: 'publication.freeze.validate-inputs', operation: () => readRuntimePackages({ sourceRoot, inputs: input, checkNpm: false }) });
  if (runtimeNames.some(runtime => runtimes[runtime].manifest.sourceCommit !== sourceCommit)) {
    throw new Error('Package differs from selected source');
  }
  return installFrozenSnapshot({
    snapshot,
    receiptPath,
    prepare: async ({ staging }) => {
      const validated = await observe({ phase: 'publication.freeze.copy', operation: () => assembleValidatedPackage({ sourceRoot, runtimes, output: staging, checkNpm: false }) });
      return freezeSnapshot({ snapshot: staging, validated, branch, repository, verifySource, measure: observe });
    },
  });
}

async function installFrozenSnapshot({ snapshot, receiptPath, prepare }: { snapshot: string; receiptPath: string; prepare: ({ staging }: { staging: string }) => Promise<FrozenPublication> }): Promise<FrozenPublication> {
  await using temporary = await temporaryDirectory({ prefix: join(dirname(receiptPath), 'bicore-receipt-') });
  const candidate = join(temporary.root, 'receipt.json');
  return await withOutputDirectories({
    destinations: [snapshot],
    replace: false,
    prepare: async ({ paths: [staging] }) => {
      if (staging === undefined) {
        throw new Error('Missing publication snapshot');
      }
      const frozen = await prepare({ staging });
      await writeFile(candidate, JSON.stringify(frozen, undefined, 2) + '\n', { flag: 'wx', mode: 0o600 });
      return frozen;
    },
    finalize: async () => {
      // Install the complete receipt exclusively, rolling back the snapshot if it cannot be linked.
      await directory({ path: dirname(receiptPath) });
      await link(candidate, receiptPath);
    },
  });
}

async function freezeSnapshot({ snapshot, validated, branch, repository, verifySource, measure }: { snapshot: string; validated: ValidatedPackage; branch: string; repository: string; verifySource: VerifySource; measure: MeasurePublication }): Promise<FrozenPublication> {
  await measure({ phase: 'publication.freeze.public-exports', operation: () => checkPublicExports({ directory: snapshot, runtimes: validated.runtimes }) });
  const data = await createReportData({ validated, verifySource });
  const sourceCommit = validated.manifest.sourceCommit;
  const manifest = await readBoundFile({ root: snapshot, entry: { path: 'manifest.json', ...validated.manifestIdentity } });
  const integrityBytes = encodeArtifactIntegrity({ repository, sourceCommit, fingerprint: data.source.treeFingerprint, manifest });
  const integrity = bytesIdentity({ data: integrityBytes });
  await writeFile(join(snapshot, artifactIntegrityPath), integrityBytes, { flag: 'wx' });
  const files = new Map(validated.files);
  files.set(artifactIntegrityPath, { path: artifactIntegrityPath, ...integrity });
  await measure({ phase: 'publication.freeze.npm-pack', operation: () => checkNpmPack({ directory: snapshot, files }) });
  const tree = await measure({
    phase: 'publication.freeze.git-stage',
    operation: async () => {
      await localGit({ cwd: snapshot, args: ['init', '-q'] });
      await localGit({ cwd: snapshot, args: ['config', 'user.name', 'github-actions[bot]'] });
      await localGit({ cwd: snapshot, args: ['config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com'] });
      await localGit({ cwd: snapshot, args: ['add', '--all'] });
      return gitObjectId({ value: (await localGit({ cwd: snapshot, args: ['write-tree'] })).toString('utf8').trim() });
    },
  });
  await measure({ phase: 'publication.freeze.git-verify', operation: () => verifyGitTree({ work: snapshot, tree, files }) });
  const identities = Object.fromEntries(runtimeNames.map(runtime => {
    const entry = validated.files.get(artifactDirectories[runtime] + '/manifest.json');
    if (!entry) {
      throw new Error('Missing frozen runtime manifest');
    }
    return [runtime, { bytes: entry.bytes, sha256: entry.sha256 }];
  }));
  const llamaIdentity = identities['llama-cpp'];
  const imageIdentity = identities['stable-diffusion-cpp'];
  const speechIdentity = identities['speech-cpp'];
  if (!llamaIdentity || !imageIdentity || !speechIdentity) {
    throw new Error('Missing frozen manifest identity');
  }
  return { schemaVersion: 3, sourceCommit, tree, branch, manifest: validated.manifestIdentity, integrity, runtimeManifests: { 'llama-cpp': llamaIdentity, 'stable-diffusion-cpp': imageIdentity, 'speech-cpp': speechIdentity }, report: { sha256: bytesIdentity({ data: Buffer.from(canonicalJson({ value: data })) }).sha256, data } };
}

/** Consume only this job's private state; later arbitrary artifacts use revalidation. */
export async function confirmPublication({ work, receipt, commit }: { work: string; receipt: string; commit: string }): Promise<ConfirmedPublication> {
  gitObjectId({ value: commit });
  const frozen = await readMetadata({ path: receipt, decode: decodeFrozenPublication });
  checkFrozen({ frozen });
  const tree = gitObjectId({ value: (await localGit({ cwd: work, args: ['rev-parse', '--verify', `${commit}^{tree}`] })).toString('utf8').trim() });
  if (tree !== frozen.tree) {
    throw new Error('Published commit differs from frozen Git tree');
  }
  assertSame({ actual: bytesIdentity({ data: await gitBlob({ work, tree: commit, path: 'manifest.json' }) }), expected: frozen.manifest, message: 'Published manifest differs from frozen receipt' });
  assertSame({ actual: bytesIdentity({ data: await gitBlob({ work, tree: commit, path: artifactIntegrityPath }) }), expected: frozen.integrity, message: 'Published integrity envelope differs from frozen receipt' });
  for (const runtime of runtimeNames) {
    assertSame({ actual: bytesIdentity({ data: await gitBlob({ work, tree: commit, path: artifactDirectories[runtime] + '/manifest.json' }) }), expected: frozen.runtimeManifests[runtime], message: 'Published runtime manifest differs from frozen receipt' });
  }
  return { commit, frozen };
}

/** Fresh verification is the fallback for a different job or lost private receipt. */
export async function revalidatePublication({ sourceRoot, packageDirectory, work, commit, verifySource }: { sourceRoot: string; packageDirectory: string; work: string; commit: string; verifySource: VerifySource }): Promise<ReportData> {
  const validated = await validatePackage({ sourceRoot, directory: packageDirectory, requireClean: true, checkNpmPack: true });
  if (validated.integrity === undefined) {
    throw new Error('Published artifact is missing its integrity envelope');
  }
  const tree = gitObjectId({ value: (await localGit({ cwd: work, args: ['rev-parse', '--verify', `${gitObjectId({ value: commit })}^{tree}`] })).toString('utf8').trim() });
  await verifyGitTree({ work, tree, files: validated.files });
  const data = await createReportData({ validated, verifySource });
  assertSame({ actual: validated.integrity.document.source.tree_fingerprint, expected: data.source.treeFingerprint, message: 'Published tree fingerprint differs from verified source' });
  return data;
}
