import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { decodeFrozenPublication, type SourceFacts, type ReportData, type FrozenPublication, type JsonObject, type Validation } from '#dto/shared';
import { artifactDirectories, runtimeNames } from '#tools/shared/contracts';
import { isContained, optionalStat, directory } from '#tools/shared/files';
import { validatePackage, type ValidatedPackage, type RuntimeManifests } from '#tools/package/aggregate';
import { assertSame, bytesIdentity, copyBoundTree, readMetadata, checkNpmPack } from '#tools/package/files';
import { canonicalJson } from '#tools/abi/schema';
import { gitBlob, gitObjectId, localGit, verifyGitTree } from './git.js';

export type VerifySource = ({ sourceCommit, manifests }: { sourceCommit: string; manifests: RuntimeManifests }) => Promise<SourceFacts>;
export type ConfirmedPublication = { commit: string; frozen: FrozenPublication };

export function checkFrozen({ frozen }: { frozen: FrozenPublication }): void {
  if (frozen.schemaVersion !== 2) {
    throw new Error('Unsupported private frozen state');
  }
  if (frozen.branch !== 'artifacts' && !frozen.branch.startsWith('artifacts/')) {
    throw new Error('Refusing to publish over a source branch');
  }
  if (frozen.sourceCommit !== frozen.report.data.sourceCommit || frozen.sourceCommit !== frozen.report.data.source.sourceCommit) {
    throw new Error('Frozen report source mismatch');
  }
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
  return {
    sourceCommit: validated.manifest.sourceCommit,
    package: { name: 'browser-inference-core', version: validated.metadata.version },
    source: facts,
    runtimes: {
      'llama-cpp': { manifest: artifactDirectories['llama-cpp'] + '/manifest.json', abiVersion: llama.abiVersion, schemaSha256: llama.schemaSha256, upstreams: { llama: llama.llamaCommit }, profiles: profiles({ values: llama.profiles }) },
      'stable-diffusion-cpp': { manifest: artifactDirectories['stable-diffusion-cpp'] + '/manifest.json', abiVersion: image.abiVersion, schemaSha256: image.schemaSha256, upstreams: image.upstreams, profiles: profiles({ values: image.profiles }) },
    },
  };
}

export async function freezePackage({ sourceRoot, packageDirectory, work, receipt, branch, expectedSource, verifySource }: { sourceRoot: string; packageDirectory: string; work: string; receipt: string; branch: string; expectedSource: string; verifySource: VerifySource }): Promise<FrozenPublication> {
  if (branch !== 'artifacts' && !branch.startsWith('artifacts/')) {
    throw new Error('Refusing to publish over a source branch');
  }
  await localGit({ cwd: sourceRoot, args: ['check-ref-format', '--branch', branch] });
  const input = resolve(packageDirectory);
  const snapshot = resolve(work);
  const receiptPath = resolve(receipt);
  if (isContained({ root: input, path: snapshot }) || isContained({ root: input, path: receiptPath }) || isContained({ root: snapshot, path: receiptPath }) || isContained({ root: receiptPath, path: snapshot })) {
    throw new Error('Publication state must be outside the runtime package');
  }
  if (await optionalStat({ path: snapshot }) || await optionalStat({ path: receiptPath })) {
    throw new Error('Publication snapshot and receipt must be fresh');
  }
  await directory({ path: dirname(snapshot), create: true });
  await directory({ path: dirname(receiptPath), create: true });
  const validated = await validatePackage({ sourceRoot, directory: input, requireClean: true, checkNpmPack: false });
  if (validated.manifest.sourceCommit !== gitObjectId({ value: expectedSource })) {
    throw new Error('Package differs from selected source');
  }
  // Copy verification is its own byte boundary. The typed validated input is
  // shared afterwards; the private snapshot needs no second manifest decoder.
  await copyBoundTree({ source: input, destination: snapshot, files: validated.files });
  await checkNpmPack({ directory: snapshot, files: validated.files });
  const data = await createReportData({ validated, verifySource });
  await localGit({ cwd: snapshot, args: ['init', '-q'] });
  await localGit({ cwd: snapshot, args: ['config', 'user.name', 'github-actions[bot]'] });
  await localGit({ cwd: snapshot, args: ['config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com'] });
  await localGit({ cwd: snapshot, args: ['add', '--all'] });
  const tree = gitObjectId({ value: (await localGit({ cwd: snapshot, args: ['write-tree'] })).toString('utf8').trim() });
  await verifyGitTree({ work: snapshot, tree, files: validated.files });
  const identities = Object.fromEntries(runtimeNames.map(runtime => {
    const entry = validated.files.get(artifactDirectories[runtime] + '/manifest.json');
    if (!entry) {
      throw new Error('Missing frozen runtime manifest');
    }
    return [runtime, { bytes: entry.bytes, sha256: entry.sha256 }];
  }));
  const llamaIdentity = identities['llama-cpp'];
  const imageIdentity = identities['stable-diffusion-cpp'];
  if (!llamaIdentity || !imageIdentity) {
    throw new Error('Missing frozen manifest identity');
  }
  const frozen: FrozenPublication = { schemaVersion: 2, sourceCommit: expectedSource, tree, branch, manifest: validated.manifestIdentity, runtimeManifests: { 'llama-cpp': llamaIdentity, 'stable-diffusion-cpp': imageIdentity }, report: { sha256: bytesIdentity({ data: Buffer.from(canonicalJson({ value: data })) }).sha256, data } };
  await mkdir(dirname(receiptPath), { recursive: true });
  await writeFile(receiptPath, JSON.stringify(frozen, undefined, 2) + '\n', { flag: 'wx', mode: 0o600 });
  return frozen;
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
  for (const runtime of runtimeNames) {
    assertSame({ actual: bytesIdentity({ data: await gitBlob({ work, tree: commit, path: artifactDirectories[runtime] + '/manifest.json' }) }), expected: frozen.runtimeManifests[runtime], message: 'Published runtime manifest differs from frozen receipt' });
  }
  return { commit, frozen };
}

/** Fresh verification is the fallback for a different job or lost private receipt. */
export async function revalidatePublication({ sourceRoot, packageDirectory, work, commit, verifySource }: { sourceRoot: string; packageDirectory: string; work: string; commit: string; verifySource: VerifySource }): Promise<ReportData> {
  const validated = await validatePackage({ sourceRoot, directory: packageDirectory, requireClean: true, checkNpmPack: true });
  const tree = gitObjectId({ value: (await localGit({ cwd: work, args: ['rev-parse', '--verify', `${gitObjectId({ value: commit })}^{tree}`] })).toString('utf8').trim() });
  await verifyGitTree({ work, tree, files: validated.files });
  return createReportData({ validated, verifySource });
}
