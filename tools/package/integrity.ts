import { load, dump, JSON_SCHEMA } from 'js-yaml';
import { join } from 'node:path';
import { decodeArtifactIntegrity, type ArtifactIntegrity, type FileIdentity, type TreeFingerprint } from '#dto/shared';
import { optionalStat } from '#tools/shared/files';
import { assertSame, bytesIdentity, readMetadataBytes } from './files.js';

export const artifactIntegrityPath = 'browser-inference-core-artifact-integrity.yaml';
export type ValidatedIntegrity = { document: ArtifactIntegrity; identity: FileIdentity };

export function checkTreeFingerprint({ fingerprint }: { fingerprint: TreeFingerprint }): void {
  if (!/^[a-f0-9]{64}$/u.test(fingerprint.value) || !fingerprint.reproduce.trim()) {
    throw new Error('Invalid source tree fingerprint');
  }
}

export function encodeArtifactIntegrity({ repository, sourceCommit, fingerprint, manifest }: { repository: string; sourceCommit: string; fingerprint: TreeFingerprint; manifest: Buffer }): Buffer {
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/u.test(repository)) {
    throw new Error('Expected a GitHub owner/repository for artifact integrity');
  }
  checkTreeFingerprint({ fingerprint });
  const identity = bytesIdentity({ data: manifest });
  const document: ArtifactIntegrity = {
    schema_version: 1,
    kind: 'browser_inference_core_artifact_integrity',
    source: { repository: `https://github.com/${repository}`, commit: sourceCommit, tree_fingerprint: fingerprint },
    manifest: { path: 'manifest.json', size: identity.bytes, sha256: identity.sha256, contents: new TextDecoder('utf-8', { fatal: true }).decode(manifest) },
  };
  const comments = [
    '# The exact bytes of this YAML are the GitHub Actions attestation subject.',
    '# Verify the attestation for the expected repository and build.yml workflow first.',
    `# gh attestation verify ${artifactIntegrityPath} --repo ${repository} \\`,
    `#   --signer-workflow ${repository}/.github/workflows/build.yml --deny-self-hosted-runners`,
    '# Commands in signed files still require review; never automatically execute them.',
    '# tree_fingerprint covers committed paths, modes, blob bytes and recursive submodules.',
    '# It excludes the index, working-tree changes and untracked files. Initialize recursive',
    '# submodules at source.commit, run reproduce at that checkout root, and compare value.',
    '# manifest.contents preserves the exact manifest.json bytes, including its final newline.',
    '# Verify manifest.json against size/sha256, then every root/runtime inventory entry.',
    '# This envelope is outside those inventories to avoid a manifest/self-hash cycle.',
    '',
  ].join('\n');
  return Buffer.from(comments + dump(document, { schema: JSON_SCHEMA, noRefs: true, lineWidth: -1 }));
}

/** Bind the sole allowed envelope to the already verified aggregate manifest. */
export async function readArtifactIntegrity({ directory, manifest, sourceCommit }: { directory: string; manifest: Buffer; sourceCommit: string }): Promise<ValidatedIntegrity | undefined> {
  const path = join(directory, artifactIntegrityPath);
  if (!(await optionalStat({ path }))) {
    return undefined;
  }
  const raw = await readMetadataBytes({ path });
  const document = decodeArtifactIntegrity({ input: load(new TextDecoder('utf-8', { fatal: true }).decode(raw), { schema: JSON_SCHEMA }) });
  if (!/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/u.test(document.source.repository) || document.source.commit !== sourceCommit) {
    throw new Error('Integrity envelope source differs from package');
  }
  checkTreeFingerprint({ fingerprint: document.source.tree_fingerprint });
  assertSame({ actual: { bytes: document.manifest.size, sha256: document.manifest.sha256 }, expected: bytesIdentity({ data: manifest }), message: 'Integrity envelope manifest identity mismatch' });
  if (!Buffer.from(document.manifest.contents).equals(manifest)) {
    throw new Error('Integrity envelope does not embed the exact manifest bytes');
  }
  return { document, identity: bytesIdentity({ data: raw }) };
}
