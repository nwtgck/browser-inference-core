import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { dump, load, JSON_SCHEMA } from 'js-yaml';
import { decodeArtifactIntegrity, type ArtifactIntegrity } from '#dto/shared';
import { artifactIntegrityPath, encodeArtifactIntegrity, readArtifactIntegrity } from '#tools/package/integrity';
import { bytesIdentity } from '#tools/package/files';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { fixtureSource, fixtureTreeFingerprint } from './fixtures.test-support.js';

test('YAML preserves exact UTF-8 JSON bytes, whitespace and comments as signed file data', async () => {
  await using temporary = await temporaryDirectory();
  const directory = temporary.root;
  for (const text of ['{"text":"# : & < > 日本語", "number": 1}\n', '{"text":"spaces  "}', '{\r\n  "text": "CRLF"\r\n}\r\n', '{"text":"trailing newlines"}\n\n']) {
    const manifest = Buffer.from(text);
    const raw = encodeArtifactIntegrity({ repository: 'fixture/browser-inference-core', sourceCommit: fixtureSource, fingerprint: fixtureTreeFingerprint, manifest });
    await writeFile(join(directory, artifactIntegrityPath), raw);
    const integrity = await readArtifactIntegrity({ directory, manifest, sourceCommit: fixtureSource });
    assert.equal(integrity?.document.manifest.contents, text);
    assert.deepEqual(integrity?.identity, bytesIdentity({ data: raw }));
    assert.deepEqual({ bytes: integrity?.document.manifest.size, sha256: integrity?.document.manifest.sha256 }, bytesIdentity({ data: manifest }));
  }
  await rm(join(directory, artifactIntegrityPath));
  assert.equal(await readArtifactIntegrity({ directory, manifest: Buffer.from('{}'), sourceCommit: fixtureSource }), undefined);
  assert.throws(() => encodeArtifactIntegrity({ repository: 'fixture\n/evil', sourceCommit: fixtureSource, fingerprint: fixtureTreeFingerprint, manifest: Buffer.from('{}') }), /owner\/repository/u);
  assert.throws(() => encodeArtifactIntegrity({ repository: 'fixture/core', sourceCommit: fixtureSource, fingerprint: fixtureTreeFingerprint, manifest: Buffer.from([0xff]) }));
});

test('an envelope rejects changed source, embedded bytes, metadata, YAML ambiguity and symlink substitution', async () => {
  await using temporary = await temporaryDirectory();
  const directory = temporary.root;
  const path = join(directory, artifactIntegrityPath);
  const manifest = Buffer.from('{"payload":"trusted"}\n');
  const raw = encodeArtifactIntegrity({ repository: 'fixture/browser-inference-core', sourceCommit: fixtureSource, fingerprint: fixtureTreeFingerprint, manifest });
  const original = decodeArtifactIntegrity({ input: load(raw.toString(), { schema: JSON_SCHEMA }) });
  const changes: ((document: ArtifactIntegrity) => void)[] = [
    document => {
      document.source.commit = 'c'.repeat(40);
    },
    document => {
      document.source.repository = 'https://example.com/fixture/core';
    },
    document => {
      document.source.tree_fingerprint.value = 'not a digest';
    },
    document => {
      document.manifest.sha256 = 'c'.repeat(64);
    },
    document => {
      document.manifest.size++;
    },
    document => {
      document.manifest.contents = document.manifest.contents.trimEnd();
    },
  ];
  for (const change of changes) {
    const document = structuredClone(original);
    change(document);
    await writeFile(path, dump(document));
    await assert.rejects(readArtifactIntegrity({ directory, manifest, sourceCommit: fixtureSource }));
  }
  for (const text of [raw + 'schema_version: 1\n', raw + 'unrecognized: true\n', raw + 'unrecognized: !!js/function function() {}\n']) {
    await writeFile(path, text);
    await assert.rejects(readArtifactIntegrity({ directory, manifest, sourceCommit: fixtureSource }));
  }
  await writeFile(join(directory, 'other.yaml'), raw);
  await rm(path);
  await symlink('other.yaml', path);
  await assert.rejects(readArtifactIntegrity({ directory, manifest, sourceCommit: fixtureSource }), /Linked|regular/u);
  assert.deepEqual(await readFile(join(directory, 'other.yaml')), raw);
});
