import { createHash } from 'node:crypto';
import { access, lstat } from 'node:fs/promises';
import { join } from 'node:path';
import type { TreeFingerprint } from '#dto/shared';
import { gitObjectId, localGit, localGitEnvironment } from '#tools/publication/git';
import { runProcess, startProcess } from '#tools/shared/process';
import { BinaryReader } from '#tools/shared/streams';

async function readBlobDigest({ reader, oid }: { reader: BinaryReader; oid: string }): Promise<string> {
  const header = await reader.line({ limit: 256 });
  const [, object, size] = /^([a-f0-9]+) blob ([0-9]+)$/u.exec(header) ?? [];
  let remaining = Number(size);
  if (object !== oid || !Number.isSafeInteger(remaining)) {
    throw new Error(`Missing or invalid tree fingerprint blob: ${oid} (${header})`);
  }
  const digest = createHash('sha256');
  while (remaining > 0) {
    const chunk = await reader.take({ count: Math.min(remaining, 1024 * 1024) });
    digest.update(chunk);
    remaining -= chunk.length;
  }
  if ((await reader.take({ count: 1 }))[0] !== 10) {
    throw new Error(`Invalid tree fingerprint blob delimiter: ${oid}`);
  }
  return digest.digest('hex');
}

async function headCommit({ root }: { root: string }): Promise<string> {
  return (await localGit({ cwd: root, args: ['rev-parse', '--verify', 'HEAD'] })).toString('utf8').trimEnd();
}

async function hashSubmodule({ root, path, oid }: { root: string; path: Buffer; oid: string }): Promise<string | undefined> {
  const name = path.toString('utf8');
  if (!Buffer.from(name).equals(path)) {
    // Bash can pass raw pathname bytes as cwd. Node cannot encode these bytes in a string.
    return undefined;
  }
  const cwd = join(root, name);
  if (!(await lstat(cwd)).isDirectory()) {
    throw new Error(`Submodule is not a directory: ${JSON.stringify(name)}`);
  }
  await access(join(cwd, '.git'));
  if (await headCommit({ root: cwd }) !== oid) {
    throw new Error(`Submodule commit mismatch: ${JSON.stringify(name)}`);
  }
  return hashCommittedTree({ root: cwd, commit: oid });
}

/** Returns undefined only when a gitlink pathname needs the byte-oriented Bash recipe. */
async function hashCommittedTree({ root, commit }: { root: string; commit: string }): Promise<string | undefined> {
  const prefix = await localGit({ cwd: root, args: ['rev-parse', '--show-prefix'] });
  if (!prefix.equals(Buffer.from('\n'))) {
    throw new Error('Run at a repository root.');
  }
  const listing = await localGit({ cwd: root, args: ['ls-tree', '-rz', '--full-tree', `${commit}^{commit}`] });
  const child = startProcess({ command: 'git', args: ['--no-replace-objects', 'cat-file', '--batch'], cwd: root, env: localGitEnvironment() });
  const reader = new BinaryReader({ stream: child.stdout });
  const digest = createHash('sha256');
  try {
    for (let offset = 0; offset < listing.length;) {
      const end = listing.indexOf(0, offset);
      const separator = listing.indexOf(9, offset);
      if (end < 0 || separator < 0 || separator + 1 >= end) {
        throw new Error('Invalid Git tree fingerprint record');
      }
      const header = listing.toString('latin1', offset, separator);
      const [, entry, oid] = /^(100644 blob|100755 blob|120000 blob|160000 commit) ([a-f0-9]{40})$/u.exec(header) ?? [];
      if (entry === undefined || oid === undefined) {
        throw new Error(`Unsupported Git entry: ${header}`);
      }
      const path = listing.subarray(separator + 1, end);
      let value: string | undefined;
      if (entry === '160000 commit') {
        value = await hashSubmodule({ root, path, oid });
        if (value === undefined) {
          return undefined;
        }
      } else {
        child.stdin.write(`${oid}\n`);
        value = await readBlobDigest({ reader, oid });
      }
      digest.update(`${entry.slice(0, 6)}\0`).update(path).update(`\0${value}\0`);
      offset = end + 1;
    }
    child.stdin.end();
    await reader.expectEnd();
    await child.completed;
    return digest.digest('hex');
  } finally {
    child.stdin.destroy();
    child.stdout.destroy();
    child.kill();
    await child.completed.catch(() => undefined);
  }
}

async function reproduceFingerprint({ root, recipe }: { root: string; recipe: string }): Promise<string> {
  const environment = localGitEnvironment();
  const result = await runProcess({ command: 'bash', args: ['--noprofile', '--norc', '-c', recipe], cwd: root, env: { ...environment, overrides: { ...environment.overrides, BASH_ENV: undefined, ENV: undefined } } });
  const value = result.stdout.toString('utf8');
  if (!/^[a-f0-9]{64}\n$/u.test(value)) {
    throw new Error('Invalid committed tree fingerprint output');
  }
  return value.trimEnd();
}

/** Hash the committed tree and publish the independent Bash reproduction recipe. */
export async function captureTreeFingerprint({ sourceRoot, sourceCommit }: { sourceRoot: string; sourceCommit: string }): Promise<TreeFingerprint> {
  async function checkHead(): Promise<void> {
    if (await headCommit({ root: sourceRoot }) !== sourceCommit) {
      throw new Error('Tree fingerprint source differs from selected commit');
    }
  }
  await checkHead();
  const commit = gitObjectId({ value: sourceCommit });
  const recipe = await localGit({ cwd: sourceRoot, args: ['cat-file', 'blob', `${commit}:.github/scripts/treefingerprint.sh`] });
  const reproduce = new TextDecoder('utf-8', { fatal: true }).decode(recipe);
  const value = await hashCommittedTree({ root: sourceRoot, commit }) ?? await reproduceFingerprint({ root: sourceRoot, recipe: reproduce });
  await checkHead();
  return { algorithm: 'sha256', scheme: 'git-committed-tree-sha256-v1', value, reproduce };
}
