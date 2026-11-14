import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { decodeNpmLock, type NpmLock, type NpmPackageEntry } from '#dto/shared';
import { packageName } from '#tools/shared/contracts';
import { optionalStat, directory as directoryPath } from '#tools/shared/files';
import { readMetadata, readMetadataBytes, assertSame } from '#tools/package/files';
import { gitObjectId } from './git.js';

export function repositoryName({ value }: { value: string }): string {
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/u.test(value) || value.split('/').some(part => part === '.' || part === '..')) {
    throw new Error('Invalid GitHub repository name');
  }
  return value;
}

export type NpmRequest = { repository: string; commit: string; version: string; package: typeof packageName; specifier: string };
export type NpmResolution = { lockfileVersion: 3; rootDependencies: Record<string, string>; packageEntry: NpmPackageEntry; specifier: string };

export async function prepareNpmResolution({ directory, repository, commit, version }: { directory: string; repository: string; commit: string; version: string }): Promise<NpmRequest> {
  const specifier = `github:${repositoryName({ value: repository })}#${gitObjectId({ value: commit })}`;
  if (!version) {
    throw new Error('Expected a package version');
  }
  const request: NpmRequest = { repository, commit, version, package: packageName, specifier };
  if (await optionalStat({ path: directory })) {
    throw new Error('npm resolution requires a fresh directory');
  }
  await directoryPath({ path: dirname(directory), create: true });
  await mkdir(directory);
  await writeFile(join(directory, 'package.json'), JSON.stringify({ name: 'runtime-consumer-lock-fixture', version: '0.0.0', private: true, dependencies: { [packageName]: specifier } }, undefined, 2) + '\n');
  await writeFile(join(directory, 'empty.npmrc'), '');
  await writeFile(join(directory, 'empty-global.npmrc'), '');
  return request;
}

export function extractNpmResolution({ lock, repository, commit, version }: { lock: NpmLock; repository: string; commit: string; version: string }): NpmResolution {
  repositoryName({ value: repository });
  gitObjectId({ value: commit });
  if (lock.lockfileVersion !== 3) {
    throw new Error('Expected npm lockfile version 3');
  }
  const key = `node_modules/${packageName}`;
  const specifier = `github:${repository}#${commit}`;
  assertSame({ actual: Object.keys(lock.packages).sort(), expected: ['', key], message: 'Unexpected transitive npm dependency' });
  const dependencies = lock.packages['']?.dependencies;
  assertSame({ actual: dependencies, expected: { [packageName]: specifier }, message: 'npm changed the requested dependency' });
  const entry = lock.packages[key];
  if (!entry || entry.version !== version || entry.resolved === undefined) {
    throw new Error('npm resolved another runtime version or omitted its resolution');
  }
  const url = new URL(entry.resolved);
  if (!['git+https:', 'git+ssh:'].includes(url.protocol) || url.hostname !== 'github.com' || url.pathname.replace(/\.git$/u, '').toLowerCase() !== '/' + repository.toLowerCase() || url.search || url.password || url.port || !['', 'git'].includes(url.username) || url.hash !== '#' + commit) {
    throw new Error('npm resolved another repository or commit');
  }
  for (const field of ['hasInstallScript', 'dependencies', 'optionalDependencies', 'peerDependencies', 'peerDependenciesMeta', 'bundleDependencies', 'bundledDependencies'] as const) {
    const value = entry[field];
    if (value !== undefined && value !== false && !(typeof value === 'object' && value !== null && Object.keys(value).length === 0)) {
      throw new Error('Unexpected npm install-time behavior');
    }
  }
  return { lockfileVersion: 3, rootDependencies: dependencies ?? {}, packageEntry: entry, specifier };
}

export async function readNpmResolution({ directory, expected }: { directory: string; expected: { repository: string; commit: string; version: string } }): Promise<NpmResolution> {
  const specifier = `github:${repositoryName({ value: expected.repository })}#${gitObjectId({ value: expected.commit })}`;
  const fixture: unknown = JSON.parse((await readMetadataBytes({ path: join(directory, 'package.json') })).toString('utf8'));
  assertSame({ actual: fixture, expected: { name: 'runtime-consumer-lock-fixture', version: '0.0.0', private: true, dependencies: { [packageName]: specifier } }, message: 'npm resolution fixture changed' });
  const lock = await readMetadata({ path: join(directory, 'package-lock.json'), decode: decodeNpmLock });
  return extractNpmResolution({ lock, ...expected });
}
