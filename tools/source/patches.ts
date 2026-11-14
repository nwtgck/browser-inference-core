import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import type { PatchReceipt } from '#dto/shared';
import { containedPath, directory, fileIdentity, optionalStat, regularFile, relativePath } from '#tools/shared/files';
import { runProcess } from '#tools/shared/process';
import { gitEnvironment } from './capture.js';

export type Patch = { file: string; target: string; reason: string; files: readonly string[]; mode?: 'apply' | 'reverse' | undefined };
export type { PatchReceipt } from '#dto/shared';

async function optionalDigest({ root, name }: { root: string; name: string }): Promise<string | undefined> {
  const path = containedPath({ root, relative: name });
  // Reject linked existing ancestors, including when the final added file is absent.
  let parent = dirname(path);
  const missing: string[] = [];
  while (!await optionalStat({ path: parent })) {
    missing.push(parent);
    parent = dirname(parent);
  }
  await directory({ path: parent });
  if (missing.length || !await optionalStat({ path })) {
    return undefined;
  }
  return (await fileIdentity({ path })).sha256;
}

/** A recipe author limits scope; observed numstat paths cannot authorize themselves. */
export async function applyPatch({ sourceRoot, patchRoot, patch: declaration }: { sourceRoot: string; patchRoot: string; patch: Patch }): Promise<PatchReceipt> {
  const patch = declaration;
  relativePath({ value: patch.file });
  if (!patch.file.endsWith('.patch') || !patch.files.length || new Set(patch.files).size !== patch.files.length) {
    throw new Error('Expected an explicit patch and nonempty distinct scope');
  }
  for (const name of patch.files) {
    relativePath({ value: name });
  }
  sourceRoot = await directory({ path: sourceRoot });
  const raw = await readFile(await regularFile({ path: containedPath({ root: patchRoot, relative: patch.file }) }));
  const text = new TextDecoder('utf-8', { fatal: true }).decode(raw);
  if (/^(?:rename |copy |GIT binary patch|Binary files |(?:old|new|deleted) (?:file )?mode (?!100(?:644|755)$))/mu.test(text)) {
    throw new Error(`Unsupported source patch representation: ${patch.file}`);
  }
  const temporary = await mkdtemp(join(tmpdir(), 'bicore-patch-'));
  try {
    const captured = join(temporary, 'input.patch');
    await writeFile(captured, raw);
    const mode = patch.mode ?? 'apply';
    let direction: string[];
    switch (mode) {
    case 'apply':
      direction = [];
      break;
    case 'reverse':
      direction = ['--reverse'];
      break;
    default: {
      const _ex: never = mode;
      throw new Error(`Unhandled patch direction: ${_ex}`);
    }
    }
    async function apply({ args }: { args: readonly string[] }): Promise<Buffer> {
      return (await runProcess({ command: 'git', args: ['-c', 'core.autocrlf=false', 'apply', '--no-index', ...direction, ...args, captured], cwd: sourceRoot, env: gitEnvironment({ ceiling: dirname(sourceRoot) }) })).stdout;
    }
    const changes = (await apply({ args: ['--numstat', '-z'] })).toString('utf8').split('\0').filter(Boolean).map((row) => {
      const match = /^\d+\t\d+\t([\s\S]+)$/u.exec(row);
      if (!match?.[1]) {
        throw new Error('Expected a text-only source patch with explicit file paths');
      }
      return relativePath({ value: match[1] });
    });
    if (!changes.length || new Set(changes).size !== changes.length) {
      throw new Error('Empty or duplicate patch changes');
    }
    if (changes.length !== patch.files.length || changes.some((name) => !patch.files.includes(name))) {
      throw new Error(`Patch changes files outside its inventory: ${patch.file}`);
    }
    const before = new Map<string, string | undefined>();
    for (const name of patch.files) {
      before.set(name, await optionalDigest({ root: sourceRoot, name }));
    }
    await apply({ args: ['--check', '--whitespace=error-all'] });
    await apply({ args: ['--whitespace=error-all'] });
    const files: PatchReceipt['files'] = [];
    for (const name of patch.files) {
      const beforeSha256 = before.get(name);
      const afterSha256 = await optionalDigest({ root: sourceRoot, name });
      files.push({ path: name, ...(beforeSha256 === undefined ? {} : { beforeSha256 }), ...(afterSha256 === undefined ? {} : { afterSha256 }) });
    }
    return { file: patch.file, target: patch.target, reason: patch.reason, mode, sha256: createHash('sha256').update(raw).digest('hex'), files };
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

export async function applyRecipe({ sources, patchRoot, patches }: { sources: Readonly<Record<string, string>>; patchRoot: string; patches: readonly Patch[] }): Promise<PatchReceipt[]> {
  const seen = new Set<string>();
  const selected = patches.map((patch) => {
    const sourceRoot = sources[patch.target];
    if (!sourceRoot) {
      throw new Error(`Unknown patch target: ${patch.target}`);
    }
    if (seen.has(patch.file)) {
      throw new Error(`Duplicate patch declaration: ${patch.file}`);
    }
    seen.add(patch.file);
    return { patch, sourceRoot };
  });
  const records: PatchReceipt[] = [];
  for (const { patch, sourceRoot } of selected) {
    records.push(await applyPatch({ sourceRoot, patchRoot, patch }));
  }
  return records;
}
