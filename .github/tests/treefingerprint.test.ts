import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { chmod, mkdir, readFile, rename, rm, symlink, unlink, writeFile } from 'node:fs/promises';
import { delimiter, dirname, join, resolve } from 'node:path';
import { test } from 'node:test';
import { gitEnvironment } from '#tools/source/capture';
import { captureTreeFingerprint } from '#tools/source/fingerprint';
import { ProcessError, runProcess } from '#tools/shared/process';
import { READ_ENV_PATH } from '#tools/shared/node-env';
import { commitFiles, git, initializeRepository, temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { stubEnvironment } from '#tests/tools/shared/node-env.test-support';

const scriptPath = '.github/scripts/treefingerprint.sh';
const script = resolve(scriptPath);

function sha256({ bytes }: { bytes: Uint8Array | string }): string {
  return createHash('sha256').update(bytes).digest('hex');
}

async function repository({ root }: { root: string }): Promise<void> {
  await initializeRepository({ root });
  await commitFiles({ root });
}

async function installRecipe({ root }: { root: string }): Promise<Buffer> {
  const bytes = await readFile(script);
  const path = join(root, scriptPath);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, bytes);
  return bytes;
}

async function publicationRepository({ root }: { root: string }): Promise<string> {
  await initializeRepository({ root });
  await installRecipe({ root });
  return commitFiles({ root });
}

async function fingerprint({ root, invocation }: { root: string; invocation: 'standalone' | 'conditional' }) {
  return runProcess({
    command: 'bash',
    args: invocation === 'conditional'
      ? ['-c', 'if source "$1"; then exit 0; else exit $?; fi', 'test', script]
      : [script],
    cwd: root,
    env: gitEnvironment({ ceiling: undefined }),
  });
}

async function bashDigest({ root }: { root: string }): Promise<string> {
  const result = await fingerprint({ root, invocation: 'standalone' });
  assert.match(result.stdout.toString(), /^[0-9a-f]{64}\n$/u);
  return result.stdout.toString().trim();
}

async function digest({ root }: { root: string }): Promise<string> {
  const sourceCommit = await git({ root, args: ['rev-parse', 'HEAD'] });
  const expected = await bashDigest({ root });
  const captured = await captureTreeFingerprint({ sourceRoot: root, sourceCommit });
  assert.equal(captured.value, expected);
  return captured.value;
}

async function mustFail({ root, sourceCommit }: { root: string; sourceCommit: string }): Promise<void> {
  for (const invocation of ['standalone', 'conditional'] as const) {
    await assert.rejects(fingerprint({ root, invocation }), (error: unknown) => {
      assert.ok(error instanceof ProcessError);
      assert.notEqual(error.exitCode, 0);
      assert.equal(error.stdout.toString(), '', 'A failed input must never produce a plausible digest');
      return true;
    });
  }
  await assert.rejects(captureTreeFingerprint({ sourceRoot: root, sourceCommit }));
}

test('tree fingerprints match independent empty-tree and binary blob, mode and symlink vectors', async () => {
  await using temporary = await temporaryDirectory();
  const root = join(temporary.root, 'repo');
  await repository({ root });
  assert.equal(await bashDigest({ root }), sha256({ bytes: '' }));
  const sourceCommit = await git({ root, args: ['rev-parse', 'HEAD'] });
  await assert.rejects(captureTreeFingerprint({ sourceRoot: root, sourceCommit }));
  const recipe = await installRecipe({ root });
  const large = Buffer.alloc(2 * 1024 * 1024 + 3, Buffer.from([0, 10, 255, 11]));
  const files = [
    { mode: '100644', name: 'a', bytes: Buffer.from([104, 101, 108, 108, 111, 0, 255, 10]) },
    { mode: '100755', name: 'b', bytes: Buffer.from('exec\n') },
    { mode: '120000', name: 'c', bytes: Buffer.from('a\n') },
    { mode: '100644', name: 'empty', bytes: Buffer.alloc(0) },
    { mode: '100644', name: 'large', bytes: large },
    { mode: '100644', name: 'last', bytes: Buffer.from('after large blob') },
  ];
  for (const file of files) {
    if (file.mode === '120000') {
      await symlink(file.bytes.toString(), join(root, file.name));
    } else {
      await writeFile(join(root, file.name), file.bytes, { mode: file.mode === '100755' ? 0o755 : 0o644 });
    }
  }
  await commitFiles({ root });
  const records = [
    { mode: '100644', name: scriptPath, bytes: recipe },
    ...files,
  ].map(({ mode, name, bytes }) => Buffer.from(`${mode}\0${name}\0${sha256({ bytes })}\0`));
  assert.equal(await digest({ root }), sha256({ bytes: Buffer.concat(records) }));
});

test('raw pathnames reproduce across clones and different commits with the same tree', async () => {
  await using temporary = await temporaryDirectory();
  const root = join(temporary.root, 'repo');
  await publicationRepository({ root });
  for (const name of [' space', '-dash', 'tab\tfile', 'new\nline', 'quotes\'"', '日本語', 'a.b', 'a/leaf']) {
    await mkdir(dirname(join(root, name)), { recursive: true });
    await writeFile(join(root, name), name);
  }
  await writeFile(Buffer.concat([Buffer.from(`${root}/`), Buffer.from([255, 254])]), 'non-UTF-8 name');
  await symlink('new\nline', join(root, 'symlink'));
  await commitFiles({ root });
  const original = await digest({ root });
  await commitFiles({ root });
  assert.equal(await digest({ root }), original);
  const clone = join(temporary.root, 'clone');
  await git({ root, args: ['clone', '-q', '--no-hardlinks', root, clone] });
  assert.equal(await digest({ root: clone }), original);
  assert.equal((await fingerprint({ root, invocation: 'conditional' })).stdout.toString(), `${original}\n`);
});

test('ordinary Git paths are hashed without executing the published reproduction recipe', async () => {
  await using temporary = await temporaryDirectory();
  const root = join(temporary.root, 'repo');
  const sourceCommit = await publicationRepository({ root });
  const expected = await bashDigest({ root });
  const bin = join(temporary.root, 'bin');
  await mkdir(bin);
  await writeFile(join(bin, 'bash'), '#!/bin/sh\nexit 99\n', { mode: 0o755 });
  using environment = stubEnvironment({ values: { PATH: bin + delimiter + (READ_ENV_PATH() ?? '') } });
  const captured = await captureTreeFingerprint({ sourceRoot: root, sourceCommit });
  assert.equal(captured.value, expected);
  assert.equal(captured.reproduce, await readFile(script, 'utf8'));
});

test('committed bytes, paths, executable modes and symlink targets all affect the fingerprint', async () => {
  await using temporary = await temporaryDirectory();
  const root = join(temporary.root, 'repo');
  await publicationRepository({ root });
  const file = join(root, 'file');
  await writeFile(file, 'first');
  await commitFiles({ root });
  let previous = await digest({ root });
  for (const mutate of [
    () => writeFile(file, 'second'),
    () => chmod(file, 0o755),
    () => rename(file, join(root, 'renamed')),
    () => symlink('renamed', join(root, 'link')),
    async () => {
      await unlink(join(root, 'link'));
      await symlink('./renamed', join(root, 'link'));
    },
  ]) {
    await mutate();
    await commitFiles({ root });
    const current = await digest({ root });
    assert.notEqual(current, previous);
    previous = current;
  }
});

test('committed fingerprints ignore worktree/index changes, checkout conversion, and replacement refs', async () => {
  await using temporary = await temporaryDirectory();
  const root = join(temporary.root, 'repo');
  await publicationRepository({ root });
  await writeFile(join(root, 'file'), 'committed\n');
  await commitFiles({ root });
  const original = await digest({ root });
  await writeFile(join(root, 'file'), 'worktree\r\n');
  await writeFile(join(root, 'untracked'), 'not committed');
  await git({ root, args: ['add', 'file'] });
  await git({ root, args: ['config', 'core.autocrlf', 'true'] });
  const status = await git({ root, args: ['status', '--porcelain=v1'] });
  assert.equal(await digest({ root }), original);
  assert.equal(await git({ root, args: ['status', '--porcelain=v1'] }), status);
  const replacementFile = join(temporary.root, 'replacement');
  await writeFile(replacementFile, 'replacement');
  const blob = await git({ root, args: ['rev-parse', 'HEAD:file'] });
  const replacement = await git({ root, args: ['hash-object', '-w', replacementFile] });
  await git({ root, args: ['replace', blob, replacement] });
  assert.equal(await digest({ root }), original);
});

test('wrong directories and missing Git objects fail without emitting a digest even under a conditional', async () => {
  await using temporary = await temporaryDirectory();
  const root = join(temporary.root, 'repo');
  const sourceCommit = await publicationRepository({ root });
  const inner = join(root, 'inner');
  const elsewhere = join(temporary.root, 'elsewhere');
  await mkdir(inner);
  await mkdir(elsewhere);
  await mustFail({ root: inner, sourceCommit });
  await mustFail({ root: elsewhere, sourceCommit });
  for (const object of ['HEAD:inner/file', 'HEAD:inner', 'HEAD^{tree}']) {
    await using broken = await temporaryDirectory();
    await publicationRepository({ root: broken.root });
    await mkdir(join(broken.root, 'inner'));
    await writeFile(join(broken.root, 'inner/file'), 'unique nonempty tree');
    const brokenCommit = await commitFiles({ root: broken.root });
    const oid = await git({ root: broken.root, args: ['rev-parse', object] });
    await unlink(join(broken.root, '.git/objects', oid.slice(0, 2), oid.slice(2)));
    await mustFail({ root: broken.root, sourceCommit: brokenCommit });
  }
});

async function addSubmodule({ root, source, path }: { root: string; source: string; path: string }): Promise<string> {
  await repository({ root: source });
  await writeFile(join(source, 'source'), 'child source');
  await commitFiles({ root: source });
  await git({ root, args: ['-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', source, path] });
  await commitFiles({ root });
  return join(root, path);
}

test('recursive submodules bind content and require the recorded Git link at every level', async () => {
  await using temporary = await temporaryDirectory();
  const root = join(temporary.root, 'repo');
  await publicationRepository({ root });
  const child = await addSubmodule({ root, source: join(temporary.root, 'child'), path: 'vendor/child' });
  const nested = await addSubmodule({ root: child, source: join(temporary.root, 'grandchild'), path: 'nested' });
  const originalCommit = await commitFiles({ root });
  const expected = await digest({ root });
  const clone = join(temporary.root, 'clone');
  await git({ root, args: ['clone', '-q', '--no-hardlinks', root, clone] });
  await git({ root: clone, args: ['config', 'submodule.vendor/child.url', child] });
  await git({ root: clone, args: ['-c', 'protocol.file.allow=always', 'submodule', 'update', '--init', '--recursive'] });
  assert.equal(await digest({ root: clone }), expected);
  await writeFile(join(nested, 'source'), 'changed grandchild');
  await commitFiles({ root: nested });
  await mustFail({ root, sourceCommit: originalCommit });
  await commitFiles({ root: child });
  await mustFail({ root, sourceCommit: originalCommit });
  const sourceCommit = await commitFiles({ root });
  assert.notEqual(await digest({ root }), expected);
  const blob = await git({ root: nested, args: ['rev-parse', 'HEAD:source'] });
  const gitDir = await git({ root: nested, args: ['rev-parse', '--absolute-git-dir'] });
  await unlink(join(gitDir, 'objects', blob.slice(0, 2), blob.slice(2)));
  await mustFail({ root, sourceCommit });
  await rm(child, { recursive: true });
  await mkdir(child);
  await mustFail({ root, sourceCommit });
});

test('non-UTF-8 gitlink paths remain reproducible and still require the recorded checkout', async () => {
  await using temporary = await temporaryDirectory();
  const root = join(temporary.root, 'repo');
  await publicationRepository({ root });
  const child = join(temporary.root, 'child');
  await repository({ root: child });
  await writeFile(join(child, 'content'), 'committed child input');
  await commitFiles({ root: child });
  const rawPath = Buffer.concat([Buffer.from(`${root}/child-`), Buffer.from([255, 254])]);
  await rename(child, rawPath);
  const sourceCommit = await commitFiles({ root });
  await digest({ root });
  await rename(rawPath, child);
  await symlink(child, rawPath);
  await mustFail({ root, sourceCommit });
});

test('the inline subshell leaves caller directory, options, variables and functions unchanged', async () => {
  await using temporary = await temporaryDirectory();
  await repository({ root: temporary.root });
  await runProcess({
    command: 'bash',
    args: ['-c', `\
before_pwd=$PWD
before_flags=$-
before_locale=\${LC_ALL-unset}
before_lazy=\${GIT_NO_LAZY_FETCH-unset}
source "$1" >/dev/null || exit
test "$PWD" = "$before_pwd" || exit
test "$-" = "$before_flags" || exit
test "\${LC_ALL-unset}" = "$before_locale" || exit
test "\${GIT_NO_LAZY_FETCH-unset}" = "$before_lazy" || exit
! declare -F fingerprint >/dev/null
`, 'test', script],
    cwd: temporary.root,
    env: gitEnvironment({ ceiling: undefined }),
  });
});

test('readonly caller controls and function conflicts fail closed even in a conditional', async () => {
  await using temporary = await temporaryDirectory();
  await repository({ root: temporary.root });
  for (const setup of [
    'readonly GIT_NO_REPLACE_OBJECTS=0',
    'fingerprint() { printf "%064d\\n" 0; }; readonly -f fingerprint',
  ]) {
    await assert.rejects(runProcess({
      command: 'bash',
      args: ['-c', `${setup}; if source "$1"; then exit 0; else exit $?; fi`, 'test', script],
      cwd: temporary.root,
      env: gitEnvironment({ ceiling: undefined }),
    }), (error: unknown) => error instanceof ProcessError && error.exitCode !== 0 && error.stdout.length === 0);
  }
});
