import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { localGit, localGitEnvironment, verifyGitTree } from '#tools/publication/git';
import { freezePackage } from '#tools/publication/artifacts';
import { packageFiles } from '#tools/package/files';
import { write, sourceRoot, fixtureSource } from '#tests/tools/package/fixtures.test-support';
import { runProcess } from '#tools/shared/process';
import { stubEnvironment } from '#tests/tools/shared/node-env.test-support';

test('streaming Git verification accepts empty and multichunk blobs and ignores later worktree mutations', async () => {
  await using temporary = await temporaryDirectory();
  const work = temporary.root;
  await write({ path: join(work, 'empty'), data: '' });
  await write({ path: join(work, 'unicode-資料 space.txt'), data: Buffer.alloc(2 * 1024 * 1024 + 37, 255) });
  const files = await packageFiles({ root: work, exclude: [] });
  await localGit({ cwd: work, args: ['init', '-q'] });
  await localGit({ cwd: work, args: ['add', '--all'] });
  const tree = (await localGit({ cwd: work, args: ['write-tree'] })).toString('utf8').trim();
  await verifyGitTree({ work, tree, files });
  await writeFile(join(work, 'unicode-資料 space.txt'), 'Later worktree data');
  await verifyGitTree({ work, tree, files });
  await localGit({ cwd: work, args: ['add', '--all'] });
  const changed = (await localGit({ cwd: work, args: ['write-tree'] })).toString('utf8').trim();
  await assert.rejects(verifyGitTree({ work, tree: changed, files }), /Git blob/u);
});

test('Git transformations, ignored omissions, extra blobs and symbolic-link tree modes cannot bypass byte validation', async t => {
  for (const transformation of ['line_endings', 'ignored', 'extra', 'symlink'] as const) {
    await t.test(transformation, async () => {
      await using temporary = await temporaryDirectory();
      const work = temporary.root;
      await write({ path: join(work, 'README.md'), data: 'Original bytes\r\n' });
      const files = await packageFiles({ root: work, exclude: [] });
      await localGit({ cwd: work, args: ['init', '-q'] });
      switch (transformation) {
      case 'line_endings': await localGit({ cwd: work, args: ['config', 'core.autocrlf', 'true'] });
        break;
      case 'ignored': await write({ path: join(work, '.git/info/exclude'), data: 'README.md\n' });
        break;
      case 'extra': await write({ path: join(work, 'unlisted'), data: 'Extra blob' });
        break;
      case 'symlink': await rm(join(work, 'README.md'));
        await symlink('somewhere', join(work, 'README.md'));
        break;
      }
      await localGit({ cwd: work, args: ['add', '--all'] });
      const tree = (await localGit({ cwd: work, args: ['write-tree'] })).toString('utf8').trim();
      await assert.rejects(verifyGitTree({ work, tree, files }), /Git (blob|tree)|publication Git tree/u);
    });
  }
});

test('publication Git environment excludes caller overrides and preserves unrelated unknown values', async () => {
  const values = { GIT_DIR: '/unrelated', GIT_WORK_TREE: '/unrelated', GIT_INDEX_FILE: '/unrelated/index', GIT_AUTHOR_NAME: 'Caller', GIT_COMMITTER_NAME: 'Caller', GIT_CONFIG_COUNT: '1', GIT_CONFIG_KEY_0: 'core.autocrlf', GIT_CONFIG_VALUE_0: 'true', GIT_CONFIG_GLOBAL: '/untrusted-config' };
  using environment = stubEnvironment({ values: { ...values, GIT_FUTURE_OVERRIDE: 'untrusted', BICORE_UNRELATED_TRANSFER: 'preserved' } });
  const names = [...Object.keys(values), 'GIT_FUTURE_OVERRIDE', 'BICORE_UNRELATED_TRANSFER', 'GIT_CONFIG_NOSYSTEM', 'GIT_ALLOW_PROTOCOL', 'GIT_NO_LAZY_FETCH', 'GIT_NO_REPLACE_OBJECTS', 'GIT_TERMINAL_PROMPT'];
  const result = await runProcess({ command: process.execPath, args: ['-e', `const fs=require('node:fs');for(const name of ${JSON.stringify(names)})fs.writeSync(1,name+'='+String(process.env[name])+'\\n');`], cwd: sourceRoot, env: localGitEnvironment() });
  const transferred = new Map(result.stdout.toString('utf8').trim().split('\n').map(line => {
    const separator = line.indexOf('=');
    return [line.slice(0, separator), line.slice(separator + 1)];
  }));
  for (const name of Object.keys(values).filter(name => name !== 'GIT_CONFIG_GLOBAL')) {
    assert.equal(transferred.get(name), 'undefined');
  }
  assert.equal(transferred.get('GIT_FUTURE_OVERRIDE'), 'undefined');
  assert.equal(transferred.get('BICORE_UNRELATED_TRANSFER'), 'preserved');
  assert.equal(transferred.get('GIT_CONFIG_GLOBAL'), '/dev/null');
  assert.equal(transferred.get('GIT_CONFIG_NOSYSTEM'), '1');
  assert.equal(transferred.get('GIT_ALLOW_PROTOCOL'), '');
  assert.equal(transferred.get('GIT_NO_LAZY_FETCH'), '1');
  assert.equal(transferred.get('GIT_NO_REPLACE_OBJECTS'), '1');
  assert.equal(transferred.get('GIT_TERMINAL_PROMPT'), '0');
});

test('freeze rejects source branches, overlapping, existing and parent-linked state before verification or writes', async () => {
  await using temporary = await temporaryDirectory();
  const root = temporary.root;
  const packageDirectory = join(root, 'package');
  let checks = 0;
  const options = {
    sourceRoot,
    packageDirectory,
    work: join(root, 'snapshot'),
    receipt: join(root, 'frozen.json'),
    branch: 'artifacts',
    expectedSource: fixtureSource,
    verifySource: async () => {
      checks++;
      throw new Error('Must reject before source verification');
    },
  };
  await assert.rejects(freezePackage({ ...options, branch: 'main' }), /source branch/u);
  await assert.rejects(freezePackage({ ...options, work: join(packageDirectory, 'work') }), /outside/u);
  await assert.rejects(freezePackage({ ...options, receipt: join(packageDirectory, 'frozen.json') }), /outside/u);
  await assert.rejects(freezePackage({ ...options, receipt: root }), /outside/u);
  await write({ path: options.receipt, data: 'keep receipt' });
  await assert.rejects(freezePackage(options), /fresh/u);
  assert.equal(await readFile(options.receipt, 'utf8'), 'keep receipt');
  await rm(options.receipt);
  await mkdir(join(root, 'target'));
  await symlink(join(root, 'target'), join(root, 'linked'));
  await assert.rejects(freezePackage({ ...options, work: join(root, 'linked/work') }), /Linked/u);
  assert.equal(checks, 0);
});
