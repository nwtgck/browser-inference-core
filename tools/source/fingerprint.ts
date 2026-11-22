import type { TreeFingerprint } from '#dto/shared';
import { gitBlob, localGit, localGitEnvironment } from '#tools/publication/git';
import { runProcess } from '#tools/shared/process';

/** Run exactly the committed, independently reproducible recipe we publish. */
export async function captureTreeFingerprint({ sourceRoot, sourceCommit }: { sourceRoot: string; sourceCommit: string }): Promise<TreeFingerprint> {
  async function checkHead(): Promise<void> {
    if ((await localGit({ cwd: sourceRoot, args: ['rev-parse', '--verify', 'HEAD'] })).toString('utf8').trim() !== sourceCommit) {
      throw new Error('Tree fingerprint source differs from selected commit');
    }
  }
  await checkHead();
  const reproduce = new TextDecoder('utf-8', { fatal: true }).decode(await gitBlob({ work: sourceRoot, tree: sourceCommit, path: '.github/scripts/treefingerprint.sh' }));
  const environment = localGitEnvironment();
  const result = await runProcess({ command: 'bash', args: ['--noprofile', '--norc', '-c', reproduce], cwd: sourceRoot, env: { ...environment, overrides: { ...environment.overrides, BASH_ENV: undefined, ENV: undefined } } });
  const value = result.stdout.toString('utf8');
  if (!/^[a-f0-9]{64}\n$/u.test(value)) {
    throw new Error('Invalid committed tree fingerprint output');
  }
  await checkHead();
  return { algorithm: 'sha256', scheme: 'git-committed-tree-sha256-v1', value: value.trimEnd(), reproduce };
}
