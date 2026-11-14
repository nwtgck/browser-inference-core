import { mkdtemp, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gitEnvironment } from '#tools/source/capture';
import { runProcess } from '#tools/shared/process';

export async function temporaryDirectory(): Promise<{ root: string; [Symbol.asyncDispose](): Promise<void> }> {
  const root = await mkdtemp(join(tmpdir(), 'bicore-tools-test-'));
  return {
    root,
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
}

export async function git({ root, args }: { root: string; args: readonly string[] }): Promise<string> {
  return (await runProcess({ command: 'git', args: ['-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', '-C', root, ...args], cwd: root, env: gitEnvironment({ ceiling: undefined }) })).stdout.toString('utf8').trimEnd();
}

export async function initializeRepository({ root }: { root: string }): Promise<void> {
  await mkdir(root, { recursive: true });
  await git({ root, args: ['init', '-q'] });
}

export async function commitFiles({ root }: { root: string }): Promise<string> {
  await git({ root, args: ['add', '-A'] });
  await git({ root, args: ['commit', '--allow-empty', '-qm', 'fixture'] });
  return git({ root, args: ['rev-parse', 'HEAD'] });
}
