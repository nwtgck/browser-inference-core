import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { runProcess } from '#tools/shared/process';
import type { Upstreams } from '#dto/speech-cpp';
import type { SourceRole } from '#runtimes/speech-cpp/tools/prepare';
import { observeSource } from '#tools/source/capture';
import { runtimeDirectory } from '#runtimes/speech-cpp/tools/config';

export async function git({ root, args }: { root: string; args: string[] }): Promise<string> {
  return (await runProcess({ command: 'git', args: ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', ...args], cwd: root })).stdout.toString('utf8').trim();
}

export async function write({ root, name, text }: { root: string; name: string; text: string }): Promise<void> {
  const destination = path.join(root, name);
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, text);
}

export async function sourceFixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'bicore-speech-source-'));
  const repository = path.join(root, 'repository');
  await mkdir(repository);
  await git({ root: repository, args: ['init', '-q'] });
  const declarations = {
    'speech': { path: 'vendor/speech', repository: 'fixture/speech' },
    ggml: { path: 'vendor/ggml', repository: 'fixture/ggml' },
  };
  const sources: Record<SourceRole, string> = {
    'speech': path.join(repository, runtimeDirectory, declarations['speech'].path),
    ggml: path.join(repository, runtimeDirectory, declarations.ggml.path),
  };
  const commits: Record<SourceRole, string> = { 'speech': '', ggml: '' };
  for (const role of ['speech', 'ggml'] as const) {
    await mkdir(sources[role], { recursive: true });
    await git({ root: sources[role], args: ['init', '-q'] });
    await write({ root: sources[role], name: role === 'ggml' ? 'include/ggml.h' : 'a.txt', text: role === 'ggml' ? '// real selected header\n' : 'one\nbefore\n' });
    await git({ root: sources[role], args: ['add', '-A'] });
    await git({ root: sources[role], args: ['commit', '-qm', 'upstream fixture'] });
    commits[role] = await git({ root: sources[role], args: ['rev-parse', 'HEAD'] });
    await git({ root: repository, args: ['update-index', '--add', '--cacheinfo', `160000,${commits[role]},${runtimeDirectory}/${declarations[role].path}`] });
  }
  await write({ root: repository, name: `${runtimeDirectory}/config/upstreams.json`, text: JSON.stringify(declarations) });
  await write({ root: repository, name: `${runtimeDirectory}/config/profiles.json`, text: JSON.stringify({ 'webgpu-wasm64-jspi': { memory64: true, webgpu: true, jspi: true, asyncify: false, pthreads: false, maximumMemory: 17179869184 } }) });
  await write({ root: repository, name: `${runtimeDirectory}/config/variants.json`, text: JSON.stringify({ browser: { assertions: 0, environment: 'web,worker' } }) });
  await git({ root: repository, args: ['add', `${runtimeDirectory}/config`] });
  await git({ root: repository, args: ['commit', '-qm', 'parent fixture'] });
  const revision = await git({ root: repository, args: ['rev-parse', 'HEAD'] });
  const patchRoot = path.join(root, 'patches');
  await mkdir(patchRoot);
  const patch = `\
diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,2 @@
 one
-before
+after
`;
  await write({ root: patchRoot, name: 'fix.patch', text: patch });
  const upstreams: Upstreams = {
    'speech': { ...declarations['speech'], commit: commits['speech'] },
    ggml: { ...declarations.ggml, commit: commits.ggml },
  };
  return {
    root,
    repository,
    revision,
    source: await observeSource({ checkoutRoot: repository, mode: { kind: 'release', expectedCommit: revision } }),
    sources,
    upstreams,
    declarations,
    patchRoot,
    output: path.join(root, 'prepared'),
    recipe: [{ file: 'fix.patch', target: 'speech', reason: 'Fixture source change', files: ['a.txt'] }],
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
}
