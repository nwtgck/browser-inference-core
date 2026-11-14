import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { runProcess } from '#tools/shared/process';
import type { Upstreams } from '#dto/stable-diffusion-cpp';
import type { SourceRole } from '#runtimes/stable-diffusion-cpp/tools/prepare';
import { observeSource } from '#tools/source/capture';
import { runtimeDirectory } from '#runtimes/stable-diffusion-cpp/tools/config';

export async function git({ root, args }: { root: string; args: string[] }): Promise<string> {
  return (await runProcess({ command: 'git', args: ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', ...args], cwd: root })).stdout.toString('utf8').trim();
}

export async function write({ root, name, text }: { root: string; name: string; text: string }): Promise<void> {
  const destination = path.join(root, name);
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, text);
}

export async function sourceFixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'bicore-sd-source-'));
  const repository = path.join(root, 'repository');
  await mkdir(repository);
  await git({ root: repository, args: ['init', '-q'] });
  const declarations = {
    'stable-diffusion': { path: 'vendor/sd', repository: 'fixture/sd' },
    ggml: { path: 'vendor/ggml', repository: 'fixture/ggml', sourceSubdirectory: 'ggml' },
  };
  const sources: Record<SourceRole, string> = {
    'stable-diffusion': path.join(repository, runtimeDirectory, declarations['stable-diffusion'].path),
    ggml: path.join(repository, runtimeDirectory, declarations.ggml.path),
  };
  const commits: Record<SourceRole, string> = { 'stable-diffusion': '', ggml: '' };
  for (const role of ['stable-diffusion', 'ggml'] as const) {
    await mkdir(sources[role], { recursive: true });
    await git({ root: sources[role], args: ['init', '-q'] });
    await write({ root: sources[role], name: role === 'ggml' ? 'ggml/include/ggml.h' : 'a.txt', text: role === 'ggml' ? '// real selected header\n' : 'one\nbefore\n' });
    await git({ root: sources[role], args: ['add', '-A'] });
    await git({ root: sources[role], args: ['commit', '-qm', 'upstream fixture'] });
    commits[role] = await git({ root: sources[role], args: ['rev-parse', 'HEAD'] });
    await git({ root: repository, args: ['update-index', '--add', '--cacheinfo', `160000,${commits[role]},${runtimeDirectory}/${declarations[role].path}`] });
  }
  await write({ root: repository, name: `${runtimeDirectory}/config/upstreams.json`, text: JSON.stringify(declarations) });
  await write({ root: repository, name: `${runtimeDirectory}/config/profiles.json`, text: JSON.stringify({ 'webgpu-wasm32-jspi': { memory64: false, webgpu: true, jspi: true, asyncify: false, pthreads: false, maximumMemory: 4294967296 } }) });
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
    'stable-diffusion': { ...declarations['stable-diffusion'], commit: commits['stable-diffusion'] },
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
    recipe: [{ file: 'fix.patch', target: 'stable-diffusion', reason: 'Fixture source change', files: ['a.txt'] }],
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
}
