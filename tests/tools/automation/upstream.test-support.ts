import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createUpdateRequest, upstreamPath } from '#tools/automation/upstream/operations';
import { planUpdate, selectUpstream } from '#tools/automation/upstream/plan';
import { commitFiles, git, initializeRepository } from '#tests/tools/shared/fixtures.test-support';

/** Both direct-operation and literal-workflow tests use the same real source layout. */
export async function createUpdaterFixture({ root }: { root: string }) {
  const checkoutRoot = join(root, 'source');
  const upstreamRoot = join(root, 'upstream');
  await initializeRepository({ root: upstreamRoot });
  await writeFile(join(upstreamRoot, 'source'), 'original\n');
  const previous = await commitFiles({ root: upstreamRoot });
  await writeFile(join(upstreamRoot, 'source'), 'updated\n');
  const target = await commitFiles({ root: upstreamRoot });
  await initializeRepository({ root: checkoutRoot });
  const runtime = join(checkoutRoot, 'runtimes/llama-cpp');
  await mkdir(join(runtime, 'config'), { recursive: true });
  await writeFile(join(runtime, 'config/toolchain.json'), '{"artifactBranch":"artifacts"}\n');
  await git({ root: checkoutRoot, args: ['-c', 'protocol.file.allow=always', 'submodule', 'add', upstreamRoot, upstreamPath] });
  await git({ root: join(checkoutRoot, upstreamPath), args: ['checkout', '--detach', previous] });
  const source = await commitFiles({ root: checkoutRoot });
  const request = await createUpdateRequest({ checkoutRoot, repository: 'example/core', base: 'feature/base' });
  const selection = selectUpstream({ target: { kind: 'custom', ref: target }, releases: [] });
  const plan = planUpdate({ request, selection, resolvedCommit: target, resolvedRef: target, relation: 'ahead', allowNonFastForward: false });
  assert.equal(plan.status, 'planned');
  if (plan.status !== 'planned') {
    throw new Error('Expected an update fixture');
  }
  return { checkoutRoot, upstreamRoot, previous, target, source, request, plan };
}
