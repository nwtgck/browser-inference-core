import { randomBytes } from 'node:crypto';
import { lstat, mkdtemp, readdir, realpath, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import type { BrowserSession, FileEntry, FileIdentity, RuntimeName } from '#dto/shared';
import type { RuntimeManifest, RuntimePackageInput, RuntimeValidation } from '#tools/package/runtime';
import { packageFiles, assertSame, bytesIdentity, readMetadataBytes } from '#tools/package/files';
import { directory, fileIdentity, isContained, treeIdentity } from '#tools/shared/files';
import { parseJson } from '#tools/shared/json';
import { replaceFileUnchanged } from '#tools/shared/output';
import type { ChildEnvironment, runProcess } from '#tools/shared/process';
import { withValidationManifest, manifestIo, type ManifestIo } from './manifest.js';

export type SmokeResult = { profile: string; variant: string; passed: boolean };
export type ExecutionSettings = { selectedSourceCommit?: string | undefined; runId?: string | undefined; runAttempt?: string | undefined; job?: string | undefined };
type TestIdentity = FileIdentity & { executable: boolean };
type TreeMetadata = Record<string, { kind: 'directory' } | { kind: 'file'; executable: boolean }>;
export type ValidationReceipt<Runtime extends RuntimeName, Result extends SmokeResult> = BrowserSession & {
  runtime: Runtime;
  status: 'complete';
  finalSnapshotSha256: string;
  finalManifestSha256: string;
  nodeAsyncifyPassed?: boolean;
  results: Result[];
};
export type SmokePlan<Manifest extends RuntimeManifest, Result extends SmokeResult> = {
  validated: RuntimeValidation<Manifest>;
  profiles: readonly string[];
  variants: readonly string[];
  testFiles: Readonly<Record<string, string>>;
  generateFixture: (({ output }: { output: string }) => Promise<void>) | undefined;
  nodeAsyncify: boolean;
  testWebGpu: boolean;
  decodeEnvelope: ({ input }: { input: unknown }) => { session: BrowserSession; results: Result[] };
  checkedResults: ({ results }: { results: readonly Result[] }) => void;
  updateManifest: ({ manifest, results }: { manifest: Manifest; results: readonly Result[] }) => Manifest;
  finalVerify: ({ input }: { input: RuntimePackageInput<Manifest> }) => Promise<RuntimeValidation<Manifest>>;
};

function snapshotDigest({ files, metadata }: { files: ReadonlyMap<string, FileEntry>; metadata: TreeMetadata }): string {
  const observed = Object.entries(metadata).sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0).map(([name, entry]) => {
    switch (entry.kind) {
    case 'directory': return [name, entry];
    case 'file': {
      const file = files.get(name);
      if (file === undefined) {
        throw new Error(`Missing snapshot file: ${name}`);
      }
      return [name, { ...entry, bytes: file.bytes, sha256: file.sha256 }];
    }
    default: {
      const exhaustive: never = entry;
      throw new Error(String(exhaustive));
    }
    }
  });
  return bytesIdentity({ data: Buffer.from(JSON.stringify(observed)) }).sha256;
}

async function observedFile({ file }: { file: string }): Promise<TestIdentity> {
  return { ...await fileIdentity({ path: file }), executable: ((await lstat(file)).mode & 0o111) !== 0 };
}

// These private observations preserve filesystem snapshot behavior. They are
// neither package payload declarations nor another persisted DTO protocol.
async function treeMetadata({ root }: { root: string }): Promise<TreeMetadata> {
  const result = new Map<string, TreeMetadata[string]>();
  const visit = async ({ relative }: { relative: string }): Promise<void> => {
    for (const name of await readdir(path.join(root, relative))) {
      const entry = relative === '' ? name : `${relative}/${name}`;
      const stats = await lstat(path.join(root, entry));
      if (stats.isDirectory()) {
        result.set(entry, { kind: 'directory' });
        await visit({ relative: entry });
      } else if (stats.isFile()) {
        result.set(entry, { kind: 'file', executable: (stats.mode & 0o111) !== 0 });
      } else {
        throw new Error(`Linked or non-regular smoke input: ${entry}`);
      }
    }
  };
  await visit({ relative: '' });
  return Object.fromEntries(result);
}

export async function testIdentity({ files, fixturePath }: { files: Readonly<Record<string, string>>; fixturePath: string | undefined }): Promise<Record<string, TestIdentity>> {
  const result: Record<string, TestIdentity> = {};
  for (const [name, file] of Object.entries(files)) {
    result[name] = await observedFile({ file });
  }
  if (fixturePath !== undefined) {
    result['synthetic-model.gguf'] = await observedFile({ file: fixturePath });
  }
  return result;
}

export function executionIdentity<Runtime extends RuntimeName>({ runtime, sourceCommit, settings }: { runtime: Runtime; sourceCommit: string; settings: ExecutionSettings }): Omit<BrowserSession, 'snapshotSha256' | 'testInputsSha256' | 'testWebGpu'> & { runtime: Runtime } {
  if (!/^[a-f0-9]{40}$/u.test(sourceCommit)) {
    throw new Error('Invalid source commit');
  }
  if ((settings.selectedSourceCommit ?? sourceCommit) !== sourceCommit) {
    throw new Error('Package does not match selected source commit');
  }
  const runId = settings.runId ?? 'local';
  const runAttempt = settings.runAttempt ?? '1';
  const job = settings.job ?? 'local';
  if ((runId !== 'local' && !/^[0-9]{1,32}$/u.test(runId)) || !/^[1-9][0-9]{0,8}$/u.test(runAttempt) || !/^[A-Za-z0-9_-]{1,100}$/u.test(job)) {
    throw new Error('Invalid run identity');
  }
  return { schemaVersion: 1, runtime, sourceCommit, runId, runAttempt, job, sessionId: randomBytes(16).toString('hex') };
}

export async function clearValidationReceipt({ packageDirectory, receiptPath }: { packageDirectory: string; receiptPath: string }): Promise<void> {
  await directory({ path: packageDirectory });
  if (isContained({ root: packageDirectory, path: receiptPath })) {
    throw new Error('Validation receipts must stay outside the package');
  }
  // Check parents before unlinking; a linked outside parent must not reach the package.
  await directory({ path: path.dirname(receiptPath), create: true });
  await rm(receiptPath, { force: true });
}

export async function validateSnapshot<Manifest extends RuntimeManifest, Result extends SmokeResult>({ checkoutRoot, receiptPath, settings, childEnvironment, runProcess: run, plan, io }: { checkoutRoot: string; receiptPath: string; settings: ExecutionSettings; childEnvironment: ChildEnvironment | undefined; runProcess: typeof runProcess; plan: SmokePlan<Manifest, Result>; io: ManifestIo }): Promise<ValidationReceipt<Manifest['runtime'], Result>> {
  const input = plan.validated.input;
  const directory = input.directory;
  const manifestPath = path.join(directory, 'manifest.json');
  const before = await readMetadataBytes({ path: manifestPath });
  assertSame({ actual: bytesIdentity({ data: before }), expected: input.manifestIdentity, message: 'Manifest changed while capturing test snapshot' });
  assertSame({ actual: Object.keys(input.manifest.profiles).sort(), expected: [...plan.profiles].sort(), message: 'Smoke requires every configured profile' });
  for (const profile of Object.values(input.manifest.profiles)) {
    assertSame({ actual: Object.keys(profile.variants).sort(), expected: [...plan.variants].sort(), message: 'Smoke requires every configured variant' });
  }
  const identity = executionIdentity({ runtime: input.manifest.runtime, sourceCommit: input.manifest.sourceCommit, settings });
  const snapshot = new Map(plan.validated.files);
  const metadata = await treeMetadata({ root: directory });
  const implementation = await testIdentity({ files: plan.testFiles, fixturePath: undefined });
  // Canonicalize only our newly owned directory, never untrusted payload paths.
  const temporary = await realpath(await mkdtemp(path.join(tmpdir(), 'bic-smoke-')));
  try {
    let fixturePath: string | undefined;
    if (plan.generateFixture !== undefined) {
      fixturePath = path.join(temporary, 'fixture.gguf');
      await plan.generateFixture({ output: fixturePath });
    }
    assertSame({ actual: await testIdentity({ files: plan.testFiles, fixturePath: undefined }), expected: implementation, message: 'Test implementation changed while generating fixture' });
    const tests = await testIdentity({ files: plan.testFiles, fixturePath });
    const session: BrowserSession & { runtime: Manifest['runtime'] } = { ...identity, snapshotSha256: snapshotDigest({ files: snapshot, metadata }), testInputsSha256: treeIdentity({ files: tests }).sha256, testWebGpu: plan.testWebGpu };
    const assertSnapshot = async ({ expected }: { expected: ReadonlyMap<string, FileEntry> }): Promise<void> => {
      assertSame({ actual: await packageFiles({ root: directory, exclude: [] }), expected, message: 'Tested package snapshot changed' });
      assertSame({ actual: await treeMetadata({ root: directory }), expected: metadata, message: 'Tested package snapshot changed its filesystem metadata' });
      assertSame({ actual: await testIdentity({ files: plan.testFiles, fixturePath }), expected: tests, message: 'Test implementation or fixture changed' });
    };
    await assertSnapshot({ expected: snapshot });
    const resultsPath = path.join(temporary, 'results.json');
    const overrides = { BIC_BROWSER_RESULTS_FILE: resultsPath, BIC_BROWSER_SESSION_JSON: JSON.stringify(session) };
    let env: ChildEnvironment;
    switch (childEnvironment?.kind) {
    case undefined: env = { kind: 'inherit', removePrefixes: [], overrides };
      break;
    case 'inherit': env = { ...childEnvironment, overrides: { ...childEnvironment.overrides, ...overrides } };
      break;
    case 'replace': env = { ...childEnvironment, values: { ...childEnvironment.values, ...overrides } };
      break;
    default: {
      const exhaustive: never = childEnvironment;
      throw new Error(`Unhandled validation child environment: ${String(exhaustive)}`);
    }
    }
    const cwd = path.join(checkoutRoot, 'runtimes', input.manifest.runtime);
    const harness = path.join(checkoutRoot, 'tests/runtimes', input.manifest.runtime.replaceAll('-', '_'));
    if (plan.nodeAsyncify) {
      await run({ command: process.execPath, args: [path.join(harness, 'asyncify-rewind.mjs'), path.join(directory, 'profiles/webgpu-wasm32-asyncify/test')], cwd, env, capture: 'file' });
      await assertSnapshot({ expected: snapshot });
    }
    await run({ command: process.execPath, args: [path.join(harness, 'browser-smoke.mjs'), directory, ...(fixturePath === undefined ? [] : [fixturePath])], cwd, env, capture: 'file' });
    await assertSnapshot({ expected: snapshot });
    const envelope = parseJson({ text: (await readMetadataBytes({ path: resultsPath })).toString('utf8'), decode: plan.decodeEnvelope });
    assertSame({ actual: envelope.session, expected: session, message: 'Browser result is not bound to this test session' });
    plan.checkedResults({ results: envelope.results });
    const manifest = plan.updateManifest({ manifest: input.manifest, results: envelope.results });
    const finalBytes = Buffer.from(JSON.stringify(manifest, undefined, 2) + '\n');
    const manifestIdentity = bytesIdentity({ data: finalBytes });
    const expected = new Map(snapshot);
    expected.set('manifest.json', { path: 'manifest.json', ...manifestIdentity });
    await withValidationManifest({
      manifestPath,
      before,
      after: finalBytes,
      io,
      finish: async () => {
        await assertSnapshot({ expected });
        await plan.finalVerify({ input: { ...input, manifest, manifestIdentity } });
        await assertSnapshot({ expected });
      },
    });
    const receipt: ValidationReceipt<Manifest['runtime'], Result> = {
      ...session,
      status: 'complete',
      finalSnapshotSha256: snapshotDigest({ files: expected, metadata }),
      finalManifestSha256: manifestIdentity.sha256,
      ...(plan.nodeAsyncify ? { nodeAsyncifyPassed: true } : {}),
      results: envelope.results,
    };
    const candidate = path.join(temporary, 'receipt.json');
    await writeFile(candidate, JSON.stringify(receipt, undefined, 2) + '\n');
    await replaceFileUnchanged({ candidate, output: receiptPath });
    return receipt;
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

export { manifestIo };
