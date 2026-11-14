import assert from 'node:assert/strict';
import { stubEnvironment } from '#tests/tools/shared/node-env.test-support';
import { chmod, lstat, mkdir, readFile, realpath, rm, symlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileIdentity } from '#tools/shared/files';
import { ProcessError, type runProcess } from '#tools/shared/process';
import { validateSavedRuntimePackage } from '#tools/validation/runtime';
import { clearValidationReceipt } from '#tools/validation/snapshot';
import { imageResults, llamaResults, readFixtureManifest, validationFixture } from './fixtures.test-support.js';

test('both runtimes bind the fresh session, smoke every configured pair and pack exactly once', async () => {
  for (const runtime of ['llama-cpp', 'stable-diffusion-cpp'] as const) {
    await using fixture = await validationFixture({ runtime });
    const manifestPath = path.join(fixture.packageDirectory, 'manifest.json');
    const before = await readFixtureManifest({ fixture });
    before.futureMetadata = { preserved: ['value'] };
    await writeFile(manifestPath, JSON.stringify(before));
    await chmod(manifestPath, 0o640);
    await writeFile(fixture.receiptPath, 'stale success');
    const receipt = await fixture.validate();
    assert.equal(receipt.runtime, runtime);
    assert.equal(receipt.results.length, fixture.profiles.length * fixture.variants.length);
    assert.equal(fixture.finalCalls, 1);
    assert.deepEqual(fixture.commands.map((command) => command.phase), runtime === 'llama-cpp' ? ['node_asyncify', 'chromium_smoke'] : ['chromium_smoke']);
    const after = await readFixtureManifest({ fixture });
    assert.equal((await lstat(manifestPath)).mode & 0o777, 0o640);
    for (const profile of fixture.profiles) {
      for (const variant of fixture.variants) {
        const provenance = after.profiles[profile]?.variants[variant];
        assert.ok(provenance);
        assert.equal(provenance.validation.browserSmoke, true);
        const previous = before.profiles[profile]?.variants[variant];
        assert.ok(previous);
        provenance.validation = previous.validation;
      }
    }
    assert.deepEqual(after, before);
    assert.deepEqual(JSON.parse(await readFile(fixture.receiptPath, 'utf8')), receipt);
    const identity = await fileIdentity({ path: manifestPath });
    assert.equal(receipt.finalManifestSha256, identity.sha256);
  }
});

test('snapshot evidence rejects replayed session fields, malformed flags and incomplete results before final pack', async () => {
  await using fixture = await validationFixture({ runtime: 'stable-diffusion-cpp' });
  const manifestPath = path.join(fixture.packageDirectory, 'manifest.json');
  const original = await readFile(manifestPath);
  const changes: NonNullable<typeof fixture.resultMutator>[] = [
    ({ session, results }) => ({ session: { ...session, sessionId: '0'.repeat(32) }, results }),
    ({ session, results }) => ({ session: { ...session, runId: '122' }, results }),
    ({ session, results }) => ({ session: { ...session, runAttempt: '1' }, results }),
    ({ session, results }) => ({ session: { ...session, sourceCommit: 'b'.repeat(40) }, results }),
    ({ session, results }) => ({ session: { ...session, testWebGpu: 'false' }, results }),
    ({ session, results }) => ({ session, results: results.slice(1) }),
    ({ session, results }) => ({ session, results: results.map(() => results[0]) }),
    ({ session, results }) => ({ session, results: [...results, results[0]] }),
    ({ session, results }) => ({ session, results: results.map((result) => ({ ...result, passed: 'true' })) }),
    ({ session, results }) => ({ session, results: results.map((result) => ({ ...result, passed: 1 })) }),
    ({ session, results }) => ({ session, results: results.map((result) => result.variant === 'test' ? { ...result, graphWalk: false } : result) }),
    ({ session, results }) => ({ session, results: results.map((result) => result.variant === 'test' ? { ...result, webgpuPerformance: [] } : result) }),
  ];
  for (const resultMutator of changes) {
    fixture.resultMutator = resultMutator;
    await writeFile(fixture.receiptPath, 'old success');
    await assert.rejects(fixture.validate());
    assert.equal(fixture.finalCalls, 0);
    await assert.rejects(readFile(fixture.receiptPath), { code: 'ENOENT' });
    assert.deepEqual(await readFile(manifestPath), original);
  }
});

test('same-sized payload changes, rewritten manifests and every indirect harness input invalidate the snapshot', async () => {
  for (const mutation of ['payload', 'payload-mode', 'empty-directory', 'manifest', 'manifest-rehash', 'contract-json', 'contract-module', 'arithmetic', 'file-mount'] as const) {
    const runtime = mutation === 'file-mount' ? 'llama-cpp' : 'stable-diffusion-cpp';
    await using fixture = await validationFixture({ runtime });
    const manifestPath = path.join(fixture.packageDirectory, 'manifest.json');
    fixture.afterProcess = async () => {
      switch (mutation) {
      case 'payload-mode': {
        const manifest = await readFixtureManifest({ fixture });
        const payload = manifest.files.find((file) => file.path.endsWith('core.wasm'));
        assert.ok(payload);
        const target = path.join(fixture.packageDirectory, payload.path);
        await chmod(target, ((await lstat(target)).mode & 0o777) ^ 0o111);
        break;
      }
      case 'empty-directory': await mkdir(path.join(fixture.packageDirectory, 'unexpected-empty-directory'));
        break;
      case 'payload': {
        const manifest = await readFixtureManifest({ fixture });
        const payload = manifest.files.find((file) => file.path.endsWith('core.wasm'));
        assert.ok(payload);
        const target = path.join(fixture.packageDirectory, payload.path);
        const bytes = await readFile(target);
        bytes[bytes.length - 1] = bytes[bytes.length - 1]! ^ 1;
        await writeFile(target, bytes);
        break;
      }
      case 'manifest': {
        const manifest = await readFixtureManifest({ fixture });
        manifest.futureMetadata = { rewrite: true };
        await writeFile(manifestPath, JSON.stringify(manifest));
        break;
      }
      case 'manifest-rehash': {
        const manifest = await readFixtureManifest({ fixture });
        const payload = manifest.files.find((file) => file.path.endsWith('core.wasm'));
        assert.ok(payload);
        const target = path.join(fixture.packageDirectory, payload.path);
        const bytes = await readFile(target);
        bytes[bytes.length - 1] = bytes[bytes.length - 1]! ^ 1;
        await writeFile(target, bytes);
        Object.assign(payload, await fileIdentity({ path: target }));
        await writeFile(manifestPath, JSON.stringify(manifest));
        break;
      }
      default: {
        const suffix = { 'contract-json': 'browser-smoke-contract.json', 'contract-module': 'browser-smoke-contract.mjs', arithmetic: 'arithmetic-probes.mjs', 'file-mount': 'file-mount-lifetime.mjs' }[mutation];
        const target = Object.entries(fixture.testFiles).find(([name]) => name.endsWith(suffix))?.[1];
        assert.ok(target);
        await writeFile(target, Buffer.concat([await readFile(target), Buffer.from(' ')]));
      }
      }
    };
    await assert.rejects(fixture.validate(), /snapshot changed|implementation or fixture changed/u);
    assert.equal(fixture.finalCalls, 0);
    await assert.rejects(readFile(fixture.receiptPath), { code: 'ENOENT' });
  }
});

test('child failure and missing fresh result never reuse an old success receipt', async () => {
  await using fixture = await validationFixture({ runtime: 'stable-diffusion-cpp' });
  const failure = new ProcessError({ command: 'fixture-browser', exitCode: 27, stderr: Buffer.from('fixture failure'), reason: 'exited with 27' });
  fixture.childFailure = failure;
  await writeFile(fixture.receiptPath, 'old success');
  await assert.rejects(fixture.validate(), (error: unknown) => error === failure);
  assert.equal(fixture.finalCalls, 0);
  await assert.rejects(readFile(fixture.receiptPath), { code: 'ENOENT' });
  fixture.childFailure = undefined;
  fixture.writeResults = false;
  await assert.rejects(fixture.validate());
  assert.equal(fixture.finalCalls, 0);
});

test('browser JSON rejects duplicate keys and nonstandard numbers at the external boundary', async () => {
  await using fixture = await validationFixture({ runtime: 'stable-diffusion-cpp' });
  for (const resultText of ['{"session":{},"session":{},"results":[]}', '{"session":{},"results":NaN}']) {
    fixture.resultText = resultText;
    await assert.rejects(fixture.validate());
    assert.equal(fixture.finalCalls, 0);
    await assert.rejects(readFile(fixture.receiptPath), { code: 'ENOENT' });
  }
});

test('generated fixture bytes are bound and only the owned temporary directory is canonicalized', async () => {
  await using fixture = await validationFixture({ runtime: 'llama-cpp' });
  const actual = path.join(fixture.root, 'actual-temp');
  const alias = path.join(fixture.root, 'temp-alias');
  await mkdir(actual);
  await symlink(actual, alias);
  using environment = stubEnvironment({ values: { TMPDIR: alias } });
  fixture.afterProcess = async () => {
    const command = fixture.commands.at(-1);
    assert.ok(command);
    assert.equal(path.dirname(command.resultPath), await realpath(path.dirname(command.resultPath)));
    if (command.phase === 'chromium_smoke') {
      const model = command.args[2];
      assert.ok(model);
      await writeFile(model, 'changed fixture');
    }
  };
  await assert.rejects(fixture.validate(), /implementation or fixture changed/u);
  assert.equal(fixture.finalCalls, 0);
});

test('later source build output is never reimported into the tested saved package', async () => {
  await using fixture = await validationFixture({ runtime: 'stable-diffusion-cpp' });
  fixture.afterProcess = async () => {
    const target = path.join(fixture.checkoutRoot, 'build', 'runtime', 'core.wasm');
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, 'unrelated later compilation');
  };
  await fixture.validate();
  assert.equal(fixture.finalCalls, 1);
});

test('npm failure restores the original and npm mutation cannot leave a success receipt', async () => {
  await using fixture = await validationFixture({ runtime: 'stable-diffusion-cpp' });
  const manifestPath = path.join(fixture.packageDirectory, 'manifest.json');
  const original = await readFile(manifestPath);
  fixture.finalFailure = new Error('npm failed');
  await assert.rejects(fixture.validate(), /npm failed/u);
  assert.equal(fixture.finalCalls, 1);
  assert.deepEqual(await readFile(manifestPath), original);
  await assert.rejects(readFile(fixture.receiptPath), { code: 'ENOENT' });
  fixture.finalFailure = undefined;
  fixture.afterFinal = async () => {
    await writeFile(path.join(fixture.packageDirectory, 'unexpected.txt'), 'changed by npm');
  };
  await assert.rejects(fixture.validate(), /snapshot changed/u);
  assert.equal(fixture.finalCalls, 1);
  assert.deepEqual(await readFile(manifestPath), original);
  await assert.rejects(readFile(fixture.receiptPath), { code: 'ENOENT' });
});

test('dirty provenance and source mismatches fail before any smoke, and receipts cannot remove payloads', async () => {
  await using fixture = await validationFixture({ runtime: 'stable-diffusion-cpp' });
  fixture.settings.selectedSourceCommit = 'b'.repeat(40);
  await assert.rejects(fixture.validate(), /source commit/u);
  assert.deepEqual(fixture.commands, []);
  fixture.settings.selectedSourceCommit = 'a'.repeat(40);
  const manifest = await readFixtureManifest({ fixture });
  for (const profile of fixture.profiles) {
    for (const variant of fixture.variants) {
      const provenance = manifest.profiles[profile]?.variants[variant];
      assert.ok(provenance);
      provenance.sourceDirty = true;
    }
  }
  const manifestPath = path.join(fixture.packageDirectory, 'manifest.json');
  await writeFile(manifestPath, JSON.stringify(manifest));
  await assert.rejects(fixture.validate(), /dirty/u);
  assert.deepEqual(fixture.commands, []);
  const original = await readFile(manifestPath);
  await assert.rejects(clearValidationReceipt({ packageDirectory: fixture.packageDirectory, receiptPath: manifestPath }), /outside/u);
  assert.deepEqual(await readFile(manifestPath), original);
  await rm(fixture.receiptPath, { force: true });
});

test('saved-package entry connects current typed configuration to mandatory external smoke and final pack', async () => {
  for (const runtime of ['llama-cpp', 'stable-diffusion-cpp'] as const) {
    await using fixture = await validationFixture({ runtime });
    const checkoutRoot = process.cwd();
    const scripts: string[] = [];
    const run: typeof runProcess = async ({ command, args, cwd, env }) => {
      assert.equal(command, process.execPath);
      assert.equal(cwd, path.join(checkoutRoot, 'runtimes', runtime));
      assert.ok(args[0]);
      scripts.push(path.basename(args[0]));
      assert.ok(env);
      const values = env.kind === 'inherit' ? env.overrides : env.values;
      assert.ok(values.BIC_BROWSER_RESULTS_FILE);
      assert.ok(values.BIC_BROWSER_SESSION_JSON);
      if (scripts.at(-1) === 'browser-smoke.mjs') {
        const session: unknown = JSON.parse(values.BIC_BROWSER_SESSION_JSON);
        let results: ReturnType<typeof llamaResults> | ReturnType<typeof imageResults>;
        switch (runtime) {
        case 'llama-cpp': results = llamaResults({ profiles: fixture.profiles, variants: fixture.variants });
          break;
        case 'stable-diffusion-cpp': results = imageResults({ profiles: fixture.profiles, variants: fixture.variants, gpu: true });
          break;
        default: {
          const exhaustive: never = runtime;
          throw new Error(String(exhaustive));
        }
        }
        await writeFile(values.BIC_BROWSER_RESULTS_FILE, JSON.stringify({ session, results }));
      }
      return { stdout: Buffer.alloc(0), stderr: Buffer.alloc(0), exitCode: 0 };
    };
    fixture.settings.testWebGpu = '1';
    const receipt = await validateSavedRuntimePackage({ checkoutRoot, runtime, packageDirectory: fixture.packageDirectory, receiptPath: fixture.receiptPath, settings: fixture.settings, childEnvironment: undefined, runProcess: run });
    assert.equal(receipt.runtime, runtime);
    assert.equal(receipt.status, 'complete');
    switch (runtime) {
    case 'llama-cpp': assert.deepEqual(scripts, ['asyncify-rewind.mjs', 'browser-smoke.mjs']);
      assert.equal(receipt.testWebGpu, false);
      break;
    case 'stable-diffusion-cpp': assert.deepEqual(scripts, ['browser-smoke.mjs']);
      assert.equal(receipt.testWebGpu, true);
      break;
    default: {
      const exhaustive: never = runtime;
      throw new Error(String(exhaustive));
    }
    }
    assert.deepEqual(JSON.parse(await readFile(fixture.receiptPath, 'utf8')), receipt);
  }
});
