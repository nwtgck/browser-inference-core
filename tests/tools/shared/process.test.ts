import assert from 'node:assert/strict';
import test from 'node:test';
import { spawn } from 'node:child_process';
import filesystem, { open, readFile, writeFile, type FileHandle } from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout } from 'node:timers/promises';
import { ProcessError, runProcess, splitCommandLine } from '#tools/shared/process';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { stubEnvironment } from '#tests/tools/shared/node-env.test-support';

test('process stdout remains binary and both diagnostic streams survive failure', async () => {
  const result = await runProcess({ command: 'python3', args: ['-c', 'import sys; sys.stdout.buffer.write(bytes([0,255,13,10])); sys.stderr.write("diagnostic")'], cwd: process.cwd() });
  assert.deepEqual(result.stdout, Buffer.from([0, 255, 13, 10]));
  assert.equal(result.stderr.toString(), 'diagnostic');
  await assert.rejects(runProcess({ command: 'python3', args: ['-c', 'import sys; print("case failure"); print("compile failure", file=sys.stderr); sys.exit(7)'], cwd: process.cwd() }), (error: unknown) => {
    assert.ok(error instanceof ProcessError);
    assert.equal(error.exitCode, 7);
    assert.match(error.stdout.toString(), /case failure/u);
    assert.match(error.stderr.toString(), /compile failure/u);
    return true;
  });
});

test('spawn errors, signals and timeouts cannot be mistaken for successful exit', async () => {
  await assert.rejects(runProcess({ command: '/missing/bicore-test-command', args: [], cwd: process.cwd() }), /ENOENT/u);
  await assert.rejects(runProcess({ command: process.execPath, args: ['-e', 'process.kill(process.pid,"SIGTERM")'], cwd: process.cwd() }), /SIGTERM/u);
  await assert.rejects(runProcess({ command: process.execPath, args: ['-e', 'setInterval(()=>{},1000)'], cwd: process.cwd(), timeoutMs: 50 }), /timed out/u);
});

test('compile commands decode POSIX quoting without executing expansions', () => {
  assert.deepEqual(splitCommandLine({ command: `clang++ -I'some dir' -DNAME="two words" -DESC=one\\ two '' "a\\q"` }), ['clang++', '-Isome dir', '-DNAME=two words', '-DESC=one two', '', 'a\\q']);
  for (const command of ['clang++ $(id)', 'clang++ $HOME', 'clang++ `id`', 'clang++ *.cpp', 'clang++ a; cat secret', "clang++ 'unterminated", 'CXX=clang++ tool']) {
    assert.throws(() => splitCommandLine({ command }));
  }
});

test('child environments preserve unknown inherited values and distinguish deletion from replacement', async () => {
  using parent = stubEnvironment({ values: { BICORE_CHILD_UNKNOWN: 'parent', BICORE_CHILD_REMOVED: 'remove' } });
  const args = ['-e', "require('node:fs').writeSync(1,[process.env.BICORE_CHILD_UNKNOWN,process.env.BICORE_CHILD_REMOVED,process.env.BICORE_CHILD_ADDED].map(value=>value===undefined?'absent':value).join('|'));"];
  const inherited = await runProcess({ command: process.execPath, args, cwd: process.cwd() });
  assert.equal(inherited.stdout.toString(), 'parent|remove|absent');
  const env = { kind: 'inherit', removePrefixes: [], overrides: { BICORE_CHILD_REMOVED: undefined, BICORE_CHILD_ADDED: '' } } as const;
  const changed = await runProcess({ command: process.execPath, args, cwd: process.cwd(), env, capture: 'file' });
  assert.equal(changed.stdout.toString(), 'parent|absent|');
  using live = stubEnvironment({ values: { BICORE_CHILD_UNKNOWN: 'later' } });
  const current = await runProcess({ command: process.execPath, args, cwd: process.cwd(), env });
  assert.equal(current.stdout.toString(), 'later|absent|');
  const replaced = await runProcess({ command: process.execPath, args, cwd: process.cwd(), env: { kind: 'replace', values: { BICORE_CHILD_ADDED: 'only' } } });
  assert.equal(replaced.stdout.toString(), 'absent|absent|only');
});

test('file capture preserves actual Node stdout/stderr bytes and rejects failed exits', async () => {
  const result = await runProcess({ command: process.execPath, args: ['-e', 'process.stdout.write(Buffer.from([0,255,13,10]));process.stderr.write("diagnostic")'], cwd: process.cwd(), capture: 'file' });
  assert.deepEqual(result.stdout, Buffer.from([0, 255, 13, 10]));
  assert.equal(result.stderr.toString(), 'diagnostic');
  await assert.rejects(runProcess({ command: process.execPath, args: ['-e', 'console.log("failed case");console.error("failure diagnostic");process.exitCode=7'], cwd: process.cwd(), capture: 'file' }), (error: unknown) => {
    assert.ok(error instanceof ProcessError);
    assert.equal(error.exitCode, 7);
    assert.match(error.stdout.toString(), /failed case/u);
    assert.match(error.stderr.toString(), /failure diagnostic/u);
    return true;
  });
});

for (const outcome of ['success', 'failed_exit', 'spawn_error', 'second_open_error'] as const) {
  test(`file capture closes acquired descriptors before removal after ${outcome}`, async context => {
    const open = filesystem.open;
    const rm = filesystem.rm;
    const handles: FileHandle[] = [];
    let removed = false;
    const interceptedOpen = context.mock.method(filesystem, 'open', async (...args: Parameters<typeof open>) => {
      switch (outcome) {
      case 'second_open_error':
        if (handles.length === 1) {
          throw new Error('second capture file open failed');
        }
        break;
      case 'success':
      case 'failed_exit':
      case 'spawn_error': break;
      default: {
        const _ex: never = outcome;
        throw new Error(`Unhandled file capture outcome: ${_ex}`);
      }
      }
      const handle = await open(...args);
      handles.push(handle);
      return handle;
    });
    const interceptedRemoval = context.mock.method(filesystem, 'rm', async (...args: Parameters<typeof rm>) => {
      assert.ok(handles.length > 0);
      for (const handle of handles) {
        assert.equal(handle.fd, -1, 'Capture descriptor must close before removing its directory');
      }
      await rm(...args);
      removed = true;
    });
    context.after(async () => {
      interceptedOpen.mock.restore();
      interceptedRemoval.mock.restore();
      syncBuiltinESMExports();
      for (const handle of handles) {
        if (handle.fd !== -1) {
          await handle.close();
        }
      }
    });
    syncBuiltinESMExports();
    let expectedHandles = 2;
    switch (outcome) {
    case 'success':
      await runProcess({ command: process.execPath, args: ['-e', ''], cwd: process.cwd(), capture: 'file' });
      break;
    case 'failed_exit':
      await assert.rejects(runProcess({ command: process.execPath, args: ['-e', 'process.exitCode=7'], cwd: process.cwd(), capture: 'file' }), /exited with 7/u);
      break;
    case 'spawn_error':
      await assert.rejects(runProcess({ command: '/missing/bicore-test-command', args: [], cwd: process.cwd(), capture: 'file' }), /ENOENT/u);
      break;
    case 'second_open_error':
      expectedHandles = 1;
      await assert.rejects(runProcess({ command: process.execPath, args: ['-e', ''], cwd: process.cwd(), capture: 'file' }), /second capture file open failed/u);
      break;
    default: {
      const _ex: never = outcome;
      throw new Error(`Unhandled file capture outcome: ${_ex}`);
    }
    }
    assert.equal(removed, true);
    assert.equal(handles.length, expectedHandles);
  });
}

test('tee publishes output before exit while preserving both streams and exit status', { timeout: 15000 }, async () => {
  for (const exitCode of [0, 7]) {
    await using temporary = await temporaryDirectory();
    const stdoutPath = join(temporary.root, 'live-stdout');
    const stderrPath = join(temporary.root, 'live-stderr');
    await using stdout = await open(stdoutPath, 'wx');
    await using stderr = await open(stderrPath, 'wx');
    const child = spawn(process.execPath, [fileURLToPath(new URL('./process-tee.test-support.js', import.meta.url)), temporary.root, 'tee', String(exitCode)], { stdio: ['ignore', stdout.fd, stderr.fd] });
    let stopped = false;
    const completed = new Promise<{ code: number | null; error: Error | undefined }>(resolve => {
      child.once('error', error => resolve({ code: null, error }));
      child.once('close', code => resolve({ code, error: undefined }));
    }).then(result => {
      stopped = true;
      return result;
    });
    await using execution = {
      completed,
      async [Symbol.asyncDispose]() {
        child.kill();
        await completed;
      },
    };
    const prefix = Buffer.concat([Buffer.from([0, 255, 13, 10]), Buffer.from('[1/2] first.o\n')]);
    const deadline = Date.now() + 6000;
    while (!(await readFile(stdoutPath)).equals(prefix) || (await readFile(stderrPath)).toString() !== 'first warning\n') {
      assert.equal(stopped, false, 'The command exited before live output was observed');
      assert.ok(Date.now() < deadline, 'Live output was not forwarded');
      await setTimeout(10);
    }
    assert.equal(stopped, false);
    // The fake command cannot finish until its first output has been observed.
    await writeFile(join(temporary.root, 'release'), 'continue');
    assert.deepEqual(await execution.completed, { code: 0, error: undefined });
    const expectedStdout = Buffer.concat([prefix, Buffer.from('[2/2] second.o\n')]);
    const expectedStderr = Buffer.from('first warning\nfinal diagnostic\n');
    assert.deepEqual(await readFile(stdoutPath), Buffer.concat([expectedStdout, Buffer.from('parent stdout remains open\n')]));
    assert.deepEqual(await readFile(stderrPath), Buffer.concat([expectedStderr, Buffer.from('parent stderr remains open\n')]));
    const result = JSON.parse(await readFile(join(temporary.root, 'result.json'), 'utf8'));
    assert.equal(result.stdout, expectedStdout.toString('hex'));
    assert.equal(result.stderr, expectedStderr.toString('hex'));
    assert.equal(result.exitCode, exitCode);
    if (exitCode !== 0) {
      assert.match(result.failureMessage, /final diagnostic/u);
      assert.match(result.failureMessage, /second[.]o/u);
    }
  }
});

test('ordinary capture does not publish captured output', async () => {
  await using temporary = await temporaryDirectory();
  await writeFile(join(temporary.root, 'release'), 'continue');
  const result = await runProcess({ command: process.execPath, args: [fileURLToPath(new URL('./process-tee.test-support.js', import.meta.url)), temporary.root, 'pipe', '0'], cwd: temporary.root, capture: 'file' });
  assert.equal(result.stdout.length, 0);
  assert.equal(result.stderr.length, 0);
  const captured = JSON.parse(await readFile(join(temporary.root, 'result.json'), 'utf8'));
  assert.equal(captured.exitCode, 0);
  assert.equal(captured.stdout, Buffer.concat([Buffer.from([0, 255, 13, 10]), Buffer.from('[1/2] first.o\n[2/2] second.o\n')]).toString('hex'));
  assert.equal(captured.stderr, Buffer.from('first warning\nfinal diagnostic\n').toString('hex'));
});

test('tee waits for both destinations after one fails and preserves command failures', async () => {
  for (const exitCode of [0, 7]) {
    await using temporary = await temporaryDirectory();
    await runProcess({ command: process.execPath, args: [fileURLToPath(new URL('./process-tee-failure.test-support.js', import.meta.url)), temporary.root, String(exitCode)], cwd: temporary.root, capture: 'file' });
    const result = JSON.parse(await readFile(join(temporary.root, 'result.json'), 'utf8'));
    assert.equal(result.pendingAtFailure, 0, 'The other destination must finish before rejection');
    assert.ok(result.completedAtFailure > 0, 'The delayed destination must have received output');
    if (exitCode === 0) {
      assert.match(result.failure.message, /stdout sink failure/u);
    } else {
      assert.equal(result.failure.name, 'ProcessError');
      assert.equal(result.failure.exitCode, 7);
      assert.equal(result.failure.stdout, 'out');
      assert.equal(result.failure.stderr, 'err');
    }
  }
});
