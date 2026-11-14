import assert from 'node:assert/strict';
import test from 'node:test';
import { ProcessError, runProcess, splitCommandLine } from '#tools/shared/process';

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
