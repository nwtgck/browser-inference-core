import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { test } from 'node:test';
import { ProcessError, runProcess } from '#tools/shared/process';
import { readRecords } from '#tools/ci/summary';

const cli = resolve('build/tooling/tools/ci/cli.js');

test('disabled timing passes large command output through without creating logs', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-command-off-'));
  try {
    const directory = join(root, 'metrics');
    const env = { ...process.env, BIC_METRICS_MODE: 'off', BIC_METRICS_DIR: directory };
    const result = await runProcess({ command: process.execPath, args: [cli, 'run', '--phase', 'test.large', '--', process.execPath, '-e', 'process.stdout.write("x".repeat(1048576))'], cwd: process.cwd(), env, capture: 'file' });
    assert.equal(result.stdout.length, 1048576);
    await assert.rejects(stat(directory));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('external command preserves streams and failure despite optional logger errors', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-command-test-'));
  try {
    const broken = join(root, 'not-a-directory');
    await writeFile(broken, 'fixture');
    const env = { ...process.env, BIC_METRICS_MODE: 'basic', BIC_METRICS_DIR: broken };
    await assert.rejects(runProcess({ command: process.execPath, args: [cli, 'run', '--phase', 'test.command', '--', process.execPath, '-e', 'console.log("out"); console.error("err"); process.exit(7)'], cwd: process.cwd(), env, capture: 'file' }), error => {
      assert.ok(error instanceof ProcessError);
      assert.equal(error.exitCode, 7);
      assert.equal(error.stdout.toString('utf8'), 'out\n');
      assert.match(error.stderr.toString('utf8'), /err/u);
      assert.match(error.stderr.toString('utf8'), /Optional telemetry/u);
      return true;
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('enabled command logs failure but never command arguments or environment secrets', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-command-record-'));
  try {
    const directory = join(root, 'metrics');
    const env = { ...process.env, BIC_METRICS_MODE: 'basic', BIC_METRICS_DIR: directory, GH_TOKEN: 'SECRET-FIXTURE' };
    await assert.rejects(runProcess({ command: process.execPath, args: [cli, 'run', '--phase', 'test.command', '--', process.execPath, '-e', 'const hidden="ARGUMENT-SECRET"; process.exit(7)'], cwd: process.cwd(), env, capture: 'file' }), error => error instanceof ProcessError && error.exitCode === 7);
    const { records, errors } = await readRecords({ directory });
    assert.equal(errors, 0);
    const end = records.find(record => record.event === 'span_end');
    assert.ok(end && end.event === 'span_end');
    assert.equal(end.result, 'failure');
    assert.equal(end.exitCode, 7);
    const content = (await Promise.all((await readdir(directory)).map(file => readFile(join(directory, file), 'utf8')))).join('');
    assert.doesNotMatch(content, /ARGUMENT-SECRET|SECRET-FIXTURE/u);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('SIGTERM reaches the child and the wrapper exits by the same signal', { skip: process.platform === 'win32', timeout: 12000 }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-command-signal-'));
  const ready = join(root, 'ready');
  const stopped = join(root, 'stopped');
  const code = `const fs = require('node:fs'); process.on('SIGTERM', () => { fs.writeFileSync(${JSON.stringify(stopped)}, 'yes'); process.exit(0); }); fs.writeFileSync(${JSON.stringify(ready)}, String(process.pid)); setInterval(() => {}, 1000);`;
  const child = spawn(process.execPath, [cli, 'run', '--phase', 'test.cancel', '--', process.execPath, '-e', code], {
    cwd: process.cwd(),
    env: { ...process.env, BIC_METRICS_MODE: 'basic', BIC_METRICS_DIR: join(root, 'metrics') },
    stdio: 'ignore',
  });
  const completed = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => child.on('close', (code, signal) => resolve({ code, signal })));
  try {
    let found = false;
    for (let attempt = 0; attempt < 200; attempt++) {
      try {
        await stat(ready);
        found = true;
        break;
      } catch {
        await delay(20);
      }
    }
    assert.equal(found, true);
    child.kill('SIGTERM');
    const result = await completed;
    assert.equal(result.signal, 'SIGTERM');
    assert.equal(await readFile(stopped, 'utf8'), 'yes');
    const { records } = await readRecords({ directory: join(root, 'metrics') });
    const end = records.find(record => record.event === 'span_end');
    assert.ok(end && end.event === 'span_end');
    assert.equal(end.result, 'cancelled');
  } finally {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill('SIGKILL');
    }
    try {
      const pid = Number(await readFile(ready, 'utf8'));
      if (pid > 1) {
        process.kill(pid, 'SIGKILL');
      }
    } catch { /* Child already exited. */ }
    await rm(root, { recursive: true, force: true });
  }
});

test('cancellation kills a child that ignores SIGTERM within the grace period', { skip: process.platform === 'win32', timeout: 12000 }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-command-ignore-'));
  const ready = join(root, 'ready');
  const code = `const fs = require('node:fs'); process.on('SIGTERM', () => {}); fs.writeFileSync(${JSON.stringify(ready)}, String(process.pid)); setInterval(() => {}, 1000);`;
  const child = spawn(process.execPath, [cli, 'run', '--phase', 'test.ignore', '--', process.execPath, '-e', code], {
    cwd: process.cwd(),
    env: { ...process.env, BIC_METRICS_MODE: 'basic', BIC_METRICS_DIR: join(root, 'metrics') },
    stdio: 'ignore',
  });
  const completed = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => child.on('close', (code, signal) => resolve({ code, signal })));
  try {
    let found = false;
    for (let attempt = 0; attempt < 200; attempt++) {
      try {
        await stat(ready);
        found = true;
        break;
      } catch {
        await delay(20);
      }
    }
    assert.equal(found, true);
    const started = Date.now();
    child.kill('SIGTERM');
    const result = await completed;
    assert.equal(result.signal, 'SIGTERM');
    assert.ok(Date.now() - started < 6000);
  } finally {
    if (child.exitCode === null && child.signalCode === null) {
      child.kill('SIGKILL');
    }
    try {
      const pid = Number(await readFile(ready, 'utf8'));
      if (pid > 1) {
        process.kill(pid, 'SIGKILL');
      }
    } catch { /* Child already exited. */ }
    await rm(root, { recursive: true, force: true });
  }
});

test('a terminating parent cannot leave a signal-ignoring grandchild holding the output pipe', { skip: process.platform === 'win32', timeout: 12000 }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-command-grandchild-'));
  const ready = join(root, 'ready');
  const grandchild = `const fs = require('node:fs'); process.on('SIGTERM', () => {}); fs.writeFileSync(${JSON.stringify(ready)}, String(process.pid)); setInterval(() => {}, 1000);`;
  const code = `const cp = require('node:child_process'); process.on('SIGTERM', () => process.exit(0)); cp.spawn(process.execPath, ['-e', ${JSON.stringify(grandchild)}], {stdio: 'inherit'}); setInterval(() => {}, 1000);`;
  const child = spawn(process.execPath, [cli, 'run', '--phase', 'test.grandchild', '--', process.execPath, '-e', code], {
    cwd: process.cwd(),
    env: { ...process.env, BIC_METRICS_MODE: 'basic', BIC_METRICS_DIR: join(root, 'metrics') },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.resume();
  child.stderr.resume();
  const completed = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => child.on('close', (code, signal) => resolve({ code, signal })));
  let timeout: NodeJS.Timeout | undefined;
  try {
    let found = false;
    for (let attempt = 0; attempt < 200; attempt++) {
      try {
        await stat(ready);
        found = true;
        break;
      } catch {
        await delay(20);
      }
    }
    assert.equal(found, true);
    child.kill('SIGTERM');
    const deadline = new Promise<never>((_resolve, reject) => {
      timeout = setTimeout(() => reject(new Error('Grandchild kept the output pipe open')), 6000);
    });
    const result = await Promise.race([completed, deadline]);
    assert.equal(result.signal, 'SIGTERM');
  } finally {
    clearTimeout(timeout);
    if (child.exitCode === null && child.signalCode === null) {
      child.kill('SIGKILL');
    }
    try {
      const pid = Number(await readFile(ready, 'utf8'));
      if (pid > 1) {
        process.kill(pid, 'SIGKILL');
      }
    } catch { /* Child already exited. */ }
    await rm(root, { recursive: true, force: true });
  }
});

test('separate CI steps summarize their own saved observations', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-command-summary-'));
  try {
    const directory = join(root, 'metrics');
    const env = { ...process.env, BIC_METRICS_MODE: 'basic', BIC_METRICS_DIR: directory };
    await runProcess({ command: process.execPath, args: [cli, 'run', '--phase', 'fixture.complete', '--', process.execPath, '-e', 'process.exit(0)'], cwd: process.cwd(), env, capture: 'file' });
    const result = await runProcess({ command: process.execPath, args: [cli, 'summary', '--directory', directory], cwd: process.cwd(), env, capture: 'file' });
    assert.match(result.stdout.toString('utf8'), /fixture\.complete/u);
    assert.match(await readFile(join(directory, 'summary.json'), 'utf8'), /"fixture\.complete"/u);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
