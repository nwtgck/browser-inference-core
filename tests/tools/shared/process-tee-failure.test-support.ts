import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Writable } from 'node:stream';
import { ProcessError, runProcess } from '#tools/shared/process';

const [root, exit] = process.argv.slice(2);
if (root === undefined || exit === undefined) {
  throw new Error('Expected fixture directory and exit code');
}
const originalStdout = Object.getOwnPropertyDescriptor(process, 'stdout');
const originalStderr = Object.getOwnPropertyDescriptor(process, 'stderr');
if (originalStdout === undefined || originalStderr === undefined) {
  throw new Error('Expected standard process output descriptors');
}
let completedWrites = 0;
let pendingWrites = 0;
const bad = new Writable({
  write(_chunk, _encoding, callback) {
    callback(new Error('stdout sink failure'));
  },
});
const slow = new Writable({
  write(_chunk, _encoding, callback) {
    pendingWrites++;
    setTimeout(() => {
      completedWrites++;
      pendingWrites--;
      callback();
    }, 150);
  },
});
let failure: { name: string; message: string; exitCode: number | null | undefined; stdout: string | undefined; stderr: string | undefined } | undefined;
let pendingAtFailure: number | undefined;
let completedAtFailure: number | undefined;
Object.defineProperty(process, 'stdout', { configurable: true, value: bad });
Object.defineProperty(process, 'stderr', { configurable: true, value: slow });
try {
  await runProcess({ command: 'python3', args: ['-c', 'import os, sys; os.write(1,b"out"); os.write(2,b"err"); sys.exit(int(sys.argv[1]))', exit], cwd: root, capture: 'tee' });
} catch (error) {
  if (!(error instanceof Error)) {
    throw error;
  }
  pendingAtFailure = pendingWrites;
  completedAtFailure = completedWrites;
  failure = {
    name: error.name,
    message: error.message,
    exitCode: error instanceof ProcessError ? error.exitCode : undefined,
    stdout: error instanceof ProcessError ? error.stdout.toString('utf8') : undefined,
    stderr: error instanceof ProcessError ? error.stderr.toString('utf8') : undefined,
  };
} finally {
  Object.defineProperty(process, 'stdout', originalStdout);
  Object.defineProperty(process, 'stderr', originalStderr);
}
await writeFile(join(root, 'result.json'), JSON.stringify({ failure, pendingAtFailure, completedAtFailure }));
