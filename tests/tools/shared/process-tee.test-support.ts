import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ProcessError, runProcess } from '#tools/shared/process';

const [root, mode, exit] = process.argv.slice(2);
if (root === undefined || exit === undefined) {
  throw new Error('Expected fixture directory and exit code');
}
let capture: 'pipe' | 'tee';
switch (mode) {
case 'pipe': capture = 'pipe';
  break;
case 'tee': capture = 'tee';
  break;
default: throw new Error('Expected pipe or tee fixture mode');
}
const script = `\
import os, sys, time
sys.stdout.buffer.write(bytes([0, 255, 13, 10]))
print('[1/2] first.o', flush=True)
print('first warning', file=sys.stderr, flush=True)
deadline = time.monotonic() + 5
while not os.path.exists('release'):
    if time.monotonic() > deadline:
        sys.exit(9)
    time.sleep(0.01)
print('[2/2] second.o', flush=True)
print('final diagnostic', file=sys.stderr, flush=True)
sys.exit(int(sys.argv[1]))
`;
let stdout: Buffer;
let stderr: Buffer;
let exitCode: number | null;
let failureMessage: string | undefined;
try {
  ({ stdout, stderr, exitCode } = await runProcess({ command: 'python3', args: ['-c', script, exit], cwd: root, capture }));
} catch (error) {
  if (!(error instanceof ProcessError)) {
    throw error;
  }
  ({ stdout, stderr, exitCode } = error);
  failureMessage = error.message;
}
await writeFile(join(root, 'result.json'), JSON.stringify({ stdout: stdout.toString('hex'), stderr: stderr.toString('hex'), exitCode, failureMessage }));
switch (capture) {
case 'tee':
  process.stdout.write('parent stdout remains open\n');
  process.stderr.write('parent stderr remains open\n');
  break;
case 'pipe': break;
default: {
  const _ex: never = capture;
  throw new Error(`Unknown fixture capture: ${String(_ex)}`);
}
}
