import { READ_ENV_PATH } from '#tools/shared/node-env';
import assert from 'node:assert/strict';
import { cp, mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { test } from 'node:test';
import { ProcessError, runProcess } from '#tools/shared/process';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { checkoutRoot, executable, list, object, text } from './workflow.test-support.js';

test('the offline launcher preserves stdin, argument boundaries, output and failures', async () => {
  await using temporary = await temporaryDirectory();
  const root = join(temporary.root, 'checkout with spaces');
  const runner = join(temporary.root, 'runner with spaces');
  await mkdir(join(root, '.github/scripts'), { recursive: true });
  await cp(join(checkoutRoot, '.github/scripts/run-offline.sh'), join(root, '.github/scripts/run-offline.sh'));
  await executable({
    root,
    name: 'docker',
    source: `\
const fs=require('node:fs');
fs.writeFileSync(process.env.CALLS,JSON.stringify({args:process.argv.slice(2),input:fs.readFileSync(0,'utf8')}));
process.stdout.write('original stdout\\n');
process.stderr.write('original stderr\\n');
process.exit(Number(process.env.DOCKER_EXIT));
`,
  });
  const input = '  literal $GH_TOKEN and spaces\n';
  const args = ['-e', 'HOME=/tmp', '-e', 'EXTRA=value with spaces'];
  const script = `\
offline_docker_args=(
  -e HOME=/tmp
  -e 'EXTRA=value with spaces'
)
.github/scripts/run-offline.sh "\${offline_docker_args[@]}" "$BIC_OFFLINE_IMAGE" <<'OFFLINE'
  literal $GH_TOKEN and spaces
OFFLINE
`;
  for (const code of [0, 7, 125]) {
    const image = code === 125 ? '' : `fixture:exit-${code}`;
    const operation = runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', script], cwd: root, capture: 'file', env: { kind: 'inherit', removePrefixes: [], overrides: { PATH: `${join(root, 'bin')}:${READ_ENV_PATH() ?? ''}`, GITHUB_WORKSPACE: root, RUNNER_TEMP: runner, BIC_OFFLINE_IMAGE: image, GH_TOKEN: 'private-token', GITHUB_TOKEN: 'private-token', CALLS: join(root, 'call.json'), DOCKER_EXIT: String(code) } } });
    if (code === 0) {
      const result = await operation;
      assert.equal(result.stdout.toString(), 'original stdout\n');
      assert.equal(result.stderr.toString(), 'original stderr\n');
    } else {
      await assert.rejects(operation, (error: unknown) => error instanceof ProcessError && error.exitCode === code && error.stdout.toString() === 'original stdout\n' && error.stderr.toString() === 'original stderr\n');
    }
    const call = object({ value: JSON.parse(await readFile(join(root, 'call.json'), 'utf8')) });
    const forwarded = list({ value: call.args }).map(value => text({ value }));
    assert.equal(call.input, input);
    assert.deepEqual(forwarded.slice(-args.length - 4), [...args, image, 'bash', '-euo', 'pipefail']);
    assert.ok(forwarded.includes('--pull=never'));
    assert.equal(forwarded[forwarded.indexOf('--network') + 1], 'none');
    assert.equal(forwarded[forwarded.indexOf('-w') + 1], root);
    assert.ok(forwarded.includes(`${root}:${root}`));
    assert.ok(forwarded.includes(`${runner}:${runner}`));
    assert.ok(forwarded.includes(`HOME=${runner}/bicore-home`));
    assert.ok(forwarded.includes('NPM_CONFIG_OFFLINE=true'));
    assert.ok(forwarded.includes('GIT_NO_LAZY_FETCH=1'));
    assert.ok(forwarded.includes(`PLAYWRIGHT_BROWSERS_PATH=${root}/.tools/browser-binaries`));
    assert.ok(forwarded.includes('LCB_SOURCE_COMMIT'));
    assert.ok(forwarded.includes('BIC_METRICS_MODE'));
    assert.ok(forwarded.includes('CCACHE_CONFIGPATH'));
    assert.ok(!forwarded.some(value => /^(GH_TOKEN|GITHUB_TOKEN)(=|$)/u.test(value)));
  }
  const missingCall = join(root, 'missing-image-call.json');
  await assert.rejects(runProcess({ command: 'bash', args: ['--noprofile', '--norc', '-eo', 'pipefail', '-c', script], cwd: root, capture: 'file', env: { kind: 'inherit', removePrefixes: [], overrides: { PATH: `${join(root, 'bin')}:${READ_ENV_PATH() ?? ''}`, GITHUB_WORKSPACE: root, RUNNER_TEMP: runner, BIC_OFFLINE_IMAGE: undefined, CALLS: missingCall, DOCKER_EXIT: '125' } } }), (error: unknown) => error instanceof ProcessError && error.exitCode === 125);
  const missing = object({ value: JSON.parse(await readFile(missingCall, 'utf8')) });
  assert.deepEqual(list({ value: missing.args }).slice(-4), ['', 'bash', '-euo', 'pipefail']);
});

