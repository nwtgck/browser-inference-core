import { READ_ENV_PATH } from '#tools/shared/node-env';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { cp, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { test } from 'node:test';
import { runInNewContext } from 'node:vm';
import { pathToFileURL } from 'node:url';
import { load } from 'js-yaml';
import ts from 'typescript';
import { ProcessError, runProcess } from '#tools/shared/process';
import llamaProfiles from '#runtimes/llama-cpp/config/profiles.json' with { type: 'json' };
import imageProfiles from '#runtimes/stable-diffusion-cpp/config/profiles.json' with { type: 'json' };
import speechProfiles from '#runtimes/speech-cpp/config/profiles.json' with { type: 'json' };
import { temporaryDirectory, git } from '#tests/tools/shared/fixtures.test-support';
import { actionSteps, checkoutRoot, executable, executeShell, jobs, jobSteps, list, namedStep, object, offlineScript, readCalls, text, workflow } from './workflow.test-support.js';

// These assertions exercise orchestration, isolation and actual acquisition and
// publication shell. Native/model behavior belongs to the runtime tests.
test('the runtime matrices stay independent until their package gates join', () => {
  const dependencies = {
    'host-and-publication-checks': [],
    'llama-cpp-compile': [],
    'llama-cpp-package': ['llama-cpp-compile'],
    'stable-diffusion-cpp-checks': [],
    'stable-diffusion-cpp-compile': [],
    'stable-diffusion-cpp-package': ['stable-diffusion-cpp-compile'],
    'speech-cpp-checks': ['speech-cpp-package'],
    'speech-cpp-compile': [],
    'speech-cpp-package': ['speech-cpp-compile'],
    'llama-cpp-validate': ['llama-cpp-package'],
    'stable-diffusion-cpp-validate': ['stable-diffusion-cpp-package'],
    publish: ['llama-cpp-package', 'stable-diffusion-cpp-package', 'speech-cpp-package'],
    'consumer-report': ['publish'],
  };
  assert.deepEqual(Object.keys(jobs).sort(), Object.keys(dependencies).sort());
  for (const [name, needs] of Object.entries(dependencies)) {
    const job = object({ value: jobs[name] });
    assert.deepEqual(job.needs ?? [], needs);
    assert.equal(job.concurrency, undefined);
    assert.equal(object({ value: job.strategy ?? {} })['max-parallel'], undefined);
  }
  for (const [name, profiles, count] of [['llama-cpp-compile', llamaProfiles, 5], ['stable-diffusion-cpp-compile', imageProfiles, 3], ['speech-cpp-compile', speechProfiles, 1]] as const) {
    const strategy = object({ value: object({ value: jobs[name] }).strategy });
    const matrix = object({ value: strategy.matrix });
    assert.equal(strategy['fail-fast'], false);
    assert.deepEqual(list({ value: matrix.profile }).sort(), Object.keys(profiles).sort());
    assert.equal(list({ value: matrix.profile }).length, count);
    assert.deepEqual(matrix.variant, ['browser', 'test']);
  }
  for (const runtime of ['llama', 'stable-diffusion', 'speech']) {
    assert.equal(object({ value: jobs[`${runtime}-cpp-compile`] }).name, `Compile ${runtime}.cpp / ` + '${{ matrix.profile }} / ${{ matrix.variant }}');
    assert.equal(object({ value: jobs[`${runtime}-cpp-package`] }).name, `Package and verify ${runtime}.cpp`);
  }
  assert.equal(workflow.concurrency, undefined);
  const events = object({ value: workflow.on });
  assert.deepEqual(object({ value: events.pull_request }).types, ['opened', 'reopened', 'synchronize']);
  for (const event of ['push', 'pull_request']) {
    assert.deepEqual(object({ value: events[event] })['branches-ignore'], ['artifacts', 'artifacts/**']);
  }
});

test('jobs use the immutable source head, locked dependencies and one strict emit', () => {
  for (const name of Object.keys(jobs)) {
    const checkout = actionSteps({ job: name, action: 'actions/checkout' });
    assert.equal(checkout.length, 1);
    assert.deepEqual(object({ value: checkout[0] }).with, { ref: '${{ env.LCB_SOURCE_COMMIT }}', submodules: false, 'persist-credentials': false });
    const setup = actionSteps({ job: name, action: 'actions/setup-node' });
    assert.equal(setup.length, 1);
    assert.equal(object({ value: object({ value: setup[0] }).with })['node-version'], name === 'speech-cpp-package' ? '24.13.1' : '22.16.0');
  }
  assert.equal(object({ value: workflow.env }).LCB_SOURCE_COMMIT, '${{ github.event.pull_request.head.sha || github.sha }}');
  assert.deepEqual(workflow.permissions, { contents: 'read' });
  assert.deepEqual(object({ value: jobs.publish }).permissions, { contents: 'write', attestations: 'write', 'id-token': 'write' });
});

test('publication signs only the verified frozen envelope digest before any append-only push', () => {
  const steps = jobSteps({ name: 'publish' });
  const freeze = steps.findIndex(step => step.id === 'freeze');
  const attest = steps.findIndex(step => typeof step.uses === 'string' && step.uses.startsWith('actions/attest@'));
  const publish = steps.findIndex(step => step.id === 'publish');
  assert.ok(freeze >= 0 && freeze < attest && attest < publish);
  assert.equal(actionSteps({ job: 'publish', action: 'actions/attest' }).length, 1);
  const step = object({ value: steps[attest] });
  assert.deepEqual(step.with, { 'subject-name': 'browser-inference-core-artifact-integrity.yaml', 'subject-digest': 'sha256:${{ steps.freeze.outputs.integrity-sha256 }}', 'create-storage-record': false });
  assert.equal(step.if, undefined);
  assert.equal(step['continue-on-error'], undefined);
  assert.equal(object({ value: steps[publish] }).if, undefined);
  for (const name of Object.keys(jobs).filter(name => name !== 'publish')) {
    assert.equal(actionSteps({ job: name, action: 'actions/attest' }).length, 0);
  }
});

test('upstream update state is initialized on the runner and exported for later steps', async () => {
  const document = object({ value: load(await readFile(join(checkoutRoot, '.github/workflows/update-llama-cpp.yml'), 'utf8')) });
  const updater = object({ value: object({ value: document.jobs }).update });
  assert.equal(object({ value: updater.env }).UPDATE_STATE, undefined);
  const steps = list({ value: updater.steps }).map(value => object({ value }));
  const acquisition = steps.find(step => step.name === 'Acquire tooling dependencies (online)');
  assert.ok(acquisition);
  await using temporary = await temporaryDirectory();
  const root = temporary.root;
  const runner = join(root, 'runner temp');
  const githubEnv = join(root, 'github env');
  await mkdir(runner);
  await executable({
    root,
    name: 'npm',
    source: `\
require('node:fs').appendFileSync(process.env.CALLS,JSON.stringify({command:'npm',args:process.argv.slice(2)})+'\\n');
`,
  });
  await executeShell({ root, script: text({ value: acquisition.run }), environment: { RUNNER_TEMP: runner, GITHUB_ENV: githubEnv, CALLS: join(root, 'calls.jsonl') } });
  const exported = (await readFile(githubEnv, 'utf8')).trimEnd();
  assert.equal(exported, `UPDATE_STATE=${join(runner, 'upstream-update')}`);
  const nextState = join(root, 'next-step-state');
  await executeShell({ root, script: 'printf \'%s\' "$UPDATE_STATE" > "$STATE_FILE"', environment: { UPDATE_STATE: exported.slice('UPDATE_STATE='.length), STATE_FILE: nextState } });
  assert.equal(await readFile(nextState, 'utf8'), join(runner, 'upstream-update'));
  assert.deepEqual((await readCalls({ root })).map(call => call.args), [['ci', '--ignore-scripts', '--no-audit', '--no-fund']]);
});

test('each job installs its root tooling lock and emits tools once', async () => {
  for (const name of Object.keys(jobs)) {
    await using temporary = await temporaryDirectory();
    const root = temporary.root;
    const recorder = `\
const fs=require('node:fs');
const path=require('node:path');
const args=process.argv.slice(2);
const command=path.basename(process.argv[1]);
fs.appendFileSync(process.env.CALLS,JSON.stringify({command,args,environment:{cwd:process.cwd()}})+'\\n');
if(command==='git') process.stdout.write(process.env.LCB_SOURCE_COMMIT+'\\n');
if(command==='ccache') process.stdout.write('ccache fixture\\n');
if(command==='node'&&args[1]==='cache-plan'){
  const output=args[args.indexOf('--output')+1];
  fs.mkdirSync(path.dirname(output),{recursive:true});
  fs.writeFileSync(output,JSON.stringify({dawnKey:'dawn',emKey:'em',emRestorePrefix:'em-prefix',ccKey:'cc',ccRestorePrefix:'prefix',emStagingPath:'em',ccStagingPath:'cc',environment:{EM_CACHE:'fixture'}}));
}
`;
    for (const command of ['npm', 'node', 'git', 'ccache']) {
      await executable({ root, name: command, source: recorder });
    }
    await mkdir(join(root, 'toolchain'));
    await cp(join(checkoutRoot, 'toolchain/config.json'), join(root, 'toolchain/config.json'));
    const runtime = text({ value: object({ value: object({ value: jobs[name] }).env }).BIC_METRICS_RUNTIME });
    const compile = name.endsWith('-compile');
    const environment = { HOME: join(root, 'home'), GITHUB_WORKSPACE: root, GITHUB_OUTPUT: join(root, 'output'), GITHUB_ENV: join(root, 'env'), LCB_SOURCE_COMMIT: 'a'.repeat(40), BIC_METRICS_RUNTIME: runtime, BIC_METRICS_PROFILE: runtime === 'speech-cpp' ? 'webgpu-wasm32-jspi' : runtime === 'stable-diffusion-cpp' ? 'webgpu-wasm32-jspi' : 'cpu-wasm32', BIC_METRICS_VARIANT: 'browser', BIC_OFFLINE_ENVIRONMENT: 'fixture', RUNNER_OS: 'Linux', RUNNER_ARCH: 'X64', CALLS: join(root, 'calls.jsonl') };
    const acquisition = namedStep({ job: name, name: 'Acquire locked development dependencies (online)' });
    await executeShell({ root, script: text({ value: acquisition.run }), environment });
    const emit = namedStep({ job: name, name: compile ? 'Emit tools and plan cache partitions' : 'Emit the checked-out tools and verify source identity' });
    await executeShell({ root, script: offlineScript({ step: emit }), environment });
    const calls = await readCalls({ root });
    const npm = calls.filter(call => call.command === 'npm');
    assert.ok(npm.every(call => call.environment?.cwd === root));
    const compiler = calls.filter(call => call.command === 'node' && call.args[0] === 'node_modules/typescript/bin/tsc');
    const noticeChecks = calls.filter(call => call.command === 'node' && call.args[1] === 'check-notices');
    assert.deepEqual(noticeChecks.map(call => call.args), name === 'llama-cpp-compile' ? [['build/tooling/runtimes/llama-cpp/tools/cli.js', 'check-notices', '--checkout-root', root]] : []);
    if (compile) {
      assert.deepEqual(npm.map(call => call.args), [['ci', '--ignore-scripts', '--no-audit', '--no-fund']]);
      assert.deepEqual(compiler.map(call => call.args), [['node_modules/typescript/bin/tsc', '--project', `tsconfig.${runtime}.json`]]);
      const cache = calls.find(call => call.command === 'node' && call.args[1] === 'cache-plan');
      assert.ok(cache);
      for (const check of noticeChecks) {
        assert.ok(calls.indexOf(check) < calls.indexOf(cache));
      }
      const args = [...cache.args];
      args[args.indexOf('--checkout-root') + 1] = checkoutRoot;
      await runProcess({ command: process.execPath, args, cwd: checkoutRoot, env: { kind: 'inherit', removePrefixes: [], overrides: { BIC_METRICS_MODE: 'off' } } });
      const planned = object({ value: JSON.parse(await readFile(join(root, 'build/cache-plan.json'), 'utf8')) });
      assert.equal(object({ value: planned.environment }).EM_CACHE, join(root, '.tools/emsdk/upstream/emscripten/cache'));
      assert.equal(planned.ccStagingPath, `.cache/restored/${environment.BIC_METRICS_RUNTIME}/${environment.BIC_METRICS_PROFILE}/browser/cc`);
      assert.equal(planned.emKey, text({ value: planned.emRestorePrefix }) + environment.LCB_SOURCE_COMMIT);
      assert.ok((await readFile(join(root, 'output'), 'utf8')).split('\n').includes('em-prefix=em-prefix'));
    } else {
      assert.deepEqual(npm.map(call => call.args), [['ci', '--ignore-scripts', '--no-audit', '--no-fund'], ['run', name.endsWith('-package') || name === 'publish' || name === 'consumer-report' ? 'tooling:build:production' : 'tooling:build']]);
      assert.deepEqual(compiler, []);
    }
  }
});

test('the speech compile project emits independently resolvable runtime and workflow entries', async () => {
  await using temporary = await temporaryDirectory();
  const root = temporary.root;
  const emitted = join(root, 'build/tooling');
  await mkdir(join(root, 'runtimes/speech-cpp'), { recursive: true });
  // Keep package imports and dependencies in the isolated tree, so an earlier
  // full tooling build cannot satisfy missing entries or their imports.
  for (const name of ['package.json', 'node_modules', 'toolchain', 'runtimes/speech-cpp/config']) {
    await cp(join(checkoutRoot, name), join(root, name), { recursive: true, dereference: true });
  }
  const configuration = ts.getParsedCommandLineOfConfigFile(join(checkoutRoot, 'tsconfig.speech-cpp.json'), { outDir: emitted }, {
    ...ts.sys,
    onUnRecoverableConfigFileDiagnostic: diagnostic => assert.fail(ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')),
  });
  assert.ok(configuration);
  const program = ts.createProgram({ rootNames: configuration.fileNames, options: configuration.options, configFileParsingDiagnostics: configuration.errors });
  const result = program.emit();
  assert.deepEqual([...ts.getPreEmitDiagnostics(program), ...result.diagnostics].map(diagnostic => ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')), []);
  assert.equal(result.emitSkipped, false);
  for (const entry of ['runtimes/speech-cpp/tools/cli.js', 'tools/ci/cli.js', 'tools/toolchain/cli.js']) {
    await import(pathToFileURL(join(emitted, entry)).href);
  }
});

test('cache policy and publication trust are evaluated independently', () => {
  function condition({ expression, context }: { expression: unknown; context: Record<string, unknown> }): boolean {
    const source = text({ value: expression });
    const value: unknown = runInNewContext(source.startsWith('${{') ? source.slice(3, -2).trim() : source, context, { timeout: 100 });
    assert.equal(typeof value, 'boolean');
    return value === true;
  }
  for (const name of ['llama-cpp-compile', 'stable-diffusion-cpp-compile', 'speech-cpp-compile']) {
    const saves = actionSteps({ job: name, action: 'actions/cache/save' });
    const restores = actionSteps({ job: name, action: 'actions/cache/restore' });
    const build = namedStep({ job: name, name: 'Prepare, build and stage the selected runtime' });
    assert.equal(saves.length, 3);
    assert.equal(restores.length, 3);
    for (const restore of restores) {
      const options = object({ value: restore.with });
      const save = saves.find(step => step.id === `save-${text({ value: restore.id })}`);
      assert.ok(save);
      assert.deepEqual(save.with, { key: options.key, path: options.path });
      assert.equal(options['restore-keys'], restore.id === 'cc' ? '${{ steps.plan.outputs.cc-prefix }}' : restore.id === 'em' ? '${{ steps.plan.outputs.em-prefix }}' : undefined);
    }
    for (const step of restores) {
      assert.equal(step['continue-on-error'], true);
      assert.equal(condition({ expression: step.if, context: { env: { BIC_CACHE_MODE: 'cold' } } }), false);
      assert.equal(condition({ expression: step.if, context: { env: { BIC_CACHE_MODE: 'enabled' } } }), true);
    }
    for (const save of saves) {
      assert.equal(save['continue-on-error'], true);
      const contexts = [
        { event: 'push', ref: 'develop', actor: 'human', mode: 'enabled', succeeded: true, hit: 'false', allowed: true },
        { event: 'pull_request', ref: '7/merge', actor: 'human', mode: 'enabled', succeeded: true, hit: 'false', allowed: true },
        { event: 'push', ref: 'feature/change', actor: 'human', mode: 'enabled', succeeded: true, hit: 'false', allowed: true },
        { event: 'push', ref: 'develop', actor: 'dependabot[bot]', mode: 'enabled', succeeded: true, hit: 'false', allowed: true },
        { event: 'pull_request', ref: '8/merge', actor: 'dependabot[bot]', mode: 'enabled', succeeded: true, hit: 'false', allowed: true },
        { event: 'workflow_dispatch', ref: 'feature/change', actor: 'human', mode: 'enabled', succeeded: true, hit: 'false', allowed: true },
        { event: 'push', ref: 'feature/change', actor: 'human', mode: 'cold', succeeded: true, hit: 'false', allowed: false },
        { event: 'pull_request', ref: '7/merge', actor: 'human', mode: 'enabled', succeeded: false, hit: 'false', allowed: false },
        { event: 'workflow_dispatch', ref: 'feature/change', actor: 'human', mode: 'enabled', succeeded: true, hit: 'true', allowed: false },
      ];
      for (const scenario of contexts) {
        const github = { event_name: scenario.event, ref_name: scenario.ref, actor: scenario.actor, repository: 'owner/repo', event: { repository: { default_branch: 'develop' }, pull_request: { head: { repo: { full_name: 'someone/fork' } } } } };
        const steps = Object.fromEntries(['dawn', 'em', 'cc'].map(name => [name, { outputs: { 'cache-hit': scenario.hit } }]));
        const expression = text({ value: save.if }).replaceAll('.cache-hit', "['cache-hit']");
        assert.equal(condition({ expression, context: { github, steps, env: { BIC_CACHE_MODE: scenario.mode }, success: () => scenario.succeeded } }), scenario.allowed);
        // Step env permits data contexts; status functions belong to step if.
        const eligible = condition({ expression: object({ value: build.env }).SAVE_CACHES, context: { env: { BIC_CACHE_MODE: scenario.mode } } });
        assert.equal(eligible, scenario.mode === 'enabled');
        assert.equal(condition({ expression: build.if, context: { success: () => scenario.succeeded } }), scenario.succeeded);
      }
    }
  }
  for (const [event, actor, head, expected] of [['push', 'human', 'someone/fork', true], ['pull_request', 'human', 'owner/repo', true], ['pull_request', 'human', 'someone/fork', false], ['push', 'dependabot[bot]', 'owner/repo', false]] as const) {
    const github = { event_name: event, actor, repository: 'owner/repo', event: { pull_request: { head: { repo: { full_name: head } } } } };
    assert.equal(condition({ expression: object({ value: jobs.publish }).if, context: { github } }), expected);
  }
});

test('profile artifacts transfer staged runtimes and notices to the matching package only', () => {
  for (const runtime of ['llama-cpp', 'stable-diffusion-cpp', 'speech-cpp']) {
    const compile = `${runtime}-compile`;
    const build = `${runtime}-package`;
    const upload = actionSteps({ job: compile, action: 'actions/upload-artifact' }).filter(step => object({ value: step.with })['if-no-files-found'] === 'error');
    assert.equal(upload.length, 1);
    const options = object({ value: object({ value: upload[0] }).with });
    assert.equal(options.name, compile + '-${{ matrix.profile }}-${{ matrix.variant }}');
    const paths = text({ value: options.path }).trim().split('\n');
    assert.deepEqual(paths, [`build/runtimes/${runtime}/stage/`, `build/runtimes/${runtime}/toolchain-licenses/`]);
    const download = actionSteps({ job: build, action: 'actions/download-artifact' });
    assert.equal(download.length, 1);
    assert.deepEqual(object({ value: download[0] }).with, { pattern: `${compile}-*`, path: `build/runtimes/${runtime}/`, 'merge-multiple': true });
    const packaged = actionSteps({ job: build, action: 'actions/upload-artifact' }).find(step => object({ value: step.with })['if-no-files-found'] === 'error');
    assert.equal(object({ value: packaged?.with }).name, build);
    const names = jobSteps({ name: build }).map(step => step.name);
    assert.ok(names.indexOf('Assemble runtime-only package') < names.indexOf('Prepare Chromium (online)'));
    assert.ok(names.indexOf('Prepare Chromium (online)') < names.indexOf('Test packaged browser startup'));
    const finalizer = namedStep({ job: build, name: 'Test packaged browser startup' });
    assert.match(offlineScript({ step: finalizer }), /validation runtime --phase startup --runtime /u);
    assert.equal(finalizer.if, undefined);
    assert.equal(finalizer['continue-on-error'], undefined);
  }
});

test('speech native checks consume the package and Node interop gates publication', async () => {
  await using temporary = await temporaryDirectory();
  const root = temporary.root;
  const recorder = `\
const fs=require('node:fs');
const path=require('node:path');
const args=process.argv.slice(2);
const command=path.basename(process.argv[1]);
fs.appendFileSync(process.env.CALLS,JSON.stringify({command,args,environment:{BICORE_SPEECH_ABI_SCHEMA:process.env.BICORE_SPEECH_ABI_SCHEMA??''}})+'\\n');
if(command==='node'&&args.includes('tests/runtimes/speech-cpp/node-core-contract.mjs')) process.exit(Number(process.env.PROBE_EXIT));
`;
  for (const command of ['node', 'npm', 'ctest', 'git']) {
    await executable({ root, name: command, source: recorder });
  }
  const native = namedStep({ job: 'speech-cpp-checks', name: 'Test speech tooling, host and current native contracts' });
  const packaged = namedStep({ job: 'speech-cpp-checks', name: 'Test the packaged speech public API' });
  const download = actionSteps({ job: 'speech-cpp-checks', action: 'actions/download-artifact' })[0];
  assert.match(text({ value: object({ value: jobs['speech-cpp-checks'] }).if }), /^!cancelled\(\)/u);
  assert.equal(download?.if, "!cancelled() && steps.speech-check-tools.outcome == 'success' && needs.speech-cpp-package.result == 'success'");
  assert.equal(packaged.if, "!cancelled() && steps.speech-api-input.outcome == 'success'");
  assert.equal(native.if, undefined);
  assert.equal(native['continue-on-error'], undefined);
  const environment = { HOME: join(root, 'home'), GITHUB_WORKSPACE: root, LCB_SOURCE_COMMIT: 'a'.repeat(40), CALLS: join(root, 'calls.jsonl'), PROBE_EXIT: '0' };
  for (const job of ['speech-cpp-checks', 'speech-cpp-compile', 'speech-cpp-package']) {
    const checkout = namedStep({ job, name: 'Check out only this runtime\'s pinned upstreams' });
    await executeShell({ root, script: text({ value: checkout.run }), environment });
  }
  assert.deepEqual((await readCalls({ root })).map(call => ({ command: call.command, args: call.args })), Array.from({ length: 3 }, () => ({
    command: 'git',
    args: ['submodule', 'update', '--init', '--checkout', '--depth=1', '--', 'runtimes/speech-cpp/vendor/speech.cpp', 'runtimes/speech-cpp/vendor/ggml'],
  })));
  await executeShell({ root, script: offlineScript({ step: native }), environment });
  await executeShell({ root, script: offlineScript({ step: packaged }), environment });
  const calls = await readCalls({ root });
  const build = calls.find(call => call.command === 'node' && call.args[1] === 'build');
  assert.ok(build);
  assert.equal(build.args[0], 'build/tooling/runtimes/speech-cpp/tools/cli.js');
  for (const [flag, expected] of [['--kind', 'native'], ['--profile', 'webgpu-wasm32-jspi'], ['--variant', 'test'], ['--expected-commit', environment.LCB_SOURCE_COMMIT]] as const) {
    assert.equal(build.args[build.args.indexOf(flag) + 1], expected);
  }
  const ctest = calls.find(call => call.command === 'ctest');
  assert.ok(ctest);
  assert.deepEqual(ctest.args, ['--test-dir', 'build/runtimes/speech-cpp/build/native/webgpu-wasm32-jspi/test', '--output-on-failure']);
  const contracts = calls.find(call => call.command === 'node' && call.args[0] === '--test');
  assert.ok(contracts);
  assert.equal(contracts.environment?.BICORE_SPEECH_ABI_SCHEMA, join(root, 'build/runtimes/speech-cpp/build/native/webgpu-wasm32-jspi/test/generated/schema.json'));
  assert.ok(calls.indexOf(build) < calls.indexOf(ctest) && calls.indexOf(ctest) < calls.indexOf(contracts));
  const packagedApi = calls.find(call => call.args[1]?.endsWith('/current-api.test.js'));
  assert.equal(packagedApi?.environment?.BICORE_SPEECH_ABI_SCHEMA, join(root, 'build/runtime-packages/speech-cpp/profiles/webgpu-wasm32-jspi/browser/abi.json'));

  const assemble = namedStep({ job: 'speech-cpp-package', name: 'Assemble runtime-only package' });
  const startup = namedStep({ job: 'speech-cpp-package', name: 'Test packaged browser startup' });
  const probe = namedStep({ job: 'speech-cpp-package', name: 'Test the actual Node Wasm boundary' });
  await executeShell({ root, script: offlineScript({ step: assemble }), environment });
  await executeShell({ root, script: offlineScript({ step: startup }), environment });
  await executeShell({ root, script: offlineScript({ step: probe }), environment });
  const packaging = (await readCalls({ root })).slice(calls.length);
  const assembly = packaging[0];
  assert.ok(assembly);
  assert.deepEqual(assembly.args.slice(0, 5), ['build/tooling/tools/cli.js', 'package', 'runtime', '--runtime', 'speech-cpp']);
  assert.deepEqual(assembly.args.flatMap((arg, index) => arg === '--license-root' ? [assembly.args[index + 1]] : []), [join(root, 'build/runtimes/speech-cpp/toolchain-licenses/emscripten'), join(root, 'build/runtimes/speech-cpp/toolchain-licenses/emdawnwebgpu_pkg')]);
  assert.deepEqual(packaging[1]?.args, ['build/tooling/tools/cli.js', 'validation', 'runtime', '--phase', 'startup', '--runtime', 'speech-cpp', '--checkout-root', root, '--package', join(root, 'build/runtime-packages/speech-cpp'), '--receipt', join(root, 'build/runtime-validation/speech-cpp.json')]);
  assert.deepEqual(packaging[2]?.args, ['--experimental-wasm-jspi', 'tests/runtimes/speech-cpp/node-core-contract.mjs', join(root, 'build/runtime-packages/speech-cpp')]);
  await assert.rejects(executeShell({ root, script: offlineScript({ step: probe }), environment: { ...environment, PROBE_EXIT: '7' } }));

  const downloads = actionSteps({ job: 'publish', action: 'actions/download-artifact' }).map(step => step.with);
  assert.deepEqual(downloads, [
    { name: 'llama-cpp-package', path: 'build/package-inputs/llama-cpp/' },
    { name: 'stable-diffusion-cpp-package', path: 'build/package-inputs/stable-diffusion-cpp/' },
    { name: 'speech-cpp-package', path: 'build/package-inputs/speech-cpp/' },
  ]);
});

test('host checks run with Dawn alone and the SDK checks can still run after a host test failure', async () => {
  await using temporary = await temporaryDirectory();
  const root = temporary.root;
  const recorder = `\
const fs=require('node:fs');
const path=require('node:path');
fs.appendFileSync(process.env.CALLS,JSON.stringify({command:path.basename(process.argv[1]),args:process.argv.slice(2),environment:process.env.BICORE_EMSDK_ROOT?{sdk:process.env.BICORE_EMSDK_ROOT,node:process.env.BICORE_EMSDK_NODE,config:process.env.EM_CONFIG}:{}})+'\\n');
`;
  for (const command of ['node', 'npm', 'python3']) {
    await executable({ root, name: command, source: recorder });
  }
  const job = 'host-and-publication-checks';
  const step = namedStep({ job, name: 'Test host, tooling and current native contracts' });
  const prepare = namedStep({ job, name: 'Prepare pinned Dawn for the host and SDK checks' });
  const readSdk = namedStep({ job, name: 'Read the pinned compiler settings' });
  const acquireSdk = namedStep({ job, name: 'Acquire pinned compiler (online)' });
  const checkSdk = namedStep({ job, name: 'Test the pinned SDK Wasm and generated declarations' });
  assert.equal(step.if, undefined);
  assert.equal(step['continue-on-error'], undefined);
  assert.equal(prepare.id, 'host-dawn');
  const ordered = jobSteps({ name: job });
  const positions = [prepare, step, readSdk, acquireSdk, checkSdk].map(phase => ordered.indexOf(phase));
  assert.ok(positions.every((position, index) => position >= 0 && (index === 0 || position > (positions[index - 1] ?? -1))));
  assert.equal(readSdk.if, "!cancelled() && steps.host-dawn.outcome == 'success'");
  assert.equal(acquireSdk.if, "!cancelled() && steps.plan.outcome == 'success'");
  assert.equal(checkSdk.if, "!cancelled() && steps.host-sdk.outcome == 'success'");
  assert.equal(acquireSdk.run, namedStep({ job: 'llama-cpp-compile', name: 'Acquire pinned compiler (online)' }).run);
  const sdk = join(root, '.tools/emsdk');
  await mkdir(sdk, { recursive: true });
  await writeFile(join(sdk, 'emsdk_env.sh'), 'export EMSDK_NODE="$FIXTURE_NODE"\n');
  const environment = { HOME: join(root, 'home'), GITHUB_WORKSPACE: root, CALLS: join(root, 'calls.jsonl'), FIXTURE_NODE: process.execPath };
  for (const phase of [prepare, step, checkSdk]) {
    await executeShell({ root, script: offlineScript({ step: phase }), environment });
  }
  const calls = await readCalls({ root });
  assert.deepEqual(calls[1], { command: 'node', args: ['build/tooling/runtimes/llama-cpp/tools/cli.js', 'check-notices', '--checkout-root', root], environment: {} });
  const preparation = ['build/tooling/tools/cli.js', 'toolchain', 'prepare', '--checkout-root', root, '--tools-root', join(root, '.tools')];
  assert.deepEqual(calls.filter(call => call.args[1] === 'toolchain').map(call => call.args), [[...preparation, '--dawn-only'], preparation]);
  const python = calls.filter(call => call.command === 'python3');
  assert.deepEqual(python.map(call => call.args), [[
    '-m', 'unittest',
    'tests.runtimes.llama_cpp.source.test_native_quantization',
    'tests.runtimes.llama_cpp.source.test_q4_reference',
    'tests.runtimes.llama_cpp.source.test_iq3_fixtures',
    'tests.runtimes.llama_cpp.test_cmake_property_snapshot',
  ]]);
  const sdkTests = calls.filter(call => call.args.includes('build/tooling/tests/tools/toolchain/tsd-sdk.test.js'));
  assert.deepEqual(sdkTests, [{ command: 'node', args: ['--test', 'build/tooling/tests/tools/abi/wasm.test.js', 'build/tooling/tests/tools/toolchain/tsd-sdk.test.js'], environment: { sdk, node: process.execPath, config: join(sdk, '.emscripten') } }]);
});

test('detailed regressions download the startup-checked package for the same runtime', async () => {
  for (const [runtime, schemaVariable] of [['llama-cpp', 'BICORE_LLAMA_ABI_SCHEMA'], ['stable-diffusion-cpp', 'BICORE_SD_ABI_SCHEMA']] as const) {
    await using temporary = await temporaryDirectory();
    const root = temporary.root;
    const job = `${runtime}-validate`;
    const step = namedStep({ job, name: 'Validate the saved runtime package' });
    const packagePath = join(root, 'build/runtime-packages', runtime);
    assert.deepEqual(actionSteps({ job, action: 'actions/download-artifact' }).map(download => download.with), [{ name: `${runtime}-package`, path: `build/runtime-packages/${runtime}/` }]);
    assert.equal(step['continue-on-error'], undefined);
    await executable({
      root,
      name: 'node',
      source: `\
const args = process.argv.slice(2);
const schema = process.env.SCHEMA_VARIABLE;
const environment = process.env[schema] ? { [schema]: process.env[schema] } : {};
require('node:fs').appendFileSync(process.env.CALLS, JSON.stringify({ command: 'node', args, environment }) + '\\n');
if (args.includes('build/tooling/tests/tools/abi/current-api.test.js') && process.env.FAIL_ABI === 'true') process.exit(7);
`,
    });
    const environment = { HOME: join(root, 'home'), GITHUB_WORKSPACE: root, CALLS: join(root, 'calls.jsonl'), SCHEMA_VARIABLE: schemaVariable };
    await executeShell({ root, script: offlineScript({ step }), environment });
    const expected = [
      { command: 'node', args: ['--test', 'build/tooling/tests/tools/abi/current-api.test.js'], environment: { [schemaVariable]: join(packagePath, 'profiles/webgpu-wasm32-jspi/browser/abi.json') } },
      { command: 'node', args: ['build/tooling/tools/cli.js', 'validation', 'runtime', '--phase', 'regression', '--runtime', runtime, '--checkout-root', root, '--package', packagePath, '--receipt', join(root, `build/runtime-validation/${runtime}.json`)], environment: {} },
    ];
    assert.deepEqual(await readCalls({ root }), expected);
    await writeFile(environment.CALLS, '');
    await assert.rejects(executeShell({ root, script: offlineScript({ step }), environment: { ...environment, FAIL_ABI: 'true' } }));
    assert.deepEqual(await readCalls({ root }), expected);
  }
});

test('workflow telemetry is best effort and has a report for each job', () => {
  for (const name of Object.keys(jobs)) {
    const summary = namedStep({ job: name, name: 'Summarize optional pipeline timings' });
    assert.equal(summary.if, "always() && env.BIC_METRICS_MODE == 'basic'");
    assert.equal(summary['continue-on-error'], true);
    const upload = namedStep({ job: name, name: 'Upload optional pipeline timings' });
    assert.equal(upload['continue-on-error'], true);
    const options = object({ value: upload.with });
    assert.equal(options.path, '${{ runner.temp }}/bic-pipeline-metrics/');
    const suffix = name.endsWith('-compile') ? '-${{ matrix.profile }}-${{ matrix.variant }}' : '';
    assert.equal(options.name, `pipeline-metrics-\${{ github.run_attempt }}-${name}${suffix}`);
    assert.equal(options['include-hidden-files'], undefined);
    assert.equal(options['retention-days'], 14);
  }
});

test('all workflow shell parses and local phases actually invoke isolated Docker without credentials', async () => {
  await using temporary = await temporaryDirectory();
  const root = temporary.root;
  await runProcess({ command: 'bash', args: ['-n', '.github/scripts/run-offline.sh'], cwd: checkoutRoot });
  await executable({
    root,
    name: 'docker',
    source: `\
const fs = require('node:fs');
fs.appendFileSync(process.env.CALLS, JSON.stringify({command:'docker',args:process.argv.slice(2)})+'\\n');
`,
  });
  const hostSteps = new Set(['Read the pinned compiler settings', 'Show install commands for the published artifact', 'Preserve publication status after a reporting failure']);
  const online = new Set(['Acquire offline execution image (online)', 'Acquire locked development dependencies (online)', 'Acquire pinned Dawn headers for native contracts (online)', 'Acquire pinned compiler (online)', 'Acquire pinned Dawn archive (online on cache miss)', 'Install browser test harness (online)', 'Prepare Chromium (online)', 'Publish append-only artifact commit', 'Resolve the published npm dependency (online)']);
  for (const filename of await readdir(join(checkoutRoot, '.github/workflows'))) {
    if (!filename.endsWith('.yml')) {
      continue;
    }
    const parsed = object({ value: load(await readFile(join(checkoutRoot, '.github/workflows', filename), 'utf8')) });
    for (const job of Object.values(object({ value: parsed.jobs }))) {
      for (const value of list({ value: object({ value: job }).steps })) {
        const step = object({ value });
        if (step.run === undefined) {
          continue;
        }
        await executeShell({ root, script: 'bash -n <<\'WORKFLOW\'\n' + text({ value: step.run }) + '\nWORKFLOW\n', environment: {} });
        if (text({ value: step.run }).includes("<<'OFFLINE'")) {
          await runProcess({ command: 'bash', args: ['-n'], cwd: checkoutRoot, input: offlineScript({ step }) });
        }
      }
    }
  }
  for (const [name] of Object.entries(jobs)) {
    for (const step of jobSteps({ name })) {
      if (step.run === undefined || hostSteps.has(text({ value: step.name })) || online.has(text({ value: step.name })) || text({ value: step.name }).startsWith('Check out')) {
        continue;
      }
      await executeShell({ root, script: text({ value: step.run }), environment: { CALLS: join(root, 'calls.jsonl'), GITHUB_WORKSPACE: root, RUNNER_TEMP: root, BIC_OFFLINE_IMAGE: 'fixture' } });
    }
  }
  const calls = await readCalls({ root });
  assert.ok(calls.length >= 20);
  for (const { args } of calls) {
    assert.ok(args.includes('--pull=never'));
    assert.equal(args[args.indexOf('--network') + 1], 'none');
    assert.ok(args.includes('--cap-drop=ALL'));
    assert.ok(args.includes('--security-opt=no-new-privileges:true'));
    assert.ok(args.includes(`HOME=${root}/bicore-home`));
    assert.ok(args.includes('CI=true'));
    for (let index = 0; index < args.length; index++) {
      if (args[index] === '-e') {
        assert.ok(!/^(GH_TOKEN|GITHUB_TOKEN)(=|$)/u.test(args[index + 1] ?? ''));
      }
    }
  }
});

test('the actual image acquisition builds from a source-free context and fingerprints installed tools', async () => {
  for (const name of Object.keys(jobs)) {
    await using temporary = await temporaryDirectory();
    const root = temporary.root;
    const runner = join(root, 'runner');
    await mkdir(runner);
    await executable({
      root,
      name: 'docker',
      source: `\
const fs=require('node:fs');
const path=require('node:path');
const args=process.argv.slice(2);
fs.appendFileSync(process.env.CALLS,JSON.stringify({command:'docker',args})+'\\n');
if(args[0]==='build'){
  const context=args.at(-1);
  fs.writeFileSync(process.env.RECIPE,fs.readFileSync(path.join(context,'Dockerfile')));
  fs.writeFileSync(process.env.CONTEXT,JSON.stringify({root:context,files:fs.readdirSync(context)}));
}
if(args[0]==='image') process.stdout.write('node-fixture\\nubuntu-fixture\\n');
if(args[0]==='run') process.stdout.write('installed-package-fixture\\nnode-version-fixture\\n');
`,
    });
    await executeShell({ root, script: text({ value: namedStep({ job: name, name: 'Acquire offline execution image (online)' }).run }), environment: { RUNNER_TEMP: runner, GITHUB_WORKSPACE: join(root, 'checkout'), GITHUB_JOB: name, GITHUB_ENV: join(root, 'environment'), CALLS: join(root, 'calls.jsonl'), RECIPE: join(root, 'recipe'), CONTEXT: join(root, 'context.json') } });
    const context = object({ value: JSON.parse(await readFile(join(root, 'context.json'), 'utf8')) });
    assert.ok(text({ value: context.root }).startsWith(runner + '/'));
    assert.deepEqual(context.files, ['Dockerfile']);
    const recipe = await readFile(join(root, 'recipe'), 'utf8');
    const copies = recipe.split('\n').map(line => line.trim().split(/\s+/u)).filter(words => words[0] === 'COPY');
    assert.ok(copies.length > 0);
    for (const words of copies) {
      assert.ok(words[1]?.startsWith('--from='), 'Only an acquired base image supplies copied files');
    }
    const commands = await readCalls({ root });
    const nodeImage = name === 'speech-cpp-package' ? 'node:24.13.1-bookworm' : 'node:22.16.0-bookworm';
    const bases = recipe.split('\n').map(line => line.trim().split(/\s+/u)).filter(words => words[0] === 'FROM').map(words => words[1]);
    assert.deepEqual(bases, [nodeImage, 'ubuntu:24.04']);
    assert.deepEqual(commands.filter(call => call.args[0] === 'pull').map(call => call.args[1]), bases);
    assert.deepEqual(commands.find(call => call.args[0] === 'image')?.args.slice(4), bases);
    const observation = commands.find(call => call.args[0] === 'run');
    assert.ok(observation);
    assert.equal(observation.args[observation.args.indexOf('--network') + 1], 'none');
    assert.ok(observation.args.includes(`${join(root, 'checkout')}:${join(root, 'checkout')}`));
    assert.ok(observation.args.includes(`${runner}:${runner}`));
    assert.equal(observation.args[observation.args.indexOf('-w') + 1], join(root, 'checkout'));
    const exports = await readFile(join(root, 'environment'), 'utf8');
    assert.match(exports, /BIC_OFFLINE_ENVIRONMENT=[0-9a-f]{64}/u);
    const identity = await readFile(join(text({ value: context.root }), 'environment'), 'utf8');
    assert.ok(identity.includes('installed-package-fixture'));
    assert.ok(identity.includes('node-fixture'));
  }
});

test('Dawn acquisition verifies restored bytes, recovers failed restores and never installs a bad download', async () => {
  const script = text({ value: namedStep({ job: 'llama-cpp-compile', name: 'Acquire pinned Dawn archive (online on cache miss)' }).run });
  for (const scenario of [
    { outcome: 'success', matched: 'exact', cached: 'archive', checksum: 'good', downloads: 0, succeeds: true },
    { outcome: 'failure', matched: 'exact', cached: 'archive', checksum: 'good', downloads: 1, succeeds: true },
    { outcome: 'success', matched: 'different', cached: 'archive', checksum: 'good', downloads: 1, succeeds: true },
    { outcome: 'success', matched: 'exact', cached: 'corrupt', checksum: 'good', downloads: 1, succeeds: true },
    { outcome: 'success', matched: '', cached: undefined, checksum: 'bad', downloads: 1, succeeds: false },
  ]) {
    await using temporary = await temporaryDirectory();
    const root = temporary.root;
    await executable({
      root,
      name: 'curl',
      source: `\
const fs = require('node:fs');
const args=process.argv.slice(2);
fs.appendFileSync(process.env.CALLS,JSON.stringify({command:'curl',args})+'\\n');
fs.writeFileSync(args[args.indexOf('--output')+1],'archive');
`,
    });
    if (scenario.cached !== undefined) {
      await mkdir(join(root, '.cache/restored/dawn'), { recursive: true });
      await writeFile(join(root, '.cache/restored/dawn/emdawnwebgpu_pkg-fixture.zip'), scenario.cached);
    }
    const environment = { DAWN_TAG: 'fixture', DAWN_SHA256: scenario.checksum === 'good' ? createHash('sha256').update('archive').digest('hex') : '0'.repeat(64), REQUESTED: 'exact', MATCHED: scenario.matched, RESTORE_OUTCOME: scenario.outcome, CALLS: join(root, 'calls.jsonl') };
    if (scenario.succeeds) {
      await executeShell({ root, script, environment });
    } else {
      await assert.rejects(executeShell({ root, script, environment }));
    }
    const calls = scenario.downloads ? await readCalls({ root }) : [];
    assert.equal(calls.length, scenario.downloads);
    for (const { args } of calls) {
      assert.equal(args[args.indexOf('--proto') + 1], '=https');
      assert.equal(args[args.indexOf('--proto-redir') + 1], '=https');
      assert.ok(args.includes('https://github.com/google/dawn/releases/download/fixture/emdawnwebgpu_pkg-fixture.zip'));
    }
    assert.deepEqual(await readdir(join(root, '.tools/downloads')), scenario.succeeds ? ['emdawnwebgpu_pkg-fixture.zip'] : []);
  }
});

test('SDK acquisition checks the release, installs LL SDK development dependencies and precedes offline builds', async () => {
  for (const scenario of [
    { job: 'llama-cpp-compile', matches: true, npmExit: 0 },
    { job: 'llama-cpp-compile', matches: false, npmExit: 0 },
    { job: 'llama-cpp-compile', matches: true, npmExit: 7 },
    { job: 'stable-diffusion-cpp-compile', matches: true, npmExit: 0 },
    { job: 'stable-diffusion-cpp-compile', matches: false, npmExit: 0 },
    { job: 'speech-cpp-compile', matches: true, npmExit: 0 },
    { job: 'speech-cpp-compile', matches: false, npmExit: 0 },
  ]) {
    await using temporary = await temporaryDirectory();
    const root = temporary.root;
    const sdk = join(root, '.tools/emsdk');
    await mkdir(sdk, { recursive: true });
    await writeFile(join(sdk, 'emscripten-releases-tags.json'), JSON.stringify({ releases: { '6.0.9': (scenario.matches ? 'a' : 'b').repeat(40) } }));
    const binary = await executable({
      root,
      name: 'emsdk',
      source: `\
require('node:fs').appendFileSync(process.env.CALLS,JSON.stringify({command:'emsdk',args:process.argv.slice(2)})+'\\n');
`,
    });
    await cp(binary, join(sdk, 'emsdk'));
    for (const command of ['npm', 'docker']) {
      await executable({
        root,
        name: command,
        source: `\
const fs=require('node:fs');
const command=require('node:path').basename(process.argv[1]);
fs.appendFileSync(process.env.CALLS,JSON.stringify({command,args:process.argv.slice(2),environment:{cwd:process.cwd()}})+'\\n');
if(command==='npm') process.exit(Number(process.env.NPM_EXIT));
`,
      });
    }
    const steps = jobSteps({ name: scenario.job });
    const acquisition = namedStep({ job: scenario.job, name: 'Acquire pinned compiler (online)' });
    const build = namedStep({ job: scenario.job, name: 'Prepare, build and stage the selected runtime' });
    assert.ok(steps.indexOf(namedStep({ job: scenario.job, name: 'Acquire locked development dependencies (online)' })) < steps.indexOf(acquisition));
    assert.ok(steps.indexOf(acquisition) < steps.indexOf(build));
    const script = text({ value: acquisition.run });
    const environment = { EMSDK_VERSION: '6.0.9', EMSCRIPTEN_RELEASE: 'a'.repeat(40), CALLS: join(root, 'calls.jsonl'), NPM_EXIT: String(scenario.npmExit), GITHUB_WORKSPACE: root, RUNNER_TEMP: root, BIC_OFFLINE_IMAGE: 'fixture' };
    if (scenario.matches && scenario.npmExit === 0) {
      await executeShell({ root, script, environment });
      await executeShell({ root, script: text({ value: build.run }), environment });
    } else {
      await assert.rejects(executeShell({ root, script, environment }));
    }
    if (!scenario.matches) {
      await assert.rejects(readFile(join(root, 'calls.jsonl')));
      continue;
    }
    const calls = await readCalls({ root });
    const buildCall = scenario.npmExit === 0 ? calls.pop() : undefined;
    assert.deepEqual(calls.slice(0, 2).map(({ command, args }) => ({ command, args })), [{ command: 'emsdk', args: ['install', '6.0.9'] }, { command: 'emsdk', args: ['activate', '6.0.9'] }]);
    assert.deepEqual(calls.slice(2), scenario.job === 'llama-cpp-compile' ? [{ command: 'npm', args: ['ci', '--prefix', join(sdk, 'upstream/emscripten'), '--include=dev', '--ignore-scripts', '--no-audit', '--no-fund'], environment: { cwd: sdk } }] : []);
    if (scenario.npmExit === 0) {
      assert.ok(buildCall);
      assert.equal(buildCall.command, 'docker');
      assert.equal(buildCall.args[buildCall.args.indexOf('--network') + 1], 'none');
      assert.ok(buildCall.args.includes('NPM_CONFIG_OFFLINE=true'));
    }
  }
});

test('the actual build shell adopts complete cache restores and keeps fresh caches after failed or absent restores', async () => {
  for (const scenario of [
    { outcome: 'success', restored: true, save: false, expected: 'restored', mode: 'off', runtime: 'llama-cpp', profile: 'cpu-wasm32', variant: 'browser', emMatch: 'exact' },
    { outcome: 'failure', restored: true, save: false, expected: 'fresh', mode: 'off', runtime: 'llama-cpp', profile: 'cpu-wasm32', variant: 'browser', emMatch: 'prefix' },
    { outcome: 'success', restored: false, save: false, expected: 'fresh', mode: 'off', runtime: 'llama-cpp', profile: 'cpu-wasm32', variant: 'browser', emMatch: 'prefix' },
    { outcome: 'success', restored: true, save: true, expected: 'restored', mode: 'off', runtime: 'llama-cpp', profile: 'cpu-wasm32', variant: 'browser', emMatch: 'prefix' },
    { outcome: 'success', restored: true, save: false, expected: 'restored', mode: 'basic', runtime: 'llama-cpp', profile: 'cpu-wasm32', variant: 'browser', emMatch: 'exact' },
    { outcome: 'success', restored: true, save: false, expected: 'restored', mode: 'off', runtime: 'llama-cpp', profile: 'webgpu-wasm32-jspi', variant: 'browser', emMatch: 'exact' },
    { outcome: 'success', restored: true, save: false, expected: 'restored', mode: 'off', runtime: 'stable-diffusion-cpp', profile: 'webgpu-wasm32-jspi', variant: 'browser', emMatch: 'prefix' },
    { outcome: 'success', restored: true, save: false, expected: 'restored', mode: 'off', runtime: 'llama-cpp', profile: 'cpu-wasm32', variant: 'browser', emMatch: 'other' },
    { outcome: 'success', restored: true, save: false, expected: 'restored', mode: 'off', runtime: 'stable-diffusion-cpp', profile: 'webgpu-wasm32-jspi', variant: 'browser', emMatch: 'malformed' },
    { outcome: 'success', restored: true, save: false, expected: 'restored', mode: 'off', runtime: 'llama-cpp', profile: 'cpu-wasm32', variant: 'test', emMatch: 'exact' },
    { outcome: 'success', restored: true, save: true, expected: 'restored', mode: 'off', runtime: 'speech-cpp', profile: 'webgpu-wasm32-jspi', variant: 'browser', emMatch: 'prefix' },
    { outcome: 'failure', restored: true, save: false, expected: 'fresh', mode: 'basic', runtime: 'speech-cpp', profile: 'webgpu-wasm32-jspi', variant: 'test', emMatch: 'exact' },
  ]) {
    const script = offlineScript({ step: namedStep({ job: `${scenario.runtime}-compile`, name: 'Prepare, build and stage the selected runtime' }) });
    await using temporary = await temporaryDirectory();
    const root = temporary.root;
    const emCachePath = '.tools/emsdk/upstream/emscripten/cache';
    const ccCachePath = `.cache/ccache/${scenario.runtime}/${scenario.profile}/${scenario.variant}`;
    const emStagingPath = `.cache/restored/${scenario.runtime}/${scenario.profile}/${scenario.variant}/em`;
    const ccStagingPath = `.cache/restored/${scenario.runtime}/${scenario.profile}/${scenario.variant}/cc`;
    const prefix = 'partition-cc-';
    const ccKey = prefix + 'a'.repeat(40);
    const emRestorePrefix = 'partition-em-';
    const emKey = emRestorePrefix + 'a'.repeat(40);
    const emMatched = scenario.emMatch === 'exact' ? emKey : scenario.emMatch === 'prefix' ? emRestorePrefix + 'b'.repeat(40) : scenario.emMatch === 'other' ? 'other-partition-em-' + 'b'.repeat(40) : emRestorePrefix + 'not-a-source';
    await mkdir(join(root, 'build/tooling/tools/toolchain'), { recursive: true });
    await writeFile(join(root, 'build/tooling/package.json'), '{"type":"module"}');
    await cp(join(checkoutRoot, 'build/tooling/tools/toolchain/cache.js'), join(root, 'build/tooling/tools/toolchain/cache.js'));
    await writeFile(join(root, 'build/cache-plan.json'), JSON.stringify({ dawnKey: 'dawn', emKey, emRestorePrefix, ccKey, ccRestorePrefix: prefix, emStagingPath, ccStagingPath, emCachePath, ccCachePath }));
    for (const location of [emCachePath, ccCachePath]) {
      await mkdir(join(root, location), { recursive: true });
      await writeFile(join(root, location, 'marker'), 'fresh');
    }
    if (scenario.restored) {
      for (const location of [emStagingPath, ccStagingPath]) {
        await mkdir(join(root, location), { recursive: true });
        await writeFile(join(root, location, 'marker'), 'restored');
      }
    }
    await writeFile(join(root, '.tools/emsdk/emsdk_env.sh'), 'unset EM_CACHE\n');
    await mkdir(join(root, '.tools/downloads'), { recursive: true });
    await writeFile(join(root, '.tools/downloads/dawn.zip'), 'verified fixture');
    await executable({
      root,
      name: 'node',
      source: `\
const cp=require('node:child_process');
const fs=require('node:fs');
const args=process.argv.slice(2);
if(args[0]?.includes('/ci/cli.js')) process.exit(7);
if(args[0]==='--input-type=module'){
  const result=cp.spawnSync(process.env.REAL_NODE,args,{stdio:'inherit'});
  process.exit(result.status??1);
}
fs.appendFileSync(process.env.CALLS,JSON.stringify({command:'node',args,environment:{EM_CACHE:process.env.EM_CACHE??'',BICORE_LLAMA_ABI_SCHEMA:process.env.BICORE_LLAMA_ABI_SCHEMA??'',BICORE_SD_ABI_SCHEMA:process.env.BICORE_SD_ABI_SCHEMA??'',BICORE_SPEECH_ABI_SCHEMA:process.env.BICORE_SPEECH_ABI_SCHEMA??'',BICORE_EMSDK_ROOT:process.env.BICORE_EMSDK_ROOT??'',BICORE_EMSDK_NODE:process.env.BICORE_EMSDK_NODE??'',EM_CONFIG:process.env.EM_CONFIG??''}})+'\\n');
`,
    });
    await executable({
      root,
      name: 'ccache',
      source: `\
require('node:fs').appendFileSync(process.env.CALLS,JSON.stringify({command:'ccache',args:process.argv.slice(2)})+'\\n');
`,
    });
    const planned = join(root, emCachePath);
    await executeShell({ root, script, environment: { HOME: join(root, 'home'), GITHUB_WORKSPACE: root, LCB_SOURCE_COMMIT: 'a'.repeat(40), BIC_METRICS_MODE: scenario.mode, BIC_METRICS_RUNTIME: scenario.runtime, BIC_METRICS_PROFILE: scenario.profile, BIC_METRICS_VARIANT: scenario.variant, DAWN_MATCHED: '', DAWN_HIT: '', DAWN_OUTCOME: 'skipped', EM_CACHE: planned, EM_MATCHED: emMatched, EM_HIT: scenario.emMatch === 'exact' ? 'true' : 'false', CC_MATCHED: prefix + 'b'.repeat(40), CC_HIT: 'false', EM_OUTCOME: scenario.outcome, CC_OUTCOME: scenario.outcome, SAVE_CACHES: String(scenario.save), EMSDK_NODE: process.execPath, REAL_NODE: process.execPath, CALLS: join(root, 'calls.jsonl') } });
    const expectedEm = scenario.emMatch === 'other' || scenario.emMatch === 'malformed' ? 'fresh' : scenario.expected;
    assert.equal(await readFile(join(root, scenario.save ? emStagingPath : emCachePath, 'marker'), 'utf8'), expectedEm);
    assert.equal(await readFile(join(root, scenario.save ? ccStagingPath : ccCachePath, 'marker'), 'utf8'), scenario.expected);
    const calls = await readCalls({ root });
    const builds = calls.filter(call => call.command === 'node' && call.args[0]?.includes(`/runtimes/${scenario.runtime}/tools/cli.js`));
    assert.equal(builds.length, 1);
    const build = builds[0];
    assert.ok(build);
    assert.equal(build.environment?.EM_CACHE, planned);
    assert.equal(build.args[build.args.indexOf('--kind') + 1], 'wasm');
    assert.equal(build.args[build.args.indexOf('--variant') + 1], scenario.variant);
    assert.equal(build.args[build.args.indexOf('--expected-commit') + 1], 'a'.repeat(40));
    assert.ok(build.args.includes('--fresh'));
    assert.ok(calls.every(call => call.args[0] !== '--test'));
  }
});

test('publication freezing keeps its saved receipt and errors without echoing the receipt', async () => {
  const step = namedStep({ job: 'publish', name: 'Assemble and freeze the source-bound publication tree' });
  for (const code of [0, 7]) {
    await using temporary = await temporaryDirectory();
    const root = temporary.root;
    const source = 'a'.repeat(40);
    const tree = 'b'.repeat(40);
    const integrity = 'c'.repeat(64);
    const output = join(root, 'outputs');
    await executable({
      root,
      name: 'node',
      source: `\
const fs=require('node:fs');
const path=require('node:path');
const args=process.argv.slice(2);
if(args[1]==='publication'&&args[2]==='assemble-freeze'){
  const receipt=args[args.indexOf('--receipt')+1];
  fs.mkdirSync(path.dirname(receipt),{recursive:true});
  fs.writeFileSync(receipt,JSON.stringify({tree:process.env.TREE,sourceCommit:process.env.LCB_SOURCE_COMMIT,integrity:{sha256:process.env.INTEGRITY}}));
  process.stdout.write('receipt stdout\\n');
  process.stderr.write('freeze diagnostic\\n');
  process.exit(Number(process.env.FREEZE_EXIT));
}
process.exit(8);
`,
    });
    const operation = runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', offlineScript({ step })], cwd: root, capture: 'file', env: { kind: 'inherit', removePrefixes: [], overrides: { PATH: `${join(root, 'bin')}:${READ_ENV_PATH() ?? ''}`, HOME: join(root, 'home'), GITHUB_WORKSPACE: root, GITHUB_OUTPUT: output, GITHUB_REPOSITORY: 'owner/repo', LCB_SOURCE_COMMIT: source, TREE: tree, INTEGRITY: integrity, FREEZE_EXIT: String(code) } } });
    if (code === 0) {
      const result = await operation;
      assert.equal(result.stdout.toString(), '');
      assert.equal(result.stderr.toString(), 'freeze diagnostic\n');
      assert.equal(await readFile(output, 'utf8'), `tree=${tree}\nsource-commit=${source}\nintegrity-sha256=${integrity}\n`);
    } else {
      await assert.rejects(operation, (error: unknown) => error instanceof ProcessError && error.exitCode === code && error.stderr.toString() === 'freeze diagnostic\n' && error.stdout.toString() === '');
      await assert.rejects(readFile(output));
    }
    assert.deepEqual(JSON.parse(await readFile(join(root, 'build/publication/frozen.json'), 'utf8')), { tree, sourceCommit: source, integrity: { sha256: integrity } });
  }
});

test('inline publication appends the frozen tree and reconciles an accepted push with a lost response', async () => {
  await using temporary = await temporaryDirectory();
  const root = temporary.root;
  const remote = join(root, 'remote.git');
  await git({ root, args: ['init', '--bare', '-q', remote] });
  await executable({
    root,
    name: 'git',
    source: `\
const cp=require('node:child_process');
const fs=require('node:fs');
const args=process.argv.slice(2);
const result=cp.spawnSync(process.env.REAL_GIT,args,{stdio:'inherit'});
if(args.includes('push')&&result.status===0&&!fs.existsSync(process.env.LOST_RESPONSE)){
  fs.writeFileSync(process.env.LOST_RESPONSE,'lost');
  process.exit(1);
}
process.exit(result.status??1);
`,
  });
  const gitExecutable = (await runProcess({ command: 'bash', args: ['-c', 'command -v git'], cwd: root })).stdout.toString().trim();
  const script = text({ value: namedStep({ job: 'publish', name: 'Publish append-only artifact commit' }).run });
  let parent: string | undefined;
  const commits: string[] = [];
  for (let index = 0; index < 2; index++) {
    const workspace = join(root, `source-${index}`);
    const work = join(workspace, 'build/publication/package');
    await mkdir(work, { recursive: true });
    await git({ root: work, args: ['init', '-q'] });
    await writeFile(join(work, 'core.wasm'), `runtime ${index}`);
    await writeFile(join(work, 'package.json'), JSON.stringify({ name: 'browser-inference-core', version: `1.0.${index}`, private: true, files: ['core.wasm'] }));
    await git({ root: work, args: ['add', '.'] });
    const tree = await git({ root: work, args: ['write-tree'] });
    const output = join(root, `output-${index}`);
    await executeShell({ root, script, environment: { GITHUB_WORKSPACE: workspace, RUNNER_TEMP: root, GITHUB_OUTPUT: output, FROZEN_TREE: tree, FROZEN_SOURCE: String(index + 1).repeat(40), ARTIFACT_BRANCH: 'artifacts', PUBLISH_REMOTE: remote, PUBLISH_ATTEMPTS: '2', GH_TOKEN: 'test-only', REAL_GIT: gitExecutable, LOST_RESPONSE: join(root, `lost-${index}`) } });
    const commit = (await readFile(output, 'utf8')).trim().slice('commit='.length);
    assert.equal(await git({ root: remote, args: ['rev-parse', 'refs/heads/artifacts'] }), commit);
    assert.equal(await git({ root: remote, args: ['show', '-s', '--format=%T', commit] }), tree);
    assert.equal(await git({ root: remote, args: ['show', '-s', '--format=%P', commit] }), parent ?? '');
    assert.equal(await git({ root: remote, args: ['show', '-s', '--format=%an <%ae>|%cn <%ce>', commit] }), 'github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>|github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>');
    commits.push(commit);
    parent = commit;
  }
  await assert.rejects(executeShell({ root, script, environment: { GITHUB_WORKSPACE: root, RUNNER_TEMP: root, GITHUB_OUTPUT: join(root, 'rejected-output'), FROZEN_TREE: 'a'.repeat(40), FROZEN_SOURCE: 'b'.repeat(40), ARTIFACT_BRANCH: 'feature/source', PUBLISH_REMOTE: remote, PUBLISH_ATTEMPTS: '2', GH_TOKEN: 'test-only', REAL_GIT: gitExecutable, LOST_RESPONSE: join(root, 'rejected-response') } }));
  assert.equal(await git({ root: remote, args: ['rev-parse', 'refs/heads/artifacts'] }), parent);
  // The earlier artifact remains installable after a later append. Use only a
  // local Git transport; the workflow's real npm flags and isolated config run.
  const first = commits[0];
  assert.ok(first);
  const fixture = join(root, 'build/consumer-lock');
  await mkdir(fixture, { recursive: true });
  await writeFile(join(fixture, 'package.json'), JSON.stringify({ name: 'local-artifact-consumer', version: '0.0.0', private: true, dependencies: { 'browser-inference-core': `git+${pathToFileURL(remote).href}#${first}` } }));
  await writeFile(join(fixture, 'empty.npmrc'), '');
  await writeFile(join(fixture, 'empty-global.npmrc'), '');
  await executeShell({ root, script: text({ value: namedStep({ job: 'consumer-report', name: 'Resolve the published npm dependency (online)' }).run }), environment: { HOME: join(root, 'home'), NPM_CONFIG_OFFLINE: 'true', REAL_GIT: gitExecutable, LOST_RESPONSE: join(root, 'resolved-response') } });
  const lock = object({ value: JSON.parse(await readFile(join(fixture, 'package-lock.json'), 'utf8')) });
  const packages = object({ value: lock.packages });
  assert.deepEqual(Object.keys(packages).sort(), ['', 'node_modules/browser-inference-core']);
  const installed = object({ value: packages['node_modules/browser-inference-core'] });
  assert.equal(installed.version, '1.0.0');
  assert.ok(text({ value: installed.resolved }).endsWith('#' + first));
});

test('the published install guide survives later processing and optional source-title failures', async () => {
  await using temporary = await temporaryDirectory();
  const root = temporary.root;
  const summary = join(root, 'job summary.md');
  const previousSummary = '# Earlier job output\n\n';
  await executable({ root, name: 'node', source: 'process.exit(7);' });
  await executable({
    root,
    name: 'git',
    source: `\
if (process.env.FAIL_TITLE === 'true') process.exit(8);
process.stdout.write('feat: runtime <one> & "two"\\n');
`,
  });
  const step = namedStep({ job: 'publish', name: 'Show install commands for the published artifact' });
  const guide = namedStep({ job: 'publish', name: 'Record the published installation guide' });
  assert.ok(jobSteps({ name: 'publish' }).every(entry => entry.name !== 'Resolve the published npm dependency (online)' && entry.name !== 'Generate consumer integration metadata from the publication guide'));
  const environment = { HOME: join(root, 'home'), GITHUB_WORKSPACE: root, GITHUB_STEP_SUMMARY: summary, GITHUB_REPOSITORY: 'owner/repo', ARTIFACT_COMMIT: 'a'.repeat(40), LCB_SOURCE_COMMIT: 'b'.repeat(40), GITHUB_RUN_ID: '1234', GITHUB_RUN_ATTEMPT: '7', GITHUB_OUTPUT: join(root, 'output') };
  for (const failsTitle of ['false', 'true']) {
    await writeFile(summary, previousSummary);
    await executeShell({ root, script: text({ value: step.run }), environment: { ...environment, FAIL_TITLE: failsTitle } });
    await assert.rejects(executeShell({ root, script: offlineScript({ step: guide }), environment }));
    const displayed = await readFile(summary, 'utf8');
    assert.ok(displayed.startsWith(previousSummary));
    assert.ok(displayed.includes(`npm install github:owner/repo#${environment.ARTIFACT_COMMIT}`));
    assert.ok(displayed.includes(`npm install llama-cpp-browser-core@github:owner/repo#${environment.ARTIFACT_COMMIT}`));
    assert.ok(displayed.includes(`npm install stable-diffusion-cpp-browser-core@github:owner/repo#${environment.ARTIFACT_COMMIT}`));
    assert.ok(displayed.includes(`npm install speech-cpp-browser-core@github:owner/repo#${environment.ARTIFACT_COMMIT}`));
    assert.ok(displayed.includes(`**Source commit:** \`${environment.LCB_SOURCE_COMMIT}\``));
    assert.equal(displayed.includes('**Source commit title:**'), failsTitle === 'false');
    if (failsTitle === 'false') {
      assert.ok(displayed.includes('<code>feat: runtime &lt;one&gt; &amp; &quot;two&quot;</code>'));
    }
  }
});

test('the separate consumer job reports the original publication attempt and only adds integration details to the summary', async () => {
  await using temporary = await temporaryDirectory();
  const root = temporary.root;
  const summary = join(root, 'job summary.md');
  const previousSummary = '# Earlier job output\n\n';
  const markdown = `\
# Browser inference runtime artifact

\`\`\`sh
npm install github:owner/repo#${'a'.repeat(40)}
\`\`\`
`;
  const integrationDetails = `\
## Consumer integration details

| Runtime | ABI | Profiles |
| --- | --- | --- |
| llama-cpp | 3 | cpu-wasm32 |
`;
  await writeFile(summary, previousSummary);
  await executable({
    root,
    name: 'node',
    source: `\
const fs = require('node:fs');
const path = require('node:path');
const args = process.argv.slice(2);
fs.appendFileSync(process.env.CALLS,JSON.stringify({command:'node',args})+'\\n');
if (process.env.FAIL_REPORT === 'true') process.exit(7);
const output = args[args.indexOf('--output') + 1];
fs.mkdirSync(output, {recursive:true});
fs.writeFileSync(path.join(output, 'report.md'), process.env.REPORT_MARKDOWN);
fs.appendFileSync(args[args.indexOf('--summary') + 1], process.env.INTEGRATION_DETAILS);
`,
  });
  const job = 'consumer-report';
  const step = namedStep({ job, name: 'Generate consumer integration metadata from the publication guide' });
  const publisher = object({ value: jobs.publish });
  assert.deepEqual(publisher.outputs, { 'artifact-commit': '${{ steps.publish.outputs.commit }}', 'published-attempt': '${{ steps.publication-guide.outputs.attempt }}' });
  assert.deepEqual(actionSteps({ job, action: 'actions/download-artifact' }).map(download => download.with), [{ name: 'publication-guide-${{ needs.publish.outputs.published-attempt }}', path: 'build/publication-guide/' }]);
  assert.equal(object({ value: jobSteps({ name: job }).find(entry => entry.id === 'upload-consumer-report')?.with }).name, 'consumer-update-${{ needs.publish.outputs.published-attempt }}');
  const environment = { HOME: join(root, 'home'), GITHUB_WORKSPACE: root, GITHUB_STEP_SUMMARY: summary, GITHUB_RUN_ID: '1234', GITHUB_RUN_ATTEMPT: '12', PUBLISHED_ATTEMPT: '7', GITHUB_REPOSITORY: 'owner/repo', ARTIFACT_COMMIT: 'a'.repeat(40), LCB_SOURCE_COMMIT: 'b'.repeat(40), CALLS: join(root, 'calls.jsonl'), REPORT_MARKDOWN: markdown, INTEGRATION_DETAILS: integrationDetails };
  await executeShell({ root, script: offlineScript({ step }), environment });
  const calls = await readCalls({ root });
  assert.equal(calls.length, 1);
  const args = calls[0]?.args;
  assert.ok(args);
  assert.deepEqual(args.slice(0, 3), ['build/tooling/tools/cli.js', 'publication', 'report']);
  assert.equal(args[args.indexOf('--run-id') + 1], '1234');
  assert.equal(args[args.indexOf('--run-attempt') + 1], '7');
  assert.equal(args[args.indexOf('--commit') + 1], 'a'.repeat(40));
  assert.equal(args[args.indexOf('--guide') + 1], join(root, 'build/publication-guide'));
  assert.equal(args[args.indexOf('--source-commit') + 1], 'b'.repeat(40));
  assert.equal(args[args.indexOf('--summary') + 1], summary);
  assert.equal(await readFile(summary, 'utf8'), previousSummary + integrationDetails);
  assert.equal(await readFile(join(root, 'build/consumer-update/report.md'), 'utf8'), markdown);

  await assert.rejects(executeShell({ root, script: offlineScript({ step }), environment: { ...environment, FAIL_REPORT: 'true' } }));
  assert.equal(await readFile(summary, 'utf8'), previousSummary + integrationDetails);
});

test('the actions lock covers every workflow usage through an existing resolved action', async () => {
  const locked = object({ value: load(await readFile(join(checkoutRoot, '.github/workflows/actions.lock'), 'utf8')) });
  const indexed = object({ value: locked.workflows });
  const dependencies = object({ value: locked.dependencies });
  const filenames = (await readdir(join(checkoutRoot, '.github/workflows'))).filter(name => name.endsWith('.yml')).sort();
  assert.deepEqual(Object.keys(indexed).sort(), filenames.map(name => '.github/workflows/' + name));
  for (const filename of filenames) {
    const document = object({ value: load(await readFile(join(checkoutRoot, '.github/workflows', filename), 'utf8')) });
    const actual = new Set<string>();
    for (const job of Object.values(object({ value: document.jobs }))) {
      for (const value of list({ value: object({ value: job }).steps })) {
        const step = object({ value });
        if (step.uses === undefined) {
          continue;
        }
        const [action, version] = text({ value: step.uses }).split('@');
        assert.ok(action && version);
        const ref = action.split('/').slice(0, 2).join('/') + '@' + version;
        actual.add(ref);
        const resolved = object({ value: dependencies[ref] });
        assert.equal(resolved.ref, version);
        assert.match(text({ value: resolved.commit }), /^sha1-[a-f0-9]{40}$/u);
        assert.equal(typeof resolved.owner_id, 'number');
        assert.equal(typeof resolved.repo_id, 'number');
      }
    }
    assert.deepEqual([...actual].sort(), list({ value: indexed['.github/workflows/' + filename] }).sort(), filename);
  }
});
