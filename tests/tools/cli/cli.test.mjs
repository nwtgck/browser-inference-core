import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { after, test } from 'node:test';
import { pathToFileURL } from 'node:url';
import ts from 'typescript';

const root = resolve(import.meta.dirname, '../../..');
const temporary = await mkdtemp(join(tmpdir(), 'bicore-cli-'));
after(async () => {
  await rm(temporary, { recursive: true, force: true });
});
await symlink(join(root, 'node_modules'), join(temporary, 'node_modules'), 'dir');
await writeFile(join(temporary, 'package.json'), JSON.stringify({ type: 'module', imports: { '#source/toolchain.json': './toolchain/config.json', '#tools/*': './tools/*.js', '#runtimes/*': './runtimes/*.js' } }));
// Emit only the CLI and its passive Node helpers. Every command module is
// supplied by the test; no real preparation/build/publication module is loaded.
for (const file of ['tools/cli.ts', 'tools/shared/process.ts', 'tools/shared/contracts.ts', 'tools/shared/files.ts', 'tools/ci/metrics.ts', 'tools/toolchain/cli-options.ts']) {
  const source = await readFile(join(root, file), 'utf8');
  const { outputText } = ts.transpileModule(source, { fileName: file, compilerOptions: { target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.ESNext, verbatimModuleSyntax: true } });
  const target = join(temporary, file.replace(/\.ts$/u, '.js'));
  await mkdir(dirname(target), { recursive: true });
  await writeFile(target, outputText);
}
const entry = join(temporary, 'tools/cli.js');
const { createCli, main } = await import(pathToFileURL(entry).href);
const { ProcessError, runProcess } = await import(pathToFileURL(join(temporary, 'tools/shared/process.js')).href);
const { createMetrics } = await import(pathToFileURL(join(temporary, 'tools/ci/metrics.js')).href);

const metricsStub = async () => ({ createMetrics: () => ({ measure: async ({ operation }) => operation() }) });

function runner({ modules }) {
  const emitted = [];
  const accessedModules = [];
  const boundary = new Proxy(modules, {
    get(target, key) {
      accessedModules.push(key);
      if (Object.hasOwn(target, key)) {
        return target[key];
      }
      if (key === 'metrics') {
        return metricsStub;
      }
      return async () => {
        throw new Error(`Unexpected command module: ${String(key)}`);
      };
    },
  });
  return {
    emitted,
    accessedModules,
    async parse(argv) {
      let help = '';
      await createCli({
        modules: boundary,
        writeJson: async value => {
          emitted.push(value);
        },
      }).parseAsync(argv, {}, (error, _values, output) => {
        if (error) {
          throw error;
        }
        help = output;
      });
      return help;
    },
  };
}

const commands = [
  { name: 'prepare', argv: ['toolchain', 'prepare', '--checkout-root', 'checkout', '--tools-root', 'tools'], required: ['checkout-root', 'tools-root'] },
  { name: 'cache-plan', argv: ['toolchain', 'cache-plan', '--runtime', 'llama-cpp', '--profile', 'cpu', '--variant', 'browser', '--checkout-root', 'checkout', '--work-root', 'work', '--source-commit', 'revision', '--ccache-version', '4', '--environment', 'ci', '--os', 'linux', '--arch', 'x64', '--output', 'plan.json'], required: ['runtime', 'profile', 'variant', 'checkout-root', 'work-root', 'source-commit', 'ccache-version', 'environment', 'os', 'arch', 'output'] },
  { name: 'runtime', argv: ['package', 'runtime', '--runtime', 'llama-cpp', '--checkout-root', 'checkout', '--build-root', 'build', '--output', 'package', '--license-root', 'licenses'], required: ['runtime', 'checkout-root', 'build-root', 'output', 'license-root'] },
  { name: 'assemble', argv: ['package', 'assemble', '--checkout-root', 'checkout', '--inputs', 'inputs', '--output', 'package'], required: ['checkout-root', 'inputs', 'output'] },
  { name: 'verify', argv: ['package', 'verify', '--checkout-root', 'checkout', '--directory', 'package'], required: ['checkout-root', 'directory'] },
  { name: 'freeze', argv: ['publication', 'freeze', '--checkout-root', 'checkout', '--directory', 'package', '--work-root', 'frozen', '--receipt', 'receipt.json', '--branch', 'artifacts', '--source-commit', 'revision', '--source-work-root', 'source-work', '--python', 'python3'], required: ['checkout-root', 'directory', 'work-root', 'receipt', 'branch', 'source-commit', 'source-work-root', 'python'] },
  { name: 'confirm', argv: ['publication', 'confirm', '--work-root', 'frozen', '--receipt', 'receipt.json', '--commit', 'artifact'], required: ['work-root', 'receipt', 'commit'] },
  { name: 'npm-fixture', argv: ['publication', 'npm-fixture', '--directory', 'npm', '--repository', 'owner/repository', '--commit', 'artifact', '--version', '1.2.3'], required: ['directory', 'repository', 'commit', 'version'] },
  { name: 'report', argv: ['publication', 'report', '--work-root', 'frozen', '--receipt', 'receipt.json', '--commit', 'artifact', '--repository', 'owner/repository', '--npm-fixture', 'npm', '--output', 'report.md'], required: ['work-root', 'receipt', 'commit', 'repository', 'npm-fixture', 'output'] },
  { name: 'validation-runtime', argv: ['validation', 'runtime', '--runtime', 'llama-cpp', '--checkout-root', 'checkout', '--package', 'package', '--receipt', 'browser-receipt.json'], required: ['runtime', 'checkout-root', 'package', 'receipt'] },
];

for (const command of commands) {
  test(`required options are enforced before ${command.name} dispatch`, async () => {
    for (const name of command.required) {
      const index = command.argv.indexOf(`--${name}`);
      const argv = [...command.argv.slice(0, index), ...command.argv.slice(index + 2)];
      await assert.rejects(runner({ modules: {} }).parse(argv), error => error.message.includes(name) && /Missing required/u.test(error.message));
    }
    await assert.rejects(runner({ modules: {} }).parse([...command.argv, '--unknown-option', 'value']), /Unknown argument/u);
  });
}

test('invalid commands, runtime choices and removed options never dispatch', async () => {
  for (const argv of [[], ['unknown'], ['toolchain'], ['toolchain', 'unknown'], ['package', 'verify', '--checkout-root', 'checkout', '--directory', 'package', '--runtime', 'unknown'], ['toolchain', 'prepare', '--source-root', 'checkout', '--tools-root', 'tools'], [...commands[5].argv, '--dependency-root', 'dependencies']]) {
    await assert.rejects(runner({ modules: {} }).parse(argv), error => !error.message.includes('Unexpected command module'));
  }
});

test('scalar options reject repeated or empty values before dispatch', async () => {
  for (const argv of [
    [...commands[0].argv, '--checkout-root', 'other'],
    ['toolchain', 'prepare', '--checkout-root=', '--tools-root', 'tools'],
    [...commands[2].argv, '--runtime', 'stable-diffusion-cpp'],
  ]) {
    await assert.rejects(runner({ modules: {} }).parse(argv), error => /one.*value/u.test(error.message));
  }
});

test('root, group and leaf help have no dispatch side effects', async () => {
  const originalMode = process.env.BIC_METRICS_MODE;
  process.env.BIC_METRICS_MODE = 'basic';
  let loads = 0;
  try {
    for (const argv of [['--help'], ['toolchain', '--help'], ['package', '--help'], ['publication', '--help'], ['validation', '--help'], ['toolchain', 'prepare', '--help'], ['publication', 'freeze', '--help'], ['validation', 'runtime', '--help']]) {
      const help = await runner({
        modules: {
          metrics: async () => {
            loads++;
            throw new Error('Metrics must not load for help');
          },
        },
      }).parse(argv);
      assert.match(help, /bicore/u);
      assert.doesNotMatch(help, /--source-root|dependency-root/u);
      if (argv.includes('freeze')) {
        assert.match(help, /upstream WGSL generator/u);
      }
    }
    assert.equal(loads, 0);
  } finally {
    if (originalMode === undefined) {
      delete process.env.BIC_METRICS_MODE;
    } else {
      process.env.BIC_METRICS_MODE = originalMode;
    }
  }
});

test('prepare forwards the required boolean and preserves argv', async () => {
  for (const [flag, dawnOnly] of [ [[], false], [['--dawn-only'], true], [['--dawn-only=false'], false] ]) {
    const calls = [];
    const cli = runner({
      modules: {
        toolchainPrepare: async () => ({
          prepareToolchain: async args => {
            calls.push(args);
          },
        }),
      },
    });
    const argv = Object.freeze([...commands[0].argv, ...flag]);
    await cli.parse(argv);
    assert.deepEqual(calls, [{ checkoutRoot: resolve('checkout'), toolsRoot: resolve('tools'), dawnOnly }]);
    assert.deepEqual(argv, [...commands[0].argv, ...flag]);
    assert.deepEqual(cli.emitted, [{ value: { toolsRoot: resolve('tools'), status: 'complete' }, output: undefined }]);
  }
});

test('cache-plan selects only the requested runtime and passes typed configuration', async () => {
  for (const runtime of ['llama-cpp', 'stable-diffusion-cpp']) {
    const selected = { profile: { suspension: 'native' }, variant: { browser: true } };
    const plan = { identity: 'planned' };
    const environment = { CCACHE_DIR: '/cache' };
    const calls = [];
    const cli = runner({
      modules: {
        runtimeConfiguration: async args => {
          calls.push(args);
          return selected;
        },
        cachePlanCli: async () => ({
          runCachePlan: async args => {
            calls.push(args);
            return { ...plan, environment };
          },
        }),
      },
    });
    const argv = [...commands[1].argv];
    argv[argv.indexOf('--runtime') + 1] = runtime;
    await cli.parse(argv);
    assert.deepEqual(calls, [
      { runtime, checkoutRoot: resolve('checkout'), profile: 'cpu', variant: 'browser' },
      { runtime, checkoutRoot: resolve('checkout'), workRoot: resolve('work'), profile: 'cpu', variant: 'browser', profileConfiguration: selected.profile, variantConfiguration: selected.variant, sourceCommit: 'revision', ccacheVersion: '4', os: 'linux', arch: 'x64', environment: 'ci' },
    ]);
    assert.deepEqual(cli.emitted, [{ value: { ...plan, environment }, output: resolve('plan.json') }]);
  }
});

test('runtime packaging preserves repeated roots and profiles and selects one runtime', async () => {
  for (const runtime of ['llama-cpp', 'stable-diffusion-cpp']) {
    const calls = [];
    const cli = runner({
      modules: {
        runtimePackage: async selection => {
          assert.deepEqual(selection, { runtime });
          return {
            buildPackage: async args => {
              calls.push(args);
              return { manifest: { sourceCommit: 'revision' } };
            },
          };
        },
      },
    });
    const argv = [...commands[2].argv, '--license-root', 'other licenses', '--profile', 'cpu', '--profile', 'gpu', '--defer-npm-pack'];
    argv[argv.indexOf('--runtime') + 1] = runtime;
    await cli.parse(argv);
    assert.deepEqual(calls, [{ sourceRoot: resolve('checkout'), buildRoot: resolve('build'), destination: resolve('package'), profiles: ['cpu', 'gpu'], licenseRoots: [resolve('licenses'), resolve('other licenses')], checkNpmPack: false }]);
  }
});

test('aggregate assembly uses its ordinary API and enables the npm check by default', async () => {
  const calls = [];
  const cli = runner({
    modules: {
      aggregate: async () => ({
        assemblePackage: async args => {
          calls.push(args);
          return { manifest: { sourceCommit: 'revision' } };
        },
      }),
    },
  });
  await cli.parse(commands[3].argv);
  assert.deepEqual(calls, [{ sourceRoot: resolve('checkout'), inputs: resolve('inputs'), destination: resolve('package'), checkNpmPack: true }]);
});

test('verify selects one runtime or validates both through the aggregate API', async () => {
  for (const runtime of [undefined, 'llama-cpp', 'stable-diffusion-cpp']) {
    const calls = [];
    const verify = async args => {
      calls.push(args);
      return { manifest: { sourceCommit: 'revision' } };
    };
    const cli = runner({
      modules: runtime === undefined
        ? { aggregate: async () => ({ validatePackage: verify }) }
        : {
          runtimeVerify: async selection => {
            assert.deepEqual(selection, { runtime });
            return { verifyRuntimePackage: verify };
          },
        },
    });
    await cli.parse([...commands[4].argv, ...(runtime === undefined ? [] : ['--runtime', runtime])]);
    assert.deepEqual(calls, [{ sourceRoot: resolve('checkout'), directory: resolve('package'), requireClean: true, checkNpmPack: true }]);
  }
});

test('freeze connects real source verification arguments without a dependency root', async () => {
  const manifests = { selected: 'manifests' };
  const facts = { verified: true };
  const frozen = { sourceCommit: 'actual-source' };
  const calls = [];
  const cli = runner({
    modules: {
      publication: async () => ({
        freezePackage: async args => {
          const { verifySource, ...options } = args;
          calls.push(options);
          assert.equal(await verifySource({ sourceCommit: 'actual-source', manifests }), facts);
          return frozen;
        },
      }),
      publicationSources: async () => ({
        verifyPublicationSources: async args => {
          calls.push(args);
          return facts;
        },
      }),
    },
  });
  await cli.parse(commands[5].argv);
  assert.deepEqual(calls, [
    { sourceRoot: resolve('checkout'), packageDirectory: resolve('package'), work: resolve('frozen'), receipt: resolve('receipt.json'), branch: 'artifacts', expectedSource: 'revision' },
    { sourceRoot: resolve('checkout'), workRoot: resolve('source-work'), sourceCommit: 'actual-source', manifests, python: 'python3' },
  ]);
  assert.deepEqual(cli.emitted, [{ value: frozen, output: undefined }]);
});

test('confirm and report reuse the confirmed frozen state', async () => {
  const publication = { commit: 'artifact', frozen: { sourceCommit: 'revision', tree: 'tree', report: { data: { package: { version: 'confirmed-version' } } } } };
  const resolution = { lock: 'resolution' };
  const calls = [];
  const modules = {
    publication: async () => ({
      confirmPublication: async args => {
        calls.push(args);
        return publication;
      },
    }),
    npmResolution: async () => ({
      readNpmResolution: async args => {
        calls.push(args);
        return resolution;
      },
    }),
    report: async () => ({
      writeConfirmedReport: async args => {
        calls.push(args);
      },
    }),
  };
  await runner({ modules }).parse(commands[6].argv);
  await runner({ modules }).parse(commands[8].argv);
  assert.deepEqual(calls, [
    { work: resolve('frozen'), receipt: resolve('receipt.json'), commit: 'artifact' },
    { work: resolve('frozen'), receipt: resolve('receipt.json'), commit: 'artifact' },
    { directory: resolve('npm'), expected: { repository: 'owner/repository', commit: 'artifact', version: 'confirmed-version' } },
    { publication, repository: 'owner/repository', resolution, output: resolve('report.md'), run: undefined },
  ]);
});

test('npm-fixture only prepares the local workflow input', async () => {
  const calls = [];
  const result = { specifier: 'selected' };
  const cli = runner({
    modules: {
      npmResolution: async () => ({
        prepareNpmResolution: async args => {
          calls.push(args);
          return result;
        },
      }),
    },
  });
  await cli.parse(commands[7].argv);
  assert.deepEqual(calls, [{ directory: resolve('npm'), repository: 'owner/repository', commit: 'artifact', version: '1.2.3' }]);
  assert.deepEqual(cli.emitted, [{ value: result, output: undefined }]);
});

test('async command failures propagate their original process status', async () => {
  const failure = new ProcessError({ command: 'stub', stderr: Buffer.from('detail'), exitCode: 17, reason: 'failed' });
  const cli = runner({
    modules: {
      toolchainPrepare: async () => ({
        prepareToolchain: async () => {
          throw failure;
        },
      }),
    },
  });
  await assert.rejects(cli.parse(commands[0].argv), error => error === failure);
});

test('the executable boundary reports help, validation failures and process exit errors', async () => {
  const failure = new ProcessError({ command: 'fixture', stderr: Buffer.from('upstream failed'), exitCode: 17, reason: 'failed' });
  for (const [argv, status, pattern] of [
    [['--help'], 0, /toolchain/u],
    [['toolchain', 'prepare', '--help'], 0, /checkout-root/u],
    [['toolchain', 'prepare'], 1, /Missing required/u],
    [['package', 'unknown'], 1, /Unknown (?:command|argument)/u],
    [commands[0].argv, 17, /fixture: failed/u],
  ]) {
    const output = [];
    const code = await main({
      argv,
      modules: {
        metrics: metricsStub,
        toolchainPrepare: async () => ({
          prepareToolchain: async () => {
            throw failure;
          },
        }),
      },
      writeJson: async () => {
        assert.fail('No command result expected');
      },
      writeText: entry => {
        output.push(entry);
      },
    });
    assert.equal(code, status);
    assert.match(output.map(entry => entry.text).join(''), pattern);
    assert.ok(output.every(entry => entry.stream === (status === 0 ? 'stdout' : 'stderr')));
  }
});

test('runtime validation forwards the environment and process boundary without starting a browser', async () => {
  for (const runtime of ['llama-cpp', 'stable-diffusion-cpp']) {
    const receipt = { runtime, checked: true };
    const cli = runner({
      modules: {
        validation: async () => ({
          validateSavedRuntimePackage: async args => {
            assert.equal(args.checkoutRoot, resolve('checkout'));
            assert.equal(args.runtime, runtime);
            assert.equal(args.packageDirectory, resolve('package'));
            assert.equal(args.receiptPath, resolve('browser-receipt.json'));
            assert.equal(args.environment, process.env);
            assert.equal(args.runProcess, runProcess);
            return receipt;
          },
        }),
      },
    });
    const argv = [...commands[9].argv];
    argv[argv.indexOf('--runtime') + 1] = runtime;
    await cli.parse(argv);
    assert.deepEqual(cli.emitted, [{ value: receipt, output: undefined }]);
  }
});

test('basic metrics measure the typed handler in the same process', async () => {
  const originalMode = process.env.BIC_METRICS_MODE;
  process.env.BIC_METRICS_MODE = 'basic';
  try {
    const calls = [];
    let loads = 0;
    let operations = 0;
    const cli = runner({
      modules: {
        metrics: async () => {
          loads++;
          return {
            createMetrics: args => {
              assert.equal(args.env, process.env);
              assert.equal(args.checkoutRoot, temporary);
              return {
                measure: async ({ phase, packageKind, operation }) => {
                  calls.push([phase, packageKind, process.pid]);
                  return operation();
                },
              };
            },
          };
        },
        aggregate: async () => ({
          assemblePackage: async () => {
            operations++;
            return { manifest: { sourceCommit: 'revision' } };
          },
        }),
      },
    });
    await cli.parse(commands[3].argv);
    assert.equal(loads, 1);
    assert.equal(operations, 1);
    assert.deepEqual(calls, [['cli.package.assemble', 'root', process.pid]]);
    assert.deepEqual(cli.emitted, [{ value: { sourceCommit: 'revision', output: resolve('package') }, output: undefined }]);
  } finally {
    if (originalMode === undefined) {
      delete process.env.BIC_METRICS_MODE;
    } else {
      process.env.BIC_METRICS_MODE = originalMode;
    }
  }
});

test('disabled metrics do not load the metrics module or change command output', async () => {
  const originalMode = process.env.BIC_METRICS_MODE;
  process.env.BIC_METRICS_MODE = 'off';
  try {
    let operations = 0;
    let loads = 0;
    const cli = runner({
      modules: {
        metrics: async () => {
          loads++;
          assert.fail('Metrics must not load in off mode');
        },
        toolchainPrepare: async () => ({
          prepareToolchain: async () => {
            operations++;
          },
        }),
      },
    });
    await cli.parse(commands[0].argv);
    assert.equal(loads, 0);
    assert.equal(operations, 1);
    assert.deepEqual(cli.emitted, [{ value: { toolsRoot: resolve('tools'), status: 'complete' }, output: undefined }]);
  } finally {
    if (originalMode === undefined) {
      delete process.env.BIC_METRICS_MODE;
    } else {
      process.env.BIC_METRICS_MODE = originalMode;
    }
  }
});

test('metrics loading failures preserve the single operation and its original error', async context => {
  const originalMode = process.env.BIC_METRICS_MODE;
  process.env.BIC_METRICS_MODE = 'basic';
  const warnings = [];
  context.mock.method(process.stderr, 'write', text => {
    warnings.push(text);
    return true;
  });
  try {
    const original = new ProcessError({ command: 'upstream', stderr: Buffer.from('original'), exitCode: 19, reason: 'failed' });
    let operations = 0;
    const cli = runner({
      modules: {
        metrics: async () => {
          throw new Error('Private import detail');
        },
        toolchainPrepare: async () => ({
          prepareToolchain: async () => {
            operations++;
            throw original;
          },
        }),
      },
    });
    await assert.rejects(cli.parse(commands[0].argv), error => error === original);
    assert.equal(operations, 1);
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /Optional telemetry unavailable/u);
    assert.doesNotMatch(warnings[0], /Private import detail/u);
  } finally {
    if (originalMode === undefined) {
      delete process.env.BIC_METRICS_MODE;
    } else {
      process.env.BIC_METRICS_MODE = originalMode;
    }
  }
});

test('actual metric spans retain cancellation and the original exception', async () => {
  const logRoot = await mkdtemp(join(tmpdir(), 'bicore-cli-metrics-'));
  const originalMode = process.env.BIC_METRICS_MODE;
  const originalDirectory = process.env.BIC_METRICS_DIR;
  process.env.BIC_METRICS_MODE = 'basic';
  process.env.BIC_METRICS_DIR = logRoot;
  try {
    let observer;
    const original = new Error('Operation cancelled');
    original.name = 'AbortError';
    let operations = 0;
    const cli = runner({
      modules: {
        metrics: async () => ({
          createMetrics: args => {
            observer = createMetrics(args);
            return observer;
          },
        }),
        validation: async () => ({
          validateSavedRuntimePackage: async () => {
            operations++;
            throw original;
          },
        }),
      },
    });
    await assert.rejects(cli.parse(commands[9].argv), error => error === original);
    assert.equal(operations, 1);
    assert.equal(observer.records.length, 2);
    assert.equal(observer.records[0].phase, 'cli.validation.runtime');
    assert.equal(observer.records[0].packageKind, 'llama');
    assert.equal(observer.records[1].result, 'cancelled');
    assert.equal(observer.records[1].exitCode, null);
    assert.deepEqual(cli.emitted, []);
  } finally {
    if (originalMode === undefined) {
      delete process.env.BIC_METRICS_MODE;
    } else {
      process.env.BIC_METRICS_MODE = originalMode;
    }
    if (originalDirectory === undefined) {
      delete process.env.BIC_METRICS_DIR;
    } else {
      process.env.BIC_METRICS_DIR = originalDirectory;
    }
    await rm(logRoot, { recursive: true, force: true });
  }
});

test('report forwards an explicit workflow run identity as one typed pair', async () => {
  const publication = { commit: 'artifact', frozen: { sourceCommit: 'revision', report: { data: { package: { version: 'confirmed-version' } } } } };
  const resolution = { lock: 'resolution' };
  const calls = [];
  const cli = runner({
    modules: {
      publication: async () => ({ confirmPublication: async () => publication }),
      npmResolution: async () => ({ readNpmResolution: async () => resolution }),
      report: async () => ({
        writeConfirmedReport: async args => {
          calls.push(args);
        },
      }),
    },
  });
  await cli.parse([...commands[8].argv, '--run-id', '12345', '--run-attempt', '2']);
  assert.deepEqual(calls, [{ publication, repository: 'owner/repository', resolution, output: resolve('report.md'), run: { id: 12345, attempt: 2 } }]);
});

test('incomplete or invalid report run identities are rejected before dispatch', async () => {
  for (const flags of [
    ['--run-id', '12345'],
    ['--run-attempt', '2'],
    ['--run-id', '0', '--run-attempt', '2'],
    ['--run-id', '12345', '--run-attempt', '-1'],
    ['--run-id', '1.5', '--run-attempt', '2'],
    ['--run-id', '9007199254740992', '--run-attempt', '2'],
    ['--run-id', '12345', '--run-attempt', 'not-a-number'],
  ]) {
    const cli = runner({ modules: {} });
    await assert.rejects(cli.parse([...commands[8].argv, ...flags]), Error);
    assert.deepEqual(cli.accessedModules, []);
    assert.deepEqual(cli.emitted, []);
  }
});
