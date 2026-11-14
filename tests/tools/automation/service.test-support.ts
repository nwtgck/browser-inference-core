#!/usr/bin/env node
import { READ_ENV_BICORE_AUTOMATION_FIXTURE, READ_ENV_GH_TOKEN, READ_ENV_BICORE_FIXTURE_ORIGINAL_PATH, READ_ENV_BICORE_FIXTURE_REAL_GIT, READ_ENV_BICORE_FIXTURE_UPSTREAM, READ_ENV_BICORE_FIXTURE_REPOSITORY, READ_ENV_BICORE_FIXTURE_ADVANCE, READ_ENV_BRANCH } from '#tools/shared/node-env';
import { spawn } from 'node:child_process';
import { appendFile, copyFile, readFile, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';
import type { GitHubIssueComment, GitHubPullRequest, GitHubRelease, GitHubWorkflowRun } from '#dto/llama-cpp';

export type Scenario = {
  pulls: GitHubPullRequest[];
  runs: GitHubWorkflowRun[];
  comments: GitHubIssueComment[];
  currentPulls: GitHubPullRequest[] | undefined;
  currentRuns: GitHubWorkflowRun[] | undefined;
  totalCount: number | undefined;
  currentTotalCount: number | undefined;
  redirect: string;
  directArchive: boolean;
  failArtifactListing: boolean;
  failWrite: boolean;
  latest: GitHubRelease | undefined;
  releases: GitHubRelease[];
  failRelease: boolean;
  relation: string;
};

async function fixtureService(): Promise<void> {
  const root = READ_ENV_BICORE_AUTOMATION_FIXTURE();
  if (!root) {
    throw new Error('This test service requires a local fixture root');
  }
  // Only this test's typed producer writes the scenario; it is not service input.
  const scenario = JSON.parse(await readFile(join(root, 'scenario.json'), 'utf8')) as Scenario;
  const tool = basename(process.argv[1] ?? '');
  const args = process.argv.slice(2);
  await appendFile(join(root, 'calls.jsonl'), JSON.stringify({ tool, args, hasToken: !!READ_ENV_GH_TOKEN() }) + '\n');
  function argument({ name }: { name: string }): string {
    const index = args.indexOf(name);
    const value = index < 0 ? undefined : args[index + 1];
    if (value === undefined) {
      throw new Error(`Missing fixture argument: ${name}`);
    }
    return value;
  }
  function output({ value }: { value: unknown }): void {
    process.stdout.write(JSON.stringify(value) + '\n');
  }
  switch (tool) {
  case 'gh': {
    const method = argument({ name: '--method' });
    const endpoint = args[args.indexOf('--method') + 2];
    if (!endpoint) {
      throw new Error('Missing fixture endpoint');
    }
    const fields = new Map<string, string>();
    for (let index = 0; index < args.length; index++) {
      if (args[index] === '-f') {
        const entry = args[index + 1] ?? '';
        const separator = entry.indexOf('=');
        fields.set(entry.slice(0, separator), entry.slice(separator + 1));
      }
    }
    const page = Number(fields.get('page') ?? '1');
    function pageValues<T>({ values }: { values: T[] }): T[] {
      return values.slice((page - 1) * 100, page * 100);
    }
    if (method === 'POST' || method === 'PATCH') {
      await appendFile(join(root, 'writes.jsonl'), JSON.stringify({ method, endpoint, body: JSON.parse(await readFile(argument({ name: '--input' }), 'utf8')) as unknown }) + '\n');
      if (scenario.failWrite) {
        throw new Error('Fixture write response was lost');
      }
      output({ value: { id: 42 } });
    } else if (endpoint === 'repos/example/core/pulls') {
      output({ value: pageValues({ values: scenario.pulls }) });
    } else if (/^repos\/example\/core\/pulls\/\d+$/u.test(endpoint)) {
      const number = Number(endpoint.split('/').at(-1));
      const pull = (scenario.currentPulls ?? scenario.pulls).find(value => value.number === number);
      await writeFile(join(root, 'refreshing'), 'true');
      output({ value: pull });
    } else if (endpoint === 'repos/example/core/actions/workflows/build.yml/runs') {
      let refreshing = false;
      try {
        refreshing = await readFile(join(root, 'refreshing'), 'utf8') === 'true';
      } catch {
        // Initial scan precedes the first immediate pre-write refresh.
      }
      const values = refreshing ? scenario.currentRuns ?? scenario.runs : scenario.runs;
      const total = refreshing ? scenario.currentTotalCount ?? scenario.totalCount : scenario.totalCount;
      output({ value: { total_count: total ?? values.length, workflow_runs: pageValues({ values }) } });
    } else if (/^repos\/example\/core\/actions\/runs\/\d+\/artifacts$/u.test(endpoint)) {
      if (scenario.failArtifactListing) {
        throw new Error('Fixture artifact listing failed');
      }
      output({ value: { artifacts: page === 1 ? [{ id: 91, name: fields.get('name'), expired: false }] : [] } });
    } else if (/^repos\/example\/core\/issues\/\d+\/comments$/u.test(endpoint)) {
      output({ value: pageValues({ values: scenario.comments }) });
    } else if (endpoint === 'repos/ggml-org/llama.cpp/releases/latest' || endpoint === 'repos/ggml-org/llama.cpp/releases') {
      if (scenario.failRelease) {
        throw new Error('Fixture stable read denied');
      }
      output({ value: endpoint.endsWith('/latest') ? scenario.latest : pageValues({ values: scenario.releases }) });
    } else if (endpoint.startsWith('repos/ggml-org/llama.cpp/compare/')) {
      output({ value: { status: scenario.relation } });
    } else {
      throw new Error(`Unexpected test service endpoint: ${endpoint}`);
    }
    break;
  }
  case 'curl': {
    const url = args.at(-1) ?? '';
    if (url.startsWith('https://api.github.com/repos/example/core/actions/artifacts/')) {
      if (!READ_ENV_GH_TOKEN() || args.includes('--location')) {
        throw new Error('Authenticated artifact requests must not follow redirects');
      }
      await writeFile(argument({ name: '--dump-header' }), `HTTP/2 ${scenario.directArchive ? '200' : '302'}\r\nLocation: ${scenario.redirect}\r\n`);
      if (scenario.directArchive) {
        await copyFile(join(root, 'report.zip'), argument({ name: '--output' }));
      } else {
        await writeFile(argument({ name: '--output' }), '');
      }
      process.stdout.write(scenario.directArchive ? '200' : '302');
    } else {
      if (READ_ENV_GH_TOKEN() || url !== scenario.redirect || !url.startsWith('https://')) {
        throw new Error('Signed storage must be HTTPS and credential-free');
      }
      await copyFile(join(root, 'report.zip'), argument({ name: '--output' }));
    }
    break;
  }
  case 'docker': {
    if (!args.includes('--network') || argument({ name: '--network' }) !== 'none') {
      throw new Error('Expected a literal network-none boundary');
    }
    const environment: NodeJS.ProcessEnv = { PATH: READ_ENV_BICORE_FIXTURE_ORIGINAL_PATH() };
    let cwd = process.cwd();
    if (args[0] !== 'run') {
      throw new Error('Only a local docker-run stand-in is supported');
    }
    let index = 1;
    for (; index < args.length; index++) {
      const arg = args[index];
      if (arg === '-e') {
        const field = args[++index] ?? '';
        const separator = field.indexOf('=');
        const name = separator < 0 ? field : field.slice(0, separator);
        // eslint-disable-next-line local-rules-node-env/node-env-policy -- The Docker test stand-in transparently transfers unknown -e NAME values; it does not interpret application settings.
        const value = separator < 0 ? process.env[name] : field.slice(separator + 1);
        if (value !== undefined) {
          environment[name] = value;
        }
      } else if (['--network', '-u', '-v', '-w'].includes(arg ?? '')) {
        const value = args[++index];
        if (arg === '-w' && value) {
          cwd = value;
        }
      } else if (arg === '--pull=never' || arg === '--rm' || arg === '-i' || arg?.startsWith('--cap-drop=') || arg?.startsWith('--security-opt=')) {
        continue;
      } else {
        index++;
        break;
      }
    }
    if (environment['GH_TOKEN'] || environment['GITHUB_TOKEN']) {
      throw new Error('Offline tools must not receive API credentials');
    }
    if (environment['CI'] !== 'true' || environment['NPM_CONFIG_OFFLINE'] !== 'true' || environment['GIT_NO_LAZY_FETCH'] !== '1') {
      throw new Error('Missing offline execution environment');
    }
    const executable = args[index];
    if (!executable) {
      throw new Error('Missing offline command');
    }
    const child = spawn(executable, args.slice(index + 1), { cwd, env: environment, stdio: 'inherit' });
    process.exitCode = await new Promise<number>((resolve, reject) => {
      child.on('error', reject);
      child.on('exit', code => resolve(code ?? 1));
    });
    break;
  }
  case 'git': {
    const realGit = READ_ENV_BICORE_FIXTURE_REAL_GIT();
    const localUpstream = READ_ENV_BICORE_FIXTURE_UPSTREAM();
    const localRepository = READ_ENV_BICORE_FIXTURE_REPOSITORY();
    if (!realGit) {
      throw new Error('Missing local Git fixture');
    }
    const sourceRemote = 'https://github.com/example/core.git';
    const advance = READ_ENV_BICORE_FIXTURE_ADVANCE();
    if (localRepository && advance && args.includes('fetch') && args.includes(sourceRemote)) {
      const branch = READ_ENV_BRANCH();
      if (!branch?.startsWith('feature/')) {
        throw new Error('The local race fixture requires a feature branch');
      }
      const update = spawn(realGit, ['-C', localRepository, 'update-ref', `refs/heads/${branch}`, advance], { stdio: 'inherit' });
      const status = await new Promise<number>((resolve, reject) => {
        update.on('error', reject);
        update.on('exit', code => resolve(code ?? 1));
      });
      if (status !== 0) {
        throw new Error('Cannot advance the local branch race fixture');
      }
    }
    const redirected = args.map(value => value === 'https://github.com/ggml-org/llama.cpp.git' ? localUpstream ?? value : value === sourceRemote ? localRepository ?? value : value);
    if (redirected.some(value => /^https?:/u.test(value))) {
      throw new Error('The local Git fixture cannot contact a remote service');
    }
    const child = spawn(realGit, ['-c', 'protocol.file.allow=always', ...redirected], { stdio: 'inherit' });
    process.exitCode = await new Promise<number>((resolve, reject) => {
      child.on('error', reject);
      child.on('exit', code => resolve(code ?? 1));
    });
    break;
  }
  default:
    throw new Error(`Unexpected fixture tool: ${tool}`);
  }
}

if (READ_ENV_BICORE_AUTOMATION_FIXTURE()) {
  await fixtureService();
}
