import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import yargs from 'yargs';
import { decodeGitHubCompare, decodeGitHubRelease, decodeGitHubReleases, type GitHubRelease } from '#dto/llama-cpp';
import { readJson } from '#tools/shared/json';
import { requiredString } from '#tools/toolchain/cli-options';
import { assertSource, checkedRef, checkoutUpstream, createUpdateRequest, errorDetails, overlayPreflight, prepareCandidate, writeCandidateReport } from './upstream/operations.js';
import { planUpdate, selectUpstream, type UpdatePlan, type UpdateTarget } from './upstream/plan.js';
import { planArchiveDownloads, planPullRequests, planReportArchives, planReportDownloads, prepareCommentWrite } from './comments/operations.js';

const planOptions = {
  'checkout-root': requiredString,
  base: requiredString,
  source: requiredString,
  previous: requiredString,
  ref: requiredString,
  'resolved-ref': requiredString,
  commit: requiredString,
  requested: { choices: ['latest', 'latest_unstable', 'custom'] as const, demandOption: true },
  'allow-non-fast-forward': { type: 'boolean', demandOption: true },
  comparison: { type: 'string' },
  relation: { type: 'string' },
} as const;

const requestOptions = {
  'checkout-root': requiredString,
  base: requiredString,
  target: { choices: ['latest', 'latest-unstable', 'custom'] as const, demandOption: true },
  'custom-ref': { type: 'string' },
} as const;

async function outputs({ env, values }: { env: NodeJS.ProcessEnv; values: Readonly<Record<string, string>> }): Promise<void> {
  const path = env['GITHUB_OUTPUT'];
  for (const [key, value] of Object.entries(values)) {
    if (/[\r\n\0]/u.test(value)) {
      throw new Error('Unsafe workflow output');
    }
    if (path) {
      await appendFile(path, `${key}=${value}\n`);
    }
  }
}

async function reported({ state, operation }: { state: string; operation: () => Promise<void> }): Promise<void> {
  try {
    await operation();
  } catch (error) {
    await mkdir(state, { recursive: true });
    const diagnostic = errorDetails({ error });
    await writeFile(join(state, 'report.json'), JSON.stringify({ status: 'failed', error: diagnostic }, undefined, 2) + '\n');
    await writeFile(join(state, 'details.md'), `**Upstream update failed.**\n\n\`\`\`text\n${diagnostic}\n\`\`\`\n`);
    throw error;
  }
}

async function selectedTarget({ target, customRef, checkoutRoot }: { target: 'latest' | 'latest-unstable' | 'custom'; customRef: string | undefined; checkoutRoot: string }): Promise<UpdateTarget> {
  switch (target) {
  case 'custom':
    return { kind: 'custom', ref: await checkedRef({ root: checkoutRoot, value: customRef ?? '' }) };
  case 'latest':
  case 'latest-unstable':
    if (customRef) {
      throw new Error('custom_ref is only valid with target=custom');
    }
    switch (target) {
    case 'latest': return { kind: 'latest' };
    case 'latest-unstable': return { kind: 'latest_unstable' };
    default: {
      const exhaustive: never = target;
      throw new Error(String(exhaustive));
    }
    }
  default: {
    const exhaustive: never = target;
    throw new Error(String(exhaustive));
  }
  }
}

async function validateRequest({ checkoutRoot, repository, base, requestedTarget, customRef }: { checkoutRoot: string; repository: string; base: string; requestedTarget: 'latest' | 'latest-unstable' | 'custom'; customRef: string | undefined }) {
  const target = await selectedTarget({ target: requestedTarget, customRef, checkoutRoot });
  const request = await createUpdateRequest({ checkoutRoot, repository, base });
  return { target, request };
}

async function workflowPlan({ repository, base, source, previous, ref, resolvedRef, commit, requested, allowNonFastForward, relation, comparison }: { repository: string; base: string; source: string; previous: string; ref: string; resolvedRef: string; commit: string; requested: UpdateTarget['kind']; allowNonFastForward: boolean; relation: string | undefined; comparison: string | undefined }): Promise<UpdatePlan> {
  const observedRelation = previous === commit ? undefined : relation ?? (comparison ? (await readJson({ path: comparison, decode: decodeGitHubCompare })).status : undefined);
  return planUpdate({ request: { repository, base, sourceCommit: source, previousCommit: previous }, selection: { requested, ref, release: undefined }, resolvedCommit: commit, resolvedRef, relation: observedRelation, allowNonFastForward });
}

export async function main({ args, env }: { args: readonly string[]; env: NodeJS.ProcessEnv }): Promise<void> {
  await yargs([...args]).scriptName('bicore-automation').exitProcess(false).version(false).strict().help().demandCommand(1)
    .parserConfiguration({ 'camel-case-expansion': false, 'parse-numbers': false })
    .fail((message, error) => {
      throw error ?? new Error(message);
    })
    .option('state', requiredString)
    .option('repository', requiredString)
    .command('upstream-validate', 'Reject invalid requests before release acquisition', builder => builder.options(requestOptions), async options => {
      await reported({
        state: options.state,
        operation: async () => {
          const { request } = await validateRequest({ checkoutRoot: options['checkout-root'], repository: options.repository, base: options.base, requestedTarget: options.target, customRef: options['custom-ref'] });
          await outputs({ env, values: { source: request.sourceCommit, previous: request.previousCommit } });
        },
      });
    })
    .command('upstream-select', 'Validate the source request and select an upstream ref', builder => builder.options(requestOptions)
      .option('releases', { type: 'string' }).option('expected-source', { type: 'string' }).option('expected-previous', { type: 'string' }), async options => {
      await reported({
        state: options.state,
        operation: async () => {
          const { target, request } = await validateRequest({ checkoutRoot: options['checkout-root'], repository: options.repository, base: options.base, requestedTarget: options.target, customRef: options['custom-ref'] });
          if (options['expected-source'] !== undefined && request.sourceCommit !== options['expected-source'] || options['expected-previous'] !== undefined && request.previousCommit !== options['expected-previous']) {
            throw new Error('Source identity changed after request validation');
          }
          let releases: GitHubRelease[];
          switch (target.kind) {
          case 'custom': releases = [];
            break;
          case 'latest': releases = options.releases === undefined ? [] : [await readJson({ path: options.releases, decode: decodeGitHubRelease })];
            break;
          case 'latest_unstable': releases = options.releases === undefined ? [] : await readJson({ path: options.releases, decode: decodeGitHubReleases });
            break;
          default: {
            const exhaustive: never = target;
            throw new Error(String(exhaustive));
          }
          }
          const selection = selectUpstream({ target, releases });
          await outputs({ env, values: { source: request.sourceCommit, previous: request.previousCommit, ref: selection.ref, requested: selection.requested } });
        },
      });
    })
    .command('upstream-plan', 'Bind the selected target to the source checkout', builder => builder.options(planOptions), async options => {
      await reported({
        state: options.state,
        operation: async () => {
          const plan = await workflowPlan({ repository: options.repository, base: options.base, source: options.source, previous: options.previous, ref: options.ref, resolvedRef: options['resolved-ref'], commit: options.commit, requested: options.requested, allowNonFastForward: options['allow-non-fast-forward'], relation: options.relation, comparison: options.comparison });
          await assertSource({ checkoutRoot: options['checkout-root'], request: plan.request, clean: true });
          switch (plan.status) {
          case 'unchanged':
            await mkdir(options.state, { recursive: true });
            await writeFile(join(options.state, 'details.md'), 'The selected llama.cpp commit is unchanged. No candidate branch or push is needed.\n');
            await outputs({ env, values: { changed: 'false', branch: '', relation: '' } });
            break;
          case 'planned':
            await outputs({ env, values: { changed: 'true', branch: plan.branch, relation: plan.relation } });
            break;
          default: {
            const exhaustive: never = plan;
            throw new Error(String(exhaustive));
          }
          }
        },
      });
    })
    .command('upstream-checkout', 'Check out the already fetched upstream commit', builder => builder.options(planOptions), async options => {
      await reported({
        state: options.state,
        operation: async () => {
          const plan = await workflowPlan({ repository: options.repository, base: options.base, source: options.source, previous: options.previous, ref: options.ref, resolvedRef: options['resolved-ref'], commit: options.commit, requested: options.requested, allowNonFastForward: options['allow-non-fast-forward'], relation: options.relation, comparison: options.comparison });
          switch (plan.status) {
          case 'planned': await checkoutUpstream({ checkoutRoot: options['checkout-root'], plan });
            break;
          case 'unchanged': break;
          default: {
            const exhaustive: never = plan;
            throw new Error(String(exhaustive));
          }
          }
        },
      });
    })
    .command('upstream-candidate', 'Preserve an existing candidate or prepare a local commit', builder => builder.options(planOptions).option('existing', { type: 'string' }), async options => {
      await reported({
        state: options.state,
        operation: async () => {
          const plan = await workflowPlan({ repository: options.repository, base: options.base, source: options.source, previous: options.previous, ref: options.ref, resolvedRef: options['resolved-ref'], commit: options.commit, requested: options.requested, allowNonFastForward: options['allow-non-fast-forward'], relation: options.relation, comparison: options.comparison });
          switch (plan.status) {
          case 'planned': {
            const candidate = await prepareCandidate({ checkoutRoot: options['checkout-root'], plan, existingCommit: options.existing || undefined, preflight: overlayPreflight });
            await writeCandidateReport({ state: options.state, candidate });
            await outputs({ env, values: { 'needs-push': String(candidate.status === 'candidate_prepared'), status: candidate.status, commit: candidate.sourceCommit, branch: plan.branch } });
            break;
          }
          case 'unchanged': break;
          default: {
            const exhaustive: never = plan;
            throw new Error(String(exhaustive));
          }
          }
        },
      });
    })
    .command('comments-pulls', 'Select open same-repository pull requests', builder => builder, async options => {
      await planPullRequests({ state: options.state, repository: options.repository });
    })
    .command('comments-runs', 'Select the latest attempt for each head', builder => builder, async options => {
      await planReportDownloads({ state: options.state, repository: options.repository });
    })
    .command('comments-artifacts', 'Select unique unexpired attempt-bound archives', builder => builder, async options => {
      await planReportArchives({ state: options.state, repository: options.repository });
    })
    .command('comments-redirects', 'Validate credential-free archive download locations', builder => builder, async options => {
      await planArchiveDownloads({ state: options.state, repository: options.repository });
    })
    .command('comments-write', 'Refresh trust and prepare one conversation comment', builder => builder.option('number', { type: 'number', demandOption: true }), async options => {
      await prepareCommentWrite({ state: options.state, repository: options.repository, number: options.number });
    })
    .parseAsync();
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main({ args: process.argv.slice(2), env: process.env });
}
