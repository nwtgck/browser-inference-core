import { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';
import { constants } from 'node:fs';
import { open } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { directory, isContained } from '#tools/shared/files';
import type { NinjaSummary } from '#tools/ci/summary';

export type MetricContext = {
  runId: string | undefined; runAttempt: string | undefined; job: string | undefined;
  sourceCommit: string | undefined; runtime: string | undefined;
  profile: string | undefined; variant: string | undefined;
};
export type MetricResult = 'success' | 'failure' | 'cancelled';
export type PackageKind = 'llama' | 'image' | 'root';
export type SpanStart = MetricContext & {
  schemaVersion: 1; mode: 'basic'; clock: 'monotonic_ns'; event: 'span_start'; phase: string; spanId: string;
  parentSpanId: string | undefined; startedNs: number; packageKind: PackageKind | undefined;
};
export type SpanEnd = Omit<SpanStart, 'event'> & {
  event: 'span_end'; elapsedNs: number; result: MetricResult; exitCode: number | null;
};
export type SpanRecord = SpanStart | SpanEnd;
export type CacheKind = 'dawn' | 'em' | 'cc';
export type CacheObservation = {
  kind: CacheKind; requestedKey: string | undefined; matchedKey: string | undefined;
  hit: string | undefined; outcome: string | undefined; state: string; saveEligible: boolean;
};
export type EnvironmentObservation = {
  cacheUseMode: string | undefined; emCache: 'unset' | 'expected' | 'different';
  compilerWrapper: 'ccache' | 'unset-or-different'; ccacheDirectory: 'set' | 'unset';
  ccacheConfiguration: 'unset' | 'expected' | 'different'; compilerIdentity: string | undefined;
};

const maximumBytes = 2 * 1024 * 1024;
const maximumRecord = 16 * 1024;
const safeLabel = /^[A-Za-z0-9_.:-]{1,600}$/u;

function label({ value }: { value: string | undefined }): string | undefined {
  return value !== undefined && safeLabel.test(value) ? value : undefined;
}

function exitCodeFor({ result }: { result: MetricResult }): number | null {
  switch (result) {
  case 'success': return 0;
  case 'failure':
  case 'cancelled': return null;
  default: {
    const exhaustive: never = result;
    throw new Error(String(exhaustive));
  }
  }
}

/** Only fixed, non-secret CI labels are recorded. */
export function metricContext({ env }: { env: NodeJS.ProcessEnv }): MetricContext {
  return {
    runId: label({ value: env['GITHUB_RUN_ID'] }),
    runAttempt: label({ value: env['GITHUB_RUN_ATTEMPT'] }),
    job: label({ value: env['GITHUB_JOB'] }),
    sourceCommit: label({ value: env['LCB_SOURCE_COMMIT'] }),
    runtime: label({ value: env['BIC_METRICS_RUNTIME'] }),
    profile: label({ value: env['BIC_METRICS_PROFILE'] }),
    variant: label({ value: env['BIC_METRICS_VARIANT'] }),
  };
}

/** A process owns one optional log; logging errors never change the operation. */
export class Metrics {
  readonly #enabled: boolean;
  readonly #directory: string | undefined;
  readonly #checkoutRoot: string;
  readonly #context: MetricContext;
  readonly #parent = new AsyncLocalStorage<string>();
  readonly #file = `events-${process.pid}-${randomUUID().replaceAll('-', '')}.jsonl`;
  readonly #events: SpanRecord[] = [];
  #ready: Promise<string> | undefined;
  #queue: Promise<void> = Promise.resolve();
  #bytes = 0;
  #warned = false;

  constructor({ enabled, directory, checkoutRoot, context }: { enabled: boolean; directory: string | undefined; checkoutRoot: string; context: MetricContext }) {
    this.#enabled = enabled;
    this.#directory = directory;
    this.#checkoutRoot = resolve(checkoutRoot);
    this.#context = context;
  }

  get records(): readonly SpanRecord[] {
    return this.#events;
  }

  #warn(): void {
    if (this.#warned) {
      return;
    }
    this.#warned = true;
    try {
      process.stderr.write('[pipeline-metrics] Optional telemetry unavailable or truncated.\n');
    } catch { /* Optional output. */ }
  }

  async #openLog(): Promise<string> {
    if (!this.#directory) {
      throw new Error('Missing metrics directory');
    }
    const targetDirectory = resolve(this.#directory);
    if (isContained({ root: this.#checkoutRoot, path: targetDirectory })) {
      throw new Error('Metrics must be outside the source tree');
    }
    await directory({ path: targetDirectory, create: true });
    const file = join(targetDirectory, this.#file);
    const created = await open(file, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK, 0o600);
    await created.close();
    return file;
  }

  async #write({ record }: { record: object }): Promise<void> {
    if (!this.#enabled) {
      return;
    }
    this.#queue = this.#queue.then(async () => {
      try {
        const raw = Buffer.from(`${JSON.stringify(record)}\n`);
        this.#ready ??= this.#openLog();
        const file = await this.#ready;
        if (raw.length > maximumRecord || this.#bytes + raw.length > maximumBytes) {
          try {
            const marker = await open(join(dirname(file), 'TRUNCATED'), constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK, 0o600);
            await marker.close();
          } catch { /* The marker may already exist. */ }
          throw new Error('Metrics budget exceeded');
        }
        const stream = await open(file, constants.O_WRONLY | constants.O_APPEND | constants.O_NOFOLLOW | constants.O_NONBLOCK);
        try {
          const info = await stream.stat();
          if (!info.isFile() || info.nlink !== 1) {
            throw new Error('Invalid metrics file');
          }
          await stream.writeFile(raw);
        } finally {
          await stream.close();
        }
        this.#bytes += raw.length;
      } catch {
        this.#warn();
      }
    });
    await this.#queue;
  }

  async begin({ phase, packageKind }: { phase: string; packageKind: PackageKind | undefined }): Promise<SpanStart | undefined> {
    if (!this.#enabled) {
      return undefined;
    }
    const record: SpanStart = {
      schemaVersion: 1,
      mode: 'basic',
      clock: 'monotonic_ns',
      event: 'span_start',
      ...this.#context,
      phase: /^[A-Za-z0-9_.:-]{1,100}$/u.test(phase) ? phase : 'unknown',
      spanId: randomUUID().replaceAll('-', ''),
      parentSpanId: this.#parent.getStore(),
      startedNs: Number(process.hrtime.bigint()),
      packageKind,
    };
    this.#events.push(record);
    await this.#write({ record });
    return record;
  }

  async end({ start, result, exitCode }: { start: SpanStart | undefined; result: MetricResult; exitCode: number | null }): Promise<void> {
    if (!start) {
      return;
    }
    const record: SpanEnd = { ...start, event: 'span_end', elapsedNs: Math.max(0, Number(process.hrtime.bigint()) - start.startedNs), result, exitCode };
    this.#events.push(record);
    await this.#write({ record });
  }

  /** Typed in-process work stays in-process and returns or throws unchanged. */
  async measure<T>({ phase, packageKind, operation }: { phase: string; packageKind: PackageKind | undefined; operation: () => Promise<T> }): Promise<T> {
    const start = await this.begin({ phase, packageKind });
    let result: MetricResult = 'success';
    try {
      return start ? await this.#parent.run(start.spanId, operation) : await operation();
    } catch (error) {
      result = error instanceof Error && error.name === 'AbortError' ? 'cancelled' : 'failure';
      throw error;
    } finally {
      await this.end({ start, result, exitCode: exitCodeFor({ result }) });
    }
  }

  async observeCache({ observation }: { observation: CacheObservation }): Promise<void> {
    if (!this.#enabled) {
      return;
    }
    await this.#write({ record: { schemaVersion: 1, event: 'cache_observation', ...this.#context, ...observation } });
    printObservation({ event: 'cache_observation', values: observation });
  }

  async observeCacheSave({ kind, outcome }: { kind: CacheKind; outcome: string }): Promise<void> {
    if (!this.#enabled) {
      return;
    }
    const values = { kind, outcome: label({ value: outcome }), state: outcome === 'success' ? 'action-completed-not-server-proof' : 'not-confirmed' };
    await this.#write({ record: { schemaVersion: 1, event: 'cache_save_observation', ...this.#context, ...values } });
    printObservation({ event: 'cache_save_observation', values });
  }

  async observeNinja({ observation }: { observation: NinjaSummary }): Promise<void> {
    if (!this.#enabled) {
      return;
    }
    await this.#write({ record: { schemaVersion: 1, event: 'ninja_observation', ...this.#context, ...observation } });
    printObservation({ event: 'ninja_observation', values: observation });
  }

  async observeEnvironment({ observation }: { observation: EnvironmentObservation }): Promise<void> {
    if (!this.#enabled) {
      return;
    }
    await this.#write({ record: { schemaVersion: 1, event: 'environment_observation', ...this.#context, ...observation } });
    printObservation({ event: 'environment_observation', values: observation });
  }
}

export function createMetrics({ env, checkoutRoot }: { env: NodeJS.ProcessEnv; checkoutRoot: string }): Metrics {
  const root = env['RUNNER_TEMP'];
  return new Metrics({
    enabled: env['BIC_METRICS_MODE'] === 'basic',
    directory: env['BIC_METRICS_DIR'] || (root ? join(root, 'bic-pipeline-metrics') : undefined),
    checkoutRoot,
    context: metricContext({ env }),
  });
}

/** Cache observations do not authorize adopting or saving a cache. */
export function cacheObservation({ kind, requested, matched, hit, outcome, saveEligible }: {
  kind: CacheKind; requested: string; matched: string; hit: string; outcome: string; saveEligible: boolean;
}): CacheObservation {
  let state: string;
  if (outcome !== 'success') {
    state = 'not-completed';
  } else if (!matched) {
    state = hit ? 'inconsistent' : 'miss';
  } else if (matched === requested && hit === 'true') {
    state = 'exact';
  } else if ((kind === 'cc' || kind === 'em') && hit === 'false' && requested.length > 40 && /^[a-f0-9]{40}$/u.test(requested.slice(-40)) &&
           matched.startsWith(requested.slice(0, -40)) && /^[a-f0-9]{40}$/u.test(matched.slice(requested.length - 40))) {
    state = 'prefix';
  } else {
    state = 'unexpected-key-or-status';
  }
  return {
    kind,
    requestedKey: label({ value: requested }),
    matchedKey: label({ value: matched }),
    hit: label({ value: hit }),
    outcome: label({ value: outcome }),
    state,
    saveEligible,
  };
}

/** Report whether cache wiring is active without logging local paths. */
export function environmentObservation({ env, workRoot }: { env: NodeJS.ProcessEnv; workRoot: string }): EnvironmentObservation {
  const location = ({ value, expected }: { value: string | undefined; expected: string }): 'unset' | 'expected' | 'different' => {
    if (!value) {
      return 'unset';
    }
    return resolve(value) === resolve(workRoot, expected) ? 'expected' : 'different';
  };
  return {
    cacheUseMode: label({ value: env['BIC_CACHE_MODE'] }),
    emCache: location({ value: env['EM_CACHE'], expected: '.tools/emsdk/upstream/emscripten/cache' }),
    compilerWrapper: env['EM_COMPILER_WRAPPER'] === 'ccache' ? 'ccache' : 'unset-or-different',
    ccacheDirectory: env['CCACHE_DIR'] ? 'set' : 'unset',
    ccacheConfiguration: location({ value: env['CCACHE_CONFIGPATH'], expected: '.cache/ccache.conf' }),
    compilerIdentity: label({ value: env['BIC_CACHE_ID'] }),
  };
}

export function printObservation({ event, values }: { event: string; values: object }): void {
  try {
    process.stdout.write(`[pipeline-metrics] ${JSON.stringify({ event, ...values })}\n`);
  } catch { /* Optional output. */ }
}
