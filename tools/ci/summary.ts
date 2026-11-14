import { lstat, opendir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { MetricContext, PackageKind, SpanEnd, SpanRecord, SpanStart } from '#tools/ci/metrics';

const maxBytes = 2 * 1024 * 1024;
const maxRecord = 16 * 1024;
const maxFiles = 128;
const maxDurationNs = 7 * 24 * 3600 * 1e9;

type Interval = { start: number; end: number };
type Category = 'object-outputs' | 'archive-outputs' | 'runtime-outputs' | 'other-outputs';
type NinjaCategory = { outputRows: number; coveredMs: number; longestOutputMs: number };
export type NinjaSummary = { status: 'unavailable' | 'unknown-format' | 'oversized' | 'malformed' } | {
  status: 'observed'; singleFreshInvocationRequired: true; outputRows: number; coveredMs: number;
  categories: Partial<Record<Category, NinjaCategory>>;
};

export function unionLength({ intervals }: { intervals: readonly Interval[] }): number {
  let end: number | undefined;
  let total = 0;
  for (const interval of [...intervals].sort((left, right) => left.start - right.start)) {
    if (interval.end < interval.start) {
      throw new Error('Negative interval');
    }
    if (end === undefined || interval.end > end) {
      total += interval.end - Math.max(interval.start, end ?? interval.start);
    }
    end = Math.max(end ?? interval.end, interval.end);
  }
  return total;
}

/** Ninja rows are output observations, not individual commands or total wall time. */
export async function ninjaSummary({ path }: { path: string }): Promise<NinjaSummary> {
  try {
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink() || info.nlink !== 1 || info.size > 8 * 1024 * 1024) {
      return { status: 'unavailable' };
    }
    const lines = (await readFile(path, 'utf8')).split('\n');
    if (lines.shift() !== '# ninja log v5') {
      return { status: 'unknown-format' };
    }
    if (lines.at(-1) === '') {
      lines.pop();
    }
    if (lines.length > 100000 || lines.some((line) => Buffer.byteLength(line) > maxRecord)) {
      return { status: 'oversized' };
    }
    const groups = new Map<Category, Interval[]>();
    for (const line of lines) {
      const fields = line.split('\t');
      if (fields.length !== 5 || !/^\d+$/u.test(fields[0] ?? '') || !/^\d+$/u.test(fields[1] ?? '')) {
        return { status: 'malformed' };
      }
      const start = Number(fields[0]);
      const end = Number(fields[1]);
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || end < start || end > 7 * 24 * 3600 * 1000) {
        return { status: 'malformed' };
      }
      const output = (fields[3] ?? '').replaceAll('\\', '/');
      let kind: Category;
      if (output.endsWith('.o') || output.endsWith('.obj')) {
        kind = 'object-outputs';
      } else if (output.endsWith('.a') || output.endsWith('.lib')) {
        kind = 'archive-outputs';
      } else if (`/${output}`.includes('/runtime/core.')) {
        kind = 'runtime-outputs';
      } else {
        kind = 'other-outputs';
      }
      const group = groups.get(kind) ?? [];
      group.push({ start, end });
      groups.set(kind, group);
    }
    const categories: Partial<Record<Category, NinjaCategory>> = {};
    const all: Interval[] = [];
    for (const [kind, intervals] of groups) {
      let longestOutputMs = 0;
      for (const interval of intervals) {
        all.push(interval);
        longestOutputMs = Math.max(longestOutputMs, interval.end - interval.start);
      }
      categories[kind] = {
        outputRows: intervals.length,
        coveredMs: unionLength({ intervals }),
        longestOutputMs,
      };
    }
    return { status: 'observed', singleFreshInvocationRequired: true, outputRows: all.length, coveredMs: unionLength({ intervals: all }), categories };
  } catch {
    return { status: 'unavailable' };
  }
}

function objectValue(input: unknown): input is Record<string, unknown> {
  return typeof input === 'object' && input !== null && !Array.isArray(input);
}

function optionalString({ value }: { value: unknown }): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

/** Project only the span fields used by the summary of this job's optional logs. */
function spanRecord({ input }: { input: unknown }): SpanRecord | undefined {
  if (!objectValue(input) || input['schemaVersion'] !== 1) {
    throw new Error('Invalid metric record');
  }
  if (input['event'] !== 'span_start' && input['event'] !== 'span_end') {
    return undefined;
  }
  if (input['mode'] !== 'basic' || input['clock'] !== 'monotonic_ns') {
    throw new Error('Unsupported timing clock');
  }
  const phase = input['phase'];
  const spanId = input['spanId'];
  const startedNs = input['startedNs'];
  if (typeof phase !== 'string' || !/^[A-Za-z0-9_.:-]{1,100}$/u.test(phase) || typeof spanId !== 'string' ||
      !/^[a-f0-9]{32}$/u.test(spanId) || typeof startedNs !== 'number' || !Number.isInteger(startedNs) || startedNs < 0) {
    throw new Error('Invalid span');
  }
  const kind = input['packageKind'];
  const packageKind: PackageKind | undefined = kind === 'llama' || kind === 'image' || kind === 'root' ? kind : undefined;
  const context: MetricContext = {
    runId: optionalString({ value: input['runId'] }),
    runAttempt: optionalString({ value: input['runAttempt'] }),
    job: optionalString({ value: input['job'] }),
    sourceCommit: optionalString({ value: input['sourceCommit'] }),
    runtime: optionalString({ value: input['runtime'] }),
    profile: optionalString({ value: input['profile'] }),
    variant: optionalString({ value: input['variant'] }),
  };
  const common = {
    schemaVersion: 1 as const,
    mode: 'basic' as const,
    clock: 'monotonic_ns' as const,
    ...context,
    phase,
    spanId,
    startedNs,
    packageKind,
    parentSpanId: optionalString({ value: input['parentSpanId'] }),
  };
  switch (input['event']) {
  case 'span_start': return { ...common, event: 'span_start' };
  case 'span_end': break;
  default: throw new Error('Invalid span event');
  }
  const elapsedNs = input['elapsedNs'];
  const result = input['result'];
  if (typeof elapsedNs !== 'number' || !Number.isInteger(elapsedNs) || elapsedNs < 0 || elapsedNs > maxDurationNs ||
      (result !== 'success' && result !== 'failure' && result !== 'cancelled')) {
    throw new Error('Invalid completed span');
  }
  const exitCode = typeof input['exitCode'] === 'number' ? input['exitCode'] : null;
  return { ...common, event: 'span_end', elapsedNs, result, exitCode };
}

/** Read diagnostic events produced by earlier steps in the same job. */
export async function readRecords({ directory }: { directory: string }): Promise<{ records: SpanRecord[]; errors: number }> {
  const records: SpanRecord[] = [];
  let errors = 0;
  let total = 0;
  let files = 0;
  try {
    const info = await lstat(directory);
    if (!info.isDirectory() || info.isSymbolicLink()) {
      return { records, errors: 1 };
    }
    const entries = await opendir(directory);
    for await (const entry of entries) {
      if (entry.name === 'TRUNCATED') {
        errors++;
        continue;
      }
      if (!entry.name.startsWith('events-') || !entry.name.endsWith('.jsonl')) {
        continue;
      }
      if (++files > maxFiles) {
        errors++;
        break;
      }
      try {
        const file = join(directory, entry.name);
        const status = await lstat(file);
        if (!status.isFile() || status.isSymbolicLink() || status.nlink !== 1 || total + status.size > maxBytes) {
          errors++;
          continue;
        }
        const raw = await readFile(file, 'utf8');
        total += Buffer.byteLength(raw);
        for (const line of raw.split('\n')) {
          if (!line) {
            continue;
          }
          if (Buffer.byteLength(line) > maxRecord) {
            errors++;
            continue;
          }
          try {
            const input: unknown = JSON.parse(line);
            const record = spanRecord({ input });
            if (record) {
              records.push(record);
            }
          } catch {
            errors++;
          }
        }
      } catch {
        errors++;
      }
    }
  } catch {
    errors++;
  }
  return { records, errors };
}

type Phase = { started: number; ended: number; failed: number; incomplete: number; inclusiveNs: number | null; exclusiveNs: number | null; exclusiveUnknown: number };
export type Summary = { schemaVersion: 1; invalidOrMissingRecords: number; note: string; phases: Record<string, Phase> };

export function summarize({ records, errors }: { records: readonly SpanRecord[]; errors: number }): Summary {
  const starts = new Map<string, SpanStart>();
  const ends = new Map<string, SpanEnd>();
  const duplicate = new Set<string>();
  const phases: Record<string, Phase> = Object.create(null);
  const bucket = ({ record }: { record: SpanRecord }): string => record.phase + (record.packageKind ? `:${record.packageKind}` : '');
  const phase = ({ record }: { record: SpanRecord }): Phase => phases[bucket({ record })] ??= {
    started: 0,
    ended: 0,
    failed: 0,
    incomplete: 0,
    inclusiveNs: 0,
    exclusiveNs: 0,
    exclusiveUnknown: 0,
  };
  for (const record of records) {
    switch (record.event) {
    case 'span_start':
      if (starts.has(record.spanId)) {
        duplicate.add(record.spanId);
        errors++;
      }
      starts.set(record.spanId, record);
      break;
    case 'span_end':
      if (ends.has(record.spanId)) {
        duplicate.add(record.spanId);
        errors++;
      }
      ends.set(record.spanId, record);
      break;
    default: {
      const exhaustive: never = record;
      throw new Error(String(exhaustive));
    }
    }
  }
  const children = new Map<string, SpanStart[]>();
  for (const start of starts.values()) {
    phase({ record: start }).started++;
    if (start.parentSpanId) {
      const group = children.get(start.parentSpanId) ?? [];
      group.push(start);
      children.set(start.parentSpanId, group);
    }
  }
  for (const [identifier, end] of ends) {
    const start = starts.get(identifier);
    if (duplicate.has(identifier) || !start ||
      (['phase', 'packageKind', 'startedNs', 'parentSpanId', 'runId', 'runAttempt', 'job', 'sourceCommit', 'runtime', 'profile', 'variant'] as const)
        .some((key) => start[key] !== end[key])) {
      errors++;
      continue;
    }
    const entry = phase({ record: end });
    entry.ended++;
    switch (end.result) {
    case 'success': break;
    case 'failure':
    case 'cancelled': entry.failed++;
      break;
    default: {
      const exhaustive: never = end.result;
      throw new Error(String(exhaustive));
    }
    }
    entry.inclusiveNs = (entry.inclusiveNs ?? 0) + end.elapsedNs;
    const intervals: Interval[] = [];
    let unknown = false;
    for (const child of children.get(identifier) ?? []) {
      const childEnd = ends.get(child.spanId);
      if (!childEnd || duplicate.has(child.spanId)) {
        unknown = true;
        break;
      }
      const begin = Math.max(end.startedNs, child.startedNs);
      const stop = Math.min(end.startedNs + end.elapsedNs, child.startedNs + childEnd.elapsedNs);
      if (stop >= begin) {
        intervals.push({ start: begin, end: stop });
      }
    }
    if (unknown) {
      entry.exclusiveUnknown++;
    } else {
      entry.exclusiveNs = (entry.exclusiveNs ?? 0) + end.elapsedNs - unionLength({ intervals });
    }
  }
  for (const entry of Object.values(phases)) {
    entry.incomplete = entry.started - entry.ended;
    if (!entry.ended) {
      entry.inclusiveNs = null;
    }
    if (!entry.ended || entry.exclusiveUnknown || errors) {
      entry.exclusiveNs = null;
    }
  }
  return {
    schemaVersion: 1,
    invalidOrMissingRecords: errors,
    note: 'Phase sums are cumulative inclusive work, NOT workflow wall time. Parent/child spans overlap.',
    phases,
  };
}
