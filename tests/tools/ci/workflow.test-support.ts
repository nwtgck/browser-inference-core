import assert from 'node:assert/strict';
import { chmod, copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { load } from 'js-yaml';
import { runProcess } from '#tools/shared/process';

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function object({ value }: { value: unknown }): Record<string, unknown> {
  assert.ok(isObject(value));
  return value;
}

export function list({ value }: { value: unknown }): unknown[] {
  assert.ok(Array.isArray(value));
  return value;
}

export function text({ value }: { value: unknown }): string {
  assert.equal(typeof value, 'string');
  assert.ok(typeof value === 'string');
  return value;
}

export const checkoutRoot = resolve(process.cwd());
export const workflow = object({ value: load(await readFile(join(checkoutRoot, '.github/workflows/build.yml'), 'utf8')) });
export const jobs = object({ value: workflow.jobs });

export function jobSteps({ name }: { name: string }): Record<string, unknown>[] {
  return list({ value: object({ value: jobs[name] }).steps }).map(value => object({ value }));
}

export function namedStep({ job, name }: { job: string; name: string }): Record<string, unknown> {
  const found = jobSteps({ name: job }).filter(value => value.name === name);
  assert.equal(found.length, 1, `${job}: ${name}`);
  return object({ value: found[0] });
}

export function actionSteps({ job, action }: { job: string; action: string }): Record<string, unknown>[] {
  return jobSteps({ name: job }).filter(value => typeof value.uses === 'string' && value.uses.split('@')[0] === action);
}

export function offlineScript({ step }: { step: Record<string, unknown> }): string {
  const script = text({ value: step.run });
  const marker = "<<'BICORE_OFFLINE'\n";
  const begin = script.indexOf(marker);
  const end = script.lastIndexOf('\nBICORE_OFFLINE');
  assert.ok(begin >= 0 && end > begin, 'Expected the explicit container stdin boundary');
  return script.slice(begin + marker.length, end) + '\n';
}

export async function executable({ root, name, source }: { root: string; name: string; source: string }): Promise<string> {
  const path = join(root, 'bin', name);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `#!${process.execPath}\n${source}`);
  await chmod(path, 0o755);
  return path;
}

export async function executeShell({ root, script, environment }: { root: string; script: string; environment: NodeJS.ProcessEnv }) {
  await mkdir(join(root, '.github/scripts'), { recursive: true });
  await copyFile(join(checkoutRoot, '.github/scripts/run-offline.sh'), join(root, '.github/scripts/run-offline.sh'));
  return runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', script], cwd: root, env: { ...process.env, ...environment, PATH: join(root, 'bin') + ':' + process.env.PATH }, timeoutMs: 20_000 });
}

export async function readCalls({ root }: { root: string }): Promise<{ command: string; args: string[]; environment?: Record<string, string> }[]> {
  const source = await readFile(join(root, 'calls.jsonl'), 'utf8');
  return source.trim().split('\n').filter(Boolean).map(line => {
    const value = object({ value: JSON.parse(line) });
    const args = list({ value: value.args }).map(value => text({ value }));
    const environment: Record<string, string> = {};
    if (value.environment !== undefined) {
      for (const [key, field] of Object.entries(object({ value: value.environment }))) {
        environment[key] = text({ value: field });
      }
    }
    return { command: text({ value: value.command }), args, environment };
  });
}
