import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { stat } from 'node:fs/promises';
import { decodeConsumerReportEnvelope, type GitHubWorkflowRun } from '#dto/llama-cpp';
import { gitObjectId } from '#tools/publication/git';
import { repositoryName } from '#tools/publication/npm-lock';
import { parseJson } from '#tools/shared/json';
import { positiveId, marker, payloadStart, payloadEnd } from './plan.js';

export const maximumArchiveBytes = 256 * 1024;

/** Bound both output streams and propagate unzip's CRC, format and exit errors. */
function archiveCommand({ command, args, limit }: { command: string; args: readonly string[]; limit: number }): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    execFile(command, [...args], { encoding: 'buffer', maxBuffer: limit, timeout: 30000 }, (error, stdout) => {
      if (error) {
        reject(error);
      } else {
        resolve(stdout);
      }
    });
  });
}

/** Read selected archive members to stdout; never extract an entry to a host path. */
export async function readConsumerReport({ path, repository, run }: { path: string; repository: string; run: GitHubWorkflowRun }): Promise<string> {
  repositoryName({ value: repository });
  if ((await stat(path)).size > maximumArchiveBytes) {
    throw new Error('Report archive exceeds the size limit');
  }
  const listing = await archiveCommand({ command: 'unzip', args: ['-Z', '-1', path], limit: 32000 });
  const names = new TextDecoder('utf-8', { fatal: true }).decode(listing).trimEnd().split('\n');
  const required = ['report.json', 'consumer-update.md', 'consumer-update.yaml'];
  for (const name of required) {
    if (names.filter(item => item === name).length !== 1) {
      throw new Error('Missing or duplicate report archive member');
    }
  }
  const details = await archiveCommand({ command: 'zipinfo', args: ['-l', path, ...required], limit: 32000 });
  if (details.toString('utf8').split('\n').some(line => line.startsWith('l'))) {
    throw new Error('Unsafe report archive entry');
  }
  // Download-only YAML is never decompressed; its uncompressed size is irrelevant.
  const envelopeBytes = await archiveCommand({ command: 'unzip', args: ['-p', path, 'report.json'], limit: 60000 });
  const markdown = await archiveCommand({ command: 'unzip', args: ['-p', path, 'consumer-update.md'], limit: 60000 });
  const envelope = parseJson({ text: new TextDecoder('utf-8', { fatal: true }).decode(envelopeBytes), decode: decodeConsumerReportEnvelope });
  positiveId({ value: envelope.runId });
  positiveId({ value: envelope.runAttempt });
  gitObjectId({ value: envelope.sourceCommit });
  gitObjectId({ value: envelope.artifactCommit });
  if (envelope.repository !== repository || envelope.runId !== run.id || envelope.runAttempt !== run.run_attempt || envelope.sourceCommit !== run.head_sha) {
    throw new Error('Report identity does not match the source workflow run');
  }
  if (createHash('sha256').update(markdown).digest('hex') !== envelope.markdownSha256) {
    throw new Error('Report Markdown digest mismatch');
  }
  const text = new TextDecoder('utf-8', { fatal: true }).decode(markdown);
  if (markdown.length > 55000 || [marker, payloadStart, payloadEnd].some(delimiter => text.includes(delimiter))) {
    throw new Error('Unexpected report delimiters or length');
  }
  if (!text.includes(`github:${repository}#${envelope.artifactCommit}`) || !text.includes(run.head_sha)) {
    throw new Error('Report body does not identify its source and artifact');
  }
  return text;
}

export function archiveRedirect({ status, headers }: { status: string; headers: string }): string {
  if (status !== '302') {
    throw new Error(`Unexpected artifact API response: ${status}`);
  }
  const locations = headers.split(/\r?\n/u).filter(line => /^location:/iu.test(line)).map(line => line.slice(line.indexOf(':') + 1).trim());
  const location = locations[0];
  if (locations.length !== 1 || !location || /[\r\n\0]/u.test(location)) {
    throw new Error('Missing or ambiguous artifact redirect');
  }
  const url = new URL(location);
  if (url.protocol !== 'https:' || !url.hostname || url.username || url.password || url.port && url.port !== '443') {
    throw new Error('Unexpected artifact download redirect');
  }
  return location;
}
