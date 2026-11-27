import { appendFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { decodeFrozenPublication, type ReportData } from '#dto/shared';
import { decodeConsumerReportEnvelope, type ConsumerReportEnvelope } from '#dto/llama-cpp';
import { runtimeNames } from '#tools/shared/contracts';
import { bytesIdentity, readMetadata, readMetadataBytes } from '#tools/package/files';
import { parseJson } from '#tools/shared/json';
import { withOutputDirectories } from '#tools/shared/output';
import { gitObjectId } from './git.js';
import { checkFrozen, type ConfirmedPublication } from './artifacts.js';
import { repositoryName, type NpmResolution } from './npm-lock.js';

export function publicationMetadata({ repository, commit, data, resolution }: { repository: string; commit: string; data: ReportData; resolution: NpmResolution }) {
  repositoryName({ value: repository });
  gitObjectId({ value: commit });
  if (resolution.specifier !== `github:${repository}#${commit}` || resolution.packageEntry.version !== data.package.version) {
    throw new Error('npm resolution differs from published package');
  }
  const raw = `https://raw.githubusercontent.com/${repository}/${commit}/`;
  return {
    schemaVersion: 2,
    runtime: { package: data.package.name, version: data.package.version, artifactCommit: commit, sourceCommit: data.sourceCommit, manifestFormatVersion: 6 },
    retrieval: { repository, artifactRawBase: raw, sourceRawBase: `https://raw.githubusercontent.com/${repository}/${data.sourceCommit}/`, install: `npm install github:${repository}#${commit}`, manifest: raw + 'manifest.json' },
    npm: resolution,
    source: data.source,
    runtimes: data.runtimes,
    consumerBoundary: 'Select each runtime, profile and Wasm instance explicitly. ABI schemas and validation scopes belong to the selected immutable package.',
  };
}

export function renderInstallGuide({ repository, commit, sourceCommit }: { repository: string; commit: string; sourceCommit: string }): string {
  repositoryName({ value: repository });
  gitObjectId({ value: commit });
  gitObjectId({ value: sourceCommit });
  const lines = [
    '# Browser inference runtime artifact', '',
    `Artifact commit: \`${commit}\``, '',
    `Source commit: \`${sourceCommit}\``, '',
    '```sh', `npm install github:${repository}#${commit}`, '```', '',
    '**Install with explicit package names:**', '',
    '```sh', `npm install llama-cpp-browser-core@github:${repository}#${commit}`, '```', '',
    '```sh', `npm install stable-diffusion-cpp-browser-core@github:${repository}#${commit}`, '```', '',
    '```sh', `npm install speech-cpp-browser-core@github:${repository}#${commit}`, '```', '',
  ];
  return lines.join('\n') + '\n';
}

function renderIntegrationDetails({ metadata }: { metadata: ReturnType<typeof publicationMetadata> }): string {
  const lines = ['| Runtime | ABI | Profiles |', '| --- | --- | --- |'];
  for (const runtime of runtimeNames) {
    const item = metadata.runtimes[runtime];
    lines.push(`| ${runtime} | ${String(item.abiVersion)} | ${Object.keys(item.profiles).join(', ')} |`);
  }
  lines.push('', 'All runtimes are independent modules. Build and smoke results do not certify real-model inference.', '', 'The root manifest binds each runtime manifest; each runtime manifest owns its payload inventory. Check the selected ABI fingerprint before calling native exports.', '', `Complete manifest: [manifest.json](${metadata.retrieval.manifest})`, '');
  for (const runtime of runtimeNames) {
    lines.push(`- [${runtime} manifest](${metadata.retrieval.artifactRawBase}${metadata.runtimes[runtime].manifest})`);
  }
  return lines.join('\n') + '\n';
}

function renderReportMarkdown({ metadata }: { metadata: ReturnType<typeof publicationMetadata> }): string {
  const { runtime: { artifactCommit, sourceCommit }, retrieval: { repository } } = metadata;
  return renderInstallGuide({ repository, commit: artifactCommit, sourceCommit }) + renderIntegrationDetails({ metadata });
}

export function renderReport({ repository, commit, data, resolution }: { repository: string; commit: string; data: ReportData; resolution: NpmResolution }): string {
  return renderReportMarkdown({ metadata: publicationMetadata({ repository, commit, data, resolution }) });
}

/** Reuse the confirmed freeze result without repeating source or package validation. */
export async function writeConfirmedReport({ publication, repository, resolution, output, run, summary }: { publication: ConfirmedPublication; repository: string; resolution: NpmResolution; output: string; run: { id: number; attempt: number } | undefined; summary: string | undefined }) {
  checkFrozen({ frozen: publication.frozen });
  const result = await writeReport({ repository, commit: publication.commit, data: publication.frozen.report.data, resolution, output, run });
  if (summary !== undefined) {
    await appendFile(summary, '## Consumer integration details\n\n' + renderIntegrationDetails({ metadata: result.metadata }));
  }
  return result;
}

function checkReportRun({ run }: { run: { id: number; attempt: number } }): void {
  if (!Number.isSafeInteger(run.id) || run.id <= 0 || !Number.isSafeInteger(run.attempt) || run.attempt <= 0) {
    throw new Error('Report requires positive workflow run and attempt identities');
  }
}

async function writeReportEnvelope({ directory, repository, commit, sourceCommit, run, markdown, frozenSha256 }: { directory: string; repository: string; commit: string; sourceCommit: string; run: { id: number; attempt: number }; markdown: string; frozenSha256: string | undefined }): Promise<void> {
  checkReportRun({ run });
  const envelope: ConsumerReportEnvelope = { schemaVersion: 1, repository, sourceCommit, artifactCommit: commit, runId: run.id, runAttempt: run.attempt, markdownSha256: bytesIdentity({ data: Buffer.from(markdown) }).sha256, frozenSha256 };
  await writeFile(join(directory, 'report.json'), JSON.stringify(envelope, undefined, 2) + '\n');
  await writeFile(join(directory, 'consumer-update.md'), markdown);
}

export async function writeConfirmedGuide({ publication, repository, output, run }: { publication: ConfirmedPublication; repository: string; output: string; run: { id: number; attempt: number } }): Promise<void> {
  checkFrozen({ frozen: publication.frozen });
  const { commit, frozen: { sourceCommit } } = publication;
  const markdown = renderInstallGuide({ repository, commit, sourceCommit });
  const receipt = Buffer.from(JSON.stringify(publication.frozen, undefined, 2) + '\n');
  await withOutputDirectories({
    destinations: [output],
    replace: true,
    prepare: async ({ paths: [directory] }) => {
      if (!directory) {
        throw new Error('Missing publication guide output');
      }
      await writeReportEnvelope({ directory, repository, commit, sourceCommit, run, markdown, frozenSha256: bytesIdentity({ data: receipt }).sha256 });
      await writeFile(join(directory, 'frozen.json'), receipt, { mode: 0o600 });
    },
  });
}

/** The guide comes from the publish job that confirmed the artifact commit. */
export async function readConfirmedGuide({ directory, repository, commit, sourceCommit, run }: { directory: string; repository: string; commit: string; sourceCommit: string; run: { id: number; attempt: number } }): Promise<ConfirmedPublication> {
  repositoryName({ value: repository });
  gitObjectId({ value: commit });
  gitObjectId({ value: sourceCommit });
  checkReportRun({ run });
  const envelope = await readMetadata({ path: join(directory, 'report.json'), decode: decodeConsumerReportEnvelope });
  if (envelope.repository !== repository || envelope.artifactCommit !== commit || envelope.sourceCommit !== sourceCommit || envelope.runId !== run.id || envelope.runAttempt !== run.attempt) {
    throw new Error('Publication guide identity does not match the confirmed workflow publication');
  }
  const markdown = await readMetadataBytes({ path: join(directory, 'consumer-update.md') });
  if (bytesIdentity({ data: markdown }).sha256 !== envelope.markdownSha256) {
    throw new Error('Publication guide Markdown digest mismatch');
  }
  const receipt = await readMetadataBytes({ path: join(directory, 'frozen.json') });
  if (bytesIdentity({ data: receipt }).sha256 !== envelope.frozenSha256) {
    throw new Error('Publication guide frozen receipt digest mismatch');
  }
  const frozen = parseJson({ text: receipt.toString('utf8'), decode: decodeFrozenPublication });
  if (frozen.sourceCommit !== sourceCommit) {
    throw new Error('Publication guide frozen source differs from the selected source');
  }
  checkFrozen({ frozen });
  return { commit, frozen };
}

export async function writeReport({ repository, commit, data, resolution, output, run }: { repository: string; commit: string; data: ReportData; resolution: NpmResolution; output: string; run: { id: number; attempt: number } | undefined }) {
  const metadata = publicationMetadata({ repository, commit, data, resolution });
  const markdown = renderReportMarkdown({ metadata });
  return withOutputDirectories({
    destinations: [output],
    replace: true,
    prepare: async ({ paths }) => {
      const directory = paths[0];
      if (!directory) {
        throw new Error('Missing report output');
      }
      await writeFile(join(directory, 'metadata.json'), JSON.stringify(metadata, undefined, 2) + '\n');
      await writeFile(join(directory, 'report.md'), markdown);
      await writeFile(join(directory, 'package-lock-entry.json'), JSON.stringify(resolution, undefined, 2) + '\n');
      if (run !== undefined) {
        await writeReportEnvelope({ directory, repository, commit, sourceCommit: data.sourceCommit, run, markdown, frozenSha256: undefined });
        // JSON is valid YAML and preserves the complete typed download metadata.
        await writeFile(join(directory, 'consumer-update.yaml'), JSON.stringify(metadata, undefined, 2) + '\n');
      }
      return { metadata, markdown };
    },
  });
}
