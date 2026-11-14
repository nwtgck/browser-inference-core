import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { ReportData } from '#dto/shared';
import type { ConsumerReportEnvelope } from '#dto/llama-cpp';
import { runtimeNames } from '#tools/shared/contracts';
import { bytesIdentity } from '#tools/package/files';
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
    runtime: { package: data.package.name, version: data.package.version, artifactCommit: commit, sourceCommit: data.sourceCommit, manifestFormatVersion: 5 },
    retrieval: { repository, artifactRawBase: raw, sourceRawBase: `https://raw.githubusercontent.com/${repository}/${data.sourceCommit}/`, install: `npm install github:${repository}#${commit}`, manifest: raw + 'manifest.json' },
    npm: resolution,
    source: data.source,
    runtimes: data.runtimes,
    consumerBoundary: 'Select each runtime, profile and Wasm instance explicitly. ABI schemas and validation scopes belong to the selected immutable package.',
  };
}

export function renderReport({ repository, commit, data, resolution }: { repository: string; commit: string; data: ReportData; resolution: NpmResolution }): string {
  const metadata = publicationMetadata({ repository, commit, data, resolution });
  const lines = [
    '# Browser inference runtime artifact', '',
    `Artifact commit: \`${metadata.runtime.artifactCommit}\``, '',
    `Source commit: \`${data.sourceCommit}\``, '',
    '```sh', metadata.retrieval.install, '```', '',
    '**Install with explicit package names:**', '',
    '```sh', `npm install llama-cpp-browser-core@github:${repository}#${commit}`, '```', '',
    '```sh', `npm install stable-diffusion-cpp-browser-core@github:${repository}#${commit}`, '```', '',
    '| Runtime | ABI | Profiles |', '| --- | --- | --- |',
  ];
  for (const runtime of runtimeNames) {
    const item = data.runtimes[runtime];
    lines.push(`| ${runtime} | ${String(item.abiVersion)} | ${Object.keys(item.profiles).join(', ')} |`);
  }
  lines.push('', 'Both runtimes are independent modules. Build and smoke results do not certify real-model inference.', '', 'The root manifest binds each runtime manifest; each runtime manifest owns its payload inventory. Check the selected ABI fingerprint before calling native exports.', '', `Complete manifest: [manifest.json](${metadata.retrieval.manifest})`, '');
  for (const runtime of runtimeNames) {
    lines.push(`- [${runtime} manifest](${metadata.retrieval.artifactRawBase}${data.runtimes[runtime].manifest})`);
  }
  return lines.join('\n') + '\n';
}

/** Reuse only the confirmed private freeze result from the same publish job. */
export async function writeConfirmedReport({ publication, repository, resolution, output, run }: { publication: ConfirmedPublication; repository: string; resolution: NpmResolution; output: string; run: { id: number; attempt: number } | undefined }) {
  checkFrozen({ frozen: publication.frozen });
  return writeReport({ repository, commit: publication.commit, data: publication.frozen.report.data, resolution, output, run });
}

export async function writeReport({ repository, commit, data, resolution, output, run }: { repository: string; commit: string; data: ReportData; resolution: NpmResolution; output: string; run: { id: number; attempt: number } | undefined }) {
  if (run !== undefined && (!Number.isSafeInteger(run.id) || run.id <= 0 || !Number.isSafeInteger(run.attempt) || run.attempt <= 0)) {
    throw new Error('Report requires positive workflow run and attempt identities');
  }
  const metadata = publicationMetadata({ repository, commit, data, resolution });
  const markdown = renderReport({ repository, commit, data, resolution });
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
        const envelope: ConsumerReportEnvelope = { schemaVersion: 1, repository, sourceCommit: data.sourceCommit, artifactCommit: commit, runId: run.id, runAttempt: run.attempt, markdownSha256: bytesIdentity({ data: Buffer.from(markdown) }).sha256 };
        await writeFile(join(directory, 'report.json'), JSON.stringify(envelope, undefined, 2) + '\n');
        await writeFile(join(directory, 'consumer-update.md'), markdown);
        // JSON is valid YAML and preserves the complete typed download metadata.
        await writeFile(join(directory, 'consumer-update.yaml'), JSON.stringify(metadata, undefined, 2) + '\n');
      }
      return { metadata, markdown };
    },
  });
}
