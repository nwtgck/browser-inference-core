import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { decodeLlamaManifest, type LlamaProvenance } from '#dto/llama-cpp';
import { verifyRuntimePackage } from '#runtimes/llama-cpp/tools/verify';
import { makeRuntime, sourceRoot } from './fixtures.test-support.js';

test('BF16 distribution metadata retains the legacy default and rejects stale build-time policy', async t => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-retained-metadata-'));
  await using cleanup = {
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
  await makeRuntime({ root, runtime: 'llama-cpp', selection: 'minimal' });
  const directory = join(root, 'package-inputs/llama-cpp');
  const path = join(directory, 'manifest.json');
  const original = decodeLlamaManifest({ input: JSON.parse(await readFile(path, 'utf8')) });
  async function change({ transform }: { transform: ({ metadata }: { metadata: LlamaProvenance }) => object }): Promise<void> {
    const profiles = Object.fromEntries(Object.entries(original.profiles).map(([name, profile]) => [name, { variants: Object.fromEntries(Object.entries(profile.variants).map(([variant, metadata]) => [variant, transform({ metadata })])) }]));
    await writeFile(path, JSON.stringify({ ...original, profiles }));
  }
  await t.test('current advertisement is accepted', async () => {
    await verifyRuntimePackage({ sourceRoot, directory, requireClean: true, checkNpmPack: false });
  });
  await t.test('missing advertisement keeps the legacy interpretation', async () => {
    await change({
      transform: ({ metadata }) => {
        const { visionBf16Storage: _advertisement, ...rest } = metadata;
        return rest;
      },
    });
    await verifyRuntimePackage({ sourceRoot, directory, requireClean: true, checkNpmPack: false });
  });
  for (const [name, transform] of [
    ['malformed advertisement', ({ metadata }: { metadata: LlamaProvenance }) => ({ ...metadata, visionBf16Storage: 'retain_checked' })],
    ['retained storage as default', ({ metadata }: { metadata: LlamaProvenance }) => ({ ...metadata, visionBf16Storage: { ...metadata.visionBf16Storage, default: 'retain_checked' } })],
    ['experimental build flag', ({ metadata }: { metadata: LlamaProvenance }) => ({ ...metadata, experimentalVisionBf16: true })],
    ['obsolete CMake equals option', ({ metadata }: { metadata: LlamaProvenance }) => ({ ...metadata, cmakeCommand: ['-DLCB_WEBGPU_BF16_STORAGE=retain-checked'] })],
    ['obsolete typed CMake option', ({ metadata }: { metadata: LlamaProvenance }) => ({ ...metadata, cmakeCommand: ['-DLCB_WEBGPU_BF16_STORAGE:STRING=retain-checked'] })],
  ] as const) {
    await t.test(name, async () => {
      await change({ transform });
      await assert.rejects(verifyRuntimePackage({ sourceRoot, directory, requireClean: true, checkNpmPack: false }));
    });
  }
});
