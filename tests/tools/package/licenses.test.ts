import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, rm, symlink } from 'node:fs/promises';
import { join } from 'node:path';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { embeddedComments, makeLicenseIndex, writeLicenseIndex, validateLicenseIndex, noticeOwner, collectNotices } from '#tools/package/licenses';
import type { PackageMetadata } from '#dto/shared';
import { write } from './fixtures.test-support.js';

const metadata: PackageMetadata = { name: 'llama-cpp-browser-core', version: '0.1.0', type: 'module', exports: {}, license: 'MIT' };

async function licenseFixture({ directory }: { directory: string }): Promise<void> {
  for (const [name, data] of [['LICENSE', 'Core notice\r\n'], ['licenses/0/LICENSE', 'llama.cpp notice'], ['licenses/0/licenses/LICENSE-jsonhpp', 'JSON complete license'], ['licenses/embedded/vendor/nlohmann/json.hpp.txt', '// SPDX-License-Identifier: MIT\n/* Attribution without keywords */\nint implementation;']] as const) {
    await write({ path: join(directory, name), data });
  }
}

test('license index groups complete notices by software with reproducible bytes and independent package metadata', async () => {
  await using temporary = await temporaryDirectory();
  const directory = temporary.root;
  await licenseFixture({ directory });
  const options = { directory, runtime: 'llama-cpp' as const, licenseRoots: ['llama.cpp'], metadata };
  const index = await writeLicenseIndex(options);
  const components = new Map(index.components.map(component => [component.id, component]));
  assert.deepEqual([...components.keys()].sort(), ['llama-cpp-browser-core', 'llama.cpp', 'nlohmann-json']);
  assert.equal(components.get('llama-cpp-browser-core')?.version, '0.1.0');
  assert.equal(components.get('llama-cpp-browser-core')?.license, 'MIT');
  assert.equal(components.get('llama.cpp')?.version, null);
  assert.equal(components.get('llama.cpp')?.license, null);
  const notices = components.get('nlohmann-json')!.notices;
  assert.equal(notices.length, 2);
  const extracted = notices.find(notice => notice.sourcePath)!;
  assert.match(await readFile(join(directory, extracted.sourcePath!), 'utf8'), /int implementation/u);
  assert.doesNotMatch(await readFile(join(directory, extracted.path), 'utf8'), /int implementation/u);
  await validateLicenseIndex({ ...options, index });
  const before = await readFile(join(directory, 'native-license-index.json'));
  await writeLicenseIndex(options);
  assert.deepEqual(await readFile(join(directory, 'native-license-index.json')), before);
  await write({ path: join(directory, 'licenses/0/vendor/new-library/LICENSE'), data: 'New library notice' });
  const extended = await writeLicenseIndex(options);
  assert(extended.components.some(component => component.id.includes('new-library')));
  await validateLicenseIndex({ ...options, index: extended });
});

test('license validation rejects missing, duplicated, reassigned, extracted and unindexed documents', async () => {
  await using temporary = await temporaryDirectory();
  const directory = temporary.root;
  await licenseFixture({ directory });
  const options = { directory, runtime: 'llama-cpp' as const, licenseRoots: ['llama.cpp'], metadata };
  const index = await writeLicenseIndex(options);
  for (const mutation of ['missing', 'duplicate', 'owner'] as const) {
    const changed = structuredClone(index);
    switch (mutation) {
    case 'missing': changed.components.pop();
      break;
    case 'duplicate': changed.components.push(changed.components[0]!);
      break;
    case 'owner': changed.components[0]!.name = 'Wrong owner';
      break;
    }
    await assert.rejects(validateLicenseIndex({ ...options, index: changed }), /index differs/u);
  }
  const extracted = index.components.flatMap(component => component.notices).find(notice => notice.sourcePath)!;
  await write({ path: join(directory, extracted.path), data: 'Missing attribution' });
  await assert.rejects(validateLicenseIndex({ ...options, index }), /Extracted notice differs/u);
  await writeLicenseIndex(options);
  await write({ path: join(directory, 'licenses/0/NOTICE'), data: 'New notice' });
  await assert.rejects(validateLicenseIndex({ ...options, index }), /index differs/u);
});

test('empty, unreviewed and linked license sources and linked index destinations fail without overwriting targets', async () => {
  await using temporary = await temporaryDirectory();
  const directory = temporary.root;
  await licenseFixture({ directory });
  const options = { directory, runtime: 'llama-cpp' as const, licenseRoots: ['llama.cpp'], metadata };
  const original = join(directory, 'licenses/0/LICENSE');
  for (const data of ['', ' \n']) {
    await write({ path: original, data });
    await assert.rejects(writeLicenseIndex(options), /Empty/u);
  }
  await write({ path: original, data: 'Notice' });
  for (const name of ['licenses/embedded/unreviewed.h.txt', 'licenses/0/unexpected-tree/library/LICENSE']) {
    await write({ path: join(directory, name), data: '// Unknown owner' });
    await assert.rejects(makeLicenseIndex(options), /Unclassified/u);
    await rm(join(directory, name));
  }
  await symlink(directory, join(directory, 'licenses/0/linked'));
  await assert.rejects(writeLicenseIndex(options), /Linked/u);
  await rm(join(directory, 'licenses/0/linked'));
  const target = join(directory, 'untouched');
  await write({ path: target, data: 'original' });
  await symlink(target, join(directory, 'native-license-index.json'));
  await assert.rejects(writeLicenseIndex(options), /Linked/u);
  assert.equal(await readFile(target, 'utf8'), 'original');
});

test('software ownership distinguishes independent roots and nested packages while rejecting unreviewed paths', () => {
  assert.notEqual(noticeOwner({ root: 'emscripten', relative: 'node_modules/source-map/LICENSE' })[0], noticeOwner({ root: 'emscripten', relative: 'node_modules/tool/node_modules/source-map/LICENSE' })[0]);
  assert.equal(noticeOwner({ root: 'emscripten', relative: 'node_modules/@scope/lib/LICENSE' })[1], '@scope/lib');
  assert.notEqual(noticeOwner({ root: 'ggml', relative: 'LICENSE' })[0], noticeOwner({ root: 'stable-diffusion.cpp', relative: 'ggml/LICENSE' })[0]);
  assert.deepEqual(noticeOwner({ root: 'stable-diffusion.cpp', relative: 'src/core/cuda/sageattention/LICENSE' }), ['stable-diffusion.cpp/sageattention', 'SageAttention (stable-diffusion.cpp)']);
  assert.equal(noticeOwner({ root: 'stable-diffusion.cpp', relative: 'ggml/src/ggml-cuda/sageattention/LICENSE' })[0], noticeOwner({ root: 'ggml', relative: 'ggml/src/ggml-cuda/sageattention/LICENSE' })[0]);
  for (const relative of ['src/core/cuda/unreviewed/LICENSE', 'src/core/cuda/sageattention-extra/LICENSE']) {
    assert.throws(() => noticeOwner({ root: 'stable-diffusion.cpp', relative }), /Unclassified/u);
  }
  for (const relative of ['../x', '/x', 'a//b', 'a/./b', 'C:/x', 'a\\b', 'a\0b', '']) {
    assert.throws(() => noticeOwner({ root: 'llama.cpp', relative }));
  }
});

test('notice collection skips unrelated links and refuses linked or empty original notices', async () => {
  await using temporary = await temporaryDirectory();
  const source = join(temporary.root, 'source');
  await write({ path: join(source, 'LICENSE'), data: 'Complete original\r\n' });
  await symlink('/missing-unrelated', join(source, 'unrelated'));
  const destination = join(temporary.root, 'output');
  assert.equal(await collectNotices({ source, destination, required: true }), 1);
  assert.equal(await readFile(join(destination, 'LICENSE'), 'utf8'), 'Complete original\r\n');
  await symlink('/missing-notice', join(source, 'LICENSE-linked'));
  await assert.rejects(collectNotices({ source, destination, required: true }), /Linked native notice/u);
});

test('complete comment archives preserve character literals, raw strings, splices and line endings', () => {
  for (const literal of ["L'a'", "u'a'", "U'a'", "u8'a'", "'a'", "0xdead'beef"]) {
    const result = embeddedComments({ original: literal + '; // By A. Contributor\nchar c=\'x\'; /* Terms unchanged */' });
    assert.match(result, /By A\. Contributor/u);
    assert.match(result, /Terms unchanged/u);
  }
  assert.equal(embeddedComments({ original: 'const char *s="/* fake */"; auto r=R"tag(// fake)tag"; char c=\'/\';\n/* Actual notice */' }), '/* Actual notice */');
  for (const original of ['int x;', '/* unterminated', '"unterminated\n/* notice */', 'R"oops(unterminated', '/* A */ char *s="Copyright 2026 B";', '??/\n* legal text */']) {
    assert.equal(embeddedComments({ original }), original);
  }
  assert.equal(embeddedComments({ original: '// A\r\n// B\r\nint x;\r\n/* C */' }), '// A\r\n// B\r\n\n\n/* C */');
  const original = '/\\\n* Author *\\\n/\nint x;\n// Continued \\\nwithout keywords\nint y;';
  const extracted = embeddedComments({ original });
  assert.match(extracted, /Author/u);
  assert.match(extracted, /without keywords/u);
  assert.doesNotMatch(extracted, /int [xy]/u);
});
