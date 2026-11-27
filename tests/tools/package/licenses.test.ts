import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, rm, symlink } from 'node:fs/promises';
import { join } from 'node:path';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { checkNoticeSource, embeddedComments, makeLicenseIndex, writeLicenseIndex, validateLicenseIndex, noticeOwner, collectNotices } from '#tools/package/licenses';
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

test('llama.cpp UI notices retain their original bytes under each upstream package and vendored library', async () => {
  await using temporary = await temporaryDirectory();
  const source = join(temporary.root, 'llama.cpp');
  const directory = join(temporary.root, 'package');
  const destination = join(directory, 'licenses/0');
  const notices = [
    ['ui/apps/chat/LICENSE', '@ggml-org/llama-chat'],
    ['ui/apps/chat/alias/LICENSE', 'llama-chat'],
    ['ui/apps/chat/src/lib/vendors/big-integer/LICENSE', 'big-integer'],
    ['ui/apps/chat/src/lib/vendors/nerdamer-prime/LICENSE', 'nerdamer-prime'],
    ['ui/lib/harness/LICENSE', '@ggml-org/llama-harness'],
    ['ui/lib/ui/LICENSE', '@ggml-org/llama-ui'],
    ['ui/lib/voice/LICENSE', '@ggml-org/llama-voice'],
  ] as const;
  await write({ path: join(directory, 'LICENSE'), data: 'Core notice' });
  for (const [relative, id] of notices) {
    await write({ path: join(source, relative), data: `${id} notice\r\n` });
  }
  assert.equal(await collectNotices({ source, destination, required: true }), notices.length);
  const options = { directory, runtime: 'llama-cpp' as const, licenseRoots: ['llama.cpp'], metadata };
  const index = await writeLicenseIndex(options);
  for (const [relative, id] of notices) {
    const path = `licenses/0/${relative}`;
    assert.deepEqual(index.components.find(component => component.id === id)?.notices, [{ path }]);
    assert.deepEqual(await readFile(join(directory, path)), await readFile(join(source, relative)));
  }
  await validateLicenseIndex({ ...options, index });
});

test('source preflight and package writing report unclassified owners and retain original notices in the index', async context => {
  const warnings = context.mock.method(process.stderr, 'write', () => true);
  await using temporary = await temporaryDirectory();
  const source = join(temporary.root, 'llama.cpp');
  const directory = join(temporary.root, 'package');
  await write({ path: join(source, 'LICENSE'), data: 'Upstream license' });
  await write({ path: join(source, 'ui/lib/harness/LICENSE'), data: 'Harness license' });
  assert.equal(await checkNoticeSource({ source }), 2);
  assert.equal(warnings.mock.callCount(), 0);
  const originals = [
    ['unexpected-tree/library/LICENSE', '\uFEFFTerms with a BOM\r\n'],
    ['ui/lib/new-library/COPYRIGHT', 'Copyright © library contributors\r\n'],
    ['ui/lib/new-library/COPYING.txt', ' Redistribution terms\n\n'],
    ['ui/lib/new-library/LICENSE.APACHE', 'Apache license terms\r\n'],
    ['ui/lib/new-library/LICENSE.Apache-2.0.txt', 'Apache license terms as a text document\n'],
    ['ui/lib/new-library/LICENSE.MIT', 'MIT license terms\n'],
    ['ui/lib/new-library/LICENSE.BSD-3-Clause', 'BSD license terms\r\n'],
    ['ui/lib/new-library/LICENSE.CC0-1.0', 'CC0 terms\r\n'],
    ['ui/lib/new-library/LICENSE.MPL-2.0', 'MPL license terms\n'],
    ['ui/lib/new-library/LICENSE.LGPL-2.1', 'LGPL license terms\n'],
    ['ui/lib/new-library/LICENSE.Zlib', 'Zlib license terms\n'],
    ['ui/lib/new-library/LICENSE.extra', 'Additional license terms\n'],
    ['ui/lib/new-library/LICENSE.upstream_terms-1.2.md', 'Supplemental terms\n'],
  ] as const;
  for (const [path, data] of originals) {
    await write({ path: join(source, path), data });
  }
  assert.equal(await checkNoticeSource({ source }), 2 + originals.length);
  assert.equal(warnings.mock.callCount(), originals.length);
  for (const [path] of originals) {
    assert(warnings.mock.calls.some(call => call.arguments[0] === `Warning: Unclassified native notice owner: llama.cpp/${path}\n`));
  }
  warnings.mock.resetCalls();
  await write({ path: join(directory, 'LICENSE'), data: 'Core notice' });
  assert.equal(await collectNotices({ source, destination: join(directory, 'licenses/0'), required: true }), 2 + originals.length);
  const options = { directory, runtime: 'llama-cpp' as const, licenseRoots: ['llama.cpp'], metadata };
  const index = await writeLicenseIndex(options);
  for (const [path, data] of originals) {
    const origin = `llama.cpp/${path}`;
    const packaged = `licenses/0/${path}`;
    assert.deepEqual(index.components.find(component => component.id === `unclassified:${origin}`), {
      id: `unclassified:${origin}`,
      name: `Unclassified owner (${origin})`,
      version: null,
      license: null,
      notices: [{ path: packaged }],
    });
    assert.deepEqual(await readFile(join(directory, packaged)), Buffer.from(data));
    assert(warnings.mock.calls.some(call => call.arguments[0] === `Warning: Unclassified native notice owner: ${origin} (${packaged})\n`));
  }
  assert.equal(warnings.mock.callCount(), originals.length);
  await validateLicenseIndex({ ...options, index });
  assert.equal(warnings.mock.callCount(), originals.length);
});

test('unclassified IDs distinguish notice roots and remain stable when a uniquely named root moves', async () => {
  await using temporary = await temporaryDirectory();
  const directory = temporary.root;
  await write({ path: join(directory, 'LICENSE'), data: 'Core notice' });
  for (const position of [0, 1, 2]) {
    await write({ path: join(directory, `licenses/${position}/lib/component/LICENSE.md`), data: 'Upstream license' });
  }
  const options = { directory, runtime: 'llama-cpp' as const, metadata };
  const first = await makeLicenseIndex({ ...options, licenseRoots: ['llama.cpp', 'toolchain', 'llama.cpp'] });
  const second = await makeLicenseIndex({ ...options, licenseRoots: ['toolchain', 'llama.cpp', 'llama.cpp'] });
  const firstIds = first.index.components.filter(component => component.id.startsWith('unclassified:')).map(component => component.id);
  const secondIds = second.index.components.filter(component => component.id.startsWith('unclassified:')).map(component => component.id);
  assert.equal(new Set(firstIds).size, 3);
  assert.equal(new Set(secondIds).size, 3);
  assert(firstIds.includes('unclassified:toolchain/lib/component/LICENSE.md'));
  assert(secondIds.includes('unclassified:toolchain/lib/component/LICENSE.md'));
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
  await assert.rejects(validateLicenseIndex({ ...options, index }), /Unrecognized native notice file/u);
});

test('malformed, unrelated and linked license inputs and linked index destinations fail without overwriting targets', async () => {
  await using temporary = await temporaryDirectory();
  const directory = temporary.root;
  await licenseFixture({ directory });
  const options = { directory, runtime: 'llama-cpp' as const, licenseRoots: ['llama.cpp'], metadata };
  const original = join(directory, 'licenses/0/LICENSE');
  for (const data of ['', ' \n']) {
    await write({ path: original, data });
    await assert.rejects(writeLicenseIndex(options), /Empty/u);
  }
  await write({ path: original, data: Uint8Array.of(0xff) });
  await assert.rejects(writeLicenseIndex(options), /not valid for encoding utf-8/u);
  await write({ path: original, data: 'Notice' });
  const invalid = [
    ['licenses/embedded/unreviewed.h.txt', /Unclassified embedded notice/u],
    ['licenses/embedded/vendor/nlohmann/json.hpp.bin', /Unclassified embedded notice/u],
    ['licenses/0/ui/apps/chat/LICENSE..txt', /Unrecognized native notice file/u],
    ['licenses/0/ui/lib/new-library/LICENSE.MIT.js', /Unrecognized native notice file/u],
    ['licenses/0/ui/lib/new-library/LICENSE.WASM', /Unrecognized native notice file/u],
    ['licenses/0/ui/lib/new-library/LICENSED.txt', /Unrecognized native notice file/u],
    ['licenses/0/ui/lib/unrelated.txt', /Unrecognized native notice file/u],
    ['licenses/0/invalid:path/LICENSE', /Unsafe native notice path/u],
    ['licenses/2/ui/lib/unreviewed/LICENSE', /Unknown license root/u],
  ] as const;
  for (const [name, message] of invalid) {
    await write({ path: join(directory, name), data: '// Invalid input' });
    await assert.rejects(makeLicenseIndex(options), message);
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

test('software ownership distinguishes independent roots and nested packages and leaves unknown owners unclassified', () => {
  assert.notEqual(noticeOwner({ root: 'emscripten', relative: 'node_modules/source-map/LICENSE' })?.[0], noticeOwner({ root: 'emscripten', relative: 'node_modules/tool/node_modules/source-map/LICENSE' })?.[0]);
  assert.equal(noticeOwner({ root: 'emscripten', relative: 'node_modules/@scope/lib/LICENSE' })?.[1], '@scope/lib');
  assert.notEqual(noticeOwner({ root: 'ggml', relative: 'LICENSE' })?.[0], noticeOwner({ root: 'stable-diffusion.cpp', relative: 'ggml/LICENSE' })?.[0]);
  assert.deepEqual(noticeOwner({ root: 'stable-diffusion.cpp', relative: 'src/core/cuda/sageattention/LICENSE' }), ['stable-diffusion.cpp/sageattention', 'SageAttention (stable-diffusion.cpp)']);
  assert.equal(noticeOwner({ root: 'stable-diffusion.cpp', relative: 'ggml/src/ggml-cuda/sageattention/LICENSE' })?.[0], noticeOwner({ root: 'ggml', relative: 'ggml/src/ggml-cuda/sageattention/LICENSE' })?.[0]);
  for (const relative of ['src/core/cuda/unreviewed/LICENSE', 'src/core/cuda/sageattention-extra/LICENSE']) {
    assert.equal(noticeOwner({ root: 'stable-diffusion.cpp', relative }), undefined);
  }
  assert.throws(() => noticeOwner({ root: 'emscripten', relative: 'node_modules/@scope/LICENSE' }), /Unowned node_modules notice/u);
  for (const relative of ['../x', '/x', 'a//b', 'a/./b', 'C:/x', 'a\\b', 'a\0b', '']) {
    assert.throws(() => noticeOwner({ root: 'llama.cpp', relative }));
  }
});

test('notice collection and preflight skip unrelated links and reject missing or linked required notices', async () => {
  await using temporary = await temporaryDirectory();
  const source = join(temporary.root, 'source');
  await write({ path: join(source, 'unrelated.txt'), data: 'Unrelated file' });
  const destination = join(temporary.root, 'output');
  await assert.rejects(checkNoticeSource({ source }), /No license notices/u);
  await assert.rejects(collectNotices({ source, destination, required: true }), /No license notices/u);
  await write({ path: join(source, 'LICENSE'), data: 'Complete original\r\n' });
  await symlink('/missing-unrelated', join(source, 'unrelated'));
  assert.equal(await collectNotices({ source, destination, required: true }), 1);
  assert.equal(await readFile(join(destination, 'LICENSE'), 'utf8'), 'Complete original\r\n');
  await symlink('/missing-notice', join(source, 'LICENSE-linked'));
  await assert.rejects(checkNoticeSource({ source }), /Linked native notice/u);
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
