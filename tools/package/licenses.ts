import { readdir, lstat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import type { RuntimeName, LicenseIndex, LicenseComponent, LicenseNotice, PackageMetadata } from '#dto/shared';
import { regularFile, relativePath, directory as directoryPath, optionalStat } from '#tools/shared/files';
import { assertSame, copyRegular, readRegularBytes } from './files.js';

export const embeddedNotices: Record<RuntimeName, Readonly<Record<string, readonly [string, string]>>> = {
  'llama-cpp': {
    'vendor/miniaudio/miniaudio.h': ['miniaudio', 'miniaudio'],
    'vendor/stb/stb_image.h': ['stb_image', 'stb_image'],
    'vendor/nlohmann/json.hpp': ['nlohmann-json', 'nlohmann/json'],
    'vendor/nlohmann/json_fwd.hpp': ['nlohmann-json', 'nlohmann/json'],
    'vendor/sheredom/subprocess.h': ['subprocess.h', 'subprocess.h'],
    'vendor/hash/sha1/sha1.c': ['sha1', 'SHA-1 by Steve Reid'],
  },
  'stable-diffusion-cpp': { 'json.hpp': ['nlohmann-json', 'nlohmann/json'], 'stb_image.h': ['stb_image', 'stb_image'], 'stb_image_resize.h': ['stb_image_resize', 'stb_image_resize'], 'stb_image_write.h': ['stb_image_write', 'stb_image_write'] },
};

/** Keep every comment and attribution. Ambiguous C/C++ syntax retains the entire input. */
export function embeddedComments({ original }: { original: string }): string {
  if (original.includes('??/')) {
    return original;
  }
  const pieces: string[] = [];
  const positions: number[] = [];
  const removed: number[] = [];
  let previous = 0;
  let total = 0;
  for (const splice of original.matchAll(/\\\r?\n/gu)) {
    pieces.push(original.slice(previous, splice.index));
    positions.push(splice.index - total);
    total += splice[0].length;
    removed.push(total);
    previous = splice.index + splice[0].length;
  }
  pieces.push(original.slice(previous));
  const source = pieces.join('');
  function physical({ position }: { position: number }): number {
    let left = 0;
    let right = positions.length;
    while (left < right) {
      const middle = Math.floor((left + right) / 2);
      if (positions[middle]! <= position) {
        left = middle + 1;
      } else {
        right = middle;
      }
    }
    return position + (removed[left - 1] ?? 0);
  }
  const comments: string[] = [];
  const spans: [number, number][] = [];
  function record({ start, end }: { start: number; end: number }): void {
    const left = physical({ position: start });
    const right = physical({ position: end - 1 }) + 1;
    const previous = spans.at(-1);
    if (previous && !original.slice(previous[1], left).trim()) {
      previous[1] = right;
      comments[comments.length - 1] = original.slice(previous[0], right);
    } else {
      spans.push([left, right]);
      comments.push(original.slice(left, right));
    }
  }
  let i = 0;
  while (i < source.length) {
    const start = i;
    if (source.startsWith('/*', i)) {
      const end = source.indexOf('*/', i + 2);
      if (end < 0) {
        return original;
      }
      i = end + 2;
      record({ start, end: i });
    } else if (source.startsWith('//', i)) {
      const end = source.indexOf('\n', i + 2);
      i = end < 0 ? source.length : end + 1;
      record({ start, end: i });
    } else if (source.startsWith('R"', i)) {
      const match = /^R"([^\s()\\]{0,16})\(/u.exec(source.slice(i));
      if (!match) {
        return original;
      }
      const delimiter = ')' + match[1] + '"';
      const end = source.indexOf(delimiter, i + match[0].length);
      if (end < 0) {
        return original;
      }
      i = end + delimiter.length;
    } else if (/\d/u.test(source[i] ?? '') && (i === 0 || !/[A-Za-z0-9_]/u.test(source[i - 1] ?? ''))) {
      const match = /^[0-9][A-Za-z0-9_.']*/u.exec(source.slice(i));
      if (!match) {
        return original;
      }
      i += match[0].length;
    } else if (source[i] === '"' || source[i] === "'") {
      const quote = source[i++];
      while (i < source.length && source[i] !== quote) {
        if (source[i] === '\\') {
          i += 2;
        } else if (source[i] === '\r' || source[i] === '\n') {
          return original;
        } else {
          i++;
        }
      }
      if (i >= source.length) {
        return original;
      }
      i++;
    } else {
      i++;
    }
  }
  if (!comments.length) {
    return original;
  }
  for (const match of original.matchAll(/copyright\s*(?:\(c\)|[0-9])|SPDX-(?:FileCopyrightText|License-Identifier)|permission is hereby granted|public domain|redistribution and use/giu)) {
    if (!spans.some(([start, end]) => start <= match.index && match.index < end)) {
      return original;
    }
  }
  return comments.join('\n\n');
}

export function noticeOwner({ root, relative }: { root: string; relative: string }): readonly [string, string] {
  relativePath({ value: relative });
  if (relative.includes(':')) {
    throw new Error('Unsafe native notice path');
  }
  if (root === 'llama.cpp' && relative === 'licenses/LICENSE-jsonhpp') {
    return ['nlohmann-json', 'nlohmann/json'];
  }
  const parts = relative.split('/');
  if (parts.includes('node_modules')) {
    const at = parts.lastIndexOf('node_modules') + 1;
    const stop = at + (parts[at]?.startsWith('@') ? 2 : 1);
    if (stop >= parts.length) {
      throw new Error(`Unowned node_modules notice: ${relative}`);
    }
    return [root + '/' + parts.slice(0, stop).join('/'), parts.slice(at, stop).join('/')];
  }
  const rules: Record<string, Record<string, readonly [string, string]>> = {
    'llama.cpp': {
      'vendor/cpp-httplib/': ['cpp-httplib', 'cpp-httplib'],
      'vendor/hash/xxhash/': ['xxhash', 'xxHash'],
      'vendor/hash/sha256/': ['sha256', 'SHA-256 (vendor/hash/sha256)'],
      'vendor/hash/rotate-bits/': ['rotate-bits', 'rotate-bits'],
      'tools/ui/src/lib/vendors/big-integer/': ['big-integer', 'big-integer'],
      'tools/ui/src/lib/vendors/nerdamer-prime/': ['nerdamer-prime', 'nerdamer-prime'],
      'gguf-py/': ['gguf-py', 'gguf-py'],
    },
    'stable-diffusion.cpp': {
      'src/core/cuda/sageattention/': ['stable-diffusion.cpp/sageattention', 'SageAttention (stable-diffusion.cpp)'],
      'ggml/src/ggml-cuda/sageattention/': ['ggml/sageattention', 'SageAttention (ggml)'],
      'ggml/': ['stable-diffusion.cpp/ggml', 'ggml (stable-diffusion.cpp)'],
      'thirdparty/LICENSE.darts_clone.txt': ['darts-clone', 'Darts-clone'],
      'examples/server/frontend/': ['stable-diffusion.cpp/frontend', 'stable-diffusion.cpp server frontend'],
    },
    ggml: { 'ggml/src/ggml-cuda/sageattention/': ['ggml/sageattention', 'SageAttention (ggml)'] },
    emscripten: { 'system/lib/libc/musl/': ['emscripten/musl', 'musl'], 'system/lib/openmp/src/thirdparty/ittnotify/': ['emscripten/ittnotify', 'ittnotify'], 'system/include/SDL/': ['emscripten/SDL', 'SDL'], 'cache/sysroot/include/SDL/': ['emscripten/SDL', 'SDL'], 'test/sounds/': ['emscripten/test-sounds', 'Emscripten test sounds'] },
  };
  for (const [prefix, owner] of Object.entries(rules[root] ?? {})) {
    if (relative === prefix || relative.startsWith(prefix)) {
      return owner;
    }
  }
  for (const marker of ['system/lib/', 'third_party/', 'thirdparty/', 'vendor/']) {
    if (relative.startsWith(marker)) {
      const tail = relative.slice(marker.length).split('/');
      if (tail.length < 2 || !tail[0]) {
        throw new Error(`Unclassified native notice: ${root}/${relative}`);
      }
      return [`${root}/${marker}${tail[0]}`, tail[0]];
    }
  }
  if (root === 'emdawnwebgpu_pkg' && (relative.startsWith('webgpu/') || relative.startsWith('webgpu_cpp/'))) {
    return [`emdawnwebgpu_pkg/${parts[0]}`, `emdawnwebgpu_pkg / ${parts[0]}`];
  }
  if (parts.length === 1 || (root === 'llama.cpp' && relative.startsWith('cmake/')) || root === 'ggml') {
    return [root, root];
  }
  throw new Error(`Unclassified native notice: ${root}/${relative}`);
}

async function noticeFiles({ root, prefix }: { root: string; prefix: string }): Promise<string[]> {
  await directoryPath({ path: root });
  const result: string[] = [];
  for (const name of (await readdir(root)).sort()) {
    if (name === '.git') {
      continue;
    }
    const file = join(root, name);
    const mode = await lstat(file);
    const relative = prefix + name;
    if (mode.isSymbolicLink()) {
      throw new Error(`Linked native notice: ${relative}`);
    }
    if (mode.isDirectory()) {
      result.push(...await noticeFiles({ root: file, prefix: relative + '/' }));
    } else if (mode.isFile()) {
      result.push(relative);
    } else {
      throw new Error(`Non-regular native notice: ${relative}`);
    }
  }
  return result;
}

async function noticeBytes({ path }: { path: string }): Promise<Buffer> {
  const raw = await readRegularBytes({ path, maxBytes: 32 * 1024 ** 2 });
  if (!new TextDecoder('utf-8', { fatal: true }).decode(raw).trim()) {
    throw new Error(`Empty or oversized native notice: ${path}`);
  }
  return raw;
}

export async function makeLicenseIndex({ directory: root, runtime, licenseRoots, metadata }: { directory: string; runtime: RuntimeName; licenseRoots: readonly string[]; metadata: PackageMetadata }) {
  const components = new Map<string, LicenseComponent>();
  const extracted = new Map<string, Buffer>();
  const paths = ['LICENSE', ...(await noticeFiles({ root: join(root, 'licenses'), prefix: 'licenses/' })).filter(name => !name.startsWith('licenses/extracted/'))];
  for (const relative of paths) {
    const raw = await noticeBytes({ path: join(root, relative) });
    let notice: LicenseNotice = { path: relative };
    let owner: readonly [string, string];
    if (relative === 'LICENSE') {
      owner = [`${runtime}-browser-core`, `${runtime}-browser-core`];
    } else if (relative.startsWith('licenses/embedded/')) {
      const source = relative.slice('licenses/embedded/'.length, -'.txt'.length);
      const identity = embeddedNotices[runtime][source];
      if (!identity) {
        throw new Error(`Unclassified embedded notice: ${relative}`);
      }
      owner = identity;
      const text = Buffer.from(embeddedComments({ original: raw.toString('utf8') }));
      if (!text.equals(raw)) {
        const output = `licenses/extracted/${source}.txt`;
        extracted.set(output, text);
        notice = { path: output, sourcePath: relative };
      }
    } else {
      let tail = relative.slice('licenses/'.length);
      let sourceRoot: string;
      switch (runtime) {
      case 'llama-cpp': {
        const at = tail.indexOf('/');
        const position = tail.slice(0, at);
        sourceRoot = licenseRoots[Number(position)] ?? '';
        if (!/^(0|[1-9][0-9]*)$/u.test(position) || !sourceRoot) {
          throw new Error(`Unknown license root: ${relative}`);
        }
        tail = tail.slice(at + 1);
        break;
      }
      case 'stable-diffusion-cpp': {
        if (tail.startsWith('toolchain/')) {
          tail = tail.slice('toolchain/'.length);
        }
        const at = tail.indexOf('/');
        const rootName = tail.slice(0, at);
        tail = tail.slice(at + 1);
        sourceRoot = rootName === 'stable-diffusion' ? 'stable-diffusion.cpp' : rootName;
        break;
      }
      default: {
        const exhaustive: never = runtime;
        throw new Error(String(exhaustive));
      }
      }
      owner = noticeOwner({ root: sourceRoot, relative: tail });
    }
    const [id, name] = owner;
    let component = components.get(id);
    if (!component) {
      component = { id, name, version: relative === 'LICENSE' ? metadata.version : null, license: relative === 'LICENSE' ? metadata.license ?? null : null, notices: [] };
      components.set(id, component);
    }
    if (component.name !== name) {
      throw new Error(`Ambiguous native component: ${id}`);
    }
    component.notices.push(notice);
  }
  return { index: { components: [...components.entries()].sort(([a], [b]) => a < b ? -1 : 1).map(([, value]) => value) }, extracted };
}

export async function writeLicenseIndex({ directory, runtime, licenseRoots, metadata }: { directory: string; runtime: RuntimeName; licenseRoots: readonly string[]; metadata: PackageMetadata }): Promise<LicenseIndex> {
  const { index, extracted } = await makeLicenseIndex({ directory, runtime, licenseRoots, metadata });
  for (const [relative, raw] of extracted) {
    const output = join(directory, relative);
    await directoryPath({ path: dirname(output), create: true });
    if (await optionalStat({ path: output })) {
      await regularFile({ path: output });
    }
    await writeFile(output, raw);
  }
  if (await optionalStat({ path: join(directory, 'native-license-index.json') })) {
    await regularFile({ path: join(directory, 'native-license-index.json') });
  }
  await writeFile(join(directory, 'native-license-index.json'), JSON.stringify(index, undefined, 2) + '\n');
  return index;
}

export async function validateLicenseIndex({ directory, runtime, licenseRoots, metadata, index: actual }: { directory: string; runtime: RuntimeName; licenseRoots: readonly string[]; metadata: PackageMetadata; index: LicenseIndex }): Promise<void> {
  const { index, extracted } = await makeLicenseIndex({ directory, runtime, licenseRoots, metadata });
  assertSame({ actual, expected: index, message: 'Native license index differs from collected notices' });
  const all = await noticeFiles({ root: join(directory, 'licenses'), prefix: 'licenses/' });
  assertSame({ actual: all.filter(name => name.startsWith('licenses/extracted/')).sort(), expected: [...extracted.keys()].sort(), message: 'Extracted notice inventory differs' });
  for (const [name, raw] of extracted) {
    assertSame({ actual: await noticeBytes({ path: join(directory, name) }), expected: raw, message: `Extracted notice differs from its source: ${name}` });
  }
}

export async function collectNotices({ source, destination, required }: { source: string; destination: string; required: boolean }): Promise<number> {
  await directoryPath({ path: source });
  let count = 0;
  async function collect({ root, prefix }: { root: string; prefix: string }): Promise<void> {
    for (const name of (await readdir(root)).sort()) {
      if (name === '.git') {
        continue;
      }
      const path = join(root, name);
      const mode = await lstat(path);
      const notice = /^(LICENSE|COPYING|COPYRIGHT)/iu.test(name);
      if (mode.isSymbolicLink()) {
        if (notice) {
          throw new Error(`Linked native notice: ${path}`);
        }
        continue;
      }
      if (mode.isDirectory()) {
        await collect({ root: path, prefix: prefix + name + '/' });
      } else if (mode.isFile() && notice) {
        await noticeBytes({ path });
        await copyRegular({ source: path, destination: join(destination, prefix + name) });
        count++;
      }
    }
  }
  await collect({ root: source, prefix: '' });
  if (required && !count) {
    throw new Error(`No license notices: ${source}`);
  }
  return count;
}
