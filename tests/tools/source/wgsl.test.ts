import assert from 'node:assert/strict';
import { copyFile, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { generateWgsl } from '#tools/source/wgsl';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

test('unchanged upstream WGSL embedder owns generation, with stable output mtime and failure preservation', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = join(temporary.root, 'shaders');
  await mkdir(sourceRoot);
  await mkdir(join(temporary.root, 'generator'));
  const generator = join(temporary.root, 'generator', 'embed_wgsl.py');
  await copyFile(resolve('runtimes/llama-cpp/vendor/llama.cpp/ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py'), generator);
  await writeFile(join(sourceRoot, 'fixture.wgsl'), '@compute @workgroup_size(1) fn main() {}\n');
  const output = join(temporary.root, 'result', 'shaders.hpp');
  await generateWgsl({ sourceRoot, generator, output });
  const header = await readFile(output, 'utf8');
  assert.match(header, /fixture/u);
  assert.match(header, /@compute/u);
  const before = await stat(output);
  await generateWgsl({ sourceRoot, generator, output });
  assert.equal((await stat(output)).mtimeMs, before.mtimeMs);
  assert.deepEqual(await readdir(join(temporary.root, 'result')), ['shaders.hpp']);
  await writeFile(generator, 'raise RuntimeError("generator diagnostic")\n');
  await assert.rejects(generateWgsl({ sourceRoot, generator, output }), /generator diagnostic/u);
  assert.equal(await readFile(output, 'utf8'), header);
  assert.deepEqual(await readdir(join(temporary.root, 'result')), ['shaders.hpp']);
});
