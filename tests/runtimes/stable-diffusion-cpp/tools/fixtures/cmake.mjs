#!/usr/bin/env node
import { READ_ENV_BIC_TEST_CHECKOUT, READ_ENV_BIC_TEST_COMPILER_FAILURE, READ_ENV_BIC_TEST_MUTATE } from '#tools/shared/node-env';
// Synthetic external compiler boundary; no source is compiled or published.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const args = process.argv.slice(2);
if (args[0] === '--build') {
  if (READ_ENV_BIC_TEST_COMPILER_FAILURE() === '1') {
    throw new Error('Synthetic compiler failure');
  }
  const build = args[1];
  assert.ok(build);
  const selection = JSON.parse(await readFile(path.join(build, 'compiler-selection.json'), 'utf8'));
  const source = path.join(selection.compileRoot, 'runtimes/stable-diffusion-cpp/native/abi/metadata.cpp');
  await mkdir(path.join(build, 'target-sources'), { recursive: true });
  await writeFile(path.join(build, 'compile_commands.json'), JSON.stringify([{ directory: build, file: source, arguments: ['synthetic-c++', '-c', source] }]));
  await writeFile(path.join(build, 'target-sources/core.txt'), `${path.dirname(source)}\n${source}\n`);
  await writeFile(path.join(build, 'target-sources/core.txt.declared'), `${path.dirname(source)}\n${source}\n`);
  if (READ_ENV_BIC_TEST_MUTATE() === 'prepared') {
    await writeFile(path.join(selection.preparedRoot, 'patch-provenance.json'), '{"changed":true}\n');
  }
  if (READ_ENV_BIC_TEST_MUTATE() === 'snapshot') {
    await writeFile(path.join(selection.compileRoot, 'runtimes/stable-diffusion-cpp/config/variants.json'), '{"changed":true}\n');
  }
} else {
  const selected = (key) => args.find((value) => value.startsWith(`-D${key}=`))?.slice(key.length + 3);
  const build = args[args.indexOf('-B') + 1];
  const checkoutRoot = selected('SDCB_CHECKOUT_ROOT');
  const compileRoot = selected('SDCB_COMPILE_ROOT');
  const preparedRoot = path.dirname(selected('SDCB_SOURCE'));
  assert.ok(build && checkoutRoot && compileRoot && preparedRoot);
  const receipt = JSON.parse(await readFile(path.join(preparedRoot, 'patch-provenance.json'), 'utf8'));
  assert.equal(receipt.runtime, 'stable-diffusion-cpp');
  assert.ok(receipt.patches.length > 0);
  assert.equal(checkoutRoot, READ_ENV_BIC_TEST_CHECKOUT());
  assert.notEqual(compileRoot, checkoutRoot);
  await writeFile(path.join(build, 'compiler-selection.json'), JSON.stringify({ compileRoot, preparedRoot }));
}
