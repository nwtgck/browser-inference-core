import assert from 'node:assert/strict';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { runProcess } from '#tools/shared/process';
import { sourceFixture, write } from './source.test-support.js';

const fixtureDirectory = path.resolve('tests/runtimes/stable-diffusion-cpp/tools/fixtures');
const runtimeDirectory = 'runtimes/stable-diffusion-cpp';

async function configure({ variant, jspi, memory64, identities }: { variant: string; jspi: boolean; memory64: boolean; identities: unknown }) {
  const fixture = await sourceFixture();
  try {
    const sourceRoot = fixture.repository;
    const runtime = path.join(sourceRoot, runtimeDirectory);
    const toolingRoot = path.join(fixture.root, 'emit');
    const output = path.join(fixture.root, 'configured');
    const sd = path.join(fixture.root, 'sd');
    const ggml = path.join(fixture.root, 'ggml');
    await mkdir(path.join(runtime, 'native/abi'), { recursive: true });
    for (const name of ['bridge.cpp', 'metadata.cpp']) {
      await copyFile(path.join(fixtureDirectory, 'configure-only.cpp'), path.join(runtime, 'native/abi', name));
    }
    for (const name of ['sd', 'ggml']) {
      await write({ root: fixture.root, name: `${name}/include/${name === 'sd' ? 'stable-diffusion' : 'ggml'}.h`, text: '// configure-only public header\n' });
    }
    await copyFile(path.join(fixtureDirectory, 'upstream-CMakeLists.txt'), path.join(sd, 'CMakeLists.txt'));
    await copyFile(path.join(fixtureDirectory, 'configure-only.cpp'), path.join(sd, 'configure-only.cpp'));
    await write({ root: sourceRoot, name: 'tools/abi/Bindings.cmake', text: await readFile(path.join(fixtureDirectory, 'BicoreAbi.cmake'), 'utf8') });
    await write({ root: sourceRoot, name: 'tools/shared/TargetSources.cmake', text: await readFile(path.resolve('tools/shared/TargetSources.cmake'), 'utf8') });
    await write({ root: toolingRoot, name: `${runtimeDirectory}/tools/bindings.js`, text: '// configure-only ABI entry\n' });
    await write({ root: runtime, name: 'CMakeLists.txt', text: `${await readFile(path.resolve(runtimeDirectory, 'CMakeLists.txt'), 'utf8')}\ninclude("${path.join(fixtureDirectory, 'inspect.cmake')}")\n` });
    await copyFile(path.resolve(runtimeDirectory, 'config/variants.json'), path.join(runtime, 'config/variants.json'));
    for (const name of ['wasm-probes.cpp', 'qwen-timestep-probe.cpp', 'bf16-weights-probe.cpp', 'graph-walk-probe.cpp', 'conv3d-bias-probe.cpp', 'webgpu-performance-probe.cpp']) {
      const destination = path.join(sourceRoot, 'tests/runtimes/stable_diffusion_cpp', name);
      await mkdir(path.dirname(destination), { recursive: true });
      await copyFile(path.join(fixtureDirectory, 'configure-only.cpp'), destination);
    }
    const receipt = path.join(fixture.root, 'receipt.json');
    await writeFile(receipt, JSON.stringify({ upstreams: identities }));
    await runProcess({
      command: 'cmake',
      args: ['-S', runtime, '-B', output, '-DEMSCRIPTEN=ON',
        `-DSDCB_SOURCE=${sd}`, `-DSDCB_GGML_SOURCE=${ggml}`, `-DSDCB_PROVENANCE=${receipt}`,
        `-DSDCB_CHECKOUT_ROOT=${sourceRoot}`, `-DSDCB_COMPILE_ROOT=${sourceRoot}`, `-DSDCB_TOOLING_ROOT=${toolingRoot}`, `-DSDCB_NODE=${process.execPath}`,
        `-DSDCB_VARIANT=${variant}`, `-DSDCB_JSPI=${jspi ? 'ON' : 'OFF'}`, `-DSDCB_MEMORY64=${memory64 ? 'ON' : 'OFF'}`,
        `-DSDCB_MAXIMUM_MEMORY=${memory64 ? '17179869184' : '4294967296'}`,
      ],
      cwd: fixture.root,
    });
    const data: Record<string, string> = {};
    for (const name of ['upstream-identities', 'abi-inputs', 'link-options', 'defines', 'link-inputs', 'sources', 'backend-defines', 'backend-options']) {
      data[name] = await readFile(path.join(output, `${name}.txt`), 'utf8');
    }
    return { fixture, data, sourceRoot };
  } catch (error) {
    await fixture[Symbol.asyncDispose]();
    throw error;
  }
}

const identities = { 'stable-diffusion': { commit: 'a'.repeat(40) }, ggml: { commit: 'b'.repeat(40) } };

test('evaluated CMake targets preserve browser/test and suspension boundaries', async () => {
  for (const variant of ['browser', 'test']) {
    for (const jspi of [false, true]) {
      const configured = await configure({ variant, jspi, memory64: false, identities });
      await using fixture = configured.fixture;
      const { data, sourceRoot } = configured;
      assert.equal(data['upstream-identities'], `${'a'.repeat(40)}-bicore;${'b'.repeat(40)}-bicore;bicore`);
      assert.ok(data['abi-inputs']?.startsWith(`core;stable-diffusion-cpp;${sourceRoot};`));
      assert.equal(data['backend-defines']?.split(';').filter((value) => value === 'GGML_WEBGPU_BATCH_PARAM_UPLOADS=1').length, 1);
      assert.ok(data['link-options']?.includes(jspi ? '-sJSPI=1' : '-sASYNCIFY=1'));
      assert.ok(data['link-inputs']?.includes('exports.json'));
      assert.equal(data['sources']?.includes('/wasm-probes.cpp'), variant === 'test');
      assert.equal(data['defines']?.includes('SDCB_VARIANT_TEST=1'), variant === 'test');
      assert.ok(data['defines']?.includes('GGML_MAX_NAME=160'));
      if (!jspi) {
        assert.equal(data['backend-options']?.includes('-exceptions'), false);
      }
    }
  }
});

test('evaluated CMake supports memory64 JSPI and rejects unsupported modes and malformed pins', async () => {
  const configured = await configure({ variant: 'browser', jspi: true, memory64: true, identities });
  await using fixture = configured.fixture;
  assert.ok(configured.data['link-options']?.includes('-sMEMORY64=1'));
  assert.ok(configured.data['link-options']?.includes('-sMAXIMUM_MEMORY=17179869184'));
  await assert.rejects(configure({ variant: 'browser', jspi: false, memory64: true, identities }), /memory64 profile requires JSPI/);
  await assert.rejects(configure({ variant: 'browser', jspi: true, memory64: false, identities: {} }), /Invalid upstream identity/);
});
