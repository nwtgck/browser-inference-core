import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, symlink, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { validatePackage } from '#tools/package/aggregate';
import { artifactDirectories } from '#tools/shared/contracts';
import { wasmExports, readWasmExports } from '#tools/package/runtime';
import { embeddedComments } from '#tools/package/licenses';
import { runProcess } from '#tools/shared/process';
import { hostFiles } from '#tools/package/runtime';
import { fileIdentity } from '#tools/shared/files';
import { packageFixture, sourceRoot, rehashRuntime, fixtureWasm, fixtureAbi, rewriteLlamaTargetMetadata } from './fixtures.test-support.js';
import { imageManifestCapabilities } from '#runtimes/stable-diffusion-cpp/tools/verify';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

async function validate({ directory }: { directory: string }) {
  return validatePackage({ sourceRoot, directory, requireClean: true, checkNpmPack: false });
}

test('root inventory owns five entries and derives complete payload closure through runtime manifests', async () => {
  await using fixture = await packageFixture();
  const result = await validate({ directory: fixture.directory });
  assert.equal(result.manifest.formatVersion, 5);
  assert.equal(result.manifest.files.length, 5);
  assert(result.files.size > result.manifest.files.length + 1);
  assert(result.manifest.files.every(entry => !entry.path.endsWith('.wasm')));
});

test('public runtime pairs and host modules resolve with their own packaged Wasm', async () => {
  await using fixture = await packageFixture();
  const result = await runProcess({
    command: process.execPath,
    args: ['--input-type=module', '-e', `\
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
for (const runtime of ['llama-cpp', 'stable-diffusion-cpp']) {
  const host = await import('browser-inference-core/' + runtime + '/host');
  assert.equal(typeof host.attachCore, 'function');
  const profile = runtime === 'llama-cpp' ? 'cpu-wasm32' : 'webgpu-wasm32-jspi';
  for (const variant of ['browser', 'test']) {
    const entry = 'browser-inference-core/' + runtime + '/profiles/' + profile + '/' + variant + '/core.mjs';
    const factory = (await import(entry)).default;
    assert.equal(typeof factory, 'function');
    const wasm = new URL('core.wasm', import.meta.resolve(entry));
    assert.equal(wasm.href, import.meta.resolve(entry.replace('.mjs', '.wasm')));
    const module = await factory({ wasmBinary: new Uint8Array(await readFile(wasm)) });
    assert.equal(module[runtime === 'llama-cpp' ? '_fixture_version' : '_sd_browser_bf16_retention_version'](), 1);
  }
}
`],
    cwd: fixture.directory,
  });
  assert.equal(result.exitCode, 0);
  for (const runtime of ['llama-cpp', 'stable-diffusion-cpp'] as const) {
    for (const name of hostFiles) {
      assert.deepEqual(await readFile(join(fixture.directory, artifactDirectories[runtime], 'host', name)), await readFile(join(sourceRoot, 'runtimes', runtime, 'host', name)));
    }
  }
});

test('profile provenance, dirty diagnostics and required assets survive assembly and reject rehashed omissions', async t => {
  await using fixture = await packageFixture();
  const rootPath = join(fixture.directory, 'manifest.json');
  const innerPath = join(fixture.directory, artifactDirectories['llama-cpp'], 'manifest.json');
  const originalRoot = await readFile(rootPath);
  const originalInner = await readFile(innerPath);
  async function restore() {
    await writeFile(innerPath, originalInner);
    await writeFile(rootPath, originalRoot);
  }
  for (const mutation of ['missing_variant', 'wrong_variant', 'wrong_configuration', 'dirty'] as const) {
    await t.test(mutation, async () => {
      await rehashRuntime({
        directory: fixture.directory,
        runtime: 'llama-cpp',
        change: manifest => {
          if (manifest.runtime !== 'llama-cpp') {
            throw new Error('Wrong test runtime');
          }
          const variants = manifest.profiles['cpu-wasm32']!.variants;
          switch (mutation) {
          case 'missing_variant': delete variants['test'];
            break;
          case 'wrong_variant': variants['browser']!.variant = 'test';
            break;
          case 'wrong_configuration': variants['browser']!.variantConfiguration = variants['test']!.variantConfiguration;
            break;
          case 'dirty': variants['browser']!.sourceDirty = true;
            variants['browser']!['sourceStatusAfterBuild'] = ['?? changed-input.cpp'];
            break;
          }
        },
      });
      await assert.rejects(validate({ directory: fixture.directory }), mutation === 'dirty' ? /cpu-wasm32\/browser.*sourceStatusAfterBuild.*changed-input.cpp/u : /[Vv]ariant/u);
      await restore();
    });
  }
  const module = join(fixture.directory, artifactDirectories['llama-cpp'], 'profiles/cpu-wasm32/browser/core.mjs');
  await rehashRuntime({
    directory: fixture.directory,
    runtime: 'llama-cpp',
    change: manifest => {
      if (manifest.runtime !== 'llama-cpp') {
        throw new Error('Wrong test runtime');
      }
    manifest.profiles['cpu-wasm32']!.variants['browser']!.browserModulePreparation = { input: { bytes: 1, sha256: '1'.repeat(64) }, output: { bytes: 0, sha256: '0'.repeat(64) }, alreadyAnnotated: false };
    },
  });
  await assert.rejects(validate({ directory: fixture.directory }), /preparation output mismatch/u);
  const identity = await fileIdentity({ path: module });
  identity['futureMetadata'] = { note: 'retained' };
  await rehashRuntime({
    directory: fixture.directory,
    runtime: 'llama-cpp',
    change: manifest => {
      if (manifest.runtime !== 'llama-cpp') {
        throw new Error('Wrong test runtime');
      }
    manifest.profiles['cpu-wasm32']!.variants['browser']!.browserModulePreparation!.output = identity;
    },
  });
  await validate({ directory: fixture.directory });
  await rehashRuntime({
    directory: fixture.directory,
    runtime: 'llama-cpp',
    change: manifest => {
      if (manifest.runtime !== 'llama-cpp') {
        throw new Error('Wrong test runtime');
      }
      manifest.profiles['cpu-wasm32']!.variants['browser']!.moduleDeclarationPreparation = { input: { bytes: 1, sha256: '1'.repeat(64) }, output: { bytes: 0, sha256: '0'.repeat(64) }, correctedBigIntReferences: 0 };
    },
  });
  await assert.rejects(validate({ directory: fixture.directory }), /Module declaration preparation output mismatch/u);
  const declarationIdentity = await fileIdentity({ path: module.replace(/\.mjs$/u, '.d.ts') });
  await rehashRuntime({
    directory: fixture.directory,
    runtime: 'llama-cpp',
    change: manifest => {
      if (manifest.runtime !== 'llama-cpp') {
        throw new Error('Wrong test runtime');
      }
      manifest.profiles['cpu-wasm32']!.variants['browser']!.moduleDeclarationPreparation!.output = declarationIdentity;
    },
  });
  await validate({ directory: fixture.directory });
  await restore();
  for (const relative of ['host/bindings.mjs', 'licenses/embedded/vendor/miniaudio/miniaudio.h.txt']) {
    const path = join(fixture.directory, artifactDirectories['llama-cpp'], relative);
    const bytes = await readFile(path);
    await rm(path);
    await rehashRuntime({ directory: fixture.directory, runtime: 'llama-cpp', change: () => {} });
    await assert.rejects(validate({ directory: fixture.directory }), /Missing runtime artifact/u);
    await writeFile(path, bytes);
    await restore();
  }
});

test('closure rejects modified, extra, linked and mixed-source payloads', async t => {
  await using fixture = await packageFixture();
  const payload = join(fixture.directory, artifactDirectories['llama-cpp'], 'profiles/cpu-wasm32/browser/core.wasm');
  const original = await readFile(payload);
  await t.test('bytes mismatch', async () => {
    await writeFile(payload, Buffer.alloc(original.length));
    await assert.rejects(validate({ directory: fixture.directory }), /payload bytes/u);
    await writeFile(payload, original);
  });
  await t.test('unlisted', async () => {
    const extra = join(fixture.directory, 'extra.txt');
    await writeFile(extra, 'extra');
    await assert.rejects(validate({ directory: fixture.directory }), /closure/u);
    await rm(extra);
  });
  await t.test('linked', async () => {
    await rm(payload);
    await symlink(join(fixture.root, 'missing'), payload);
    await assert.rejects(validate({ directory: fixture.directory }), /Linked/u);
    await rm(payload);
    await writeFile(payload, original);
  });
  await t.test('mixed source', async () => {
    await rehashRuntime({
      directory: fixture.directory,
      runtime: 'llama-cpp',
      change: manifest => {
        manifest.sourceCommit = 'c'.repeat(40);
        for (const profile of Object.values(manifest.profiles)) {
          const variants: Record<string, { sourceCommit: string }> = profile.variants;
          for (const value of Object.values(variants)) {
            value.sourceCommit = manifest.sourceCommit;
          }
        }
      },
    });
    await assert.rejects(validate({ directory: fixture.directory }), /Mixed source/u);
  });
});

test('rehashing a package cannot conceal changed ABI, declaration, patch or license assertions', async t => {
  for (const mutation of ['types', 'patches', 'license', 'wasm'] as const) {
    await t.test(mutation, async () => {
      await using fixture = await packageFixture();
      const runtime = mutation === 'patches' ? 'stable-diffusion-cpp' : 'llama-cpp';
      const root = join(fixture.directory, artifactDirectories[runtime]);
      if (mutation === 'types') {
        await writeFile(join(root, 'api/functions.d.ts'), 'export {};\n');
      }
      if (mutation === 'license') {
        await writeFile(join(root, 'native-license-index.json'), '{"components":[]}');
      }
      if (mutation === 'wasm') {
        await writeFile(join(root, 'profiles/cpu-wasm32/browser/core.wasm'), fixtureWasm({ names: ['other_fn'] }));
      }
      await rehashRuntime({
        directory: fixture.directory,
        runtime,
        change: manifest => {
          if (mutation === 'patches' && manifest.runtime === 'stable-diffusion-cpp') {
            for (const profile of Object.values(manifest.profiles)) {
              for (const variant of Object.values(profile.variants)) {
                variant.patches = { ...variant.patches, patches: [] };
              }
            }
          }
        },
      });
      await assert.rejects(validate({ directory: fixture.directory }), /declarations|patch provenance|license index|Wasm export/u);
    });
  }
});

test('Wasm export inspection rejects truncation and preserves the actual symbol name', async () => {
  await using temporary = await temporaryDirectory();
  const path = join(temporary.root, 'fixture.wasm');
  const data = fixtureWasm({ names: ['fixture_fn'] });
  assert.deepEqual(wasmExports({ data }), new Set(['fixture_fn']));
  assert.throws(() => wasmExports({ data: data.subarray(0, 12) }), /Truncated/u);
  await writeFile(path, data);
  assert.deepEqual(await readWasmExports({ path }), new Set(['fixture_fn']));
  await writeFile(path, data.subarray(0, 12));
  await assert.rejects(readWasmExports({ path }), /Truncated/u);
});

test('retention capability depends on the native unsigned32 signature, independently of its C typedef spelling', () => {
  const schema = fixtureAbi({ runtime: 'stable-diffusion-cpp', memory64: false, gpuProfile: false });
  schema.functions[0]!.returnType = 'unsigned int';
  assert.equal(imageManifestCapabilities({ schema }).bf16WeightRetention, true);
  schema.functions[0]!.abiType.bits = 64;
  assert.throws(() => imageManifestCapabilities({ schema }), /public native version API/u);
  schema.functions[0]!.abiType.bits = 32;
  schema.functions[0]!.parameters.push({ name: 'unexpected', cType: 'int', kind: 'signed', abiType: { kind: 'integer', bits: 32, signed: true } });
  assert.throws(() => imageManifestCapabilities({ schema }), /public native version API/u);
});

test('model graph metadata cannot disappear in one target after all byte bindings are rebuilt', async () => {
  await using fixture = await packageFixture();
  await rewriteLlamaTargetMetadata({
    directory: fixture.directory,
    transform: ({ variant, schema }) => {
      if (variant === 'test') {
        delete schema.extensions!['model-graph-v1'];
      }
    },
  });
  await assert.rejects(validate({ directory: fixture.directory }), /Missing compiled model graph boundary/u);
});

test('optional boundary declarations and types require their compiled exports', async () => {
  await using fixture = await packageFixture();
  await rewriteLlamaTargetMetadata({
    directory: fixture.directory,
    transform: ({ schema }) => {
    schema.extensions!['model-graph-v1']!.schemaHashExport = '_absent_native_symbol';
    },
  });
  await assert.rejects(validate({ directory: fixture.directory }), /uncompiled export/u);
});

test('Q8 diagnostic metadata cannot be advertised by a non-diagnostic profile', async () => {
  await using fixture = await packageFixture();
  await rewriteLlamaTargetMetadata({
    directory: fixture.directory,
    transform: ({ schema }) => {
      const base = schema.extensions!['model-graph-v1']!;
    schema.extensions!['webgpu-q8-diagnostic-v1'] = { ...base, extensionId: 'webgpu-q8-diagnostic-v1', suspension: 'jspi', functions: [...base.functions, { ...base.functions[0]!, name: 'fixture_run_v1' }] };
    },
  });
  await assert.rejects(validate({ directory: fixture.directory }), /Q8 diagnostic boundary differs/u);
});

test('unrelated saved metadata is preserved without becoming another payload identity', async () => {
  await using fixture = await packageFixture();
  await rehashRuntime({
    directory: fixture.directory,
    runtime: 'llama-cpp',
    change: manifest => {
      manifest.files[0]!['futureMetadata'] = { note: 'retained' };
      manifest.profiles['cpu-wasm32']!.variants['browser']!.validation['futureMetadata'] = { detail: true };
    },
  });
  await rehashRuntime({
    directory: fixture.directory,
    runtime: 'stable-diffusion-cpp',
    change: manifest => {
      if (manifest.runtime !== 'stable-diffusion-cpp') {
        throw new Error('Wrong test runtime');
      }
      manifest.capabilities['futureMetadata'] = { note: 'retained' };
      manifest.upstreams.ggml['futureMetadata'] = { note: 'retained' };
    },
  });
  const rootPath = join(fixture.directory, 'manifest.json');
  const root = JSON.parse(await readFile(rootPath, 'utf8')) as { files: { futureMetadata?: string }[] };
  root.files[0]!.futureMetadata = 'retained';
  await writeFile(rootPath, JSON.stringify(root));
  const validated = await validate({ directory: fixture.directory });
  assert.deepEqual(validated.runtimes['llama-cpp'].files[0]!['futureMetadata'], { note: 'retained' });
  assert.deepEqual(validated.runtimes['llama-cpp'].profiles['cpu-wasm32']!.variants['browser']!.validation['futureMetadata'], { detail: true });
});

test('embedded notice extraction keeps every attribution and falls back on ambiguous syntax', () => {
  const text = '/* Copyright first author */\n// Additional attribution\nconst char *s = "// not a comment";\n/* Alternative license */';
  const result = embeddedComments({ original: text });
  assert(result.includes('Copyright first author'));
  assert(result.includes('Additional attribution'));
  assert(result.includes('Alternative license'));
  assert(!result.includes('const char'));
  const ambiguous = '/* notice */\nR"broken(unclosed';
  assert.equal(embeddedComments({ original: ambiguous }), ambiguous);
  const outside = '/* header */\nCopyright 2026 author';
  assert.equal(embeddedComments({ original: outside }), outside);
  const spliced = '/\\\n* First attribution */\n// Second \\\r\ncontinued attribution\nint ignored;';
  assert.equal(embeddedComments({ original: spliced }), '/\\\n* First attribution */\n// Second \\\r\ncontinued attribution\n');
});

test('minified Wasm symbols remain bound to the generated public factory without disabling minification', async () => {
  const { assertWasmExports } = await import('#tools/package/wasm-exports');
  const wasmNames = new Set(['a', 'b']);
  assert.doesNotThrow(() => assertWasmExports({ source: "function assign(exportsObject) { renamedModule['_first'] = exportsObject['a']; renamedModule._second = wrap('second', exportsObject.b, 0); }", wasmNames, expected: ['_first', '_second'] }));
  assert.throws(() => assertWasmExports({ source: "Module['_first'] = wasmExports['absent'];", wasmNames, expected: ['_first'] }), /Missing actual Wasm/u);
  assert.throws(() => assertWasmExports({ source: "Module['_first'] = combine(wasmExports['a'],wasmExports['b']);", wasmNames, expected: ['_first'] }), /Ambiguous/u);
  assert.throws(() => assertWasmExports({ source: 'export default () => ({});', wasmNames: new Set(['first']), expected: ['_first'] }), /factory mapping/u);
});


test('inventories reject escaped and duplicate paths, missing payloads and redirected inner bindings', async t => {
  await using fixture = await packageFixture();
  for (const unsafe of ['../escape', '/absolute', 'C:\\absolute', 'a//b', 'a/./b', 'a/../b', 'a\\b', 'C:relative']) {
    await t.test(unsafe, async () => {
      await rehashRuntime({
        directory: fixture.directory,
        runtime: 'llama-cpp',
        change: manifest => {
          manifest.files[0]!.path = unsafe;
        },
      });
      await assert.rejects(validate({ directory: fixture.directory }), /relative source path|Unsafe artifact path/u);
    });
  }
  await t.test('duplicate inner path', async () => {
    await rehashRuntime({
      directory: fixture.directory,
      runtime: 'llama-cpp',
      change: manifest => {
        manifest.files.push(manifest.files[0]!);
      },
    });
    await assert.rejects(validate({ directory: fixture.directory }), /Duplicate/u);
  });
  await rehashRuntime({ directory: fixture.directory, runtime: 'llama-cpp', change: () => {} });
  const rootPath = join(fixture.directory, 'manifest.json');
  const rootRaw = await readFile(rootPath, 'utf8');
  const root = JSON.parse(rootRaw) as { files: { path: string; bytes: number; sha256: string }[]; runtimes: { 'llama-cpp': { manifest: string } } };
  await t.test('duplicate root path', async () => {
    root.files.push(root.files[0]!);
    await writeFile(rootPath, JSON.stringify(root));
    await assert.rejects(validate({ directory: fixture.directory }), /Duplicate/u);
    root.files.pop();
  });
  await t.test('redirected inner manifest', async () => {
    root.runtimes['llama-cpp'].manifest = 'manifest.json';
    await writeFile(rootPath, JSON.stringify(root));
    await assert.rejects(validate({ directory: fixture.directory }), /Wrong runtime manifest binding/u);
  });
  await writeFile(rootPath, rootRaw);
  await t.test('missing payload', async () => {
    await rm(join(fixture.directory, artifactDirectories['llama-cpp'], 'profiles/cpu-wasm32/browser/core.wasm'));
    await assert.rejects(validate({ directory: fixture.directory }), /payload bytes/u);
  });
});
