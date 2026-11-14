import assert from 'node:assert/strict';
import test from 'node:test';
import * as shared from '#dto/shared';
import * as llama from '#dto/llama-cpp';
import * as image from '#dto/stable-diffusion-cpp';
import type { AbiRecord, AbiSchema, AbiType, BrowserSession, CompilerInputs, FileIdentity, FrozenPublication, ReportData, RootManifest, SourceFacts } from '#dto/shared';
import type { LlamaBrowserResult, LlamaManifest, LlamaProvenance, PreparationReceipt } from '#dto/llama-cpp';
import type { StableDiffusionBrowserResult, StableDiffusionManifest, StableDiffusionProvenance, Upstreams } from '#dto/stable-diffusion-cpp';

const identity: FileIdentity = { bytes: 12, sha256: 'observed-content-digest' };
const sourceInputs = { mode: 'release' as const, publishable: true, sourceCommit: 'observed-source', files: { 'owned.cpp': identity } };
const compiledInputs: CompilerInputs = { translationUnits: { 'source/owned.cpp': identity }, targetSources: { core: { 'source/owned.cpp': identity } } };
const prepared: PreparationReceipt = {
  formatVersion: 1,
  options: { target: 'wasm', webgpu: false, gpuProfile: false, fileRows: false, qsa: true, mtmdMode: 'cpu' },
  upstream: { files: 1, sha256: 'upstream-tree' },
  inputs: { 'unit.cpp': identity },
  runtimeInputs: { 'owned.cpp': identity },
  patches: [{ file: 'change.patch', target: 'llama', reason: 'Observed change', mode: 'apply', sha256: 'patch-digest', files: [{ path: 'new.cpp', afterSha256: 'new-digest' }] }],
  selections: [],
  shaders: {},
  outputs: { 'unit.cpp': identity },
};
const llamaProvenance: LlamaProvenance = {
  profile: 'cpu',
  variant: 'browser',
  sourceCommit: 'observed-source',
  sourceDirty: false,
  configuration: { memory64: false },
  variantConfiguration: { browser: true },
  toolchain: { emsdkVersion: 'configured-version' },
  llamaCommit: 'observed-upstream',
  emccVersion: 'observed-compiler',
  nodeVersion: 'observed-node',
  sourceInputs,
  upstreamBuildInfo: { commit: 'upstream-short-id', number: 1 },
  preparedSources: prepared,
  compiledInputs,
  cmakeCommand: ['cmake', '-S', '/compile-inputs'],
  buildCache: {},
  visionBf16Storage: { version: 1, modes: ['legacy_expand', 'retain_checked'], default: 'legacy_expand' },
  browserModulePreparation: { input: identity, output: identity, alreadyAnnotated: false },
  validation: { compiled: true },
};
const llamaManifest: LlamaManifest = {
  formatVersion: 2,
  runtime: 'llama-cpp',
  abiVersion: 3,
  sourceCommit: 'observed-source',
  schemaSha256: 'abi-digest',
  llamaCommit: 'observed-upstream',
  licenseRoots: ['emscripten'],
  files: [{ path: 'profiles/cpu/browser/core.mjs', ...identity }],
  profiles: { cpu: { variants: { browser: llamaProvenance } } },
};
const upstreams: Upstreams = {
  'stable-diffusion': { path: 'vendor/image', repository: 'owner/image', commit: 'observed-image-upstream' },
  ggml: { path: 'vendor/math', repository: 'owner/math', sourceSubdirectory: 'ggml', commit: 'observed-math-upstream' },
};
const imageProvenance: StableDiffusionProvenance = {
  runtime: 'stable-diffusion-cpp',
  profile: 'cpu',
  variant: 'browser',
  sourceCommit: 'observed-source',
  sourceDirty: false,
  upstreams,
  configuration: { memory64: false },
  variantConfiguration: { browser: true },
  cmakeCommand: ['cmake'],
  patches: { runtime: 'stable-diffusion-cpp', upstreams, patches: [], vendorCheckoutModified: false },
  sourceInputs,
  preparedSources: { files: 2, sha256: 'prepared-tree' },
  compiledInputs,
  nativeCheck: false,
  toolchain: { emsdkVersion: 'configured-version' },
  buildCache: {},
  validation: { compiled: true },
};
const imageManifest: StableDiffusionManifest = {
  formatVersion: 2,
  runtime: 'stable-diffusion-cpp',
  abiVersion: 3,
  sourceCommit: 'observed-source',
  schemaSha256: 'abi-digest',
  upstreams,
  experimental: true,
  capabilities: { ggufFileOffsetBits: 64, safetensorsFileOffsetBits: 64, callerOwnedRandomAccess: true, upstreamApi: true, ggufShards: true, ggufOriginalShapes: true, bf16WeightRetention: true },
  files: [{ path: 'profiles/cpu/browser/core.mjs', ...identity }],
  profiles: { cpu: { variants: { browser: imageProvenance } } },
};
const rootManifest: RootManifest = {
  formatVersion: 5,
  sourceCommit: 'observed-source',
  runtimes: { 'llama-cpp': { manifest: 'runtimes/llama/manifest.json', manifestFormatVersion: 2 }, 'stable-diffusion-cpp': { manifest: 'runtimes/image/manifest.json', manifestFormatVersion: 2 } },
  files: [{ path: 'README.md', ...identity }],
};
const abi: AbiSchema = {
  abiVersion: 3,
  runtime: 'llama-cpp',
  pointerRepresentation: 'bigint_native',
  pointerBytes: 4,
  schemaSha256: 'abi-digest',
  functions: [],
  records: [],
  callbacks: [],
  constants: [],
  constantValues: {},
  excluded: [],
  exports: [],
  suspendingExports: [],
  callbackMetadata: { version: 2, getters: [] },
};
const browserSession: BrowserSession = {
  schemaVersion: 1,
  runtime: 'llama-cpp',
  sourceCommit: 'observed-source',
  runId: 'local',
  runAttempt: '1',
  job: 'local',
  sessionId: 'fresh-session',
  snapshotSha256: 'tested-snapshot',
  testInputsSha256: 'tested-implementation',
  testWebGpu: false,
};
const llamaBrowserResult: LlamaBrowserResult = { profile: 'cpu', variant: 'browser', passed: true, syntheticModel: true, sampledToken: 1, fileMountLifetime: { passed: true, largeOffsetReads: 2 } };
const imageBrowserResult: StableDiffusionBrowserResult = {
  profile: 'cpu',
  variant: 'test',
  passed: true,
  scope: 'Synthetic arithmetic evidence',
  graphWalk: true,
  timestep: [{ backend: 'CPU', passed: true, maximumError: 0 }],
  bf16Weights: [{ backend: 'CPU', passed: true }],
  conv3dBias: [{ backend: 'CPU', passed: true }],
  webgpuPerformance: [{ backend: 'CPU', passed: true }],
  reads: { bytes: 128 },
};

test('DTO public entry points expose named decoders without raw schemas', () => {
  for (const entry of [shared, llama, image]) {
    for (const [name, value] of Object.entries(entry)) {
      assert.equal(typeof value, 'function', name);
      assert.ok(name.startsWith('decode'), name);
    }
  }
});

test('complete typed producers pass their structures directly to typed consumers', () => {
  assert.deepEqual(llama.decodeLlamaManifest({ input: llamaManifest }), llamaManifest);
  assert.deepEqual(image.decodeStableDiffusionManifest({ input: imageManifest }), imageManifest);
  assert.deepEqual(shared.decodeRootManifest({ input: rootManifest }), rootManifest);
  const received = llama.decodeLlamaProvenance({ input: llamaProvenance });
  const outputs: Record<string, FileIdentity> = received.preparedSources.outputs;
  assert.deepEqual(outputs, prepared.outputs);
  // A loss of concrete inventory typing must fail the compiler check as well.
  // @ts-expect-error Compiled source sizes are numbers in the canonical DTO.
  const wrong: CompilerInputs = { translationUnits: { unit: { bytes: '12', sha256: 'digest' } }, targetSources: {} };
  void wrong;
});

test('unknown future artifact fields are preserved throughout stored structures', () => {
  const input = {
    ...llamaManifest,
    futureRoot: { enabled: true },
    profiles: {
      cpu: {
        futureProfile: true,
        variants: {
          browser: {
            ...llamaProvenance,
            futureProvenance: true,
            sourceInputs: { ...sourceInputs, futureSource: true, files: { 'owned.cpp': { ...identity, futureIdentity: true } } },
            preparedSources: { ...prepared, futureReceipt: true, options: { ...prepared.options, futureOption: true }, patches: [{ ...prepared.patches[0], futurePatch: true }] },
            compiledInputs: { ...compiledInputs, futureCompiler: true },
            validation: { ...llamaProvenance.validation, futureValidation: true },
          },
        },
      },
    },
  };
  const decoded = llama.decodeLlamaManifest({ input });
  assert.deepEqual(decoded, input);
  const provenance = decoded.profiles['cpu']?.variants['browser'];
  assert.ok(provenance);
  provenance.validation.browserSmoke = true;
  const expected = structuredClone(input);
  expected.profiles.cpu.variants.browser.validation.browserSmoke = true;
  assert.deepEqual(JSON.parse(JSON.stringify(decoded)), expected);
  const imageInput = { ...imageManifest, upstreams: { ...upstreams, ggml: { ...upstreams.ggml, futureIdentity: true } }, futureRoot: true };
  assert.deepEqual(image.decodeStableDiffusionManifest({ input: imageInput }), imageInput);
  const rootInput = { ...rootManifest, futureRoot: true, files: [{ ...rootManifest.files[0], futureIdentity: true }] };
  assert.deepEqual(shared.decodeRootManifest({ input: rootInput }), rootInput);
});

test('runtime decoders reject wrong required nested shapes and runtime identities', () => {
  assert.throws(() => llama.decodeLlamaManifest({ input: { ...llamaManifest, runtime: 'stable-diffusion-cpp' } }));
  assert.throws(() => llama.decodeLlamaProvenance({ input: { ...llamaProvenance, compiledInputs: { translationUnits: {}, targetSources: [] } } }));
  assert.throws(() => llama.decodePreparationReceipt({ input: { ...prepared, outputs: { 'unit.cpp': { bytes: '12', sha256: 'digest' } } } }));
  assert.throws(() => image.decodeStableDiffusionManifest({ input: { ...imageManifest, capabilities: { ...imageManifest.capabilities, callerOwnedRandomAccess: 1 } } }));
  assert.throws(() => image.decodeStableDiffusionProvenance({ input: { ...imageProvenance, patches: { ...imageProvenance.patches, upstreams: { 'stable-diffusion': upstreams['stable-diffusion'] } } } }));
  const preparation: NonNullable<LlamaProvenance['moduleDeclarationPreparation']> = { input: identity, output: identity, correctedBigIntReferences: 2 };
  assert.deepEqual(llama.decodeLlamaProvenance({ input: { ...llamaProvenance, moduleDeclarationPreparation: preparation } }).moduleDeclarationPreparation, preparation);
  assert.throws(() => llama.decodeLlamaProvenance({ input: { ...llamaProvenance, moduleDeclarationPreparation: { ...preparation, correctedBigIntReferences: '2' } } }));
});

test('structural decoding leaves file and publishability decisions at their operations', () => {
  const input = { ...rootManifest, sourceCommit: 'not-a-git-object-id', files: [{ path: '../unsafe-at-use', bytes: -1, sha256: 'not-a-digest' }] };
  assert.deepEqual(shared.decodeRootManifest({ input }), input);
  assert.equal(llama.decodeLlamaProvenance({ input: { ...llamaProvenance, validation: { compiled: false } } }).validation.compiled, false);
  const { visionBf16Storage: _visionBf16Storage, ...legacyProvenance } = llamaProvenance;
  assert.equal(llama.decodeLlamaProvenance({ input: legacyProvenance }).visionBf16Storage, undefined);
});

test('external compiler projections tolerate unrelated fields and type all AST descendants once', () => {
  assert.deepEqual(shared.decodeCompilationDatabase({ input: [{ directory: '/source', file: 'a.cpp', arguments: ['clang++'], futureToolData: true }] }), [{ directory: '/source', file: 'a.cpp', arguments: ['clang++'] }]);
  assert.throws(() => shared.decodeCompilationDatabase({ input: [{ directory: '/source', file: 1 }] }));
  const root = shared.decodeClangAst({ input: { kind: 'TranslationUnitDecl', futureCompilerData: true, inner: [{ kind: 'RecordDecl', inner: [{ kind: 'FieldDecl', name: 'field', type: { qualType: 'int', futureTypeData: true } }] }] } });
  assert.equal(root.inner?.[0]?.inner?.[0]?.type?.qualType, 'int');
  const cppRecord = shared.decodeClangAst({ input: { kind: 'CXXRecordDecl', definitionData: { isStandardLayout: false, isTriviallyCopyable: false, futureLayoutData: true }, bases: [{ type: { qualType: 'Base' }, access: 'public', isVirtual: true }], range: { begin: { offset: 12, col: 3, spellingLoc: { file: 'external-macro.h' } }, end: { offset: 40, col: 31 } } } });
  assert.equal(cppRecord.definitionData?.isStandardLayout, false);
  assert.equal(cppRecord.definitionData?.isTriviallyCopyable, false);
  assert.equal(shared.decodeClangAst({ input: { kind: 'CXXRecordDecl', definitionData: { isStandardLayout: true } } }).definitionData?.isTriviallyCopyable, undefined);
  assert.equal(cppRecord.bases?.[0]?.isVirtual, true);
  assert.equal(cppRecord.range?.begin.offset, 12);
  const staticFunction = shared.decodeClangAst({ input: { kind: 'FunctionDecl', name: 'public_prefix_helper', storageClass: 'static' } });
  assert.equal(staticFunction.storageClass, 'static');
  assert.equal(shared.decodeClangAst({ input: { kind: 'FunctionDecl', storageClass: 'future_compiler_storage' } }).storageClass, 'future_compiler_storage');
  assert.throws(() => shared.decodeClangAst({ input: { kind: 'TranslationUnitDecl', inner: [{ kind: 'RecordDecl', inner: [{ kind: 1 }] }] } }));
});

test('ABI artifact decoding keeps required structure and permits future attributes', () => {
  assert.deepEqual(shared.decodeAbiSchema({ input: { ...abi, futureAbiData: true, callbackMetadata: { ...abi.callbackMetadata, futureCallbackData: true } } }), abi);
  assert.deepEqual(shared.decodeAbiCatalog({ input: { ...abi, targets: { target: { ...abi, futureTargetData: true } }, futureCatalogData: true } }), { ...abi, targets: { target: abi } });
  assert.throws(() => shared.decodeAbiSchema({ input: { ...abi, exports: [1] } }));
  const probe = { name: 'sdc_test_probe', export: '_sdc_test_probe', returnType: 'int', returnKind: 'signed', abiType: { kind: 'integer', bits: 32, signed: true }, parameters: [], bridge: false };
  assert.deepEqual(shared.decodeAbiSchema({ input: { ...abi, testFunctions: [probe] } }).testFunctions, [probe]);
});

test('ABI array elements retain recursive type facts while legacy arrays remain readable', () => {
  const matrix: AbiType = { kind: 'array', dimensions: [2], element: { kind: 'array', abiType: { kind: 'array', dimensions: [3], element: { kind: 'signed', abiType: { kind: 'integer', bits: 32, signed: true } } } } };
  const record: AbiRecord = { name: 'Matrix', cType: 'struct Matrix', id: 1, size: 24, alignment: 4, fields: [{ id: 0, name: 'values', cType: 'int[2][3]', kind: 'array', abiType: matrix, offset: 0, size: 24 }] };
  const input: AbiSchema = { ...abi, records: [record] };
  const decoded = shared.decodeAbiSchema({ input });
  assert.deepEqual(decoded, input);
  const bits: number | undefined = decoded.records[0]?.fields[0]?.abiType.element?.abiType.element?.abiType.bits;
  assert.equal(bits, 32);
  const legacyArray = { kind: 'array', dimensions: [2, 3], bits: 32, signed: true };
  const legacy = shared.decodeAbiSchema({ input: { ...abi, records: [{ ...record, fields: [{ ...record.fields[0], abiType: legacyArray }] }] } });
  assert.equal(legacy.records[0]?.fields[0]?.abiType.element, undefined);
  assert.throws(() => shared.decodeAbiSchema({ input: { ...abi, records: [{ ...record, fields: [{ ...record.fields[0], abiType: { ...matrix, element: { kind: 'signed', abiType: 'integer' } } }] }] } }));
});

test('owned symbolic wire values use snake_case while runtime route identities retain their spelling', () => {
  assert.equal(shared.decodeAbiSchema({ input: abi }).pointerRepresentation, 'bigint_native');
  assert.throws(() => shared.decodeAbiSchema({ input: { ...abi, pointerRepresentation: 'bigint-native' } }));
  const selection = { file: 'src/llama-context.cpp', mode: 'upstream_native_context', reason: 'Observed native context selection', before: identity, after: identity };
  assert.equal(llama.decodePreparationReceipt({ input: { ...prepared, selections: [selection] } }).selections[0]?.mode, 'upstream_native_context');
  assert.throws(() => llama.decodePreparationReceipt({ input: { ...prepared, selections: [{ ...selection, mode: 'upstream-native-context' }] } }));
  assert.equal(llama.decodeLlamaManifest({ input: llamaManifest }).runtime, 'llama-cpp');
  assert.equal(image.decodeStableDiffusionManifest({ input: imageManifest }).runtime, 'stable-diffusion-cpp');
});

test('Clang projection omits implementation branches while preserving linkage, enums and declaration types', () => {
  const declaration = {
    kind: 'LinkageSpecDecl',
    language: 'C',
    inner: [
      { kind: 'FunctionDecl', name: 'public_function', type: { qualType: 'int (int)' }, inner: [{ kind: 'ParmVarDecl', name: 'value', type: { qualType: 'int' } }, { kind: 'CompoundStmt', inner: [{ kind: 42 }] }, { kind: 'DeprecatedAttr', inner: [{}] }] },
      { kind: 'EnumDecl', name: 'PublicValues', inner: [{ kind: 'EnumConstantDecl', name: 'PUBLIC_VALUE', inner: [{ kind: 'ConstantExpr', value: '9223372036854775807' }] }] },
    ],
  };
  const root = shared.decodeClangAst({
    input: {
      kind: 'TranslationUnitDecl',
      inner: [
        { kind: 'NamespaceDecl', name: 'std', inner: [{ kind: 42 }] },
        { kind: 'FunctionTemplateDecl', inner: [{ kind: 42 }] },
        { kind: 'ClassTemplateDecl', inner: [{ kind: 42 }] },
        declaration,
      ],
    },
  });
  assert.equal(root.inner?.length, 1);
  const linkage = root.inner?.[0];
  assert.equal(linkage?.language, 'C');
  assert.deepEqual(linkage?.inner?.[0]?.inner?.map(({ kind }) => kind), ['ParmVarDecl', 'DeprecatedAttr']);
  assert.equal(linkage?.inner?.[0]?.inner?.[0]?.type?.qualType, 'int');
  assert.equal(linkage?.inner?.[1]?.inner?.[0]?.inner?.[0]?.value, '9223372036854775807');
  assert.throws(() => shared.decodeClangAst({ input: { kind: 'TranslationUnitDecl', inner: [{ kind: 'LinkageSpecDecl', inner: [{ kind: 'FunctionDecl', inner: [{ kind: 'ParmVarDecl', type: 42 }] }] }] } }));
});

test('browser envelopes type admission evidence while retaining unrelated harness statistics', () => {
  const llamaEnvelope = { session: { ...browserSession, runtime: 'llama-cpp' }, results: [llamaBrowserResult] };
  const imageEnvelope = { session: { ...browserSession, runtime: 'stable-diffusion-cpp' }, results: [imageBrowserResult] };
  const candidate = { ...llamaEnvelope, futureEnvelopeData: true };
  assert.deepEqual(llama.decodeLlamaBrowserEnvelope({ input: candidate }), candidate);
  assert.deepEqual(image.decodeStableDiffusionBrowserEnvelope({ input: imageEnvelope }), imageEnvelope);
  assert.throws(() => llama.decodeLlamaBrowserEnvelope({ input: imageEnvelope }));
  assert.throws(() => image.decodeStableDiffusionBrowserEnvelope({ input: { ...imageEnvelope, results: [{ ...imageBrowserResult, graphWalk: 1 }] } }));
  assert.throws(() => llama.decodeLlamaBrowserEnvelope({ input: { ...llamaEnvelope, results: [{ ...llamaBrowserResult, syntheticModel: 'true' }] } }));
  assert.equal(llama.decodeLlamaBrowserEnvelope({ input: { ...llamaEnvelope, results: [{ ...llamaBrowserResult, passed: false }] } }).results[0]?.passed, false);
});

test('runtime provenance retains checked browser evidence without widening it to generic JSON', () => {
  const llamaValidation = { ...llamaProvenance.validation, browserSmoke: true, syntheticUntrainedModel: llamaBrowserResult };
  assert.deepEqual(llama.decodeLlamaProvenance({ input: { ...llamaProvenance, validation: llamaValidation } }).validation, llamaValidation);
  const evidence = {
    graphWalk: imageBrowserResult.graphWalk,
    timestep: imageBrowserResult.timestep,
    bf16Weights: imageBrowserResult.bf16Weights,
    conv3dBias: imageBrowserResult.conv3dBias,
    webgpuPerformance: imageBrowserResult.webgpuPerformance,
  };
  const imageValidation = { ...imageProvenance.validation, browserSmoke: true, browserSmokeScope: imageBrowserResult.scope, browserSmokeEvidence: evidence };
  const received = image.decodeStableDiffusionProvenance({ input: { ...imageProvenance, validation: imageValidation } });
  assert.deepEqual(received.validation, imageValidation);
  const backend: 'CPU' | 'WebGPU' | undefined = received.validation.browserSmokeEvidence?.timestep[0]?.backend;
  assert.equal(backend, 'CPU');
});

test('npm and license decoders retain the projections consumed by operations', () => {
  const lock = { lockfileVersion: 3, packages: { '': { dependencies: { core: 'github:owner/repo#commit' } }, 'node_modules/core': { version: '1.0', resolved: 'git+https://github.com/owner/repo#commit', hasInstallScript: true, optionalDependencies: { native: '*' } } } };
  assert.deepEqual(shared.decodeNpmLock({ input: { ...lock, futureLockData: true } }), lock);
  const npmEntry = { ...lock.packages['node_modules/core'], integrity: 'npm-owned-integrity', futureReportData: { field: true } };
  assert.deepEqual(shared.decodeNpmLock({ input: { ...lock, packages: { ...lock.packages, 'node_modules/core': npmEntry } } }).packages['node_modules/core'], npmEntry);
  assert.throws(() => shared.decodeNpmLock({ input: { ...lock, packages: { '': { dependencies: [] } } } }));
  assert.deepEqual(shared.decodeNpmPack({ input: [{ files: [{ path: 'core.wasm', size: 12, futureFileData: true }], futurePackData: true }] }), [{ files: [{ path: 'core.wasm', size: 12 }] }]);
  const license = { components: [{ id: 'component', name: 'Component', version: null, license: null, notices: [{ path: 'licenses/NOTICE' }] }] };
  assert.deepEqual(shared.decodeLicenseIndex({ input: { ...license, futureLicenseData: true } }), license);
  assert.throws(() => shared.decodeLicenseIndex({ input: { components: [{ ...license.components[0], notices: 'NOTICE' }] } }));
});

test('package metadata preserves every field used to reject install-time behavior', () => {
  const metadata = {
    name: 'runtime',
    version: '1.0',
    type: 'module',
    exports: { './api': './api.mjs' },
    scripts: { postinstall: 'unexpected-command' },
    dependencies: { dependency: '*' },
    devDependencies: {},
    optionalDependencies: {},
    workspaces: [],
    peerDependencies: {},
    peerDependenciesMeta: {},
    bundleDependencies: [],
    bundledDependencies: [],
  };
  assert.deepEqual(shared.decodePackageMetadata({ input: { ...metadata, futureMetadata: true } }), metadata);
});

test('private frozen input decodes source and report structures without a producer roundtrip', () => {
  const source: SourceFacts = {
    sourceCommit: 'observed-source',
    runtimes: {
      'llama-cpp': { upstream: 'observed-upstream', preparations: { cpu: { upstream: prepared.upstream, options: prepared.options, patches: prepared.patches.map(({ file, target, mode, sha256, reason }) => ({ file, target, mode, sha256, reason })), prepared: { files: 1, sha256: 'prepared-tree' } } }, verifiedVariants: ['cpu/browser'] },
      'stable-diffusion-cpp': { upstreams, preparation: { files: 2, sha256: 'prepared-tree' }, patches: [], verifiedVariants: ['cpu/browser'] },
    },
  };
  const data: ReportData = {
    sourceCommit: 'observed-source',
    package: { name: 'browser-inference-core', version: '1.0' },
    source,
    runtimes: {
      'llama-cpp': { manifest: 'llama/manifest.json', abiVersion: 3, schemaSha256: 'abi-digest', upstreams: { llama: 'observed-upstream' }, profiles: { cpu: { variants: { browser: { configuration: llamaProvenance.configuration, variantConfiguration: llamaProvenance.variantConfiguration, validation: { compiled: true, syntheticUntrainedModel: llamaBrowserResult } } } } } },
      'stable-diffusion-cpp': { manifest: 'image/manifest.json', abiVersion: 3, schemaSha256: 'abi-digest', upstreams, profiles: {} },
    },
  };
  const frozen: FrozenPublication = { schemaVersion: 2, sourceCommit: 'observed-source', tree: 'observed-tree', branch: 'artifacts', manifest: identity, runtimeManifests: { 'llama-cpp': identity, 'stable-diffusion-cpp': identity }, report: { sha256: 'report-digest', data } };
  assert.deepEqual(shared.decodeFrozenPublication({ input: frozen }), frozen);
  const future = { ...frozen, futureState: true, report: { ...frozen.report, data: { ...data, futureReport: { enabled: true } } } };
  assert.deepEqual(shared.decodeFrozenPublication({ input: future }), future);
  assert.throws(() => shared.decodeFrozenPublication({ input: { ...frozen, runtimeManifests: { 'llama-cpp': identity } } }));
  assert.throws(() => shared.decodeFrozenPublication({ input: { ...frozen, report: { ...frozen.report, data: { ...data, source: { ...source, sourceCommit: 1 } } } } }));
  assert.equal(shared.decodeFrozenPublication({ input: { ...frozen, branch: 'feature/forbidden-at-publication' } }).branch, 'feature/forbidden-at-publication');
});

test('GitHub automation decodes only its external projections and retains API null conventions', () => {
  const release = { draft: false, prerelease: false, published_at: null, tag_name: 'candidate-tag', html_url: 'external-release-url' };
  assert.deepEqual(llama.decodeGitHubRelease({ input: { ...release, futureApiData: true } }), release);
  assert.deepEqual(llama.decodeGitHubReleases({ input: [release] }), [release]);
  assert.deepEqual(llama.decodeGitHubCompare({ input: { status: 'external-status', futureApiData: true } }), { status: 'external-status' });
  const pull = { number: -1, state: 'open', head: { sha: 'not-a-source-id', ref: 'head-branch', repo: null } };
  assert.deepEqual(llama.decodeGitHubPullRequest({ input: pull }), pull);
  assert.deepEqual(llama.decodeGitHubPullRequests({ input: [pull] }), [pull]);
  const run = { id: 1, run_attempt: 1, head_sha: 'source-id', head_branch: null, head_repository: null, event: 'push', status: 'completed', conclusion: null, html_url: 'external-run-url', run_started_at: null };
  assert.deepEqual(llama.decodeGitHubWorkflowRuns({ input: [run] }), [run]);
  assert.deepEqual(llama.decodeGitHubArtifacts({ input: [{ id: 1, name: 'report', expired: false, futureApiData: true }] }), [{ id: 1, name: 'report', expired: false }]);
  assert.deepEqual(llama.decodeGitHubIssueComments({ input: [{ id: 1, user: { login: 'bot', type: 'Bot' }, body: null }] }), [{ id: 1, user: { login: 'bot', type: 'Bot' }, body: null }]);
  assert.throws(() => llama.decodeGitHubPullRequest({ input: { ...pull, head: { ...pull.head, sha: 1 } } }));
  assert.throws(() => llama.decodeGitHubArtifacts({ input: [{ id: 1, name: 'report', expired: 'false' }] }));
  const report = { schemaVersion: 1, repository: 'owner/repository', sourceCommit: 'source-id', artifactCommit: 'artifact-id', runId: 1, runAttempt: 1, markdownSha256: 'markdown-digest' };
  assert.deepEqual(llama.decodeConsumerReportEnvelope({ input: { ...report, futureReportData: true } }), report);
  assert.throws(() => llama.decodeConsumerReportEnvelope({ input: { ...report, runAttempt: '1' } }));
});
