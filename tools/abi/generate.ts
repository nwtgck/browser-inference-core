import { copyFile, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import type { AbiExtension, AbiFunction, AbiSchema, RuntimeName } from '#dto/shared';
import { replaceFileUnchanged, withOutputDirectories } from '#tools/shared/output';
import { isContained } from '#tools/shared/files';
import { childNodes, compilerMacros, inspectAst, inspectLayouts, readCompilation, type AstEntry, type AstNode, type AstScope } from './clang.js';
import { Declarations } from './declarations.js';
import { abiSchemaFingerprint, canonicalJson, jsType, loweredReturn, renderFunctionModule, renderFunctionTypes, renderSchemaModule } from './schema.js';

export type BindingPolicy = {
  runtime: RuntimeName;
  astScope: AstScope | undefined;
  publicFunction: ({ entry, translationUnit }: { entry: AstEntry; translationUnit: string }) => boolean;
  exclusions: Readonly<Record<string, string>>;
  acceptedDeprecated: readonly string[];
  synchronous: readonly string[];
};

function returnPublicFunction({ declarations, node }: { declarations: Declarations; node: AstNode }): AbiFunction {
  const direct = declarations.function({ node, bridge: false });
  const bridge = direct.returnKind === 'record' || direct.parameters.some((parameter) => parameter.kind === 'record');
  return { ...direct, bridge, export: '_' + (bridge ? 'bicore_' : '') + direct.name };
}

function validateBridge({ fn, actual, declarations }: { fn: AbiFunction; actual: AstNode | undefined; declarations: Declarations }): void {
  if (!actual) {
    throw new Error(`Public aggregate function ${fn.name} needs an ordinary native bridge; C++ bodies are not generated`);
  }
  const bridge = declarations.function({ node: actual, bridge: false });
  const result = loweredReturn({ kind: fn.returnKind });
  const expectedCount = fn.parameters.length + (result.output ? 1 : 0);
  if (bridge.parameters.length !== expectedCount || (result.output ? bridge.returnKind !== 'void' : canonicalJson({ value: bridge.abiType }) !== canonicalJson({ value: fn.abiType }))) {
    throw new Error(`Native bridge signature differs: ${fn.name}`);
  }
  if (result.output && bridge.parameters[0]?.kind !== 'pointer') {
    throw new Error(`Native record return storage is not a pointer: ${fn.name}`);
  }
  for (const [index, parameter] of fn.parameters.entries()) {
    const actualParameter = bridge.parameters[index + (result.output ? 1 : 0)];
    if (!actualParameter) {
      throw new Error(`Native bridge parameter differs: ${fn.name}[${index}]`);
    }
    switch (parameter.kind) {
    case 'record':
      switch (actualParameter.kind) {
      case 'pointer': break;
      default: throw new Error(`Native bridge parameter differs: ${fn.name}[${index}]`);
      }
      break;
    default:
      if (canonicalJson({ value: actualParameter.abiType }) !== canonicalJson({ value: parameter.abiType })) {
        throw new Error(`Native bridge parameter differs: ${fn.name}[${index}]`);
      }
    }
  }
}

function extension({ id, prefix, recordPrefix, declarations, entries }: { id: string; prefix: string; recordPrefix: string | undefined; declarations: Declarations; entries: readonly AstEntry[] }): AbiExtension | undefined {
  const functions = entries.filter(({ node, namespaces }) => !namespaces.length && node.kind === 'FunctionDecl' && node.name?.startsWith(prefix)).map(({ node }) => declarations.function({ node, bridge: false }));
  if (!functions.length) {
    return undefined;
  }
  const unique = [...new Map(functions.map((fn) => [fn.name, fn])).values()].sort((left, right) => left.name.localeCompare(right.name));
  const records = recordPrefix ? declarations.records().filter((record) => record.name.startsWith(recordPrefix)) : [];
  const constantValues = Object.fromEntries(Object.entries(declarations.constantValues).filter(([name]) => recordPrefix && name.startsWith(recordPrefix.toUpperCase())));
  return {
    extensionId: id,
    version: 1,
    abiVersion: 1,
    baseAbiVersion: 3,
    pointerRepresentation: 'bigint_u64',
    suspension: id === 'model-graph-v1' ? 'none' : 'jspi',
    versionExport: '_' + prefix + 'version',
    schemaHashExport: '_' + prefix + 'schema_hash',
    functions: unique,
    records,
    constants: Object.keys(constantValues).sort(),
    constantValues,
  };
}

export function renderExtensionTypes({ extension: value }: { extension: AbiExtension }): string {
  if (value.extensionId === 'webgpu-q8-diagnostic-v1') {
    const version = value.functions.find((fn) => fn.name.endsWith('_version'));
    const run = value.functions.find((fn) => fn.name.endsWith('_run_v1'));
    if (!version || !run) {
      throw new Error('Incomplete diagnostic declaration');
    }
    const type = ({ parameter }: { parameter: AbiFunction['parameters'][number] }) => ['pointer', 'record', 'u64', 'i64'].includes(parameter.kind) ? 'bigint' : 'number';
    return ['/** Derived from the compiled optional diagnostic declaration. */', 'export interface Q8DiagnosticFunctions {', `  version(${version.parameters.map((parameter) => `${parameter.name}: ${type({ parameter })}`).join(', ')}): number;`, `  runV1(${run.parameters.map((parameter) => `${parameter.name}: ${type({ parameter })}`).join(', ')}): Promise<number>;`, '}', ''].join('\n');
  }
  const lines = ['/** Synchronous native extension, derived from its real declarations. */', 'export interface ModelGraphNativeExports {'];
  for (const fn of value.functions) {
    const result = jsType({ kind: fn.returnKind });
    lines.push(`  ${fn.export}(${fn.parameters.map((parameter) => `${parameter.name}: ${['pointer', 'record', 'u64', 'i64'].includes(parameter.kind) ? 'bigint' : 'number'}`).join(', ')}): ${result};`);
  }
  lines.push('}', '');
  return lines.join('\n');
}

export function renderModuleTypes({ schema }: { schema: AbiSchema }): string {
  const rawType = ({ fn, parameter }: { fn: AbiFunction; parameter: AbiFunction['parameters'][number] | undefined }): string => {
    const kind = parameter?.kind ?? fn.returnKind;
    const native = parameter?.abiType ?? fn.abiType;
    if (kind === 'void' || !parameter && kind === 'record') {
      return 'void';
    }
    if (kind === 'record' || kind === 'pointer') {
      return schema.pointerBytes === 8 ? 'bigint' : 'number';
    }
    return native.kind === 'integer' && native.bits === 64 ? 'bigint' : 'number';
  };
  const lines = ['import type { CoreOptions } from "./module-options.js";', 'export type { CoreOptions } from "./module-options.js";', '/** Raw target exports. Use the host caller for suspension and range checks. */', 'export interface StableDiffusionModule {', '  HEAPU8: Uint8Array<ArrayBuffer>;', '  FS: {', '    mkdir(path: string): unknown;', '  };', '  addFunction(callback: (...args: (number | bigint)[]) => number | bigint | void, signature: string): number | bigint;', '  removeFunction(pointer: number | bigint): void;', '  ccall(name: string, result: "number" | "bigint" | undefined, types: readonly ("number" | "bigint")[], args: readonly (number | bigint)[], options: { async: true }): unknown;', '  _bicore_abi_version(): number;', '  _bicore_pointer_bytes(): number;', `  _bicore_schema_hash(): ${schema.pointerBytes === 8 ? 'bigint' : 'number'};`, '  _bicore_model_io_capabilities(): number;', '  _malloc(size: number): number;', '  _free(pointer: number): void;'];
  for (const fn of [...schema.functions, ...schema.testFunctions ?? []]) {
    const args = fn.parameters.map((parameter, index) => `arg${index}: ${rawType({ fn, parameter })}`);
    if (loweredReturn({ kind: fn.returnKind }).output) {
      args.unshift(`out: ${schema.pointerBytes === 8 ? 'bigint' : 'number'}`);
    }
    const result = rawType({ fn, parameter: undefined });
    lines.push(`  ${fn.export}(${args.join(', ')}): ${result}${schema.suspendingExports.includes(fn.export.slice(1)) ? ` | Promise<${result}>` : ''};`);
  }
  lines.push('}', 'export default function createModule(options?: CoreOptions): Promise<StableDiffusionModule>;', '');
  return lines.join('\n');
}

export async function generateBindings({ policy, compileRoot, compilationDatabase, translationUnit, bridgeUnit, output, runtimeDirectory }: {
  policy: BindingPolicy; compileRoot: string; compilationDatabase: string; translationUnit: string; bridgeUnit: string; output: string; runtimeDirectory: string | undefined;
}): Promise<AbiSchema> {
  const expected = path.join(compileRoot, 'runtimes', policy.runtime, 'native/abi');
  if (!isContained({ root: compileRoot, path: translationUnit }) || path.resolve(translationUnit) !== path.resolve(expected, 'metadata.cpp') || path.resolve(bridgeUnit) !== path.resolve(expected, 'bridge.cpp')) {
    throw new Error('ABI translation units must belong to this compile snapshot');
  }
  const compilation = await readCompilation({ database: compilationDatabase, translationUnit });
  const bridgeCompilation = await readCompilation({ database: compilationDatabase, translationUnit: bridgeUnit });
  const entries = await inspectAst({ compilation, scope: policy.astScope });
  const bridgeEntries = await inspectAst({ compilation: bridgeCompilation, scope: { filters: ['bicore_'], prefixes: ['bicore_'], aliases: [] } });
  const [layouts, macros] = await Promise.all([inspectLayouts({ compilation }), compilerMacros({ compilation })]);
  const declarations = new Declarations({ entries, layouts, macros, runtime: policy.runtime, translationUnit });
  const bridgeFunctions = new Map(bridgeEntries.filter(({ node }) => node.kind === 'FunctionDecl' && node.name?.startsWith('bicore_')).map(({ node }) => [node.name, node]));
  const functions = new Map<string, AbiFunction>();
  const excluded = new Map<string, string>();
  for (const [name, reason] of declarations.opaqueRecords) {
    excluded.set(name, reason);
  }
  for (const entry of entries) {
    const { node } = entry;
    if (!policy.publicFunction({ entry, translationUnit }) || node.kind !== 'FunctionDecl' || !node.name) {
      continue;
    }
    const reason = policy.exclusions[node.name]
      ?? (node.storageClass === 'static' ? 'header-local functions have no external C export' : undefined)
      ?? (node.variadic ? 'variadic C APIs require a separately reviewed ordinary interface' : undefined)
      ?? (childNodes({ node }).some((child) => child.kind === 'DeprecatedAttr') && !policy.acceptedDeprecated.includes(node.name) ? 'deprecated upstream declaration' : undefined);
    if (reason) {
      excluded.set(node.name, reason);
      continue;
    }
    const fn = returnPublicFunction({ declarations, node });
    if (fn.bridge) {
      validateBridge({ fn, actual: bridgeFunctions.get(`bicore_${fn.name}`), declarations });
    }
    functions.set(fn.name, fn);
  }
  if (!functions.size) {
    throw new Error('No public functions found in the actual target header set');
  }
  const extensions: Record<string, AbiExtension> = {};
  const helpers = ['_bicore_abi_version', '_bicore_schema_hash', '_bicore_pointer_bytes', '_malloc', '_free'];
  let moduleTypes: string | undefined;
  switch (policy.runtime) {
  case 'llama-cpp':
    for (const value of [
      extension({ id: 'model-graph-v1', prefix: 'lcb_model_graph_', recordPrefix: undefined, declarations, entries }),
      extension({ id: 'webgpu-q8-diagnostic-v1', prefix: 'lcb_webgpu_q8_diagnostic_', recordPrefix: 'ggml_webgpu_q8_diagnostic_', declarations, entries }),
    ]) {
      if (value) {
        extensions[value.extensionId] = value;
      }
    }
    break;
  case 'stable-diffusion-cpp':
    helpers.push('_bicore_model_io_capabilities');
    moduleTypes = 'core.d.ts';
    break;
  default: {
    const exhaustive: never = policy.runtime;
    throw new Error(String(exhaustive));
  }
  }
  const baseFunctions = [...functions.values()].sort((left, right) => left.name.localeCompare(right.name));
  const testFunctions: AbiFunction[] = [];
  switch (policy.runtime) {
  case 'llama-cpp': break;
  case 'stable-diffusion-cpp':
    for (const { node, namespaces } of entries) {
      if (namespaces.length || node.kind !== 'FunctionDecl' || !node.name?.startsWith('sdc_test_')) {
        continue;
      }
      const fn = returnPublicFunction({ declarations, node });
      if (fn.bridge) {
        validateBridge({ fn, actual: bridgeFunctions.get(`bicore_${fn.name}`), declarations });
      }
      if (!testFunctions.some((previous) => previous.name === fn.name)) {
        testFunctions.push(fn);
      }
    }
    testFunctions.sort((left, right) => left.name.localeCompare(right.name));
    break;
  default: {
    const exhaustive: never = policy.runtime;
    throw new Error(String(exhaustive));
  }
  }
  for (const name of declarations.suspendingTests) {
    if (!testFunctions.some((fn) => fn.name === name)) {
      throw new Error(`Suspending test registration has no compiled declaration: ${name}`);
    }
  }
  const recordList = declarations.records().filter((record) => !record.name.startsWith('ggml_webgpu_q8_diagnostic_'));
  const constantValues = Object.fromEntries(Object.entries(declarations.constantValues).filter(([name]) => !name.startsWith('GGML_WEBGPU_Q8_DIAGNOSTIC_')).sort(([left], [right]) => left.localeCompare(right)));
  const extensionExports = Object.values(extensions).flatMap((value) => [value.schemaHashExport, ...value.functions.map((fn) => fn.export)]);
  const exports = [...new Set([...helpers, ...baseFunctions.map((fn) => fn.export), ...testFunctions.map((fn) => fn.export), ...extensionExports])].sort();
  const synchronous = policy.synchronous.map((name) => {
    const fn = functions.get(name);
    if (!fn || fn.bridge) {
      throw new Error(`Synchronous leaf declaration needs review: ${name}`);
    }
    return { name, export: fn.export };
  });
  const suspendingExports = baseFunctions.filter((fn) => !policy.synchronous.includes(fn.name)).map((fn) => fn.export.slice(1));
  suspendingExports.push(...declarations.suspendingTests);
  for (const value of Object.values(extensions)) {
    switch (value.suspension) {
    case 'none': break;
    case 'jspi':
      for (const fn of value.functions) {
        if (fn.export !== value.versionExport && fn.export !== value.schemaHashExport) {
          suspendingExports.push(fn.export.slice(1));
        }
      }
      break;
    default: {
      const exhaustive: never = value.suspension;
      throw new Error(String(exhaustive));
    }
    }
  }
  suspendingExports.sort();
  const schema: AbiSchema = {
    abiVersion: 3,
    runtime: policy.runtime,
    pointerRepresentation: 'bigint_native',
    pointerBytes: declarations.pointerBytes,
    schemaSha256: '0'.repeat(64),
    functions: baseFunctions,
    records: recordList,
    callbacks: [...declarations.callbacks.values()].filter((callback) => /^(?:ggml|gguf|llama|mtmd|sd)_/u.test(callback.name)).sort((left, right) => left.name.localeCompare(right.name)),
    constants: Object.keys(constantValues),
    constantValues,
    excluded: [...excluded].map(([name, reason]) => ({ name, reason })).sort((left, right) => left.name.localeCompare(right.name)),
    exports,
    suspendingExports,
    callbackMetadata: { version: 2, getters: synchronous },
    capabilities: declarations.capabilities,
    extensions,
    ...(testFunctions.length ? { testFunctions } : {}),
  };
  schema.schemaSha256 = abiSchemaFingerprint({ schema });
  await withOutputDirectories({
    destinations: [output],
    replace: true,
    preserveUnchanged: true,
    prepare: async ({ paths: [destination] }) => {
      if (!destination) {
        throw new Error('Missing ABI output');
      }
      const files = new Map<string, string>([
        ['schema.json', JSON.stringify(schema, null, 2) + '\n'], ['schema.mjs', renderSchemaModule({ schema })],
        ['functions.d.ts', renderFunctionTypes({ schema })], ['functions.mjs', renderFunctionModule({ schema })],
        ['exports.json', JSON.stringify(exports, null, 2) + '\n'], ['jspi-exports.json', JSON.stringify(suspendingExports, null, 2) + '\n'],
        ['abi-fingerprint.json', JSON.stringify({ schemaSha256: schema.schemaSha256 }) + '\n'],
      ]);
      if (moduleTypes) {
        files.set(moduleTypes, renderModuleTypes({ schema }));
      }
      for (const [name, value] of Object.entries(extensions)) {
        const prefix = name === 'webgpu-q8-diagnostic-v1' ? 'q8-diagnostic-v1' : name;
        files.set(`${prefix}.schema.json`, JSON.stringify({ ...value, schemaSha256: schema.schemaSha256 }, null, 2) + '\n');
        files.set(`${prefix}.functions.d.ts`, renderExtensionTypes({ extension: value }));
      }
      await Promise.all([...files].map(([name, text]) => writeFile(path.join(destination, name), text)));
      if (moduleTypes) {
        await copyFile(path.join(compileRoot, 'runtimes/stable-diffusion-cpp/bindings/module-options.d.ts'), path.join(destination, 'module-options.d.ts'));
      }
      // Keep the ordinary fingerprint header's mtime when semantic metadata did
      // not change. CMake still verifies and renders the fixed template below.
      try {
        const previous = await readFile(path.join(output, 'abi-fingerprint.json'), 'utf8');
        if (previous === JSON.stringify({ schemaSha256: schema.schemaSha256 }) + '\n') {
          await copyFile(path.join(output, 'fingerprint.h'), path.join(destination, 'fingerprint.h'));
        }
      } catch (error) {
        if (!(error instanceof Error && 'code' in error && error.code === 'ENOENT')) {
          throw error;
        }
      }
    },
  });
  if (runtimeDirectory !== undefined) {
    await mkdir(runtimeDirectory, { recursive: true });
    await replaceFileUnchanged({ candidate: path.join(output, 'schema.json'), output: path.join(runtimeDirectory, 'abi.json') });
    if (moduleTypes) {
      await replaceFileUnchanged({ candidate: path.join(output, moduleTypes), output: path.join(runtimeDirectory, moduleTypes) });
      await replaceFileUnchanged({ candidate: path.join(output, 'module-options.d.ts'), output: path.join(runtimeDirectory, 'module-options.d.ts') });
    }
    for (const name of ['model-graph-v1', 'webgpu-q8-diagnostic-v1']) {
      const prefix = name === 'webgpu-q8-diagnostic-v1' ? 'q8-diagnostic-v1' : name;
      for (const suffix of ['schema.json', 'functions.d.ts']) {
        if (extensions[name]) {
          await replaceFileUnchanged({ candidate: path.join(output, `${prefix}.${suffix}`), output: path.join(runtimeDirectory, `${prefix}.${suffix}`) });
        } else {
          await rm(path.join(runtimeDirectory, `${prefix}.${suffix}`), { force: true });
        }
      }
    }
  }
  return schema;
}
