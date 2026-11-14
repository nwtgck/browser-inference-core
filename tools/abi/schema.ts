import { createHash } from 'node:crypto';
import type { AbiCatalog, AbiSchema, Kind } from '#dto/shared';

/** Canonicalize owned JSON data, never C/C++ source text. */
export function canonicalJson({ value }: { value: unknown }): string {
  const encode = ({ entry }: { entry: unknown }): string => {
    if (entry === null || typeof entry !== 'object') {
      const result = JSON.stringify(entry);
      if (result === undefined) {
        throw new TypeError('Non-JSON ABI value');
      }
      return result;
    }
    if (Array.isArray(entry)) {
      return '[' + entry.map((item: unknown) => encode({ entry: item })).join(',') + ']';
    }
    return '{' + Object.entries(entry).filter(([, item]) => item !== undefined).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => JSON.stringify(key) + ':' + encode({ entry: item })).join(',') + '}';
  };
  return encode({ entry: value });
}

export function abiSchemaFingerprint({ schema }: { schema: AbiSchema | AbiCatalog }): string {
  const { schemaSha256: _schemaSha256, ...data } = schema;
  return createHash('sha256').update(canonicalJson({ value: data })).digest('hex');
}

export function publicDeclarations({ schema }: { schema: AbiSchema }) {
  return {
    runtime: schema.runtime,
    functions: schema.functions.map(({ name, returnType, returnKind, parameters }) => ({ name, returnType, returnKind, parameters: parameters.map(({ name: parameterName, cType, kind }) => ({ name: parameterName, cType, kind })) })),
    records: schema.records.map(({ name, cType, fields }) => ({ name, cType, fields: fields.map(({ name: fieldName, cType: fieldType, kind }) => ({ name: fieldName, cType: fieldType, kind })) })),
    constants: schema.constants,
  };
}

export function assembleAbiCatalog({ schemas }: { schemas: readonly AbiSchema[] }): AbiCatalog {
  const first = schemas[0];
  if (!first) {
    throw new Error('An ABI catalog needs at least one compiled target');
  }
  const declarations = canonicalJson({ value: publicDeclarations({ schema: first }) });
  const targets: Record<string, AbiSchema> = {};
  for (const schema of schemas) {
    if (abiSchemaFingerprint({ schema }) !== schema.schemaSha256) {
      throw new Error('ABI metadata fingerprint differs from its data');
    }
    if (canonicalJson({ value: publicDeclarations({ schema }) }) !== declarations) {
      throw new Error('Runtime targets expose different public declarations');
    }
    const previous = targets[schema.schemaSha256];
    if (previous && canonicalJson({ value: previous }) !== canonicalJson({ value: schema })) {
      throw new Error('Conflicting ABI fingerprint');
    }
    targets[schema.schemaSha256] = schema;
  }
  const catalog: AbiCatalog = { ...first, targets };
  catalog.schemaSha256 = abiSchemaFingerprint({ schema: catalog });
  return catalog;
}

export function renderSchemaModule({ schema }: { schema: AbiSchema | AbiCatalog }): string {
  return 'export default ' + JSON.stringify(schema) + ';\n';
}

export function jsType({ kind }: { kind: Kind }): string {
  switch (kind) {
  case 'pointer': case 'record': case 'u64': case 'i64': return 'bigint';
  case 'void': return 'void';
  case 'boolean': case 'signed': case 'unsigned': case 'float': return 'number';
  case 'array': throw new Error('An array cannot cross a C function boundary without decay');
  default: {
    const exhaustive: never = kind;
    throw new Error(String(exhaustive));
  }
  }
}

/** Aggregate returns use caller-provided storage in the ordinary bridge. */
export function loweredReturn({ kind }: { kind: Kind }): { kind: Kind; output: boolean } {
  switch (kind) {
  case 'record': return { kind: 'void', output: true };
  default: return { kind, output: false };
  }
}

export function renderFunctionTypes({ schema }: { schema: Pick<AbiSchema, 'functions'> }): string {
  const lines = ['/** Public C API. Pointers and pointer-sized integers are normalized to bigint. */', 'export interface LowLevelFunctions {'];
  for (const fn of schema.functions) {
    const result = loweredReturn({ kind: fn.returnKind });
    const parameters: string[] = result.output ? ['out: bigint'] : [];
    for (const [index, parameter] of fn.parameters.entries()) {
      parameters.push(`arg${index}: ${jsType({ kind: parameter.kind })}`);
    }
    lines.push(`  ${fn.name}(${parameters.join(', ')}): Promise<${jsType({ kind: result.kind })}>;`);
  }
  lines.push('}', 'export type NativeScalar = number | bigint;', 'export type NativeCall = {', '  name: keyof LowLevelFunctions;', '  args: readonly NativeScalar[];', '};', 'export interface NativeInvoker {', '  number(call: NativeCall): Promise<number>;', '  bigint(call: NativeCall): Promise<bigint>;', '  void(call: NativeCall): Promise<void>;', '}', 'export function createLowLevelFunctions({ invoker }: { invoker: NativeInvoker }): LowLevelFunctions;', '');
  return lines.join('\n');
}

/** A typed JS facade, derived from the same declarations as its interface.
 * The generic native caller owns ABI conversion; this facade checks the exact
 * public result type instead of casting a dynamic map to an interface. */
export function renderFunctionModule({ schema }: { schema: Pick<AbiSchema, 'functions'> }): string {
  const lines = [
    '/** @param {{ invoker: import("./functions.js").NativeInvoker }} options @returns {import("./functions.js").LowLevelFunctions} */',
    'export function createLowLevelFunctions({ invoker }) {',
    '  return {',
  ];
  for (const fn of schema.functions) {
    const result = loweredReturn({ kind: fn.returnKind });
    const args = fn.parameters.map((_, index) => `arg${index}`);
    if (result.output) {
      args.unshift('out');
    }
    const returns = jsType({ kind: result.kind });
    lines.push(`    ${JSON.stringify(fn.name)}: (${args.join(', ')}) => invoker.${returns}({ name: ${JSON.stringify(fn.name)}, args: [${args.join(', ')}] }),`);
  }
  lines.push('  };', '}', '');
  return lines.join('\n');
}
