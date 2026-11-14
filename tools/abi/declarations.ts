import { childNodes, promotedFields, type AstEntry, type AstNode, type RecordLayout } from './clang.js';
import type { AbiFunction, AbiRecord, AbiType, Kind, RuntimeName } from '#dto/shared';

type Described = { kind: Kind; abiType: AbiType };
type RecordDeclaration = { node: AstNode; name: string; cType: string; nested?: true };

function descendants({ node }: { node: AstNode }): AstNode[] {
  return childNodes({ node }).flatMap((child) => [child, ...descendants({ node: child })]);
}

export function enumValues({ node }: { node: AstNode }): Map<string, string> {
  const values = new Map<string, string>();
  let previous = -1n;
  for (const member of childNodes({ node }).filter((child) => child.kind === 'EnumConstantDecl')) {
    if (!member.name) {
      throw new Error('Unnamed enum constant');
    }
    const expression = descendants({ node: member }).find((child) => child.kind === 'ConstantExpr' && child.value !== undefined);
    previous = expression?.value === undefined ? previous + 1n : BigInt(String(expression.value));
    values.set(member.name, previous.toString());
  }
  return values;
}

function recordKey({ type }: { type: string }): string {
  const anonymous = [...type.matchAll(/(?:unnamed|anonymous)(?: struct| union)? at (.+?:\d+:\d+)\)/gu)].at(-1)?.[1];
  return anonymous ? `anonymous:${anonymous}` : type.replace(/^(?:const )?(?:struct|union|class) /u, '').trim();
}

export class Declarations {
  readonly entries: readonly AstEntry[];
  readonly runtime: RuntimeName;
  readonly unit: string;
  readonly pointerBytes: 4 | 8;
  readonly aliases = new Map<string, string>();
  readonly enums = new Map<string, Map<string, string>>();
  readonly enumUnderlyingTypes = new Map<string, string>();
  readonly recordDeclarations = new Map<string, RecordDeclaration>();
  readonly recordTypeNames = new Map<string, string>();
  readonly opaqueRecords = new Map<string, string>();
  readonly recordLayouts = new Map<string, RecordLayout>();
  readonly registrations = new Map<string, number>();
  readonly constantValues: Record<string, string> = {};
  readonly capabilities: Record<string, number> = {};
  readonly suspendingTests = new Set<string>();
  readonly callbacks = new Map<string, { name: string; cType: string; returnType: string; returnKind: Kind; abiType: AbiType; parameters: { name: string; cType: string; kind: Kind; abiType: AbiType }[] }>();
  readonly macros: ReadonlyMap<string, string>;

  constructor({ entries, layouts, macros, runtime, translationUnit }: { entries: readonly AstEntry[]; layouts: readonly RecordLayout[]; macros: ReadonlyMap<string, string>; runtime: RuntimeName; translationUnit: string }) {
    this.entries = entries;
    this.runtime = runtime;
    this.unit = translationUnit;
    this.macros = macros;
    const pointerBytes = Number(macros.get('__SIZEOF_POINTER__'));
    if (pointerBytes !== 4 && pointerBytes !== 8) {
      throw new Error('Unsupported target pointer width');
    }
    this.pointerBytes = pointerBytes;
    const byId = new Map(entries.flatMap(({ node }) => node.id ? [[node.id, node] as const] : []));
    for (const layout of layouts) {
      this.recordLayouts.set(recordKey({ type: layout.type }), layout);
    }
    for (const { node, namespaces } of entries) {
      if (namespaces.length) {
        continue;
      }
      if (node.kind === 'TypedefDecl' && node.name && node.type) {
        this.aliases.set(node.name, node.type.desugaredQualType === node.name ? node.type.qualType : node.type.desugaredQualType ?? node.type.qualType);
        // Clang 18 wraps the owned tag in ElaboratedType. Current Emscripten
        // Clang emits a direct RecordType reference for the same typedef.
        const types = descendants({ node });
        const owned = types.find((child) => child.ownedTagDecl)?.ownedTagDecl
          ?? types.find((child) => child.kind === 'RecordType' && child.decl)?.decl;
        const record = owned ? byId.get(owned.id) : undefined;
        if (record?.completeDefinition && !node.type.qualType.includes('*') && !node.type.qualType.includes('(') && (node.name.startsWith('sd_') || node.name === 'ggml_bf16_t')) {
          this.recordDeclarations.set(node.name, { node: record, name: node.name, cType: node.name });
        }
      }
      if (['RecordDecl', 'CXXRecordDecl'].includes(node.kind) && node.completeDefinition && node.name && /^(?:llama|ggml|gguf|mtmd|sd)_/u.test(node.name)) {
        if (node.bases?.length || node.definitionData && (node.definitionData.isStandardLayout !== true || node.definitionData.isTriviallyCopyable !== true)) {
          this.opaqueRecords.set(node.name, 'C++ record is not a trivially copyable standard-layout C record; only opaque pointers may cross this ABI');
          continue;
        }
        this.recordDeclarations.set(node.name, { node, name: node.name, cType: `${node.tagUsed ?? 'struct'} ${node.name}` });
      }
      if (node.kind === 'EnumDecl' && node.name) {
        const values = enumValues({ node });
        this.enums.set(node.name, values);
        if (node.fixedUnderlyingType) {
          this.enumUnderlyingTypes.set(node.name, node.fixedUnderlyingType.qualType);
        }
        if (node.name === 'BicoreRecordLayouts') {
          for (const [name, value] of values) {
            this.registrations.set(name.replace(/^BICORE_/u, ''), Number(value));
          }
        } else if (node.name === 'BicoreMacroConstants') {
          for (const [name, value] of values) {
            this.constantValues[name.replace(/^BICORE_/u, '')] = value;
          }
        } else if (node.name === 'BicoreCapabilities') {
          for (const [name, value] of values) {
            this.capabilities[name] = Number(value);
          }
        } else if (node.name === 'BicoreSuspendingTests') {
          for (const name of values.keys()) {
            this.suspendingTests.add(name.replace(/^BICORE_/u, ''));
          }
        } else {
          for (const [name, value] of values) {
            if (/^(?:LLAMA|GGML|GGUF|MTMD)_/u.test(name) || (runtime === 'stable-diffusion-cpp' && node.loc?.includedFrom?.file === translationUnit)) {
              this.constantValues[name] = value;
            }
          }
        }
      }
    }
    this.registerNestedRecords();
    for (const { node, namespaces } of entries) {
      if (namespaces.length || node.kind !== 'TypedefDecl' || !node.name || !node.type) {
        continue;
      }
      const prototype = descendants({ node }).find((child) => child.kind === 'FunctionProtoType');
      if (prototype) {
        this.registerCallback({ name: node.name, cType: node.type.qualType, prototype });
      }
    }
    for (const name of this.registrations.keys()) {
      if (!this.recordDeclarations.has(name) && !this.opaqueRecords.has(name)) {
        throw new Error(`Registered public record has no decoded declaration: ${name}`);
      }
    }
  }

  /** A containing sizeof forces Clang to lay out its by-value children too.
   * Use source identities from the AST to bind those anonymous declarations;
   * no additional generated C++ registration or guessed field offsets. */
  private registerNestedRecords(): void {
    // Map iteration also visits newly added children, including deeper records.
    for (const declaration of this.recordDeclarations.values()) {
      const definitions = descendants({ node: declaration.node }).filter((node) => ['RecordDecl', 'CXXRecordDecl'].includes(node.kind) && node.completeDefinition);
      for (const field of this.fields({ declaration })) {
        const type = this.resolve({ type: field.cType }).replace(/(?:\[\d*\])+$/u, '').trim();
        const key = recordKey({ type });
        if (!key.startsWith('anonymous:') || type.includes('*') || this.recordTypeNames.has(key)) {
          continue;
        }
        const position = /:(\d+):(\d+)$/u.exec(key);
        if (!position) {
          throw new Error(`Anonymous record has no compiler source identity: ${field.cType}`);
        }
        const start = field.node.range?.begin.offset;
        const byOffset = start === undefined ? [] : definitions.filter((node) => node.loc?.offset === start);
        const candidates = byOffset.length ? byOffset : definitions.filter((node) => node.loc?.line === Number(position[1]) && node.loc.col === Number(position[2]));
        const node = candidates[0];
        if (!node || candidates.length !== 1) {
          throw new Error(`Cannot bind anonymous declaration ${declaration.name}.${field.name}`);
        }
        if (node.bases?.length || node.definitionData && (node.definitionData.isStandardLayout !== true || node.definitionData.isTriviallyCopyable !== true)) {
          throw new Error(`Unsupported nested C++ record ${declaration.name}.${field.name}`);
        }
        if (!this.recordLayouts.has(key)) {
          throw new Error(`Missing compiler layout for anonymous field ${declaration.name}.${field.name}`);
        }
        const name = `${declaration.name}.${field.name}`;
        this.recordDeclarations.set(name, { node, name, cType: type, nested: true });
        this.recordTypeNames.set(key, name);
      }
    }
  }

  resolve({ type }: { type: string }): string {
    let value = type.trim();
    const seen = new Set<string>();
    while (this.aliases.has(value)) {
      if (seen.has(value)) {
        throw new Error(`Cyclic public type alias: ${type}`);
      }
      seen.add(value);
      const next = this.aliases.get(value);
      if (!next || next === value) {
        break;
      }
      value = next;
    }
    return value;
  }

  describe({ type }: { type: string }): Described {
    let resolved = this.resolve({ type });
    const callback = this.callbacks.has(type) ? type : undefined;
    const dimensions: number[] = [];
    while (/\[\d*\]$/u.test(resolved) && !/\([^)]*\*[^)]*\)\s*(?:\[\d*\])+$/u.test(resolved)) {
      const match = /\[(\d*)\]$/u.exec(resolved);
      if (!match) {
        throw new Error(`Unsupported array declaration: ${type}`);
      }
      dimensions.unshift(Number(match[1] || '0'));
      resolved = resolved.slice(0, match.index).trim();
    }
    if (dimensions.length) {
      const element = this.describe({ type: resolved });
      return { kind: 'array', abiType: { kind: 'array', dimensions, element } };
    }
    if (resolved.includes('*') || resolved.includes('(&)')) {
      return { kind: 'pointer', abiType: { kind: 'pointer', bits: this.pointerBytes * 8, ...(callback ? { callback } : {}) } };
    }
    resolved = resolved.replace(/^const /u, '').replace(/ const$/u, '');
    const rawKey = recordKey({ type: resolved });
    const key = this.recordTypeNames.get(rawKey) ?? rawKey;
    const opaqueReason = this.opaqueRecords.get(key);
    if (opaqueReason) {
      throw new Error(`Unsupported by-value public C type ${type}: ${opaqueReason}`);
    }
    if (/^(?:struct|union|class) /u.test(resolved) || this.recordDeclarations.has(key)) {
      if (!this.recordDeclarations.has(key)) {
        throw new Error(`Public by-value record has no compiler-derived declaration: ${type}`);
      }
      return { kind: 'record', abiType: { kind: 'record', record: key } };
    }
    if (resolved === 'void') {
      return { kind: 'void', abiType: { kind: 'void' } };
    }
    if (resolved === 'bool' || resolved === '_Bool') {
      return { kind: 'boolean', abiType: { kind: 'integer', bits: 8, signed: false } };
    }
    switch (resolved) {
    case 'float': return { kind: 'float', abiType: { kind: 'float', bits: 32 } };
    case 'double': return { kind: 'float', abiType: { kind: 'float', bits: 64 } };
    }
    const enumName = resolved.replace(/^enum /u, '');
    const enumMembers = this.enums.get(enumName);
    if (enumMembers) {
      const fixed = this.enumUnderlyingTypes.get(enumName);
      if (fixed) {
        return this.describe({ type: fixed });
      }
      const values = [...enumMembers.values()].map((value) => BigInt(value));
      const signed = values.some((value) => value < 0n);
      const bits = values.every((value) => value >= (signed ? -2147483648n : 0n) && value <= (signed ? 2147483647n : 4294967295n)) ? 32 : 64;
      if (values.some((value) => value < (signed ? -(1n << 63n) : 0n) || value >= (signed ? 1n << 63n : 1n << 64n))) {
        throw new Error(`Unsupported public enum width: ${enumName}`);
      }
      return { kind: bits === 64 ? signed ? 'i64' : 'u64' : 'signed', abiType: { kind: 'integer', bits, signed } };
    }
    const pointerSized = ['size_t', 'ssize_t', 'uintptr_t', 'intptr_t', 'ptrdiff_t'].includes(type);
    const signed = !resolved.startsWith('unsigned ') && !(resolved === 'char' && this.macros.has('__CHAR_UNSIGNED__'));
    const width = /(?:^| )char$/u.test(resolved) ? 8 : /(?:^| )short(?: int)?$/u.test(resolved) ? 16
      : /(?:^| )long long(?: int)?$/u.test(resolved) ? 64 : /(?:^| )long(?: int)?$/u.test(resolved) ? Number(this.macros.get('__SIZEOF_LONG__')) * 8
        : /^(?:unsigned )?int$/u.test(resolved) ? Number(this.macros.get('__SIZEOF_INT__')) * 8 : undefined;
    if (!width || ![8, 16, 32, 64].includes(width)) {
      throw new Error(`Unsupported public C type: ${type} -> ${resolved}`);
    }
    const wide = pointerSized || ['uint64_t', 'int64_t', 'off_t'].includes(type) || /(?:^| )long(?: long)?(?: int)?$/u.test(resolved);
    return { kind: wide ? signed ? 'i64' : 'u64' : signed ? 'signed' : 'unsigned', abiType: { kind: 'integer', bits: width, signed, ...(pointerSized ? { pointerSized: true } : {}) } };
  }

  private registerCallback({ name, cType, prototype }: { name: string; cType: string; prototype: AstNode }): void {
    const types = childNodes({ node: prototype }).flatMap((child) => child.type ? [child.type.qualType] : []);
    const returnType = types[0];
    if (!returnType) {
      throw new Error(`Missing callback return type: ${name}`);
    }
    // Insert a placeholder before resolving recursive callback references.
    const returns = this.describe({ type: returnType });
    const parameters = types.slice(1).map((type, index) => ({ name: `arg${index}`, cType: type, ...this.describe({ type }) }));
    this.callbacks.set(name, { name, cType, returnType, returnKind: returns.kind, abiType: returns.abiType, parameters });
  }

  layout({ declaration }: { declaration: RecordDeclaration }): RecordLayout {
    const registered = this.registrations.get(declaration.name);
    if (registered === undefined && !declaration.nested) {
      throw new Error(`Public record ${declaration.name} needs an ordinary BicoreRecordLayouts type registration`);
    }
    const direct = this.recordLayouts.get(recordKey({ type: declaration.cType }));
    if (direct) {
      if (registered !== undefined && direct.size !== registered) {
        throw new Error(`Compiler layout differs from sizeof registration: ${declaration.name}`);
      }
      return direct;
    }
    if (declaration.nested) {
      throw new Error(`Missing anonymous target layout: ${declaration.name}`);
    }
    const names = this.fields({ declaration }).map((field) => field.name);
    const candidates = [...this.recordLayouts.values()].filter((candidate) => candidate.size === registered && names.every((name) => promotedFields({ layout: candidate }).has(name)));
    const first = candidates[0];
    if (!first || candidates.some((candidate) => candidate.alignment !== first.alignment || names.some((name) => promotedFields({ layout: candidate }).get(name) !== promotedFields({ layout: first }).get(name)))) {
      throw new Error(`Cannot identify anonymous public record layout: ${declaration.name}`);
    }
    return first;
  }

  fields({ declaration }: { declaration: RecordDeclaration }): { name: string; cType: string; node: AstNode }[] {
    const children = childNodes({ node: declaration.node });
    return children.filter((node) => ['FieldDecl', 'IndirectFieldDecl'].includes(node.kind) && node.name).map((node) => {
      const name = node.name;
      if (!name) {
        throw new Error('Unnamed public field');
      }
      const source = node.kind === 'FieldDecl' ? node : descendants({ node: declaration.node }).find((child) => child.kind === 'FieldDecl' && child.name === name && child.type);
      const fieldType = source?.type;
      if (!fieldType) {
        throw new Error(`Missing promoted field type: ${declaration.name}.${name}`);
      }
      if (source.isBitfield) {
        throw new Error(`Unsupported public bitfield: ${declaration.name}.${node.name}`);
      }
      return { name, cType: fieldType.qualType, node: source };
    });
  }

  sizeOf({ type }: { type: string }): number {
    const described = this.describe({ type });
    switch (described.abiType.kind) {
    case 'array': {
      const resolved = this.resolve({ type }).replace(/(?:\[\d*\])+$/u, '').trim();
      const element = this.sizeOf({ type: resolved });
      return (described.abiType.dimensions ?? []).reduce((size, count) => size * count, element);
    }
    case 'record': {
      const name = described.abiType.record;
      if (!name) {
        throw new Error('Missing record type');
      }
      const declaration = this.recordDeclarations.get(name);
      const layout = declaration ? this.layout({ declaration }) : this.recordLayouts.get(recordKey({ type: this.resolve({ type }) }));
      if (!layout) {
        throw new Error(`Missing compiler layout for nested record ${type}`);
      }
      return layout.size;
    }
    default: break;
    }
    if (described.abiType.bits === undefined) {
      throw new Error(`Type has no object size: ${type}`);
    }
    return described.abiType.bits / 8;
  }

  records(): AbiRecord[] {
    return [...this.recordDeclarations.values()].sort((left, right) => left.name.localeCompare(right.name)).map((declaration, id) => {
      const layout = this.layout({ declaration });
      const offsets = promotedFields({ layout });
      const fields = this.fields({ declaration }).map((field, fieldId) => {
        const offset = offsets.get(field.name);
        if (offset === undefined) {
          throw new Error(`Missing compiler field offset ${declaration.name}.${field.name}`);
        }
        const size = this.sizeOf({ type: field.cType });
        if (offset + size > layout.size) {
          throw new Error(`Compiler field exceeds record: ${declaration.name}.${field.name}`);
        }
        const described = this.describe({ type: field.cType });
        const anonymous = /\((?:unnamed|anonymous)(?: struct| union)? at /u.test(field.cType);
        const cType = anonymous ? `${described.kind} ${declaration.name}.${field.name}` : field.cType;
        const abiType = described.abiType;
        return { name: field.name, cType, id: fieldId, offset, size, kind: described.kind, abiType };
      });
      const cType = declaration.nested ? `${declaration.node.tagUsed ?? 'struct'} ${declaration.name}` : declaration.cType;
      return { name: declaration.name, cType, id, size: layout.size, alignment: layout.alignment, fields };
    });
  }

  function({ node, bridge }: { node: AstNode; bridge: boolean }): AbiFunction {
    if (!node.name || !node.type) {
      throw new Error('Incomplete public function declaration');
    }
    if (node.variadic) {
      throw new Error(`Variadic public function ${node.name} requires an explicit exclusion reason`);
    }
    const parametersStart = node.type.qualType.indexOf('(');
    // A direct function-pointer return starts a grouped declarator here, not
    // the parameter list. Treating its prefix `void` as the result is unsafe.
    if (parametersStart < 0 || /^\(\s*[*&]/u.test(node.type.qualType.slice(parametersStart))) {
      throw new Error(`Unsupported function return declarator: ${node.name}: ${node.type.qualType}`);
    }
    const returnType = node.type.qualType.slice(0, parametersStart).trim();
    if (!returnType) {
      throw new Error(`Unsupported function return declaration: ${node.type.qualType}`);
    }
    const returns = this.describe({ type: returnType });
    const parameters = childNodes({ node }).filter((child) => child.kind === 'ParmVarDecl').map((parameter, index) => {
      if (!parameter.type) {
        throw new Error(`Missing parameter type: ${node.name}`);
      }
      const described = this.describe({ type: parameter.type.qualType });
      if (described.kind === 'array' || described.kind === 'void') {
        throw new Error(`Unlowered public parameter: ${node.name}`);
      }
      return { name: parameter.name ?? `arg${index}`, cType: parameter.type.qualType, ...described };
    });
    return { name: node.name, export: '_' + (bridge ? 'bicore_' : '') + node.name, returnType, returnKind: returns.kind, abiType: returns.abiType, parameters, bridge };
  }
}
