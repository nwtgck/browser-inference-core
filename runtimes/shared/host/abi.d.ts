export type NativeScalar = number | bigint;
export type AbiType = {
  kind: 'void' | 'pointer' | 'integer' | 'float' | 'record' | 'array';
  bits?: number;
  signed?: boolean;
  pointerSized?: boolean;
  record?: string;
  callback?: string;
  dimensions?: number[];
  element?: {
    kind: PublicKind;
    abiType: AbiType;
  };
};
export type PublicKind = 'void' | 'pointer' | 'record' | 'array' | 'boolean' | 'signed' | 'unsigned' | 'u64' | 'i64' | 'float';
export type AbiParameter = {
  name: string;
  cType: string;
  kind: PublicKind;
  abiType: AbiType;
};
export type AbiFunction = {
  name: string;
  export: string;
  returnType: string;
  returnKind: PublicKind;
  abiType: AbiType;
  parameters: AbiParameter[];
  bridge: boolean;
};
export type AbiField = AbiParameter & {
  id: number;
  offset: number;
  size: number;
};
export type AbiRecord = {
  name: string;
  cType: string;
  id: number;
  size: number;
  alignment: number;
  fields: AbiField[];
};
export type AbiExtension = {
  extensionId: string;
  version: 1;
  abiVersion: 1;
  baseAbiVersion: 3;
  pointerRepresentation: 'bigint_u64';
  suspension: 'none' | 'jspi';
  versionExport: string;
  schemaHashExport: string;
  functions: AbiFunction[];
  records: AbiRecord[];
  constants: string[];
  constantValues: Record<string, string>;
};
export type AbiSchema = {
  abiVersion: 3;
  runtime: 'llama-cpp' | 'stable-diffusion-cpp';
  pointerBytes: 4 | 8;
  pointerRepresentation: 'bigint_native';
  schemaSha256: string;
  functions: AbiFunction[];
  records: AbiRecord[];
  constants: string[];
  constantValues: Record<string, string>;
  callbackMetadata: {
    version: 2;
    getters: {
      name: string;
      export: string;
    }[];
  };
  exports: string[];
  suspendingExports: string[];
  capabilities?: Record<string, number>;
  extensions?: Record<string, AbiExtension>;
};
export type NativeModule = {
  HEAPU8: Uint8Array;
  ccall?: (name: string, returnType: 'number' | 'bigint' | undefined, types: ('number' | 'bigint')[], args: NativeScalar[], options: { async: true }) => unknown;
};
export function toNativePointer(options: { value: bigint; pointerBytes: 4 | 8 }): NativeScalar;
export function fromNativePointer(options: { value: NativeScalar; pointerBytes: 4 | 8 }): bigint;
export function toEmscriptenPointer(options: { value: bigint; pointerBytes: 4 | 8 }): number;
export function fromEmscriptenPointer(options: { value: number; pointerBytes: 4 | 8 }): bigint;
export function selectAbiSchema(options: { module: NativeModule; schema: unknown }): AbiSchema;
export type NativeCall = {
  name: string;
  args: readonly NativeScalar[];
};
export type NativeInvoker = {
  number(call: NativeCall): Promise<number>;
  bigint(call: NativeCall): Promise<bigint>;
  void(call: NativeCall): Promise<void>;
};
export function createAbiCaller(options: {
  module: NativeModule;
  schema: unknown;
  suspension: 'direct' | 'jspi' | 'asyncify';
  gate: {
    enter(): void;
    leave(): void;
  } | undefined;
}): {
  schema: AbiSchema;
  invoker: NativeInvoker;
  call(options: NativeCall): Promise<NativeScalar | void>;
  callSync(options: NativeCall): NativeScalar | void;
};
export type AbiCore<Api> = {
  module: NativeModule;
  api: Api;
  schema: AbiSchema;
  pointerBytes: 4 | 8;
  readonly busy: boolean;
  constant(name: string): number;
  alloc(bytes: number | bigint): bigint;
  free(pointer: bigint): void;
  recordSize(name: string): number;
  recordAlignment(name: string): number;
  allocRecord(name: string): bigint;
  fieldAddress(name: string, pointer: bigint, field: string): bigint;
  getField(name: string, pointer: bigint, field: string): NativeScalar;
  setField(name: string, pointer: bigint, field: string, value: NativeScalar): void;
  bytes(pointer: bigint, length: number | bigint): Uint8Array;
  utf8(text: string): bigint;
  readUtf8(pointer: bigint, maximum?: number): string | null;
};
export function attachAbiCore<Api>(options: {
  module: NativeModule;
  schema: unknown;
  suspension: 'direct' | 'jspi' | 'asyncify';
  createApi: ({ invoker }: { invoker: NativeInvoker }) => Api;
}): AbiCore<Api>;
