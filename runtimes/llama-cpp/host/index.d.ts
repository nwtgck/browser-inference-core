import type { LowLevelFunctions } from '../api/functions.js';
import type { ModuleByProfile, Profile, Variant } from './profiles.js';
import type { AbiSchema } from './abi.js';
export { createAbiCaller, selectAbiSchema, toNativePointer, fromNativePointer, toEmscriptenPointer, fromEmscriptenPointer } from './abi.js';
export type { ModuleByProfile, Profile, Variant } from './profiles.js';
export interface CoreModuleOptions extends Record<string, unknown> {
  /** WebGPU browser variants only. Opaque host handle; no WebGPU types dependency. */
  hostNavigator?: { readonly gpu?: object };
}
export interface CoreModule {
  HEAPU8: Uint8Array;
  FS: object;
  addFunction<Args extends (number | bigint)[]>(callback: (...args: Args) => number | bigint | void, signature: string): number | bigint;
  removeFunction(pointer: number | bigint): void;
}
export interface Core<Module extends CoreModule = CoreModule> {
  module: Module;
  api: LowLevelFunctions;
  schema: AbiSchema;
  readonly busy: boolean;
  pointerBytes: number;
  constant(name: string): number;
  alloc(bytes: number | bigint): bigint;
  free(pointer: bigint): void;
  recordSize(name: string): number;
  recordAlignment(name: string): number;
  allocRecord(name: string): bigint;
  fieldAddress(name: string, pointer: bigint, field: string): bigint;
  getField(name: string, pointer: bigint, field: string): number | bigint;
  setField(name: string, pointer: bigint, field: string, value: number | bigint): void;
  bytes(pointer: bigint, length: number | bigint): Uint8Array;
  utf8(text: string): bigint;
  readUtf8(pointer: bigint, maximum?: number): string | null;
}
export function createCore<P extends Profile, V extends Variant = 'browser'>(options: {
  profile: P;
  variant?: V;
  baseURL?: URL | string;
  moduleOptions?: CoreModuleOptions;
}): Promise<Core<ModuleByProfile[P][V]>>;
export function attachCore<Module extends CoreModule>(module: Module, schema: object, options: { suspension: 'direct' | 'jspi' | 'asyncify' }): Core<Module>;
export function mountReadOnlyFile(core: Core, path: string, source: {
  size: number;
  read(destination: Uint8Array, offset: number): number;
}, options?: { maxChunkBytes?: number }): { path: string; remove(): void };
export const schema: object;
