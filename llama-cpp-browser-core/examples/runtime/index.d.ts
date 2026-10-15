import type { LowLevelFunctions } from '../../api/functions.js';
export type Profile = 'cpu-wasm32' | 'cpu-wasm64' | 'webgpu-wasm32-asyncify' | 'webgpu-wasm32-jspi' | 'webgpu-wasm64-jspi';
export type Variant = 'browser' | 'test';
export interface CoreModule {
  HEAPU8: Uint8Array;
  FS: any;
  addFunction(callback: (...args: any[]) => any, signature: string): number | bigint;
  removeFunction(pointer: number | bigint): void;
  [exportName: string]: any;
}
export interface Core {
  module: CoreModule;
  api: LowLevelFunctions;
  schema: object;
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
export function createCore(options: {
  profile: Profile;
  variant?: Variant;
  baseURL?: URL | string;
  moduleOptions?: Record<string, unknown>;
}): Promise<Core>;
export function attachCore(module: CoreModule, schema: any, options?: { suspension?: 'direct' | 'asyncify' }): Core;
export function mountReadOnlyFile(core: Core, path: string, source: {
  size: number;
  read(destination: Uint8Array, offset: number): number;
}, options?: { maxChunkBytes?: number }): { path: string; remove(): void };
export const schema: object;
