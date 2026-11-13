import type { LowLevelFunctions } from '../api/functions.js';
import type { CoreModule } from './profiles.js';
export type { CoreModule, ModuleByProfile, Profile, Variant } from './profiles.js';
export interface Core<Module extends CoreModule = CoreModule> {
  module: Module;
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
export function attachCore<Module extends CoreModule>(module: Module, schema: object, options: { suspension: 'direct' | 'asyncify' }): Core<Module>;
export function mountReadOnlyFile(core: Core, path: string, source: {
  size: number;
  read(destination: Uint8Array, offset: number): number;
}, options?: { maxChunkBytes?: number }): { path: string; remove(): void };
export const schema: object;
