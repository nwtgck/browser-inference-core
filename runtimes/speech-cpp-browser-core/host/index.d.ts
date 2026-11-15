import type { LowLevelFunctions } from '../api/functions.js';
import type { AbiCore } from './abi.js';
import type { CoreModule } from './profiles.js';
export { createAbiCaller, selectAbiSchema, toNativePointer, fromNativePointer, toEmscriptenPointer, fromEmscriptenPointer } from './abi.js';
export type { CoreModule, ModuleByProfile, Profile, Variant } from './profiles.js';
export type Core<Module extends CoreModule = CoreModule> = Omit<AbiCore<LowLevelFunctions>, 'module'> & { module: Module };
export function attachCore<Module extends CoreModule>(module: Module, schema: object, options: { suspension: 'jspi' }): Core<Module>;
export function mountReadOnlyFile(core: Core, path: string, source: {
  size: number;
  read(destination: Uint8Array, offset: number): number;
}, options?: { maxChunkBytes?: number }): { path: string; remove(): void };
export const schema: object;
