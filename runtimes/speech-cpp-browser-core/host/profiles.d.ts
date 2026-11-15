/** Generated from this package's profile/variant set. */
import type { default as Factory_0_0, SpeechModule as Module_0_0 } from '../profiles/webgpu-wasm32-jspi/browser/core.js';
import type { default as Factory_0_1, SpeechModule as Module_0_1 } from '../profiles/webgpu-wasm32-jspi/test/core.js';
export interface ModuleByProfile {
  'webgpu-wasm32-jspi': {
    'browser': Module_0_0;
    'test': Module_0_1;
  };
}
export interface FactoryByProfile {
  'webgpu-wasm32-jspi': {
    'browser': typeof Factory_0_0;
    'test': typeof Factory_0_1;
  };
}
export type Profile = keyof ModuleByProfile;
export type Variant = keyof ModuleByProfile[Profile];
export type CoreFactory = FactoryByProfile[Profile][Variant];
export type CoreModule = ModuleByProfile[Profile][Variant];
