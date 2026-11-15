/** Generated from this package's profile/variant set. */
import type { SpeechModule as Module_0_0 } from '../profiles/webgpu-wasm64-jspi/browser/core.js';
import type { SpeechModule as Module_0_1 } from '../profiles/webgpu-wasm64-jspi/test/core.js';
export interface ModuleByProfile {
  'webgpu-wasm64-jspi': {
    'browser': Module_0_0;
    'test': Module_0_1;
  };
}
export type Profile = keyof ModuleByProfile;
export type Variant = keyof ModuleByProfile[Profile];
export type CoreModule = ModuleByProfile[Profile][Variant];
