/** Generated from this package's profile/variant set. */
import type { StableDiffusionModule as Module_0_0 } from '../profiles/webgpu-wasm32-asyncify/browser/core.js';
import type { StableDiffusionModule as Module_0_1 } from '../profiles/webgpu-wasm32-asyncify/test/core.js';
import type { StableDiffusionModule as Module_1_0 } from '../profiles/webgpu-wasm32-jspi/browser/core.js';
import type { StableDiffusionModule as Module_1_1 } from '../profiles/webgpu-wasm32-jspi/test/core.js';
import type { StableDiffusionModule as Module_2_0 } from '../profiles/webgpu-wasm64-jspi/browser/core.js';
import type { StableDiffusionModule as Module_2_1 } from '../profiles/webgpu-wasm64-jspi/test/core.js';
export interface ModuleByProfile {
  'webgpu-wasm32-asyncify': {
    'browser': Module_0_0;
    'test': Module_0_1;
  };
  'webgpu-wasm32-jspi': {
    'browser': Module_1_0;
    'test': Module_1_1;
  };
  'webgpu-wasm64-jspi': {
    'browser': Module_2_0;
    'test': Module_2_1;
  };
}
export type Profile = keyof ModuleByProfile;
export type Variant = keyof ModuleByProfile[Profile];
export type CoreModule = ModuleByProfile[Profile][Variant];
