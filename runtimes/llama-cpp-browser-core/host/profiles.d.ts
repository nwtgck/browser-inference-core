/** Generated from this package's profile/variant set. */
import type { MainModule as Module_0_0 } from '../profiles/cpu-wasm32/browser/core.js';
import type { MainModule as Module_0_1 } from '../profiles/cpu-wasm32/test/core.js';
import type { MainModule as Module_1_0 } from '../profiles/webgpu-wasm32-asyncify/browser/core.js';
import type { MainModule as Module_1_1 } from '../profiles/webgpu-wasm32-asyncify/test/core.js';
import type { MainModule as Module_2_0 } from '../profiles/webgpu-wasm32-jspi/browser/core.js';
import type { MainModule as Module_2_1 } from '../profiles/webgpu-wasm32-jspi/test/core.js';
import type { MainModule as Module_3_0 } from '../profiles/webgpu-wasm64-jspi/browser/core.js';
import type { MainModule as Module_3_1 } from '../profiles/webgpu-wasm64-jspi/test/core.js';
import type { MainModule as Module_4_0 } from '../profiles/webgpu-wasm64-jspi-gpu-profile/browser/core.js';
import type { MainModule as Module_4_1 } from '../profiles/webgpu-wasm64-jspi-gpu-profile/test/core.js';
export interface ModuleByProfile {
  'cpu-wasm32': {
    'browser': Module_0_0;
    'test': Module_0_1;
  };
  'webgpu-wasm32-asyncify': {
    'browser': Module_1_0;
    'test': Module_1_1;
  };
  'webgpu-wasm32-jspi': {
    'browser': Module_2_0;
    'test': Module_2_1;
  };
  'webgpu-wasm64-jspi': {
    'browser': Module_3_0;
    'test': Module_3_1;
  };
  'webgpu-wasm64-jspi-gpu-profile': {
    'browser': Module_4_0;
    'test': Module_4_1;
  };
}
export type Profile = keyof ModuleByProfile;
export type Variant = keyof ModuleByProfile[Profile];
