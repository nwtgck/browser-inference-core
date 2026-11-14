import { artifactDirectories, runtimeNames } from '#tools/shared/contracts';
import type { RuntimeName } from '#dto/shared';

export type ExportMap = Record<string, string | { types: string; import: string } | { types: string }>;

export function runtimeExports({ runtime, prefix }: { runtime: RuntimeName; prefix: string }): ExportMap {
  const exports: ExportMap = {
    './host/abi': { types: `${prefix}/host/abi.d.ts`, import: `${prefix}/host/abi.mjs` },
    './host': { types: `${prefix}/host/index.d.ts`, import: `${prefix}/host/index.mjs` },
    './api/functions': { types: `${prefix}/api/functions.d.ts`, import: `${prefix}/api/functions.mjs` },
    './api/schema.json': `${prefix}/api/schema.json`,
    './api/schema.mjs': { types: `${prefix}/api/schema.d.ts`, import: `${prefix}/api/schema.mjs` },
    './api/exports.json': `${prefix}/api/exports.json`,
    './profiles/*/core.mjs': { types: `${prefix}/profiles/*/core.d.ts`, import: `${prefix}/profiles/*/core.mjs` },
    './profiles/*/core.wasm': `${prefix}/profiles/*/core.wasm`,
    './manifest.json': `${prefix}/manifest.json`,
    './native-license-index.json': `${prefix}/native-license-index.json`,
    './licenses/*': `${prefix}/licenses/*`,
  };
  switch (runtime) {
  case 'llama-cpp':
    exports['./api/extensions'] = { types: `${prefix}/api/extensions.d.ts` };
    for (const extension of ['model-graph-v1', 'q8-diagnostic-v1']) {
      exports[`./profiles/*/${extension}.schema.json`] = `${prefix}/profiles/*/${extension}.schema.json`;
      exports[`./profiles/*/${extension}.functions`] = { types: `${prefix}/profiles/*/${extension}.functions.d.ts` };
    }
    break;
  case 'stable-diffusion-cpp': break;
  default: {
    const exhaustive: never = runtime;
    throw new Error(String(exhaustive));
  }
  }
  return exports;
}

export function packageExports(): ExportMap {
  const exports: ExportMap = { './manifest.json': './manifest.json' };
  for (const runtime of runtimeNames) {
    for (const [name, value] of Object.entries(runtimeExports({ runtime, prefix: `./${artifactDirectories[runtime]}` }))) {
      exports[`./${runtime}/${name.slice(2)}`] = value;
    }
  }
  return exports;
}
