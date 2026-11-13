"""Public package names, physical paths and explicit runtime export maps."""
from __future__ import annotations

PACKAGE_NAME = 'browser-inference-core'
MANIFEST_VERSION = 4
RUNTIME_MANIFEST_VERSION = 2
RUNTIME_NAMES = {'llama-cpp': 'llama-cpp-browser-core',
                 'stable-diffusion-cpp': 'stable-diffusion-cpp-browser-core'}
# Logical runtime identifiers deliberately differ from published directory names.
# Source checkouts and import specifiers keep the shorter identifiers.
ARTIFACT_DIRS = {runtime: 'runtimes/' + name for runtime, name in RUNTIME_NAMES.items()}


def runtime_exports(runtime: str, prefix: str = '.') -> dict:
    if runtime not in RUNTIME_NAMES:
        raise ValueError('Unknown runtime export map')
    exports = {
        './host': {'types': prefix + '/host/index.d.ts', 'import': prefix + '/host/index.mjs'},
        './api/functions': {'types': prefix + '/api/functions.d.ts'},
        './api/schema.json': prefix + '/api/schema.json',
        './api/schema.mjs': prefix + '/api/schema.mjs',
        './api/exports.json': prefix + '/api/exports.json',
        './profiles/*/core.mjs': {'types': prefix + '/profiles/*/core.d.ts',
                                 'import': prefix + '/profiles/*/core.mjs'},
        './profiles/*/core.wasm': prefix + '/profiles/*/core.wasm',
        './manifest.json': prefix + '/manifest.json',
        './native-license-index.json': prefix + '/native-license-index.json',
        './licenses/*': prefix + '/licenses/*',
    }
    if runtime == 'llama-cpp':
        exports.update({
            './profiles/*/model-graph-v1.schema.json': prefix + '/profiles/*/model-graph-v1.schema.json',
            './profiles/*/model-graph-v1.functions': {'types': prefix + '/profiles/*/model-graph-v1.functions.d.ts'},
            './profiles/*/q8-diagnostic-v1.schema.json': prefix + '/profiles/*/q8-diagnostic-v1.schema.json',
            './profiles/*/q8-diagnostic-v1.functions': {'types': prefix + '/profiles/*/q8-diagnostic-v1.functions.d.ts'},
        })
    return exports


def package_exports() -> dict:
    exports = {'./manifest.json': './manifest.json'}
    for runtime, directory in ARTIFACT_DIRS.items():
        for name, value in runtime_exports(runtime, './' + directory).items():
            exports['./' + runtime + '/' + name.removeprefix('./')] = value
    return exports
