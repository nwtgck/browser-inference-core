# Synthetic Wasm headers below test packaging ONLY, never inference/build success.
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
from build_tools.packaging import assemble as package
from build_tools.packaging.layout import runtime_exports
from build_tools.packaging.types import write_host_profiles
SOURCE = 'a' * 40

def schema_fixture():
    return {'abiVersion': 2, 'functions': [{
        'name': 'sd_browser_bf16_retention_version',
        'export': '_sdc_sd_browser_bf16_retention_version',
        'returnType': 'uint32_t', 'returnKind': 'unsigned', 'parameters': [],
    }], 'records': [], 'constants': ['SD_TYPE_BF16']}


def schema_hash():
    return hashlib.sha256(json.dumps(schema_fixture()).encode()).hexdigest()

def schema_file(name):
    if name == 'schema.json': return json.dumps(schema_fixture())
    if name == 'schema.mjs': return 'export default ' + json.dumps({**schema_fixture(), 'schemaSha256': schema_hash()}, separators=(',', ':')) + ';\n'
    return 'fixture declarations'

def image_patch_provenance():
    # These deterministic hashes describe synthetic packaging inputs only.
    patches = [
        ('0036-webgpu-bf16-weight-kernels.patch', 'ggml'),
        ('0037-webgpu-bounded-tensor-uploads.patch', 'ggml'),
        ('0038-browser-bounded-model-reader.patch', 'stable-diffusion'),
        ('0039-browser-bf16-weight-retention.patch', 'stable-diffusion'),
    ]
    return {'patches': [{'file': name, 'target': target,
                         'sha256': hashlib.sha256(('fixture ' + name).encode()).hexdigest(), 'fixtureOnly': True}
                        for name, target in patches]}


def write_manifest(root, data):
    data['files'] = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                      'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                     for p in sorted(root.rglob('*')) if p.is_file() and p != root / 'manifest.json']
    (root / 'manifest.json').write_text(json.dumps(data))

class RuntimePackageFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs'; self.out = self.root / 'out'
        for runtime in package.RUNTIMES: self.fixture(runtime)
    def tearDown(self): self.tmp.cleanup()
    def fixture(self, runtime):
        root = self.inputs / runtime; root.mkdir(parents=True)
        validator = package.runtime_module(runtime)
        def put(relative, value):
            p = root / relative; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(value if isinstance(value, bytes) else value.encode())
        put('LICENSE', 'fixture notice')
        put('package.json', json.dumps({'name': 'llama-cpp-browser-core' if runtime == 'llama-cpp' else validator.NAME,
                                      'version': '0.1.0', 'type': 'module', 'exports': runtime_exports(runtime),
                                      'files': ['profiles/', 'licenses/', 'api/', 'host/', 'LICENSE', 'manifest.json', 'native-license-index.json']}))
        manifest = {'sourceCommit': SOURCE, 'profiles': {}}
        if runtime == 'llama-cpp':
            manifest.update(formatVersion=2, llamaCommit='b' * 40, licenseRoots=[])
            for relative in validator.EMBEDDED_NOTICE_FILES: put('licenses/embedded/' + relative + '.txt', 'fixture embedded notice')
            for name in validator.HOST_FILES: put('host/' + name, 'fixture')
            for name in ('schema.json', 'schema.mjs', 'functions.d.ts', 'exports.json'): put('api/' + name, schema_file(name))
            profiles = {'cpu-wasm32': {}}
        else:
            manifest.update(formatVersion=2, abiVersion=2, schemaSha256=schema_hash(), capabilities=validator.CAPABILITIES, runtime=runtime, upstreams={'fixture': 'b' * 40}, experimental=True)
            for name in ['stable-diffusion/LICENSE', 'ggml/LICENSE', 'toolchain/emscripten/LICENSE', 'toolchain/emdawnwebgpu_pkg/LICENSE', *['embedded/' + p + '.txt' for p in ['json.hpp', 'stb_image.h', 'stb_image_resize.h', 'stb_image_write.h']]]: put('licenses/' + name, 'fixture notice')
            for name in validator.API_FILES: put('api/'+name, schema_file(name))
            for name in validator.HELPERS: put('host/'+name, 'fixture helper')
            profiles = validator.PROFILES
        for profile, config in profiles.items():
            variants = {}
            for variant, settings in validator.VARIANTS.items():
                prefix = f'profiles/{profile}/{variant}/'
                put(prefix + 'core.wasm', b'\0asm\1\0\0\0')
                put(prefix + 'core.mjs', 'export default () => "fixture";')
                put(prefix + 'core.d.ts', 'export default function create(): string;')
                data = {'profile': profile, 'variant': variant, 'sourceCommit': SOURCE, 'sourceDirty': False,
                        'variantConfiguration': settings, 'validation': {'compiled': True, 'fixtureOnly': True, 'realModelInference': False}}
                if runtime == 'llama-cpp': data['llamaCommit'] = manifest['llamaCommit']
                else: data.update(upstreams=manifest['upstreams'], configuration=config, patches=image_patch_provenance())
                variants[variant] = data
            manifest['profiles'][profile] = {'variants': variants}
        write_host_profiles(root / 'host/profiles.d.ts', runtime, profiles, validator.VARIANTS)
        validator.write_index(root, runtime=runtime, license_roots=manifest.get('licenseRoots', []))
        write_manifest(root, manifest)
