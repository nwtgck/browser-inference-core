# Synthetic Wasm headers below test packaging ONLY, never inference/build success.
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('multi_runtime_package', ROOT / 'scripts/package_runtime.py')
package = importlib.util.module_from_spec(spec); spec.loader.exec_module(package)
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

class RuntimeValidatorImports(unittest.TestCase):
    def test_image_retention_capability_requires_the_matching_schema(self):
        validator = package.runtime_module('stable-diffusion-cpp')
        self.assertIs(validator.CAPABILITIES['bf16WeightRetention'], True)
        self.assertIs(validator.CAPABILITIES['ggufOriginalShapes'], True)
        validator.validate_bf16_retention_schema(schema_fixture())
        for key, value in [('returnType', 'int'), ('returnKind', 'signed'),
                           ('parameters', [{'kind': 'pointer', 'cType': 'void *'}]),
                           ('export', '_other')]:
            with self.subTest(field=key):
                schema = schema_fixture()
                schema['functions'][0][key] = value
                with self.assertRaisesRegex(ValueError, 'BF16 retention capability'):
                    validator.validate_bf16_retention_schema(schema)
        missing = schema_fixture(); missing['functions'] = []
        duplicate = schema_fixture(); duplicate['functions'].append(dict(duplicate['functions'][0]))
        missing_type = schema_fixture(); missing_type['constants'] = []
        for name, schema in [('missing function', missing), ('duplicate function', duplicate), ('missing type', missing_type)]:
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, 'BF16 retention capability'):
                    validator.validate_bf16_retention_schema(schema)

    def test_llama_validator_loads_from_root_with_no_runtime_scripts_on_sys_path(self):
        # A fresh isolated interpreter cannot inherit imports from earlier tests.
        command = '''
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('root_package', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
validator = module.runtime_module('llama-cpp')
assert callable(validator.validate)
validator.validate_bf16_metadata({})
try:
    validator.validate_bf16_metadata({'visionBf16Storage': {'version': 999}})
except ValueError:
    pass
else:
    raise AssertionError('The real BF16 validator must still reject incompatible metadata')
print('runtime validator imported and checked')
'''
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, '-I', '-B', '-c', command, str(ROOT / 'scripts/package_runtime.py')],
                                    cwd=temporary, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'runtime validator imported and checked')


class MultiRuntime(unittest.TestCase):
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
                                      'version': '0.1.0', 'type': 'module', 'files': ['profiles/', 'licenses/', 'api/', 'examples/', 'LICENSE', 'manifest.json', 'native-license-index.json']}))
        manifest = {'sourceCommit': SOURCE, 'profiles': {}}
        if runtime == 'llama-cpp':
            manifest.update(formatVersion=2, llamaCommit='b' * 40, licenseRoots=[])
            for relative in validator.EMBEDDED_NOTICE_FILES: put('licenses/embedded/' + relative + '.txt', 'fixture embedded notice')
            for name in validator.EXAMPLE_RUNTIME_FILES: put('examples/runtime/' + name, 'fixture')
            profiles = {'cpu-wasm32': {}}
        else:
            manifest.update(formatVersion=2, abiVersion=2, schemaSha256=schema_hash(), capabilities=validator.CAPABILITIES, runtime=runtime, upstreams={'fixture': 'b' * 40}, experimental=True)
            for name in ['stable-diffusion/LICENSE', 'ggml/LICENSE', 'toolchain/emscripten/LICENSE', 'toolchain/emdawnwebgpu_pkg/LICENSE', *['embedded/' + p + '.txt' for p in ['json.hpp', 'stb_image.h', 'stb_image_resize.h', 'stb_image_write.h']]]: put('licenses/' + name, 'fixture notice')
            for name in validator.API_FILES: put('api/'+name, schema_file(name))
            for name in validator.HELPERS: put('examples/runtime/'+name, 'fixture helper')
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
        validator.write_index(root, runtime=runtime, license_roots=manifest.get('licenseRoots', []))
        write_manifest(root, manifest)
    def test_complete_package_has_namespaces_and_resolvable_legacy_alias(self):
        package.assemble(self.inputs, self.out)
        result = package.validate(self.out)
        self.assertEqual(result['formatVersion'], 3)
        self.assertEqual(set(result['runtimes']), set(package.RUNTIMES))
        self.assertFalse((self.out / 'profiles').exists())
        self.assertEqual({path.name for path in self.out.iterdir() if path.is_dir()}, set(package.ARTIFACT_DIRS.values()))
        self.assertEqual(result['runtimes']['llama-cpp']['manifest'], 'llama-cpp-browser-core/manifest.json')
        self.assertEqual(result['runtimes']['stable-diffusion-cpp']['manifest'], 'stable-diffusion-cpp-browser-core/manifest.json')
        subprocess.run(['node', '--input-type=module', '-e', """
          const old = await import('llama-cpp-browser-core/profiles/cpu-wasm32/browser/core.mjs');
          const named = await import('llama-cpp-browser-core/llama-cpp/profiles/cpu-wasm32/browser/core.mjs');
          const image = await import('llama-cpp-browser-core/stable-diffusion-cpp/profiles/webgpu-wasm32-asyncify/browser/core.mjs');
          if (old.default !== named.default || image.default() !== 'fixture') throw Error('Wrong export mapping');
        """], cwd=self.out, check=True)
    def test_consumer_report_rebases_paths_and_keeps_image_provenance_bounded(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        spec = importlib.util.spec_from_file_location('multi_consumer', ROOT / 'scripts/consumer_metadata.py')
        reporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(reporter)
        package.assemble(self.inputs, self.out)
        # No network, npm lock generation, or real upstream build is claimed.
        data = reporter.metadata(self.out, 'example/lcore', 'd' * 40, {'specifier': 'github:example/lcore#' + 'd' * 40}, {'baseCommit': 'b' * 40})
        self.assertTrue(data['retrieval']['sourceRawBase'].endswith('/llama-cpp/'))
        self.assertEqual(data['retrieval']['sourceRawBase'], data['retrieval']['sourceRepositoryRawBase']+'llama-cpp/')
        self.assertEqual(data['runtime']['manifestFormatVersion'], 3)
        self.assertTrue(data['browserProfiles']['cpu-wasm32']['wasm']['path'].startswith('llama-cpp-browser-core/profiles/'))
        self.assertEqual(data['runtime']['llamaManifestPath'], 'llama-cpp-browser-core/manifest.json')
        sd = data['stableDiffusion']
        self.assertEqual(sd['manifest']['path'], 'stable-diffusion-cpp-browser-core/manifest.json')
        self.assertIn('#' + 'd' * 40, sd['installSeparatelyForNaidan'])
        for profile in sd['profiles'].values():
            for item in profile['variants'].values():
                self.assertEqual(set(item), {'validation', 'sourceCommit'})
                self.assertFalse(item['validation']['realModelInference'])
        text = reporter.legacy.write_report(self.root / 'report', data, '123', '1')
        self.assertLess(len(text.encode('utf8')), 55000)

    def test_mixed_sources_cannot_assemble(self):
        root = self.inputs / 'stable-diffusion-cpp'
        m = json.loads((root / 'manifest.json').read_text()); m['sourceCommit'] = 'c' * 40
        for info in m['profiles'].values():
            for data in info['variants'].values(): data['sourceCommit'] = 'c' * 40
        write_manifest(root, m)
        with self.assertRaisesRegex(ValueError, 'mix'): package.assemble(self.inputs, self.out)
    def test_image_retention_requires_complete_ordered_matching_patch_provenance(self):
        root = self.inputs / 'stable-diffusion-cpp'
        validator = package.runtime_module('stable-diffusion-cpp')
        original = json.loads((root / 'manifest.json').read_text())
        for case in ['missing', 'duplicate', 'order', 'hash', 'target', 'variant-identity']:
            manifest = json.loads(json.dumps(original))
            entries = manifest['profiles']['webgpu-wasm32-jspi']['variants']['test']['patches']['patches']
            if case == 'missing':
                entries.pop(1)
            elif case == 'duplicate':
                entries.append(dict(entries[0]))
            elif case == 'order':
                entries[0], entries[1] = entries[1], entries[0]
            elif case == 'hash':
                entries[0]['sha256'] = 'not a SHA-256'
            elif case == 'target':
                entries[0]['target'] = 'stable-diffusion'
            elif case == 'variant-identity':
                entries[0]['sha256'] = 'c' * 64
            else:
                self.fail('Unknown test mutation')
            write_manifest(root, manifest)
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, 'BF16 retention patch'):
                validator.validate(root, check_npm_pack=False)
    def test_tampering_is_detected(self):
        package.assemble(self.inputs, self.out)
        (self.out / 'stable-diffusion-cpp-browser-core/profiles/webgpu-wasm32-asyncify/browser/core.wasm').write_bytes(b'invalid')
        with self.assertRaisesRegex(ValueError, 'hash/size'): package.validate(self.out)
    def test_missing_variant_is_not_publishable(self):
        root = self.inputs / 'stable-diffusion-cpp'; m = json.loads((root / 'manifest.json').read_text())
        del m['profiles']['webgpu-wasm32-jspi']['variants']['test']; write_manifest(root, m)
        with self.assertRaisesRegex(ValueError, 'variants'): package.assemble(self.inputs, self.out)
    def test_dirty_image_source_blocks_entire_publication(self):
        root = self.inputs / 'stable-diffusion-cpp'; m = json.loads((root / 'manifest.json').read_text())
        m['profiles']['webgpu-wasm32-jspi']['variants']['test']['sourceDirty'] = True; write_manifest(root, m)
        with self.assertRaisesRegex(ValueError, 'dirty'): package.assemble(self.inputs, self.out)
    def test_assembly_requires_both_runtimes(self):
        import shutil
        shutil.rmtree(self.inputs / 'stable-diffusion-cpp')
        with self.assertRaises(FileNotFoundError): package.assemble(self.inputs, self.out)

if __name__ == '__main__': unittest.main()
