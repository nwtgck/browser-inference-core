# Synthetic Wasm headers below test packaging ONLY, never inference/build success.
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import REPOSITORY_ROOT as ROOT
from build_tools.packaging import assemble as package
from tests.support.runtime_package import (
    RuntimePackageFixture, SOURCE, package, schema_fixture, schema_hash, schema_file, write_manifest,
)
from build_tools.packaging.files import tree_identity
from tests.support.packaging_cli import run_cli

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
        # A fresh isolated interpreter receives only the project package root,
        # not runtime script directories or another test's imported modules.
        command = '''
import sys
sys.path.insert(0, sys.argv[1])
from build_tools.packaging import assemble as module
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
            result = subprocess.run([sys.executable, '-I', '-B', '-c', command, str(ROOT)],
                                    cwd=temporary, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'runtime validator imported and checked')

    def test_unselected_runtime_is_not_imported_even_when_it_is_unavailable(self):
        command = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
selected, forbidden = sys.argv[2:]
class RejectOtherRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == forbidden or fullname.startswith(forbidden + '.'):
            raise AssertionError('Unselected runtime imported: ' + fullname)
sys.meta_path.insert(0, RejectOtherRuntime())
from build_tools.packaging.assemble import runtime_module
assert callable(runtime_module(selected).validate)
assert not any(n == forbidden or n.startswith(forbidden + '.') for n in sys.modules)
"""
        for selected, other in [('llama-cpp', 'stable_diffusion_cpp'), ('stable-diffusion-cpp', 'llama_cpp')]:
            with self.subTest(runtime=selected), tempfile.TemporaryDirectory() as cwd:
                result = subprocess.run([sys.executable, '-I', '-B', '-c', command, str(ROOT),
                                         selected, 'build_tools.runtimes.' + other],
                                        cwd=cwd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class MultiRuntime(RuntimePackageFixture):
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


    def test_aggregate_cli_validates_all_three_packages_before_activation(self):
        arguments = ['--inputs', str(self.inputs), '--output', str(self.out)]
        result, calls = run_cli(self.root, 'build_tools.packaging.assemble', arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(not Path(call['cwd']).is_relative_to(self.out) for call in calls))
        self.assertEqual(Path(calls[-1]['cwd']).name, 'prepared')
        before, inode = tree_identity(self.out), self.out.stat().st_ino
        result, calls = run_cli(self.root, 'build_tools.packaging.assemble', arguments, fail_pack=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('injected npm pack failure', result.stderr)
        self.assertEqual(tree_identity(self.out), before)
        self.assertEqual(self.out.stat().st_ino, inode)
        self.assertEqual(len(calls), 3)
        self.assertEqual(list(self.out.parent.glob('bicore-output-*')), [])

    def test_aggregate_cli_defer_skips_compression_and_verify_cannot_defer(self):
        arguments = ['--inputs', str(self.inputs), '--output', str(self.out), '--defer-npm-pack']
        result, calls = run_cli(self.root, 'build_tools.packaging.assemble', arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])
        result, calls = run_cli(self.root, 'build_tools.packaging.assemble', [*arguments, '--verify-only'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('always includes npm packing', result.stderr)
        self.assertEqual(calls, [])

    def test_complete_package_exposes_only_explicit_runtime_namespaces(self):
        package.assemble(self.inputs, self.out)
        result = package.validate(self.out)
        self.assertEqual(result['formatVersion'], 4)
        self.assertEqual(set(result['runtimes']), set(package.RUNTIMES))
        self.assertFalse((self.out / 'profiles').exists())
        self.assertEqual({path.name for path in self.out.iterdir() if path.is_dir()}, {'runtimes'})
        self.assertEqual({path.name for path in (self.out / 'runtimes').iterdir()},
                         {'llama-cpp-browser-core', 'stable-diffusion-cpp-browser-core'})
        self.assertEqual(result['runtimes']['llama-cpp']['manifest'], 'runtimes/llama-cpp-browser-core/manifest.json')
        self.assertEqual(result['runtimes']['stable-diffusion-cpp']['manifest'], 'runtimes/stable-diffusion-cpp-browser-core/manifest.json')
        subprocess.run(['node', '--input-type=module', '-e', """
          const named = await import('browser-inference-core/llama-cpp/profiles/cpu-wasm32/browser/core.mjs');
          const image = await import('browser-inference-core/stable-diffusion-cpp/profiles/webgpu-wasm32-asyncify/browser/core.mjs');
          if (named.default() !== 'fixture' || image.default() !== 'fixture') throw Error('Wrong export mapping');
          for (const path of ['profiles/cpu-wasm32/browser/core.mjs', 'api/schema.json', 'examples/runtime',
                              'llama-cpp/host/index.mjs', 'llama-cpp/vendor/llama.cpp', 'stable-diffusion-cpp/tests/native.cpp']) {
            try { await import('browser-inference-core/' + path); }
            catch (error) { if (error.code === 'ERR_PACKAGE_PATH_NOT_EXPORTED') continue; throw error; }
            throw Error('Unexpected public export: ' + path);
          }
        """], cwd=self.out, check=True)
    def test_consumer_report_rebases_paths_and_keeps_image_provenance_bounded(self):
        from build_tools.publication import consumer as reporter
        package.assemble(self.inputs, self.out)
        # No network, npm lock generation, or real upstream build is claimed.
        data = reporter.metadata(self.out, 'example/lcore', 'd' * 40, {'specifier': 'github:example/lcore#' + 'd' * 40}, {'baseCommit': 'b' * 40})
        self.assertTrue(data['retrieval']['sourceRawBase'].endswith('/llama-cpp/'))
        self.assertEqual(data['retrieval']['sourceRawBase'], data['retrieval']['sourceRepositoryRawBase']+'runtimes/llama-cpp/')
        self.assertEqual(data['runtime']['manifestFormatVersion'], 4)
        self.assertTrue(data['browserProfiles']['cpu-wasm32']['wasm']['path'].startswith('runtimes/llama-cpp-browser-core/profiles/'))
        self.assertEqual(data['runtime']['llamaManifestPath'], 'runtimes/llama-cpp-browser-core/manifest.json')
        sd = data['stableDiffusion']
        self.assertEqual(sd['manifest']['path'], 'runtimes/stable-diffusion-cpp-browser-core/manifest.json')
        self.assertEqual(sd['packageExports']['host'], 'browser-inference-core/stable-diffusion-cpp/host')
        self.assertEqual(sd['packageExports']['profiles'], 'browser-inference-core/stable-diffusion-cpp/profiles/')
        self.assertIn('independent modules', sd['consumerBoundary'])
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
    def test_tampering_is_detected(self):
        package.assemble(self.inputs, self.out)
        (self.out / 'runtimes/stable-diffusion-cpp-browser-core/profiles/webgpu-wasm32-asyncify/browser/core.wasm').write_bytes(b'invalid')
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
