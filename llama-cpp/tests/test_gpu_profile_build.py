"""Profile isolation and compiled-command validation; no claim of GPU execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_gpu_profile_build import verify


class GpuProfileBuild(unittest.TestCase):
    def test_only_diagnostic_profile_enables_upstream_profiling(self):
        profiles = json.loads((ROOT / 'config/profiles.json').read_text())
        self.assertEqual(len(profiles), 5)
        self.assertNotIn('cpu-wasm64', profiles)
        diagnostic = profiles['webgpu-wasm64-jspi-gpu-profile'].copy()
        self.assertTrue(diagnostic.pop('gpuProfile'))
        self.assertEqual(diagnostic, profiles['webgpu-wasm64-jspi'])
        self.assertEqual([key for key, value in profiles.items() if value.get('gpuProfile')],
                         ['webgpu-wasm64-jspi-gpu-profile'])

    def test_backend_definition_must_match_and_have_value_one(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'compile_commands.json'
            for enabled in (False, True):
                for flag in ('', '-DGGML_WEBGPU_GPU_PROFILE=1', '-DGGML_WEBGPU_GPU_PROFILE=0',
                             '-DGGML_WEBGPU_GPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE_EXTRA=1'):
                    with self.subTest(enabled=enabled, flag=flag):
                        database.write_text(json.dumps([{'file': '/build/ggml-webgpu.cpp',
                                                         'command': 'em++ -c ggml-webgpu.cpp ' + flag}]))
                        if flag == ('-DGGML_WEBGPU_GPU_PROFILE=1' if enabled else ''):
                            verify(database, webgpu=True, enabled=enabled)
                        else:
                            with self.assertRaisesRegex(ValueError, 'definition mismatch'):
                                verify(database, webgpu=True, enabled=enabled)

    def test_missing_and_duplicate_translation_units_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'compile_commands.json'
            entry = {'file': 'generated/ggml-webgpu.cpp', 'arguments': ['em++', '-DGGML_WEBGPU_GPU_PROFILE=1']}
            for entries in ([], [entry, entry]):
                database.write_text(json.dumps(entries))
                with self.assertRaisesRegex(ValueError, 'translation units'):
                    verify(database, webgpu=True, enabled=True)
            database.write_text('[]')
            verify(database, webgpu=False, enabled=False)
            with self.assertRaisesRegex(ValueError, 'requires a WebGPU'):
                verify(database, webgpu=False, enabled=True)
            database.write_text(json.dumps([entry]))
            verify(database, webgpu=True, enabled=True)


if __name__ == '__main__':
    unittest.main()
