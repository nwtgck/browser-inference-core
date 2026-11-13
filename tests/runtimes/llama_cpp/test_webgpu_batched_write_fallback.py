"""Exact browser write fallback, current canonical kernels and generic API controls."""
import hashlib
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
sys.path[:0] = [str(ROOT.parents[1]), str(runtime_test_root('llama-cpp'))]
from tests.support import legacy_webgpu_source as source
from tests.support.llama_cpp.webgpu_stages import batched_write_fallback as fallback
from tests.runtimes.llama_cpp.test_webgpu_buffer_limit import definition
from tests.runtimes.llama_cpp.webgpu_prefix_context import without_batched_write_fallback
from tests.runtimes.llama_cpp.webgpu_embedding_test_support import legacy_fixed_sources


def generic_fixture(prepared, upstream):
    text = (prepared / 'ggml-webgpu.cpp').read_text()
    hook = re.search(r'#ifdef __EMSCRIPTEN__\n    // Keep scalar writes.*?\n#endif', text, re.S)
    if hook is None:
        raise ValueError('Missing checked browser write hook')
    backend = (upstream / 'ggml/src/ggml-backend.cpp').read_text()
    definitions = [definition(backend, 'template <typename cb_get_buffer>'),
                   definition(backend, 'static bool ggml_backend_tensor_batch_is_2d('),
                   definition(backend, 'void ggml_backend_tensor_set('),
                   definition(backend, 'void ggml_backend_tensor_set_2d('),
                   definition(backend, 'void ggml_backend_tensor_set_n(')]
    template = (runtime_test_root('llama-cpp') / 'webgpu-batched-write-fallback/generic.cpp.in').read_text()
    return template.replace('@HOOK@', hook.group()).replace('@GENERIC@', '\n\n'.join(definitions))


class BatchedWriteFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temporary = tempfile.TemporaryDirectory(prefix='batched-write-fallback-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.final = {}
        cls.prefix = {}
        for profile in (False, True):
            with without_batched_write_fallback(source):
                cls.prefix[profile] = source.prepare(cls.upstream, cls.work / f'prefix-{profile}', gpu_profile=profile).parent
            cls.final[profile] = source.prepare(cls.upstream, cls.work / f'final-{profile}', gpu_profile=profile).parent

    def test_exact_hook_inverse_and_unchanged_upstream_batch_implementations(self):
        pristine = (self.upstream / source.SOURCE_PATH).read_text()
        self.assertEqual(fallback.INPUTS, source.buffer_limit.OUTPUTS)
        for profile in (False, True):
            prefix, final = self.prefix[profile], self.final[profile]
            changed = {p.name for p in prefix.iterdir() if p.read_bytes() != (final / p.name).read_bytes()}
            self.assertEqual(changed, {'ggml-webgpu.cpp'})
            text = (final / 'ggml-webgpu.cpp').read_text()
            self.assertEqual(source.tensor_copy.digest(final / 'ggml-webgpu.cpp'), fallback.OUTPUTS[profile]['ggml-webgpu.cpp'])
            for marker in ['static void ggml_backend_webgpu_buffer_set_tensor_n(',
                           'static void ggml_backend_webgpu_buffer_get_tensor_n(']:
                self.assertEqual(definition(text, marker), definition(pristine, marker))
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                target = work / source.SOURCE_PATH
                target.parent.mkdir(parents=True)
                target.write_text(text)
                for flags in (['--check'], []):
                    subprocess.run(['git', 'apply', '--no-index', '-R', *flags,
                                    str(ROOT / source.PATCH_DIRECTORY / fallback.PATCH_NAME)],
                                   cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                self.assertEqual(target.read_bytes(), (prefix / target.name).read_bytes())
            stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            source.prepare(self.upstream, final, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})

    def test_current_kernels_survive_final_canonical_embedding(self):
        shaders = self.upstream / 'ggml/src/ggml-webgpu/wgsl-shaders'
        embed = runpy.run_path(str(shaders / 'embed_wgsl.py'))
        for final in self.final.values():
            embedding = (final / source.aligned_word_load.HEADER_NAME).read_text()
            for name in ['count_equal', 'group_norm', 'set_tensor_n', 'sum_rows']:
                self.assertIn('wgsl_' + name, embedding)
                original = (shaders / (name + '.wgsl')).read_text()
                self.assertIn(embed['expand_includes'](original, str(shaders)), embedding)
            backend = (final / 'ggml-backend.cpp').read_text()
            self.assertIn('ggml_backend_sched_prefetch_inputs(sched, split_id);', backend)
            prefetch = backend.index('// read before the previous split, wait for the copy only')
            snapshot = backend.index('// Only an explicit opt-in can consume host bytes immediately')
            self.assertLess(prefetch, snapshot)

    def test_contract_patch_input_output_drift_is_atomic(self):
        for profile in (False, True):
            existing = {p.name: p.read_bytes() for p in self.final[profile].iterdir()}
            cases = [('CONTRACT_INPUTS', name) for name in fallback.CONTRACT_INPUTS]
            cases += [(field, 'ggml-webgpu.cpp') for field in ['INPUTS', 'OUTPUTS']]
            cases += [('PATCH_SHA256', None)]
            for field, key in cases:
                with self.subTest(profile=profile, field=field, key=key):
                    if field == 'CONTRACT_INPUTS':
                        value = {**fallback.CONTRACT_INPUTS, key: '0' * 64}
                    elif field in ['INPUTS', 'OUTPUTS']:
                        value = {**getattr(fallback, field), profile: {key: '0' * 64}}
                    else:
                        value = '0' * 64
                    with patch.object(fallback, field, value), self.assertRaises(ValueError):
                        source.prepare(self.upstream, self.final[profile], gpu_profile=profile)
                    self.assertEqual(existing, {p.name: p.read_bytes() for p in self.final[profile].iterdir()})

    def test_final_legacy_inverse_rejects_patch_and_source_drift(self):
        for profile, final in self.final.items():
            result = legacy_fixed_sources(final)
            for name, expected in source.subgroup_size_control.INPUTS[profile].items():
                self.assertEqual(hashlib.sha256(result[name].encode()).hexdigest(), expected)
            with patch.object(fallback, 'PATCH_SHA256', '0' * 64):
                with self.assertRaisesRegex(ValueError, 'batched-write prefix patch identity'):
                    legacy_fixed_sources(final)
            changed = self.work / f'drift-{profile}'
            shutil.copytree(final, changed)
            with (changed / 'ggml-webgpu.cpp').open('a') as stream:
                stream.write('\n// unreviewed drift\n')
            with self.assertRaisesRegex(ValueError, 'exact optional-stage composition'):
                legacy_fixed_sources(changed)

    def test_actual_generic_api_preserves_native_hook_and_browser_scalar_order(self):
        fixture = self.work / 'generic.cpp'
        fixture.write_text(generic_fixture(self.final[False], self.upstream))
        for browser in (False, True):
            binary = self.work / f'generic-{browser}'
            command = ['c++', '-std=c++17', '-O1', '-fsanitize=undefined', '-fno-sanitize-recover=all']
            if browser:
                command.append('-D__EMSCRIPTEN__')
            subprocess.run([*command, str(fixture), '-o', str(binary)], check=True, capture_output=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn('"passed":true', result.stdout)
            self.assertIn('"browser":' + str(browser).lower(), result.stdout)


if __name__ == '__main__':
    unittest.main()
