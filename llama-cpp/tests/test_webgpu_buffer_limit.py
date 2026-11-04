"""Check the actual allocation-limit getter without allocating large buffers."""
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_buffer_limit as limit
import prepare_webgpu_source as webgpu


def definition(text, marker):
    start = text.index(marker)
    at = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[at] == '{') - (text[at] == '}')
        at += 1
    return text[start:at]


def getter_fixture(prepared):
    body = definition(prepared.read_text(), 'static size_t ggml_backend_webgpu_buffer_type_get_max_size(')
    template = (ROOT / 'tests/webgpu-buffer-limit/limits.cpp.in').read_text()
    return template.replace('@GETTER@', body)


def memset_fixture(prepared):
    text = prepared.read_text()
    template = (ROOT / 'tests/webgpu-buffer-limit/memset.cpp.in').read_text()
    macros = []
    for name in ['ROUNDUP_POW2', 'CEIL_DIV', 'WEBGPU_PARAMS_BUF_SIZE_BYTES', 'WEBGPU_STORAGE_BUF_BINDING_MULT']:
        matches = re.findall(r'^#define ' + name + r'(?:\([^\n]*|\s[^\n]*)$', text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError('Memset macro source changed: ' + name)
        macros.append(matches[0])
    return (template.replace('@MACROS@', '\n'.join(macros))
            .replace('@WORKGROUPS@', definition(text, 'static inline void compute_2d_workgroups('))
            .replace('@BINDING@', definition(text, 'static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry('))
            .replace('@MEMSET@', definition(text, 'static void ggml_backend_webgpu_buffer_memset(')))


class BufferLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='lcb-buffer-limit-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.source = ROOT / 'vendor/llama.cpp'
        cls.prepared = {p: webgpu.prepare(cls.source, cls.work / str(p), gpu_profile=p) for p in [False, True]}

    def test_actual_getter_at_native_width(self):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        self.assertIsNotNone(compiler, 'A C++ compiler is required for the actual getter contract')
        source = self.work / 'getter.cpp'
        source.write_text(getter_fixture(self.prepared[False]))
        binary = self.work / 'getter'
        subprocess.run([compiler, '-std=c++17', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        str(source), '-o', str(binary)], check=True, capture_output=True)
        result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        self.assertIn('"passed":true', result.stdout)

    def test_actual_memset_queue_and_binding_metadata(self):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        self.assertIsNotNone(compiler, 'A C++ compiler is required for the actual memset recorder')
        source = self.work / 'memset.cpp'
        source.write_text(memset_fixture(self.prepared[False]))
        binary = self.work / 'memset'
        subprocess.run([compiler, '-std=c++17', '-fsanitize=undefined', '-fno-sanitize-recover=all',
                        str(source), '-o', str(binary)], check=True, capture_output=True)
        result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        self.assertIn('"passed":true', result.stdout)

    def test_both_profiles_preserve_upstream_memset_and_all_non_getter_bytes(self):
        upstream = (self.source / webgpu.SOURCE_PATH).read_text()
        memset_marker = 'static void ggml_backend_webgpu_buffer_memset('
        for profile, path in self.prepared.items():
            with self.subTest(profile=profile):
                self.assertEqual(definition(path.read_text(), memset_marker), definition(upstream, memset_marker))
                prefix = self.work / ('prefix-' + str(profile)) / webgpu.SOURCE_PATH
                prefix.parent.mkdir(parents=True)
                shutil.copyfile(path, prefix)
                env = webgpu.patch_environment(prefix.parents[3])
                subprocess.run(['git', 'apply', '--no-index', '--reverse', '--check',
                                str(ROOT / webgpu.PATCH_DIRECTORY / limit.PATCH_NAME)],
                               cwd=prefix.parents[3], env=env, check=True, capture_output=True)
                subprocess.run(['git', 'apply', '--no-index', '--reverse',
                                str(ROOT / webgpu.PATCH_DIRECTORY / limit.PATCH_NAME)],
                               cwd=prefix.parents[3], env=env, check=True, capture_output=True)
                self.assertEqual(webgpu.tensor_copy.digest(prefix), limit.INPUTS[profile][prefix.name])
                marker = 'static size_t ggml_backend_webgpu_buffer_type_get_max_size('
                self.assertEqual(path.read_text().replace(definition(path.read_text(), marker), ''),
                                 prefix.read_text().replace(definition(prefix.read_text(), marker), ''))

    def test_historical_prefixes_keep_exact_stage_identities(self):
        import webgpu_prefix_context as prefixes
        cases = [(prefixes.without_shared_word_load, webgpu.q4_subgroup.OUTPUTS),
                 (prefixes.without_checkpoint_readback, webgpu.aligned_word_load.OUTPUTS),
                 (prefixes.without_subgroup_size_control, webgpu.checkpoint_readback.OUTPUTS),
                 (prefixes.without_q6_subgroup, webgpu.subgroup_size_control.OUTPUTS),
                 (prefixes.without_buffer_limit, webgpu.q6_subgroup.OUTPUTS)]
        for context, expected in cases:
            for profile in [False, True]:
                with self.subTest(context=context.__name__, profile=profile), context(webgpu):
                    output = webgpu.prepare(self.source, self.work / (context.__name__ + str(profile)),
                                            gpu_profile=profile).parent
                    for name, digest in expected[profile].items():
                        self.assertEqual(webgpu.tensor_copy.digest(output / name), digest)

    def test_legacy_fixed_inverse_is_exact_and_rejects_drift(self):
        from webgpu_embedding_test_support import legacy_fixed_sources
        for profile, prepared in self.prepared.items():
            with self.subTest(profile=profile):
                result = legacy_fixed_sources(prepared.parent)
                for name, expected in webgpu.subgroup_size_control.INPUTS[profile].items():
                    import hashlib
                    self.assertEqual(hashlib.sha256(result[name].encode()).hexdigest(), expected)
                with patch.object(limit, 'PATCH_SHA256', '0' * 64):
                    with self.assertRaisesRegex(ValueError, 'allocation-limit prefix patch identity'):
                        legacy_fixed_sources(prepared.parent)
                changed = self.work / ('changed-final-' + str(profile))
                shutil.copytree(prepared.parent, changed)
                with (changed / 'ggml-webgpu.cpp').open('a') as stream:
                    stream.write('\n// unexpected source drift\n')
                with self.assertRaisesRegex(ValueError, 'exact optional-stage composition'):
                    legacy_fixed_sources(changed)

    def test_changed_contract_or_patch_fails_before_source_rewrite(self):
        for kind in ['contract', 'patch', 'prefix']:
            with self.subTest(kind=kind):
                root = self.work / ('reject-' + kind)
                source = root / 'source'
                for name in limit.CONTRACT_INPUTS:
                    target = source / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.source / name, target)
                patch_root = root / 'patches'
                patch_root.mkdir(parents=True)
                shutil.copyfile(ROOT / webgpu.PATCH_DIRECTORY / limit.PATCH_NAME, patch_root / limit.PATCH_NAME)
                target = root / 'work' / webgpu.SOURCE_PATH
                target.parent.mkdir(parents=True)
                marker = 'static size_t ggml_backend_webgpu_buffer_type_get_max_size('
                text = self.prepared[False].read_text()
                target.write_text(text.replace(definition(text, marker),
                                               definition((self.source / webgpu.SOURCE_PATH).read_text(), marker)))
                self.assertEqual(webgpu.tensor_copy.digest(target), limit.INPUTS[False][target.name])
                if kind == 'contract':
                    (source / 'ggml/src/ggml-alloc.c').write_text('changed contract\n')
                elif kind == 'patch':
                    (patch_root / limit.PATCH_NAME).write_text('changed patch\n')
                else:
                    target.write_text('changed prefix\n')
                before = target.read_bytes()
                expected = {'contract': 'contract changed', 'patch': 'patch identity changed',
                            'prefix': 'composition input changed'}[kind]
                with self.assertRaisesRegex(ValueError, expected):
                    limit.prepare(target.parents[3], source, patch_root,
                                  webgpu.patch_environment(target.parents[3]), gpu_profile=False)
                self.assertEqual(target.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
