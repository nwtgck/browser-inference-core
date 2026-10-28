"""Checked MoE matrix embedding, actual factory/cache and profile syntax."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tests/webgpu-moe-group-skip'
sys.path[:0] = [str(ROOT / 'scripts'), str(HERE)]
import prepare_webgpu_source as source
import prepare_webgpu_moe_group_skip as group
import emit_factory


class MoeGroupSkip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).exists():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        cls.temp = tempfile.TemporaryDirectory(prefix='moe-group-skip-tests-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.prepared = source.prepare(cls.upstream, cls.directory / 'prepared').parent
        with patch.object(source.router_tail, 'prepare', return_value={}), \
                patch.object(source.host_encode_assembly, 'prepare'), \
                patch.object(source.f32_small_m, 'prepare'), \
                patch.object(source.moe_gather_logical32, 'prepare', return_value={}), \
                patch.object(source, 'OUTPUT_SHA256', group.OUTPUTS['ggml-webgpu.cpp']), \
                patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', group.OUTPUTS['ggml-webgpu-shader-lib.hpp']):
            cls.group_stage = source.prepare(cls.upstream, cls.directory / 'group-stage').parent

    def test_exact_matrix_change_and_canonical_sources(self):
        identity = json.loads((HERE / 'inputs.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'scripts/prepare_webgpu_moe_group_skip.py').read_bytes()).hexdigest(), identity['module_sha256'])
        for relative, expected in group.CONTRACT_INPUTS.items():
            self.assertEqual(hashlib.sha256((self.upstream / relative).read_bytes()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256((self.group_stage / 'ggml-webgpu.cpp').read_bytes()).hexdigest(), group.INPUTS['ggml-webgpu.cpp'])
        self.assertEqual(hashlib.sha256((self.group_stage / 'ggml-webgpu-shader-lib.hpp').read_bytes()).hexdigest(), group.OUTPUTS['ggml-webgpu-shader-lib.hpp'])
        self.assertFalse((self.group_stage / 'ggml-wgsl-shaders.hpp').exists())
        emitter = importlib.util.spec_from_file_location('moe_group_upstream_emitter', self.upstream / group.BASE / 'wgsl-shaders/embed_wgsl.py')
        module = importlib.util.module_from_spec(emitter)
        emitter.loader.exec_module(module)
        old = (self.upstream / group.BASE / 'wgsl-shaders/mul_mat_id.wgsl').read_text()
        new = old.replace('        if (is_valid) {\n            let k_end',
                          '        if (is_valid && output_col_base < rest_token_n) {\n            let k_end')
        self.assertEqual(hashlib.sha256(new.encode()).hexdigest(), group.OUTPUTS['wgsl-shaders/' + group.SHADER_NAME])
        expected = module.expand_includes(new, str(self.upstream / group.BASE / 'wgsl-shaders'))
        header = (self.group_stage / group.HEADER_NAME).read_text()
        chunks = re.findall(r'R"(wgsl(?:_x)*)\((.*?)\)\1"', header, re.S)
        self.assertTrue(chunks)
        self.assertEqual(''.join(text for _, text in chunks), expected)
        self.assertEqual(hashlib.sha256(header.encode()).hexdigest(), group.HEADER_SHA256)
        library = (self.group_stage / 'ggml-webgpu-shader-lib.hpp').read_text()
        # Reversing the two linking edits gives the exact previous shared header,
        # including every matvec factory, key, specialization and canonical include.
        before = library.replace('#include "' + group.HEADER_NAME + '"\n', '').replace(
            'preprocessor.preprocess(wgsl_mul_mat_id_group, defines)',
            'preprocessor.preprocess(wgsl_mul_mat_id, defines)')
        self.assertEqual(hashlib.sha256(before.encode()).hexdigest(), group.INPUTS['ggml-webgpu-shader-lib.hpp'])

    def test_drift_never_publishes_partial_or_replaces_existing_outputs(self):
        before = {path.name: path.read_bytes() for path in self.prepared.iterdir() if path.is_file()}
        for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
            values = getattr(group, field)
            for name in values:
                with self.subTest(field=field, path=name), patch.object(group, field, {**values, name: '0' * 64}):
                    with self.assertRaises(ValueError):
                        source.prepare(self.upstream, self.prepared)
                self.assertEqual(before, {path.name: path.read_bytes() for path in self.prepared.iterdir() if path.is_file()})
        for field in ('PATCH_SHA256', 'HEADER_SHA256'):
            with self.subTest(field=field), patch.object(group, field, '0' * 64):
                with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                    source.prepare(self.upstream, Path(tmp))
                self.assertEqual(before, {path.name: path.read_bytes() for path in self.prepared.iterdir() if path.is_file()})

    def test_unknown_upstream_or_intermediate_specialization_is_rejected(self):
        for where in ('upstream', 'work'):
            for relative in (group.HEADER_NAME, 'wgsl-shaders/' + group.SHADER_NAME):
                with self.subTest(where=where, path=relative), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    upstream, work = root / 'upstream', root / 'work'
                    for name in group.CONTRACT_INPUTS:
                        target = upstream / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(self.upstream / name, target)
                    for name in group.INPUTS:
                        target = work / group.BASE / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        payload = (self.group_stage / name).read_bytes()
                        if name == 'ggml-webgpu-shader-lib.hpp':
                            payload = payload.replace(('#include "' + group.HEADER_NAME + '"\n').encode(), b'').replace(
                                b'preprocessor.preprocess(wgsl_mul_mat_id_group, defines)',
                                b'preprocessor.preprocess(wgsl_mul_mat_id, defines)')
                        target.write_bytes(payload)
                    target = (upstream if where == 'upstream' else work) / group.BASE / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text('unreviewed specialization')
                    with self.assertRaisesRegex(ValueError, 'unreviewed'):
                        group.prepare(work, upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))

    def test_repeated_preparation_preserves_timestamps_and_repairs_embedding(self):
        stamps = {p.name: p.stat().st_mtime_ns for p in self.prepared.iterdir() if p.is_file()}
        source.prepare(self.upstream, self.prepared)
        self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in self.prepared.iterdir() if p.is_file()})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            source.prepare(self.upstream, output)
            (output / group.HEADER_NAME).write_text('stale generated matrix shader')
            source.prepare(self.upstream, output)
            self.assertEqual(hashlib.sha256((output / group.HEADER_NAME).read_bytes()).hexdigest(), group.HEADER_SHA256)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_matrix_factory_preserves_code_arithmetic_tiling_and_cache(self):
        directory = self.directory / 'factory'
        executable = emit_factory.build(self.upstream, self.prepared, directory)
        shaders = directory / 'shaders'
        shaders.mkdir()
        cases = [(0, 35, 7), (0, 36, 36), (1, 256, 7), (1, 1024, 36), (2, 256, 7), (2, 2560, 36)]
        for index, (kind, k, m) in enumerate(cases):
            row = k if kind == 0 else k // (32 if kind == 1 else 256)
            vec = kind == 0 and k % 4 == 0 and m % 4 == 0
            a = 4 if vec else 1
            x = 4 if vec else 3
            dst = 8 if vec else 7
            stride_a, stride_x = row + (4 if vec else 1), k + (4 if vec else 5)
            fields = [kind, k, m, 11, 4, 9, 4, a, x, dst, 3,
                      stride_a, m * stride_a + 4, stride_x, 4 * stride_x + 4, 6]
            for input_type in ('f32', 'f16'):
                name = f'case-{index}-{input_type}'
                command = [str(executable), str(shaders), name, *map(str, fields)]
                if input_type == 'f16':
                    command.append('f16')
                metadata = json.loads(subprocess.check_output(command, text=True))
                self.assertEqual(metadata['gatherParams'], [3, dst, 11, 4, 9, 6, m])
                self.assertEqual(metadata['mainParams'], [a, x, dst, k, m, 11, 4, 9, 4,
                                                         stride_a, stride_x, m * stride_a + 4, 4 * stride_x + 4])
                self.assertEqual(metadata['maxTiles'], 13)
                self.assertEqual(metadata['vectorized'], vec)
                self.assertEqual(metadata['tileK'], 8 if kind == 0 else 32)
                baseline = (shaders / (name + '-baseline.wgsl')).read_text()
                candidate = (shaders / (name + '-candidate.wgsl')).read_text()
                self.assertEqual(candidate, baseline.replace('        if (is_valid) {\n            let k_end',
                    '        if (is_valid && output_col_base < rest_token_n) {\n            let k_end'))

    @unittest.skipUnless(shutil.which('c++') and shutil.which('cc'), 'C/C++ compilers required')
    def test_self_authored_fixture_oracle_and_factory_generation(self):
        import make_fixtures
        executable = emit_factory.build(self.upstream, self.prepared, self.directory / 'fixture-host')
        work = self.directory / 'fixtures'
        manifest = make_fixtures.generate(self.upstream, work, executable)
        cases = json.loads(manifest.read_text())
        self.assertEqual(len(cases), 46)
        self.assertEqual({case['type'] for case in cases}, {'f32', 'q8', 'iq3'})
        self.assertEqual(sum(case['zeroTilesOverride'] for case in cases), 3)
        self.assertEqual(sum(case['pattern'] == 'mixed-skipped' for case in cases), 3)
        self.assertEqual(sum(case['pattern'] == 'all-skipped' for case in cases), 3)
        for case in cases:
            for name, expected in case['hashes'].items():
                self.assertEqual(hashlib.sha256((work / 'fixtures' / case['name'] / (name + '.bin')).read_bytes()).hexdigest(), expected)
            for name in ('gather', 'gather-baseline', 'baseline', 'candidate'):
                self.assertEqual(hashlib.sha256((work / 'shaders' / (case['name'] + '-' + name + '.wgsl')).read_bytes()).hexdigest(), case['shaderHashes'][name])
        for case in cases:
            self.assertNotEqual(case['shaderHashes']['gather'], case['shaderHashes']['gather-baseline'])
        profiled = emit_factory.build(self.upstream, self.prepared, self.directory / 'profile-factory', profile=True)
        for case in cases:
            result = json.loads(subprocess.check_output([str(profiled), str(work / 'shaders'), case['name'] + '-profile', *map(str, case['args'])], text=True))
            self.assertEqual(result['mainParams'], case['mainParams'])
            self.assertEqual(result['gatherParams'], case['gatherParams'])
            self.assertEqual(result['grid'], case['grid'])
            for name in ('gather', 'gather-baseline', 'baseline', 'candidate'):
                self.assertEqual((work / 'shaders' / (case['name'] + '-profile-' + name + '.wgsl')).read_bytes(),
                                 (work / 'shaders' / (case['name'] + '-' + name + '.wgsl')).read_bytes())
        for case in [cases[0], next(c for c in cases if c['experts'] == 4)]:
            name = case['name'] + '-wg1024'
            metadata = json.loads(subprocess.check_output([str(profiled), str(work / 'shaders'), name,
                                  *map(str, case['args']), 'f32', '1024'], text=True))
            self.assertEqual(metadata['gatherWorkgroupSize'], 1024)
            self.assertEqual(metadata['mainParams'], case['mainParams'])
            self.assertEqual(metadata['grid'], case['grid'])
            for variant in ('baseline', 'candidate'):
                self.assertEqual((work / 'shaders' / (name + '-' + variant + '.wgsl')).read_bytes(),
                                 (work / 'shaders' / (case['name'] + '-' + variant + '.wgsl')).read_bytes())
            self.assertNotEqual((work / 'shaders' / (name + '-gather.wgsl')).read_bytes(),
                                (work / 'shaders' / (case['name'] + '-gather.wgsl')).read_bytes())

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_full_cpp_normal_and_optional_profile_syntax(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir():
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        identity = json.loads((HERE / 'inputs.json').read_text())
        for name, expected in identity['dawn_header_sha256'].items():
            self.assertEqual(hashlib.sha256((dawn / name).read_bytes()).hexdigest(), expected)
        canonical = self.directory / 'canonical'
        canonical.mkdir()
        subprocess.run([sys.executable, str(self.upstream / group.BASE / 'wgsl-shaders/embed_wgsl.py'),
                        '--input_dir', str(self.upstream / group.BASE / 'wgsl-shaders'),
                        '--output_file', str(canonical / 'ggml-wgsl-shaders.hpp')], check=True)
        includes = [self.prepared, canonical, self.upstream / group.BASE,
                    self.upstream / 'ggml/include', self.upstream / 'ggml/src']
        command = ['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4',
                   '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS']
        command += ['-I' + str(path) for path in includes]
        command += ['-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
        for flags in ([], ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE']):
            subprocess.run(command + flags + [str(self.prepared / 'ggml-webgpu.cpp')], check=True)


if __name__ == '__main__':
    unittest.main()
