"""Source-bound F32 matvec epilogue admission, ownership and factory contracts."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tests/webgpu-f32-matvec-epilogue'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import prepare_webgpu_source as source
import prepare_webgpu_f32_matvec_epilogue as post
from test_webgpu_bind_group_cache import definition
spec = importlib.util.spec_from_file_location('f32_post_emitter', HERE / 'emit_host.py')
emitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emitter)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


class F32MatvecEpilogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='f32-matvec-epilogue-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.before, cls.after = {}, {}
        for profile in (False, True):
            # Inspect this stage's immediate historical prefix without changing its pins.
            with patch.object(post, 'prepare', return_value={}):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            cls.after[profile] = source.prepare(cls.upstream, cls.directory / f'after-{profile}', gpu_profile=profile).parent

    def test_checked_inverse_and_only_expected_outputs(self):
        identity = json.loads((HERE / 'inputs.json').read_text())
        self.assertEqual(sha(ROOT / 'scripts/prepare_webgpu_f32_matvec_epilogue.py'), identity['module_sha256'])
        for profile in (False, True):
            old, new = self.before[profile], self.after[profile]
            changed = {p.name for p in old.iterdir() if p.read_bytes() != (new / p.name).read_bytes()}
            self.assertEqual(changed, set(post.INPUTS[profile]))
            self.assertEqual({p.name for p in new.iterdir()} - {p.name for p in old.iterdir()}, {post.HEADER_NAME, post.HELPER_NAME})
            for name, expected in post.INPUTS[profile].items():
                self.assertEqual(sha(old / name), expected)
                self.assertEqual(sha(new / name), post.OUTPUTS[profile][name])
            self.assertEqual(sha(new / post.HEADER_NAME), post.HEADER_SHA256)
            self.assertEqual(sha(new / post.HELPER_NAME), post.OUTPUTS[profile][post.HELPER_NAME])
            self.assertFalse((new / 'ggml-wgsl-shaders.hpp').exists())
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); target = work / post.BASE; target.mkdir(parents=True)
                for name in post.INPUTS[profile]:
                    shutil.copyfile(new / name, target / name)
                # The added raw main is checked by the inverse as well.
                shader = target / 'wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl'; shader.parent.mkdir()
                embedded = (new / post.HEADER_NAME).read_text()
                canonical = self.upstream / post.BASE / 'wgsl-shaders'
                raw = (canonical / 'mul_mat_vec.wgsl').read_text()
                raw = raw.replace('    broadcast3: u32\n};', '    broadcast3: u32,\n    scale: f32,\n    bias: f32,\n    final_scale: f32,\n    final_bias: f32\n};')
                for value in ('row_total', 'partial_sums[partial_index(thread_id, 0)]'):
                    lines = [line for line in raw.splitlines() if line.strip().startswith('dst[') and line.endswith(' = ' + value + ';')]
                    self.assertEqual(len(lines), 1)
                    line = lines[0]; indent = line[:len(line) - len(line.lstrip())]; destination = line.strip().split(' = ')[0]
                    replacement = '\n'.join(indent + s for s in [f'let scaled = {value} * params.scale + params.bias;',
                        'let activated = 1.0 / (1.0 + exp(-scaled));', 'let result = activated * params.final_scale + params.final_bias;', destination + ' = result;'])
                    raw = raw.replace(line, replacement)
                self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(), post.OUTPUTS[profile]['wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl'])
                shader.write_text(raw);shutil.copyfile(new / post.HELPER_NAME, target / post.HELPER_NAME)
                subprocess.run(['git', 'apply', '--no-index', '-R', str(ROOT / source.PATCH_DIRECTORY / post.PATCHES[profile][0])],
                               cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                for name in post.INPUTS[profile]:
                    self.assertEqual((target / name).read_bytes(), (old / name).read_bytes())
                self.assertIn('let activated = 1.0 / (1.0 + exp(-scaled));', embedded)

    def test_contract_patch_input_output_and_embedding_drift_fail_closed(self):
        for profile in (False, True):
            for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS', 'PATCHES', 'HEADER_SHA256'):
                values = getattr(post, field)
                keys = list(values) if field == 'CONTRACT_INPUTS' else list(values[profile]) if field in ('INPUTS', 'OUTPUTS') else [None]
                for key in keys:
                    with self.subTest(profile=profile, field=field, key=key), tempfile.TemporaryDirectory() as tmp:
                        work = Path(tmp); (work / post.BASE).mkdir(parents=True)
                        for name in post.INPUTS[profile]:
                            shutil.copyfile(self.before[profile] / name, work / post.BASE / name)
                        changed = copy.deepcopy(values)
                        if field == 'CONTRACT_INPUTS': changed[key] = '0' * 64
                        elif field in ('INPUTS', 'OUTPUTS'): changed[profile][key] = '0' * 64
                        elif field == 'PATCHES': changed[profile] = (changed[profile][0], '0' * 64)
                        else: changed = '0' * 64
                        with patch.object(post, field, changed), self.assertRaises(ValueError):
                            post.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work), gpu_profile=profile)

    def test_actual_graph_gallocr_dispatch_cache_and_coefficients(self):
        configured = os.environ.get('LCB_TEST_GGML_BASE_LIBRARY')
        if configured:
            library = Path(configured).resolve()
        else:
            build = self.directory / 'native-base'
            subprocess.run(['cmake', '-S', str(self.upstream), '-B', str(build), '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=OFF',
                            '-DGGML_OPENMP=OFF', '-DGGML_CCACHE=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
                            '-DLLAMA_BUILD_TOOLS=OFF', '-DLLAMA_BUILD_COMMON=OFF', '-DLLAMA_CURL=OFF'], check=True, capture_output=True)
            subprocess.run(['cmake', '--build', str(build), '--target', 'ggml-base', '-j', '1'], check=True, capture_output=True)
            library = build / 'ggml/src/libggml-base.a'
        expected = json.loads((HERE / 'inputs.json').read_text())['processed_sha256']
        for profile in (False, True):
            out = self.directory / f'host-{profile}'
            exe = emitter.build(self.upstream, self.after[profile], out, library, profile=profile)
            (out / 'shaders').mkdir()
            result = subprocess.run([str(exe), str(out / 'shaders')], check=True, capture_output=True, text=True)
            self.assertIn('"status":"passed"', result.stdout)
            for name, digest in expected.items():
                self.assertEqual(sha(out / 'shaders' / name), digest)
            for name in ('sg-scalar', 'sg-vec4', 'wg-scalar', 'wg-vec4'):
                old = (out / 'shaders' / (name + '-none.wgsl')).read_text()
                new = (out / 'shaders' / (name + '-post.wgsl')).read_text()
                self.assertNotEqual(old, new)
                # Every unchanged accumulator/reduction byte is covered by the
                # canonical-main inverse and fixed processed factory identities.
        for path in (self.directory / 'host-False/shaders').iterdir():
            self.assertEqual(path.read_bytes(), (self.directory / 'host-True/shaders' / path.name).read_bytes())

    def test_stage_build_and_fallback_contracts(self):
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_f32_matvec_epilogue.py', cmake)
        for name, digest in post.PATCHES.values():
            self.assertIn(name, cmake);self.assertEqual(sha(ROOT / source.PATCH_DIRECTORY / name), digest)
        for name in post.CONTRACT_INPUTS:
            self.assertIn('${LCB_LLAMA_SOURCE}/' + name, cmake)
        for profile in (False, True):
            cpp = (self.after[profile] / 'ggml-webgpu.cpp').read_text()
            self.assertIn('if (auto fused = ggml_webgpu_f32_matvec_epilogue(ctx, cgraph, node_idx, num_encoded_ops)) { return fused; }', cpp)
            self.assertIn('return ggml_webgpu_mul_mat(ctx, src0, src1, node);', cpp)
            self.assertIn('ggml_webgpu_scale_unary(ctx, cgraph, node_idx, num_encoded_ops)', cpp)
            self.assertIn('params.reserve(post ? 20 : 16);', cpp)
            self.assertLess(cpp.index('if (post && !pipeline.context)'), cpp.index('params.reserve(post ? 20 : 16)'))


if __name__ == '__main__':
    unittest.main()
