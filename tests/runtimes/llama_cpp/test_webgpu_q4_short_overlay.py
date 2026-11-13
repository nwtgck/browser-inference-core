"""Source-bound Q4_K workgroup checks; no browser/model inference claim."""
from pathlib import Path
import importlib.util
import json
import struct
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
from build_tools.runtimes.llama_cpp import source as source
# Historical byte identities belong to the reference fixture, not production.
from tests.support import frozen_webgpu_source as reference_source
from tests.support import frozen_webgpu_source as legacy_source
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy as copy
from tests.support.llama_cpp.frozen_webgpu_stages import q4_short as q4
from tests.runtimes.llama_cpp.test_webgpu_q8_short_overlay import function

from tests.runtimes.llama_cpp.matmul_vec4_test_support import explicit_layout_argument
from tests.runtimes.llama_cpp.matvec_subgroup_test_support import q4_subgroup_helpers
from tests.support.llama_cpp.current_webgpu import require_recording_inputs


def prepare(upstream, output):
    source.prepare(upstream, output)


def build_host(upstream, prepared, output):
    spec = importlib.util.spec_from_file_location('q4_complete_factory',
        runtime_test_root('llama-cpp') / 'webgpu-q4-short/emit_factory.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_host(upstream, prepared, output)


class Q4ShortOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / q4.HEADER_PATH).is_file():
            if os.environ.get('CI') == 'true':
                raise RuntimeError('Set LCB_TEST_LLAMA_SOURCE for current Q4 validation')
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_checker_report_reads_the_actual_reference_receipt(self):
        # Exercise the actual CLI/report wiring, not GPU numerical behavior.
        # Native runners and the device runner are explicit fixtures here.
        path = runtime_test_root('llama-cpp') / 'webgpu-q4-short/check_gpu.py'
        spec = importlib.util.spec_from_file_location('q4_report_wiring', path)
        checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / 'work'
            module = root / 'webgpu.mjs'; module.write_text('// fixture only\n')
            receipt = {'fixtureOnly': True, 'scope': 'report path regression'}
            calls = []
            work.mkdir()
            (work / 'host-fixture.json').write_text('{"host": "observed"}')

            def fake_prepare(_upstream, output):
                output.mkdir(parents=True)
                (output / 'ggml-webgpu-shader-lib.hpp').write_text('// fixture only\n')
                (output / 'source-preparation.json').write_text('{"inputs": {"observed": true}}')

            def fake_reference(_upstream, output):
                output.mkdir(parents=True, exist_ok=True)
                executable = output / 'reference-variant'
                executable.write_text('fixture only')
                executable.with_name(executable.name + '.json').write_text(json.dumps(receipt))
                return executable

            def fake_run(command, **kwargs):
                calls.append(command)
                if str(command[0]).endswith('host-fixture'):
                    shaders = work / 'shaders'; shaders.mkdir()
                    (shaders / 'cases.json').write_text('[{}]')
                    (shaders / 'fixture.wgsl').write_text('// fixture only\n')
                elif len(command) > 1 and str(command[1]).endswith('fixtures.mjs'):
                    fixtures = work / 'fixtures'; fixtures.mkdir()
                    (fixtures / '0.bin').write_bytes(b'fixture only')
                elif str(command[0]).endswith('/reference-variant'):
                    kwargs['stdout'].write(b'fixture reference')
                return subprocess.CompletedProcess(command, 0)

            # subprocess is a shared module: use only the fixtures above inside
            # the patched block, and source_commit must not query a parent Git.
            with patch.object(checker, 'prepare', side_effect=fake_prepare), \
                 patch.object(checker, 'build_host', return_value=work / 'host-fixture'), \
                 patch.object(checker, 'build_reference', side_effect=fake_reference), \
                 patch.object(checker.subprocess, 'run', side_effect=fake_run), \
                 patch.object(checker.sys, 'argv', ['check_gpu.py', '--llama-source', str(self.upstream),
                       '--work', str(work), '--webgpu-module', str(module)]):
                self.assertEqual(checker.main(), 0)
            report = json.loads((work / 'source.json').read_text())
            self.assertEqual(report['referenceCompilation'], receipt)
            self.assertEqual(report['sourcePreparation'], {'inputs': {'observed': True}})
            self.assertEqual(report['nativeRecorder'], {'host': 'observed'})
            self.assertNotIn('upstreamContracts', report)
            self.assertIsNone(report['upstreamCommit'])
            self.assertEqual((work / 'fixtures/0.f64').read_bytes(), b'fixture reference')
            self.assertFalse((work / 'host/reference.json').exists())
            self.assertEqual(len(calls), 4)

    def test_preparation_preserves_upstream_and_shader_files(self):
        before = {p: copy.digest(self.upstream / p) for p in q4.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prepare(self.upstream, out)
            receipt = json.loads((out / 'source-preparation.json').read_text())
            self.assertEqual(copy.digest(out / 'ggml-webgpu-shader-lib.hpp'),
                             receipt['outputs']['ggml-webgpu-shader-lib.hpp']['sha256'])
            self.assertFalse((out / 'ggml-wgsl-shaders.hpp').exists())
            stamp = (out / 'ggml-webgpu-shader-lib.hpp').stat().st_mtime_ns
            prepare(self.upstream, out)
            self.assertEqual(stamp, (out / 'ggml-webgpu-shader-lib.hpp').stat().st_mtime_ns)
        self.assertEqual(before, {p: copy.digest(self.upstream / p) for p in before})



    def test_runner_rejects_invalid_commands_and_overlapping_work(self):
        runner = runtime_test_root('llama-cpp') / 'webgpu-q4-short/check_gpu.py'
        with tempfile.TemporaryDirectory() as tmp:
            for value in ('invalid', '{}', '[]', '[1]', '["", "node"]', '"node"'):
                with self.subTest(value=value):
                    result = subprocess.run([sys.executable, str(runner), '--llama-source', str(self.upstream),
                        '--work', tmp, '--webgpu-module', 'unused', '--node-command-json', value],
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('--node-command-json', result.stderr)
                    self.assertFalse((Path(tmp) / 'prepared').exists())
            result = subprocess.run([sys.executable, str(runner), '--llama-source', str(self.upstream),
                '--work', str(self.upstream), '--webgpu-module', 'unused'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('must not overlap', result.stderr)

    @unittest.skipUnless(shutil.which('g++'), 'C++ compiler required')
    def test_actual_upstream_dequantizer_reference_known_blocks(self):
        spec = importlib.util.spec_from_file_location('q4_gpu_checker', runtime_test_root('llama-cpp') / 'webgpu-q4-short/check_gpu.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            exe = checker.build_reference(self.upstream, out)
            params = [0,0,0,1,1,256,1,256,1,256,1,256,1,1,1,1]
            header = struct.pack('<5I', 0,144,256,0,16)
            for scales, qs, d, dmin, expected in [
                ([1]*4+[0]*4+[1]*4, 0x21, 0x3c00, 0, 384.0),
                ([255]*12, 255, 0x2c00, 0x2800, 14616.0),
                ([0]*12, 0, 0x3c00, 0, 0.0),
            ]:
                weights = struct.pack('<2H', d,dmin) + bytes(scales) + bytes([qs]*128)
                payload = header + weights + struct.pack('<256f', *([1.0]*256)) + struct.pack('<16I', *params)
                result = subprocess.run([str(exe)], input=payload, check=True, capture_output=True)
                self.assertEqual(struct.unpack('<2d', result.stdout), (expected, expected))
            result = subprocess.run([str(exe)], input=header, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'truncated fixture', result.stderr)

    def test_actual_selector_cache_factory_dispatch_and_fallback(self):
        require_recording_inputs(self, clang_ast=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prepare(self.upstream, out / 'prepared')
            exe = build_host(self.upstream, out / 'prepared', out / 'host')
            result = subprocess.run([str(exe), str(out / 'shaders')], check=True, capture_output=True, text=True)
            self.assertIn('actual Q4 selector/cache/factory checks passed', result.stdout)
            for kind in ('ordinary', 'indexed'):
                for size in (32, 64, 128, 256):
                    shader = (out / 'shaders' / f'{kind}-{size}.wgsl').read_text()
                    self.assertIn(f'@workgroup_size({size})', shader)
                    self.assertNotIn('#include', shader)
            print(result.stdout.strip())


    def test_current_diagnostic_complete_header_and_dispatch(self):
        require_recording_inputs(self, clang_ast=True)
        from build_tools.runtimes.llama_cpp import webgpu_plan
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            webgpu_plan.prepare(self.upstream, work / 'prepared', gpu_profile=True)
            exe = build_host(self.upstream, work / 'prepared', work / 'host')
            result = subprocess.run([str(exe), str(work / 'shaders')], check=True, capture_output=True, text=True, timeout=60)
            self.assertIn('actual Q4 selector/cache/factory checks passed', result.stdout)
            self.assertTrue(json.loads(exe.with_suffix('.json').read_text())['gpuProfile'])
            print(result.stdout.strip())

    def test_current_comment_is_accepted_without_historical_hash_updates(self):
        require_recording_inputs(self, clang_ast=True)
        from build_tools.runtimes.llama_cpp import webgpu_plan
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            prepare(self.upstream, work / 'original')
            receipt = json.loads((work / 'original' / webgpu_plan.RECEIPT).read_text())
            private = work / 'private'
            for name in ('ggml/include', 'ggml/src'):
                shutil.copytree(self.upstream / name, private / name)
            for name in receipt['inputs']:
                if name.startswith('runtime/') or (private / name).exists():
                    continue
                target = private / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.upstream / name, target)
            header = private / webgpu_plan.BASE / 'ggml-webgpu-shader-lib.hpp'
            header.write_bytes(header.read_bytes() + b'\n// unrelated Q4 comment\n')
            prepare(private, work / 'comment')
            exe = build_host(private, work / 'comment', work / 'host')
            result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=60)
            self.assertIn('actual Q4 selector/cache/factory checks passed', result.stdout)

    def test_current_unused_type_error_is_not_hidden_by_extraction(self):
        require_recording_inputs(self, clang_ast=True)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            prepare(self.upstream, work / 'prepared')
            header = work / 'prepared/ggml-webgpu-shader-lib.hpp'
            header.write_bytes(header.read_bytes() + b'\nstruct unused_q4_contract { missing_q4_type value; };\n')
            with self.assertRaises(subprocess.CalledProcessError) as failure:
                build_host(self.upstream, work / 'prepared', work / 'host')
            self.assertIn('missing_q4_type', failure.exception.stderr)


class HistoricalQ4ShortGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.support.historical_webgpu import historical_source
        cls.upstream = cls.enterClassContext(historical_source())

    def test_stage_contracts_fail_closed(self):
        for guard in ('CONTRACT_INPUTS', 'INPUT_SHA256', 'OUTPUT_SHA256', 'PATCH_SHA256'):
            values = getattr(q4, guard)
            cases = ({**values, name: '0' * 64} for name in values) if isinstance(values, dict) else ['0' * 64]
            for value in cases:
                with self.subTest(guard=guard, value=value), tempfile.TemporaryDirectory() as tmp, patch.object(q4, guard, value):
                    with self.assertRaises(ValueError): legacy_source.prepare(self.upstream, Path(tmp), patch_root=self.upstream / 'patches')
                    self.assertFalse((Path(tmp) / 'ggml-webgpu.cpp').exists())
                    self.assertFalse((Path(tmp) / 'ggml-webgpu-shader-lib.hpp').exists())


    def test_build_inputs_and_multitoken_matrix_factory_remain_source_bound(self):
        self.assertEqual(reference_source.scale_unary.INPUTS['ggml-webgpu-shader-lib.hpp'], q4.OUTPUT_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            legacy_source.prepare(self.upstream, Path(tmp), patch_root=self.upstream / 'patches')
            prepared = explicit_layout_argument((Path(tmp) / 'ggml-webgpu-shader-lib.hpp').read_text())
        original = (self.upstream / q4.HEADER_PATH).read_text()
        def matrix_factory(body):
            start = body.index('    webgpu_pipeline get_mul_mat_id_pipeline(')
            return body[start:body.index('    webgpu_pipeline get_mul_mat_id_vec_pipeline(', start)]
        # Later stages select the group shader and guard actual vec4 layouts.
        # Reverse only those explicit changes; all other factory bytes remain.
        prepared = prepared.replace(', bool vec4_aligned)', ')').replace(
            '(vec4_aligned && context.src0->ne[0] % 4', '(context.src0->ne[0] % 4')
        self.assertEqual(matrix_factory(prepared), matrix_factory(original).replace(
            'preprocessor.preprocess(wgsl_mul_mat_id, defines)',
            'preprocessor.preprocess(wgsl_mul_mat_id_group, defines)'))



if __name__ == '__main__': unittest.main()
