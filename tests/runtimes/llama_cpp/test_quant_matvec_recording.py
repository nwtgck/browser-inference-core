"""Current Q4/Q6 factories and dispatch; historical byte comparisons live separately."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from build_tools.project import runtime_root, runtime_test_root
from build_tools.runtimes.llama_cpp import webgpu_plan as plan
from tests.support.llama_cpp.current_webgpu import compile_recorder, require_recording_inputs
from tests.support.llama_cpp.dispatch_ast import current_dispatch

CASES = runtime_test_root('llama-cpp') / 'webgpu-native-recording/quant-matvec-cases.cpp'


class CurrentQuantizedMatvec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE',
                                          runtime_root('llama-cpp') / 'vendor/llama.cpp')).resolve()
        temporary = tempfile.TemporaryDirectory(prefix='current-quant-matvec-')
        cls.addClassCleanup(temporary.cleanup)
        cls.work = Path(temporary.name)

    def build_and_run(self, upstream, prepared, destination):
        fragment = destination / 'dispatch.inc'
        destination.mkdir(parents=True)
        fragment.write_text(current_dispatch(upstream, prepared))
        executable = compile_recorder(upstream, prepared, destination, CASES, fragment=fragment)
        result = subprocess.run([str(executable), str(destination / 'shaders')],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['status'], 'passed')
        self.assertGreater(data['checks'], 9000)
        self.assertGreater(data['pipelines'], 100)
        shaders = {path.name: path.read_bytes() for path in (destination / 'shaders').iterdir()}
        self.assertTrue(shaders)
        self.assertTrue(all(b'#include' not in value for value in shaders.values()))
        return data, shaders

    def test_current_normal_and_diagnostic_factories_cache_selectors_and_grid(self):
        require_recording_inputs(self, clang_ast=True)
        results = []
        for profile in (False, True):
            with self.subTest(profile=profile):
                prepared = self.work / f'prepared-{profile}'
                plan.prepare(self.upstream, prepared, gpu_profile=profile)
                results.append(self.build_and_run(self.upstream, prepared, self.work / f'host-{profile}'))
        if len(results) == 2:
            self.assertEqual(results[0], results[1])

    def test_unrelated_current_comment_does_not_require_historical_hash_updates(self):
        require_recording_inputs(self, clang_ast=True)
        private = self.work / 'private'
        for name in ('ggml/include', 'ggml/src'):
            shutil.copytree(self.upstream / name, private / name)
        original = self.work / 'receipt-inputs'
        plan.prepare(self.upstream, original)
        receipt = json.loads((original / plan.RECEIPT).read_text())
        for name in receipt['inputs']:
            if name.startswith('runtime/') or (private / name).exists():
                continue
            target = private / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.upstream / name, target)
        header = private / plan.BASE / 'ggml-webgpu-shader-lib.hpp'
        header.write_bytes(header.read_bytes() + b'\n// unrelated quantized factory documentation\n')
        prepared = self.work / 'comment'
        plan.prepare(private, prepared)
        self.build_and_run(private, prepared, self.work / 'comment-host')

    def test_invalid_unused_declaration_is_not_hidden_by_fragment_extraction(self):
        require_recording_inputs(self)
        prepared = self.work / 'invalid'
        plan.prepare(self.upstream, prepared)
        header = prepared / 'ggml-webgpu-shader-lib.hpp'
        header.write_bytes(header.read_bytes() + b'\nstruct unused_quant_contract { missing_quant_type value; };\n')
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            compile_recorder(self.upstream, prepared, self.work / 'invalid-host', CASES)
        self.assertIn('missing_quant_type', failure.exception.stderr)
