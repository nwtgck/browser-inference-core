"""Run final preparation through native recorders.

Q8 and IQ3 use the complete current factory and real Dawn declarations with a
mocked C WebGPU boundary; both keep a type-checked, AST-selected current
provider-dispatch fragment. Neither executes GPU shaders or browser inference.
"""
import importlib.util
import json
import os
from pathlib import Path
from build_tools.project import runtime_root, runtime_test_root
import subprocess
import tempfile
import unittest

from build_tools.runtimes.llama_cpp import webgpu_plan as plan

ROOT = runtime_root('llama-cpp')


class FinalFactoryContracts(unittest.TestCase):
    def test_final_preparation_feeds_current_q8_and_iq3_recorders(self):
        upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (upstream / plan.WGSL / 'embed_wgsl.py').is_file():
            message = 'Initialize upstream or set LCB_TEST_LLAMA_SOURCE'
            if os.environ.get('CI') == 'true':
                raise RuntimeError(message)
            self.skipTest(message)
        from tests.support.llama_cpp.current_webgpu import require_recording_inputs
        require_recording_inputs(self, clang_ast=True)
        for name, count, message in (
            ('webgpu-q8-subgroup', 6, 'actual subgroup selector/cache/factory/dispatch checks passed'),
            ('webgpu-iq3s-subgroup', 4, 'actual IQ3_S selector/cache/factory/dispatch checks passed'),
        ):
            with self.subTest(factory=name), tempfile.TemporaryDirectory(prefix='final-factory-') as temp:
                work = Path(temp)
                spec = importlib.util.spec_from_file_location('final_' + name.replace('-', '_'),
                                                             runtime_test_root('llama-cpp') / name / 'emit_factory.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                prepared = work / 'prepared'
                module.prepare(upstream, prepared)
                receipt = json.loads((prepared / plan.RECEIPT).read_text())
                self.assertFalse(receipt['gpuProfile'])
                self.assertEqual([record['file'] for record in receipt['patches']
                                  if record['target'] == 'llama-webgpu'], list(plan.patch_names(False)))
                self.assertIn('ggml-webgpu-aligned-word-load-shaders.hpp', receipt['outputs'])
                executable = module.build_host(upstream, prepared, work / 'host')
                result = subprocess.run([str(executable), str(work / 'shaders')],
                                        check=True, capture_output=True, text=True)
                self.assertIn(message, result.stdout)
                shaders = sorted((work / 'shaders').glob('*.wgsl'))
                self.assertEqual(len(shaders), count)
                for shader in shaders:
                    text = shader.read_text()
                    self.assertNotIn('#include', text)
                    self.assertIn('@compute', text)
                    self.assertIn('@workgroup_size', text)
