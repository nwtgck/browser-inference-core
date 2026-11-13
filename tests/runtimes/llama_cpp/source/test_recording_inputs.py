"""Current standalone preparation inputs, separate from historical byte guards.

These adapters still construct native recorders; this is not a real GPU test.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import runtime_root, runtime_test_root
from build_tools.runtimes.llama_cpp import webgpu_plan

ROOT = runtime_root('llama-cpp')


class CurrentRecordingInputs(unittest.TestCase):
    def setUp(self):
        self.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (self.upstream / webgpu_plan.WGSL / 'embed_wgsl.py').is_file():
            self.skipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        temporary = tempfile.TemporaryDirectory(prefix='current-recording-inputs-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_current_standalone_preparation_uses_final_embeddings_and_canonical(self):
        from tests.runtimes.llama_cpp.webgpu_embedding_test_support import canonical_prefix, matvec_constants
        # These inputs are current, not the frozen historical composer fixture.
        prepared = self.directory / 'standalone-final'
        for directory in ('webgpu-q8-short', 'webgpu-q8-subgroup', 'webgpu-iq3s-subgroup'):
            # Exercise each public standalone preparation entrypoint in a fresh process.
            script = """import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('standalone_factory',sys.argv[1])
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
factory.prepare(Path(sys.argv[2]),Path(sys.argv[3]))
"""
            subprocess.run([sys.executable, '-S', '-c', script,
                            str(runtime_test_root('llama-cpp') / directory / 'emit_factory.py'),
                            str(self.upstream), str(prepared)], check=True,
                           capture_output=True, text=True)
        prefix = '#include "ggml-wgsl-shaders.hpp"\n'
        constants = matvec_constants((prepared / 'ggml-webgpu-shader-lib.hpp').read_text()) + '\n'
        self.assertEqual(canonical_prefix(prefix, prepared), constants + '#include "ggml-webgpu-aligned-word-load-shaders.hpp"\n')
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp)
            (bad / 'ggml-webgpu-shader-lib.hpp').write_bytes((prepared / 'ggml-webgpu-shader-lib.hpp').read_bytes())
            with self.assertRaisesRegex(ValueError, 'embedding is missing'):
                canonical_prefix(prefix, bad)
        with self.assertRaisesRegex(ValueError, 'include changed'):
            canonical_prefix(prefix + prefix, prepared)
