"""Current copy/snapshot/scheduler selections in a deferred CPU queue model.

All three complete C++ units are type checked against current inputs before
selection. Native behavior covers selected functions, not the entire backend,
GPU execution, Emscripten, or the browser. Historical regression assertions
remain in separate classes using the fixed source archive.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from build_tools.project import runtime_root
from build_tools.runtimes.llama_cpp import webgpu_plan
from tests.support.llama_cpp.current_copies import CurrentCopies
from tests.support.llama_cpp.current_webgpu import require_recording_inputs


class CurrentCopyRecording(unittest.TestCase):
    def setUp(self):
        require_recording_inputs(self, clang_ast=True)
        self.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE',
                                            runtime_root('llama-cpp') / 'vendor/llama.cpp'))
        if not (self.upstream / 'ggml/src/ggml-cpu/ggml-cpu.cpp').is_file():
            message = 'Current copy recording requires complete llama.cpp source (LCB_TEST_LLAMA_SOURCE)'
            if os.environ.get('CI') == 'true':
                raise RuntimeError(message)
            self.skipTest(message)
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='current-copy-')))

    def record(self, upstream, *, batch, diagnostic):
        prepared = self.work / 'prepared'
        webgpu_plan.prepare(upstream, prepared, gpu_profile=diagnostic)
        recorder = CurrentCopies(upstream, prepared, batch=batch)
        for snapshot, count in ((False, 991), (True, 45842)):
            with self.subTest(snapshot=snapshot):
                binary = recorder.build(self.work / ('snapshot' if snapshot else 'ordinary'), snapshot=snapshot)
                result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=60)
                self.assertIn(f'PASS: {count} ', result.stdout)
                print(result.stdout.strip(), flush=True)
        return recorder, prepared

    def test_complete_units_and_real_functions_in_normal_mode(self):
        self.record(self.upstream, batch=False, diagnostic=False)

    def test_complete_units_and_real_functions_in_batch_mode(self):
        self.record(self.upstream, batch=True, diagnostic=False)

    def test_complete_units_and_real_functions_in_diagnostic_mode(self):
        self.record(self.upstream, batch=True, diagnostic=True)

    def test_unrelated_comments_do_not_invalidate_live_inputs(self):
        copied = self.work / 'source'
        for relative in ('ggml/include', 'ggml/src', 'src'):
            shutil.copytree(self.upstream / relative, copied / relative)
        for relative in ('ggml/src/ggml-webgpu/ggml-webgpu.cpp', 'ggml/src/ggml-backend.cpp',
                         'ggml/src/ggml-cpu/ggml-cpu.cpp'):
            target = copied / relative
            target.write_bytes(target.read_bytes() + b'\n// unrelated current-input documentation\n')
        self.record(copied, batch=False, diagnostic=False)

    def test_unused_bad_declaration_outside_selected_functions_is_rejected(self):
        prepared = self.work / 'prepared'
        webgpu_plan.prepare(self.upstream, prepared)
        target = prepared / 'ggml-backend.cpp'
        target.write_bytes(target.read_bytes() + b'\nstruct unused_copy_contract { missing_copy_type value; };\n')
        with self.assertRaises(subprocess.CalledProcessError) as error:
            CurrentCopies(self.upstream, prepared, batch=False)
        self.assertIn('missing_copy_type', error.exception.stderr)

    def test_source_change_after_selection_prevents_build(self):
        prepared = self.work / 'prepared'
        webgpu_plan.prepare(self.upstream, prepared)
        recorder = CurrentCopies(self.upstream, prepared, batch=False)
        target = prepared / 'ggml-backend.cpp'
        target.write_bytes(target.read_bytes() + b'\n// changed after parsing\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            recorder.build(self.work / 'must-not-build', snapshot=True)
        self.assertFalse((self.work / 'must-not-build').exists())


if __name__ == '__main__':
    unittest.main()
