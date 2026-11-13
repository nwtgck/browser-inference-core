"""Missing native inputs are visible locally and fatal in the required CI tier."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.support.llama_cpp.current_webgpu import require_recording_inputs


class RecordingRequirements(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='recording-input-requirements-')
        self.addCleanup(temporary.cleanup)
        self.sdk = Path(temporary.name)
        self.enterContext(patch('tests.support.llama_cpp.current_webgpu.dawn_package', return_value=self.sdk))

    def test_local_missing_headers_are_a_reported_skip(self):
        with patch.dict(os.environ, {'CI': 'false'}), patch('shutil.which', return_value='/native/c++'):
            with self.assertRaisesRegex(unittest.SkipTest, 'pinned Dawn headers'):
                require_recording_inputs(self)

    def test_ci_missing_headers_are_not_a_skip(self):
        with patch.dict(os.environ, {'CI': 'true'}), patch('shutil.which', return_value='/native/c++'):
            with self.assertRaisesRegex(RuntimeError, 'pinned Dawn headers'):
                require_recording_inputs(self)

    def test_ci_requires_clang_for_semantic_dispatch_even_with_headers_and_native_compiler(self):
        header = self.sdk / 'webgpu/include/webgpu/webgpu.h'
        header.parent.mkdir(parents=True)
        header.write_text('input presence fixture, not real SDK declarations')
        with patch.dict(os.environ, {'CI': 'true', 'CXX': 'native-cxx', 'CLANGXX': 'ast-clang'}), \
                patch('shutil.which', side_effect=lambda name: '/native/c++' if name == 'native-cxx' else None):
            with self.assertRaisesRegex(RuntimeError, 'Clang'):
                require_recording_inputs(self, clang_ast=True)

    def test_explicit_clang_and_native_compiler_names_are_respected(self):
        header = self.sdk / 'webgpu/include/webgpu/webgpu.h'
        header.parent.mkdir(parents=True)
        header.write_text('input presence fixture, not real SDK declarations')
        with patch.dict(os.environ, {'CI': 'true', 'CXX': 'native-cxx', 'CLANGXX': 'ast-clang'}), \
                patch('shutil.which', side_effect=lambda name: '/compiler/' + name) as lookup:
            require_recording_inputs(self, clang_ast=True)
        self.assertEqual([call.args[0] for call in lookup.call_args_list], ['ast-clang', 'native-cxx'])
