"""The current CPU reference compiles real code; byte identities are observations."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.support.llama_cpp.native_quantization import build_reference, source_commit

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


class CurrentNativeQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE',
                                         REPOSITORY_ROOT / 'runtimes/llama-cpp/vendor/llama.cpp')).resolve()
        missing = (not (cls.upstream / 'ggml/src/ggml-quants.c').is_file()
                   or not shutil.which(os.environ.get('CC', 'cc'))
                   or not shutil.which(os.environ.get('CXX', 'c++')))
        if missing:
            message = 'Set LCB_TEST_LLAMA_SOURCE and provide native C/C++ compilers'
            if os.environ.get('CI') == 'true':
                raise RuntimeError(message)
            raise unittest.SkipTest(message)
        cls.harness = Path(__file__).resolve().parents[1] / 'webgpu-iq3s-subgroup/oracle-main.c'

    def private_source(self, root):
        for part in ('ggml/src', 'ggml/include'):
            shutil.copytree(self.upstream / part, root / part)
        return root

    def test_complete_current_unit_and_real_header_pass_native_self_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'out'
            executable = build_reference(self.upstream, output, self.harness, name='oracle')
            result = subprocess.run([str(executable), '--self-test'], check=True,
                                    capture_output=True, text=True, timeout=30)
            self.assertIn('actual upstream IQ3_S dequantizer self-tests passed', result.stdout)
            receipt = json.loads((output / 'oracle.json').read_text())
            self.assertIn('ggml/src/ggml-quants.c', receipt['sourceSha256'])
            self.assertEqual(len(receipt['compileCommands']), 2)
            self.assertIn('ggml-quants.h', receipt['dependencyFiles']['harness.d'])
            self.assertIn('ggml-common.h', receipt['dependencyFiles']['quantization.d'])
            self.assertFalse((output / 'oracle-generated.c').exists())

    def test_unrelated_source_comment_changes_provenance_not_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            private = self.private_source(work / 'source')
            source = private / 'ggml/src/ggml-quants.c'
            source.write_bytes(source.read_bytes() + b'\n// current dequantization documentation only\n')
            executable = build_reference(private, work / 'out', self.harness, name='oracle')
            result = subprocess.run([str(executable), '--self-test'], check=True,
                                    capture_output=True, text=True, timeout=30)
            self.assertIn('self-tests passed', result.stdout)
            import hashlib
            observed = json.loads((work / 'out/oracle.json').read_text())['sourceSha256']
            self.assertEqual(observed['ggml/src/ggml-quants.c'], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertNotEqual(observed['ggml/src/ggml-quants.c'],
                                hashlib.sha256((self.upstream / 'ggml/src/ggml-quants.c').read_bytes()).hexdigest())

    def test_invalid_unused_type_outside_the_dequantizer_is_not_sliced_away(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            private = self.private_source(work / 'source')
            source = private / 'ggml/src/ggml-quants.c'
            source.write_bytes(source.read_bytes() + b'\nstruct unused_quant_contract { missing_quant_dependency value; };\n')
            with self.assertRaises(subprocess.CalledProcessError) as failure:
                build_reference(private, work / 'out', self.harness, name='oracle')
            self.assertIn('missing_quant_dependency', failure.exception.stderr)
            self.assertFalse((work / 'out/oracle').exists())
            self.assertFalse((work / 'out/oracle.json').exists())

    def test_output_cannot_overlap_inputs_or_escape_its_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            for destination in (self.upstream, self.upstream / 'forbidden', self.upstream.parent):
                with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, 'overlap'):
                    build_reference(self.upstream, destination, self.harness, name='oracle')
            for name in ('', '.', '..', '../escape', 'a/b'):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'file name'):
                    build_reference(self.upstream, Path(tmp), self.harness, name=name)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_source_only_input_does_not_inherit_a_parent_git_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            subprocess.run(['git', 'init', '-q', str(parent)], check=True)
            source = parent / 'source'; source.mkdir()
            # No .git at this source: never query the enclosing worktree.
            self.assertIsNone(source_commit(source))
