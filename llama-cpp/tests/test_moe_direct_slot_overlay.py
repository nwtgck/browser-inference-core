"""Source, provenance and actual upstream CMake embedding checks; no GPU/build."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_moe_direct_slot as moe


class MoeDirectSlotOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.source / moe.SHADER_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        cls.patch = ROOT / moe.PATCH_DIRECTORY / moe.PATCH_NAME

    def test_exact_patch_generated_header_only_changes_intended_shader(self):
        revision = moe.verify_reviewed_source(self.source)
        before = {p: (self.source / p).read_bytes() for p in moe.REVIEWED_REVISIONS[revision]}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'overlay'
            result = moe.prepare(self.source, output, self.patch)
            self.assertEqual(moe.digest(result['shader']), moe.PATCHED_SHADER_SHA256)
            baseline = Path(temporary) / 'baseline.hpp'
            subprocess.run([sys.executable, str(self.source / moe.SHADER_DIRECTORY / 'embed_wgsl.py'),
                            '--input_dir', str(self.source / moe.SHADER_DIRECTORY),
                            '--output_file', str(baseline)], check=True)
            base = baseline.read_text(); candidate = result['header'].read_text()
            start = 'static const char wgsl_mul_mat_id_vec_part0[] = '
            end = 'static const char wgsl_mul_mat_vec_part0[] = '
            self.assertEqual(base[:base.index(start)], candidate[:candidate.index(start)])
            self.assertEqual(base[base.index(end):], candidate[candidate.index(end):])
            self.assertIn('let selected_slot = wg_linear / output_groups;', candidate)
            self.assertNotIn('gathered_count_ids', candidate[candidate.index(start):candidate.index(end)])
            mtimes = {key: path.stat().st_mtime_ns for key, path in result.items()}
            again = moe.prepare(self.source, output, self.patch)
            self.assertEqual(mtimes, {key: path.stat().st_mtime_ns for key, path in again.items()})
        self.assertEqual(before, {p: (self.source / p).read_bytes() for p in before})

    def test_overlap_is_rejected_before_any_write(self):
        for output in (self.source, self.source / 'test-output', self.source.parent):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'outside'):
                moe.prepare(self.source, output, self.patch)

    def test_revision_guard_and_changed_patch_fail_closed(self):
        with patch.object(moe.subprocess, 'check_output', side_effect=[str(self.source), '0' * 40]), \
             self.assertRaisesRegex(ValueError, 'semantic review'):
            moe.verify_reviewed_source(self.source)
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / 'changed.patch'
            bad.write_bytes(self.patch.read_bytes() + b'\n')
            with patch.object(moe, 'verify_reviewed_source') as verify, \
                 self.assertRaisesRegex(ValueError, 'patch identity'):
                moe.prepare(self.source, Path(temporary) / 'output', bad)
            verify.assert_not_called()
            self.assertFalse((Path(temporary) / 'output').exists())

    def test_reviewed_backend_change_fails_before_embedding(self):
        with patch.object(moe, 'digest', return_value='0' * 64), \
             self.assertRaisesRegex(ValueError, 'semantic review'):
            moe.verify_reviewed_source(self.source)

    def test_actual_upstream_target_searches_overlay_before_generated_header(self):
        if not shutil.which('cmake'):
            self.skipTest('CMake required')
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary) / 'build'
            command = ['cmake', '-S', str(ROOT / 'tests/moe-direct-slot-overlay'), '-B', str(build),
                       '-DLCB_LLAMA_SOURCE=' + str(self.source)]
            subprocess.run(command, check=True, capture_output=True)
            before = (build / 'includes-before.txt').read_text().split(';')
            after = (build / 'includes-after.txt').read_text().split(';')
            overlay = build / 'moe-direct-slot-overlay'
            self.assertEqual(after, [str(overlay)] + before)
            self.assertTrue((overlay / 'ggml-wgsl-shaders.hpp').is_file())
            self.assertIn(str(overlay), (build / 'compile_commands.json').read_text())
            # Reconfigure from the same pristine inputs, without applying twice.
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual((build / 'includes-after.txt').read_text().split(';'), after)


if __name__ == '__main__':
    unittest.main()
