"""Late preparation failures retain complete prior sources and sibling components."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import runtime_root
from build_tools.runtimes.llama_cpp import source, webgpu_plan
from build_tools.runtimes.llama_cpp.sources import file_backed_rows, mtmd, qsa_selection

ROOT = runtime_root('llama-cpp')


def contents(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


class SourcePublication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (cls.upstream / 'include/llama.h').exists():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')

    def test_late_shader_failure_preserves_complete_output_and_success_removes_profile_files(self):
        with tempfile.TemporaryDirectory(prefix='source-publication-') as temporary:
            output = Path(temporary) / 'output'
            source.prepare(self.upstream, output, gpu_profile=True)
            before = contents(output)
            stamps = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in output.iterdir()}
            with patch.object(webgpu_plan, 'generate', side_effect=RuntimeError('last stage failed')):
                with self.assertRaisesRegex(RuntimeError, 'last stage failed'):
                    source.prepare(self.upstream, output)
            self.assertEqual(contents(output), before)
            self.assertEqual({p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in output.iterdir()}, stamps)
            source.prepare(self.upstream, output)
            self.assertNotIn('ggml-webgpu-q8-diagnostic.h', contents(output))
            self.assertNotIn('ggml-webgpu-q8-diagnostic-sha256.hpp', contents(output))

    def test_model_tree_publish_waits_for_companion_header_and_preserves_unchanged_files(self):
        with tempfile.TemporaryDirectory(prefix='rows-publication-') as temporary:
            output = Path(temporary) / 'output'
            file_backed_rows.prepare(self.upstream, output)
            before = contents(output)
            marker = output / 'obsolete.txt'
            marker.write_text('previous output marker')
            previous = contents(output)
            real_copy = shutil.copyfile
            def fail_companion(src, dst, *args, **kwargs):
                if Path(src).name == 'file-backed-rows.h':
                    raise OSError('companion copy failed')
                return real_copy(src, dst, *args, **kwargs)
            with patch.object(shutil, 'copyfile', side_effect=fail_companion), self.assertRaisesRegex(OSError, 'companion copy failed'):
                file_backed_rows.prepare(self.upstream, output)
            self.assertEqual(contents(output), previous)
            sample = output / 'src/models/models.h'
            stamp = sample.stat().st_mtime_ns
            file_backed_rows.prepare(self.upstream, output)
            self.assertEqual(contents(output), before)
            self.assertEqual(sample.stat().st_mtime_ns, stamp)

    def test_qsa_receipt_failure_keeps_previous_complete_directory(self):
        with tempfile.TemporaryDirectory(prefix='qsa-publication-') as temporary:
            root = Path(temporary)
            output = root / 'output'
            qsa_selection.prepare(self.upstream, self.upstream / qsa_selection.SOURCE_PATH, output)
            (output / 'previous.txt').write_text('previous marker')
            previous = contents(output)
            # Fail after patch application and output construction, before the
            # complete receipt can be published. No expected output hash gate.
            with patch.object(qsa_selection.json, 'dumps', side_effect=ValueError('receipt failure')):
                with self.assertRaisesRegex(ValueError, 'receipt failure'):
                    qsa_selection.prepare(self.upstream, self.upstream / qsa_selection.SOURCE_PATH, output)
            self.assertEqual(contents(output), previous)

    def test_mtmd_components_preserve_completed_siblings_on_later_failure(self):
        with tempfile.TemporaryDirectory(prefix='mtmd-components-') as temporary:
            root = Path(temporary)
            output = root / 'output'
            patches = ROOT / mtmd.PATCH_DIRECTORY
            mtmd.prepare(self.upstream, output, patches / 'mtmd-vision-bf16-runtime.patch', filename='mtmd.cpp')
            mtmd.prepare(self.upstream, output, patches / 'mtmd-vision-bf16-clip-runtime.patch')
            previous = contents(output)
            invalid = root / 'bad.patch'
            invalid.write_text('not a patch\n')
            import subprocess
            with self.assertRaises(subprocess.CalledProcessError):
                mtmd.prepare(self.upstream, output, invalid)
            self.assertEqual(contents(output), previous)
            stamp = (output / 'mtmd.cpp').stat().st_mtime_ns
            mtmd.prepare(self.upstream, output, patches / 'mtmd-vision-bf16-clip-runtime.patch')
            self.assertEqual((output / 'mtmd.cpp').stat().st_mtime_ns, stamp)
