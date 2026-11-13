"""Exercise checked coherent build-tree preparation without changing upstream."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_mtmd as mtmd
import update_llama_cpp as updater
from mtmd_fixture_support import CheckedMtmdFixture, ORIGINAL, CHANGED


class MtmdOverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='lcb-mtmd-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.patches = self.root / 'source', self.root / 'patches'
        fixture = CheckedMtmdFixture(self.source, self.patches)
        fixture.install(self)
        self.file = self.source / 'tools/mtmd/mtmd-model.cpp'

    def prepare(self, output=None, mode='webgpu'):
        return mtmd.prepare(self.source, output or self.root / 'build', mode=mode, patch_root=self.patches)

    def test_reconfigure_is_idempotent_and_preserves_the_upstream_file(self):
        tree = self.prepare()
        before = {p.relative_to(tree): p.stat().st_mtime_ns for p in tree.rglob('*') if p.is_file()}
        self.assertEqual((tree / 'mtmd-model.cpp').read_text(), CHANGED)
        self.prepare()
        self.assertEqual(before, {p.relative_to(tree): p.stat().st_mtime_ns for p in tree.rglob('*') if p.is_file()})
        self.assertEqual(self.file.read_text(), ORIGINAL)

    def test_parallel_profiles_do_not_share_patched_source_files(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            trees = list(pool.map(lambda i: self.prepare(self.root / str(i)), range(3)))
        self.assertEqual(len(set(trees)), 3)
        self.assertTrue(all((t / 'mtmd-model.cpp').read_text() == CHANGED for t in trees))
        self.assertEqual(self.file.read_text(), ORIGINAL)

    def test_changed_upstream_fails_without_overwriting_a_previous_overlay(self):
        tree = self.prepare()
        before = mtmd.inventory(tree)
        self.file.write_text('upstream drift\n')
        with self.assertRaisesRegex(ValueError, 'semantic review'):
            self.prepare()
        self.assertEqual(mtmd.inventory(tree), before)
        self.assertEqual(self.file.read_text(), 'upstream drift\n')

    def test_overlapping_source_and_output_are_rejected(self):
        for output in (self.source, self.source / 'build', self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.prepare(output)
        self.assertEqual(self.file.read_text(), ORIGINAL)

    def test_symlink_to_upstream_is_rejected(self):
        link = self.root / 'linked'
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.prepare(link)
        self.assertEqual(self.file.read_text(), ORIGINAL)

    def test_missing_source_does_not_create_a_patched_file(self):
        self.file.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or linked'):
            self.prepare()
        self.assertFalse((self.root / 'build').exists())

    def test_real_upstream_patch_and_compiler_syntax(self):
        # Full TU syntax is a separate compile gate; this test verifies the real
        # source composition and upstream allocation boundary without compiling.
        source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        import importlib.util
        spec = importlib.util.spec_from_file_location('real_mtmd_preparer', ROOT / 'scripts/prepare_mtmd.py')
        real = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real)
        original = real.inventory(source / 'tools/mtmd')
        for mode in ('cpu', 'webgpu'):
            tree = real.prepare(source, self.root / ('real-' + mode), mode=mode)
            manifest = json.loads((tree / real.MANIFEST_NAME).read_text())
            self.assertEqual(original, manifest['sourceInputs'])
            self.assertEqual(set(original), set(manifest['preparedFiles']))
            self.assertEqual(manifest['mode'], mode)
            for name in original:
                if name.startswith('models/'):
                    self.assertEqual(original[name], manifest['preparedFiles'][name])
            code = (tree / 'mtmd-subcontext.cpp').read_text()
            guard = 'if (!ggml_backend_sched_alloc_graph(backends.sched.get(), gf)) {'
            self.assertEqual(code.count(guard), 1)  # ordinary encode boundary
            if mode == 'webgpu':
                encode = code[code.index('bool mtmd_subcontext::encode'):]
                self.assertLess(encode.index(guard), encode.index('retained_plan.verify_allocated'))
                self.assertLess(encode.index(guard), encode.index('lcb_clip: matmul placement'))
            self.assertFalse((tree / 'clip.h').exists())
            self.assertFalse((tree / 'clip.cpp').exists())
        self.assertEqual(real.inventory(source / 'tools/mtmd'), original)

    def test_stale_files_and_linked_copies_are_rejected(self):
        tree = self.prepare()
        for name in ('clip.h', 'old-audio-overlay.cpp'):
            stale = tree / name
            stale.write_text('stale\n')
            with self.assertRaisesRegex(ValueError, 'stale'):
                self.prepare()
            stale.unlink()
        target = tree / 'mtmd-model.h'
        target.unlink()
        target.symlink_to(self.source / 'tools/mtmd/mtmd-model.h')
        with self.assertRaisesRegex(ValueError, 'Linked'):
            self.prepare()

    def test_all_stages_are_checked_before_publication(self):
        tree = self.prepare()
        before = mtmd.inventory(tree)
        last = self.patches / 'mtmd-non-pthread-threadpool.patch'
        last.write_bytes(last.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'patch identity'):
            self.prepare()
        self.assertEqual(mtmd.inventory(tree), before)

    def test_manifest_binds_headers_builders_and_distinct_modes(self):
        cpu, gpu = self.prepare(self.root / 'cpu', 'cpu'), self.prepare(self.root / 'gpu', 'webgpu')
        self.assertEqual((cpu / 'mtmd-model.cpp').read_text(), ORIGINAL)
        self.assertEqual((gpu / 'mtmd-model.cpp').read_text(), CHANGED)
        for tree in (cpu, gpu):
            manifest = json.loads((tree / mtmd.MANIFEST_NAME).read_text())
            self.assertEqual(manifest['preparedFiles'], {k:v for k,v in mtmd.inventory(tree).items() if k != mtmd.MANIFEST_NAME})
            self.assertEqual(manifest['sourceInputs']['models/model.cpp'], manifest['preparedFiles']['models/model.cpp'])
            self.assertEqual(manifest['preparedFiles']['mtmd-model.h']['sha256'], hashlib.sha256(CHANGED.encode()).hexdigest())

    def test_actual_git_failure_keeps_text_diagnostics_and_previous_output(self):
        target = self.prepare()
        before = mtmd.inventory(target)
        path = self.patches / 'mtmd-vision-bf16-runtime.patch'
        # Register a synthetic exact patch identity whose hunk cannot apply.
        # The actual Git command must fail, not a checksum guard or a mock.
        path.write_bytes(path.read_bytes().replace(b'-original\n', b'-different input\n'))
        mtmd.STAGES[path.name]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.prepare()
        self.assertIsInstance(caught.exception.stderr, str)
        self.assertIsInstance(caught.exception.stdout, str)
        self.assertIn('patch does not apply', caught.exception.stderr)
        self.assertEqual(mtmd.inventory(target), before)
        self.assertEqual(self.file.read_text(), ORIGINAL)

    def test_actual_git_failure_is_serializable_in_updater_reports(self):
        root = self.root / 'updater'
        source = root / 'vendor/llama.cpp'
        shutil.copytree(self.source, source)
        patches = root / mtmd.PATCH_DIRECTORY
        shutil.copytree(self.patches, patches)
        path = patches / 'mtmd-vision-bf16-runtime.patch'
        path.write_bytes(path.read_bytes().replace(b'-original\n', b'-different input\n'))
        mtmd.STAGES[path.name]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = updater.overlay_preflight(root)
        self.assertEqual(result['status'], 'failed')
        report = {'preflight': result}
        # Exercise both real reporting operations after actual Git failure.
        destination = self.root / 'report.json'
        destination.write_text(json.dumps(report, indent=2) + '\n')
        self.assertEqual(json.loads(destination.read_text()), report)
        self.assertIn('patch does not apply', updater.render_summary(report))
        self.assertIsInstance(result['error'], str)
        self.assertIn('mtmd.cpp', result['error'])
        self.assertFalse((root / 'build').exists())
        self.assertEqual((source / 'tools/mtmd/mtmd.cpp').read_text(), ORIGINAL)
