"""Synthetic staging failures preserve the previously complete artifact group."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_tools.packaging.files import tree_identity
from build_tools.runtimes.stable_diffusion_cpp import staging


class ImageStaging(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build = self.root / 'build'
        self.output = self.root / 'upload/profile/browser'
        self.notices = self.root / 'upload/toolchain-licenses'
        for part, names in (('runtime', ('core.mjs', 'core.wasm', 'core.d.ts')),
                            ('generated', staging.API_FILES)):
            (self.build / part).mkdir(parents=True)
            for name in names:
                (self.build / part / name).write_text('new synthetic payload')
        (self.build / 'provenance.json').write_text('{"fixture":true}')
        self.notice_roots = (self.root / 'emscripten', self.root / 'emdawnwebgpu_pkg')
        for root in self.notice_roots:
            root.mkdir()
            (root / 'LICENSE').write_text('full fixture notice')
        for output in (self.output, self.notices):
            output.mkdir(parents=True)
            (output / 'previous').write_text('old valid output')
        self.before = (tree_identity(self.output), tree_identity(self.notices))

    def stage(self):
        staging.stage_profile(self.build, self.output, notice_roots=self.notice_roots,
                              notice_destination=self.notices)

    def assert_previous(self):
        self.assertEqual((tree_identity(self.output), tree_identity(self.notices)), self.before)
        self.assertEqual(list(self.root.rglob('bicore-output-*')), [])

    def test_runtime_provenance_and_schema_copy_failures_keep_both_previous_outputs(self):
        for source in (self.build / 'runtime/core.wasm', self.build / 'provenance.json',
                       self.build / 'generated/schema.json'):
            with self.subTest(source=source):
                old = source.read_bytes()
                source.unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.stage()
                self.assert_previous()
                source.write_bytes(old)

    def test_notice_collection_failure_does_not_publish_the_new_runtime(self):
        (self.notice_roots[1] / 'LICENSE').unlink()
        with self.assertRaisesRegex(ValueError, 'No license notices'):
            self.stage()
        self.assert_previous()

    def test_second_activation_failure_rolls_back_the_runtime_and_notices(self):
        rename = Path.rename
        def fail(path, target):
            if path.name == 'prepared' and target == self.notices:
                raise OSError('injected notice activation')
            return rename(path, target)
        with patch.object(Path, 'rename', fail), self.assertRaisesRegex(OSError, 'notice activation'):
            self.stage()
        self.assert_previous()

    def test_success_contains_only_the_complete_payload_and_notices(self):
        self.stage()
        self.assertEqual(set(p.name for p in self.output.iterdir()), {'runtime', 'generated', 'provenance.json'})
        self.assertEqual(set(p.name for p in self.notices.iterdir()), {'emscripten', 'emdawnwebgpu_pkg'})
        self.assertEqual((self.output / 'generated/schema.json').read_text(), 'new synthetic payload')
        self.assertEqual(list(self.root.rglob('bicore-output-*')), [])
