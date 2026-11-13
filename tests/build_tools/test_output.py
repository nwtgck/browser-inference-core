"""Directory activation preserves complete earlier outputs on ordinary failures."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from build_tools.output import output_directories, preserve_unchanged


class OutputDirectories(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.a, self.b = self.root / 'a', self.root / 'b'
        for output in (self.a, self.b):
            output.mkdir(mode=0o700)
            (output / 'payload').write_text('old-' + output.name)

    def assert_original(self):
        for output in (self.a, self.b):
            self.assertEqual((output / 'payload').read_text(), 'old-' + output.name)
        self.assertEqual(list(self.root.glob('bicore-output-*')), [])

    def test_completed_group_replaces_old_outputs(self):
        with output_directories(self.a, self.b, replace=True) as outputs:
            for output in outputs:
                self.assertEqual(list(output.iterdir()), [])
                (output / 'new').write_text('complete')
            self.assertEqual((self.a / 'payload').read_text(), 'old-a')
        for output in (self.a, self.b):
            self.assertEqual([p.name for p in output.iterdir()], ['new'])
        self.assertEqual(list(self.root.glob('bicore-output-*')), [])

    def test_preparation_failure_keeps_previous_bytes_and_inodes(self):
        inode = self.a.stat().st_ino
        with self.assertRaisesRegex(OSError, 'write failure'):
            with output_directories(self.a, self.b, replace=True) as (a, b):
                (a / 'partial').write_text('unfinished')
                raise OSError('write failure')
        self.assert_original()
        self.assertEqual(self.a.stat().st_ino, inode)

    def test_second_activation_failure_restores_both_outputs(self):
        rename = Path.rename
        def fail(path, target):
            if path.name == 'prepared' and target == self.b:
                raise OSError('second activation')
            return rename(path, target)
        with patch.object(Path, 'rename', fail), self.assertRaisesRegex(OSError, 'second activation'):
            with output_directories(self.a, self.b, replace=True) as outputs:
                for output in outputs:
                    (output / 'new').write_text('candidate')
        self.assert_original()

    def test_later_backup_failure_restores_already_activated_output(self):
        rename = Path.rename
        def fail(path, target):
            if path == self.b:
                raise OSError('second backup')
            return rename(path, target)
        with patch.object(Path, 'rename', fail), self.assertRaisesRegex(OSError, 'second backup'):
            with output_directories(self.a, self.b, replace=True) as outputs:
                for output in outputs:
                    (output / 'new').write_text('candidate')
        self.assert_original()

    def test_recovery_failure_retains_and_names_the_previous_output(self):
        rename = Path.rename
        def fail(path, target):
            if path.name in ('prepared', 'previous') and target == self.b:
                raise OSError('activation and recovery')
            return rename(path, target)
        with patch.object(Path, 'rename', fail), self.assertRaisesRegex(RuntimeError, 'previous output retained at') as error:
            with output_directories(self.a, self.b, replace=True) as outputs:
                for output in outputs:
                    (output / 'new').write_text('candidate')
        previous, = self.root.glob('bicore-output-*/previous')
        self.assertIn(str(previous), str(error.exception))
        self.assertEqual((previous / 'payload').read_text(), 'old-b')
        self.assertEqual((self.a / 'payload').read_text(), 'old-a')

    def test_output_appearing_or_disappearing_during_preparation_is_rejected(self):
        new = self.root / 'new'
        with self.assertRaisesRegex(ValueError, 'appeared'):
            with output_directories(new, replace=True):
                new.mkdir()
                (new / 'keep').write_text('another writer')
        self.assertEqual((new / 'keep').read_text(), 'another writer')
        with self.assertRaisesRegex(ValueError, 'disappeared'):
            with output_directories(self.a, replace=True):
                shutil.rmtree(self.a)
        self.assertFalse(self.a.exists())
        self.assertEqual(list(self.root.glob('bicore-output-*')), [])

    def test_reuse_callback_cannot_replace_destination_without_detection(self):
        saved = self.root / 'other'
        def callback(work, previous):
            previous.rename(saved)
            previous.mkdir()
            (previous / 'independent').write_text('keep')
            return True
        with self.assertRaisesRegex(ValueError, 'destination changed'):
            with output_directories(self.a, replace=True, reuse_if=callback):
                pass
        self.assertEqual((self.a / 'independent').read_text(), 'keep')
        self.assertEqual((saved / 'payload').read_text(), 'old-a')

    def test_rechecks_later_destination_before_activation(self):
        rename = Path.rename
        def changed(path, target):
            result = rename(path, target)
            if path.name == 'prepared' and target == self.a:
                rename(self.b, self.root / 'other-b')
                self.b.mkdir()
                (self.b / 'independent').write_text('keep')
            return result
        with patch.object(Path, 'rename', changed), self.assertRaisesRegex(ValueError, 'destination changed'):
            with output_directories(self.a, self.b, replace=True):
                pass
        self.assertEqual((self.a / 'payload').read_text(), 'old-a')
        self.assertEqual((self.b / 'independent').read_text(), 'keep')

    def test_rejects_duplicates_overlap_links_and_existing_default(self):
        linked = self.root / 'link'
        linked.symlink_to(self.a, target_is_directory=True)
        for targets in ((), (self.a, self.a), (self.a, self.a / 'inside'), (linked / 'new',), (linked,), (self.a,)):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                with output_directories(*targets):
                    self.fail('invalid output accepted')
        self.assert_original()

    def test_identical_output_keeps_inodes_and_timestamps(self):
        original = self.a / 'payload'
        os.utime(original, ns=(123456789, 123456789))
        before = original.stat()
        with output_directories(self.a, replace=True, reuse_if=preserve_unchanged) as (work,):
            shutil.copy2(original, work / 'payload')
        after = original.stat()
        self.assertEqual((before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns))

    def test_changed_output_preserves_equal_file_time_without_hardlinks(self):
        original = self.a / 'payload'
        os.utime(original, ns=(123456789, 123456789))
        before = original.stat()
        with output_directories(self.a, replace=True, reuse_if=preserve_unchanged) as (work,):
            (work / 'payload').write_text(original.read_text())
            (work / 'new').write_text('added')
        after = (self.a / 'payload').stat()
        self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
        self.assertNotEqual(before.st_ino, after.st_ino)


if __name__ == '__main__':
    unittest.main()
