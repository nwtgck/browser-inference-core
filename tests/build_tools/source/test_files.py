"""Concurrent parent creation must not weaken the no-link path boundary."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_tools.source.files import directory


class SourceDirectories(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / 'shared-parent'

    def test_parent_created_by_another_profile_is_accepted(self):
        original = Path.mkdir
        def create(path, *args, **kwargs):
            if path == self.target and not path.exists():
                original(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'mkdir', create):
            self.assertEqual(directory(self.target, create=True), self.target)

    def test_concurrently_created_link_is_still_rejected(self):
        actual = self.root / 'another-directory'
        actual.mkdir()
        original = Path.mkdir
        def create(path, *args, **kwargs):
            if path == self.target and not path.exists():
                path.symlink_to(actual, target_is_directory=True)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'mkdir', create):
            with self.assertRaisesRegex(ValueError, 'Linked'):
                directory(self.target, create=True)
        self.assertEqual(list(actual.iterdir()), [])
