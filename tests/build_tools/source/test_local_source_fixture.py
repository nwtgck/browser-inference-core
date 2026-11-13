"""The archive fallback owns synthetic history and never mutates the source."""
from pathlib import Path
import tempfile
import unittest

from build_tools.source.checkout import git
from tests.support.local_source import source_repository


class LocalSourceFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'archive'
        self.source.mkdir()
        (self.source / 'input.txt').write_text('original\n')

    def test_archive_receives_explicit_fixture_identity_not_an_upstream_claim(self):
        result = source_repository(self.source, self.root / 'fixture')
        self.assertEqual((result / 'input.txt').read_bytes(), (self.source / 'input.txt').read_bytes())
        self.assertIn('not verified upstream Git history', git(result, 'log', '-1', '--format=%B'))
        self.assertFalse((self.source / '.git').exists())
        (result / 'input.txt').write_text('changed fixture\n')
        self.assertEqual((self.source / 'input.txt').read_text(), 'original\n')

    def test_existing_checkout_is_local_clone_and_keeps_commit_identity(self):
        original = source_repository(self.source, self.root / 'fixture')
        clone = source_repository(original, self.root / 'clone')
        self.assertEqual(git(original, 'rev-parse', 'HEAD'), git(clone, 'rev-parse', 'HEAD'))
        self.assertEqual(git(clone, 'status', '--porcelain'), '')

    def test_existing_or_overlapping_destination_is_not_touched(self):
        for destination in (self.source, self.source / 'nested', self.root):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                source_repository(self.source, destination)
        self.assertEqual((self.source / 'input.txt').read_text(), 'original\n')
