"""Shared local snapshot selection and source identity contracts."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from build_tools.source.inputs import copy_local, file_identities, identity, tree_identity


class LocalInputs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='bicore-local-inputs-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.output = self.root / 'captured'

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], stderr=subprocess.STDOUT)

    def write(self, relative, value):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

    def test_dirty_copy_selects_owned_tracked_and_unignored_files_without_cache_trees(self):
        self.write('.gitignore', '*.cache\n')
        changed = self.write('runtime/native.cpp', 'tracked before')
        removed = self.write('runtime/deleted.cpp', 'deleted')
        self.write('runtime/build/intermediate', 'tracked build product')
        self.write('runtime/vendor/upstream.cpp', 'excluded vendor')
        self.write('other-runtime/native.cpp', 'not selected')
        self.git('add', '.')
        changed.write_text('local edit')
        removed.unlink()
        self.write('runtime/new.cpp', 'untracked developer input')
        self.write('runtime/ignored.cache', 'ignored cache')
        self.write('runtime/build/current', 'untracked build product')
        copy_local(self.repo, self.output, include=('runtime',), exclude=('runtime/vendor',))
        self.assertEqual({name: (self.output / name).read_text() for name in file_identities(self.output)},
                         {'runtime/native.cpp': 'local edit', 'runtime/new.cpp': 'untracked developer input'})

    def test_dirty_copy_rejects_file_and_parent_links(self):
        target = self.write('outside/secret', 'never dereference')
        alias = self.repo / 'runtime/input'
        alias.parent.mkdir()
        alias.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'Linked or non-regular'):
            copy_local(self.repo, self.output, include=('runtime',))
        with self.assertRaisesRegex(ValueError, 'Linked or non-regular'):
            identity(alias)
        parent = self.repo / 'linked'
        parent.symlink_to(target.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Linked or non-directory'):
            identity(parent / target.name)

    def test_source_identities_record_relative_links_without_following_targets(self):
        self.write('source/native.cpp', 'source bytes')
        source = self.repo / 'source'
        (source / 'alias').symlink_to('native.cpp')
        files = file_identities(source)
        self.assertEqual(files['alias'], {'symlink': 'native.cpp'})
        self.assertEqual(files['native.cpp'], identity(source / 'native.cpp'))
        self.assertEqual(tree_identity(files), tree_identity(dict(reversed(list(files.items())))))
        files['alias'] = {'symlink': 'other.cpp'}
        self.assertNotEqual(tree_identity(files), tree_identity(file_identities(source)))

    def test_linked_root_or_special_entry_cannot_disappear_from_an_identity(self):
        alias = self.root / 'linked-repository'
        alias.symlink_to(self.repo, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Linked or non-directory'):
            file_identities(alias)
        if hasattr(os, 'mkfifo'):
            source = self.repo / 'source'
            source.mkdir()
            os.mkfifo(source / 'pipe')
            with self.assertRaisesRegex(ValueError, 'Non-regular captured source entry'):
                file_identities(source)
        with self.assertRaisesRegex(ValueError, 'Invalid relative'):
            file_identities(self.repo, exclude=('../escape',))

    def test_empty_local_selection_is_rejected_before_creating_output(self):
        with self.assertRaisesRegex(ValueError, 'included source'):
            copy_local(self.repo, self.output, include=())
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
