"""Pinned source contents are Git objects, not a status-checked worktree."""
import os
from pathlib import Path
import shutil
from unittest.mock import patch

from build_tools.source.snapshot import export_commit
from build_tools.source.checkout import checked_sources
from tests.build_tools.source.support import SourceFixture


class CommitSnapshot(SourceFixture):
    def setUp(self):
        super().setUp()
        self.checkout = self.sources['stable-diffusion']
        self.initialize(self.checkout)
        self.commit = self.commit_files(self.checkout)

    def initialize(self, path):
        self.git(path, 'init', '-q')
        self.git(path, 'config', 'user.name', 'fixture')
        self.git(path, 'config', 'user.email', 'fixture@example.invalid')

    def commit_files(self, path):
        self.git(path, 'add', '-A')
        self.git(path, 'commit', '--allow-empty', '-qm', 'fixture source')
        return self.git(path, 'rev-parse', 'HEAD').decode().strip()

    def test_reads_commit_despite_assume_unchanged_worktree_edit(self):
        self.git(self.checkout, 'update-index', '--assume-unchanged', 'a.txt')
        (self.checkout / 'a.txt').write_text('locally different bytes\n')
        self.assertEqual(self.git(self.checkout, 'status', '--porcelain'), b'')
        export_commit(self.checkout, self.commit, self.output)
        self.assertEqual((self.output / 'a.txt').read_text(), 'one\nbefore\n')

    def test_skip_worktree_and_missing_file_do_not_select_contents(self):
        self.git(self.checkout, 'update-index', '--skip-worktree', 'a.txt')
        (self.checkout / 'a.txt').unlink()
        export_commit(self.checkout, self.commit, self.output)
        self.assertEqual((self.output / 'a.txt').read_text(), 'one\nbefore\n')

    def test_index_and_head_changes_do_not_change_selected_commit(self):
        (self.checkout / 'a.txt').write_text('new commit\n')
        self.commit_files(self.checkout)
        (self.checkout / 'a.txt').write_text('new index\n')
        self.git(self.checkout, 'add', 'a.txt')
        export_commit(self.checkout, self.commit, self.output)
        self.assertEqual((self.output / 'a.txt').read_text(), 'one\nbefore\n')

    def test_git_replace_refs_cannot_redefine_selected_commit(self):
        (self.checkout / 'a.txt').write_text('replacement\n')
        changed = self.commit_files(self.checkout)
        self.git(self.checkout, 'replace', self.commit, changed)
        self.assertEqual(self.git(self.checkout, 'show', self.commit + ':a.txt'), b'replacement\n')
        export_commit(self.checkout, self.commit, self.output)
        self.assertEqual((self.output / 'a.txt').read_text(), 'one\nbefore\n')

    def test_attributes_and_external_filters_do_not_transform_inputs(self):
        (self.checkout / '.gitattributes').write_text('a.txt export-ignore\ntext export-subst\n')
        (self.checkout / 'text').write_text('$Format:%H$\n')
        commit = self.commit_files(self.checkout)
        export_commit(self.checkout, commit, self.output)
        self.assertTrue((self.output / 'a.txt').is_file())
        self.assertEqual((self.output / 'text').read_text(), '$Format:%H$\n')

    def test_modes_links_binary_and_large_blobs_are_preserved(self):
        (self.checkout / 'tool').write_bytes(b'run\n')
        (self.checkout / 'tool').chmod(0o755)
        (self.checkout / 'alias').symlink_to('tool')
        blob = bytes(range(256)) * 10000
        (self.checkout / 'bytes').write_bytes(blob)
        commit = self.commit_files(self.checkout)
        export_commit(self.checkout, commit, self.output)
        self.assertEqual((self.output / 'tool').stat().st_mode & 0o777, 0o755)
        self.assertEqual(os.readlink(self.output / 'alias'), 'tool')
        self.assertEqual((self.output / 'bytes').read_bytes(), blob)
        self.assertNotEqual((self.output / 'tool').stat().st_ino, (self.checkout / 'tool').stat().st_ino)

    def test_ignored_untracked_and_excluded_metadata_do_not_appear(self):
        (self.checkout / 'local').write_text('untracked')
        (self.checkout / 'build').mkdir()
        (self.checkout / 'build/input').write_text('excluded build state')
        self.git(self.checkout, 'add', 'build/input')
        self.git(self.checkout, 'commit', '-qm', 'tracked excluded build directory')
        commit = self.git(self.checkout, 'rev-parse', 'HEAD').decode().strip()
        export_commit(self.checkout, commit, self.output)
        for name in ('.git', 'local', 'build'):
            self.assertFalse((self.output / name).exists())

    def test_nested_repository_uses_parent_gitlink_not_nested_head(self):
        nested = self.checkout / 'nested'
        nested.mkdir()
        self.initialize(nested)
        (nested / 'data').write_text('pinned nested contents')
        child = self.commit_files(nested)
        self.git(self.checkout, 'update-index', '--add', '--cacheinfo', f'160000,{child},nested')
        self.git(self.checkout, 'commit', '-qm', 'nested pin')
        parent = self.git(self.checkout, 'rev-parse', 'HEAD').decode().strip()
        (nested / 'data').write_text('different nested HEAD')
        self.commit_files(nested)
        export_commit(self.checkout, parent, self.output)
        self.assertEqual((self.output / 'nested/data').read_text(), 'pinned nested contents')

    def test_missing_nested_objects_never_fall_back_to_parent_repository(self):
        nested = self.checkout / 'nested'
        nested.mkdir()
        self.git(self.checkout, 'update-index', '--add', '--cacheinfo', f'160000,{self.commit},nested')
        self.git(self.checkout, 'commit', '-qm', 'uninitialized nested pin')
        parent = self.git(self.checkout, 'rev-parse', 'HEAD').decode().strip()
        with self.assertRaisesRegex(ValueError, 'Initialize the exact'):
            export_commit(self.checkout, parent, self.output)

    def test_external_link_is_rejected_without_reading_its_target(self):
        (self.checkout / 'alias').symlink_to('../outside')
        commit = self.commit_files(self.checkout)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            export_commit(self.checkout, commit, self.output)

    def test_inherited_git_environment_cannot_redirect_reader(self):
        with patch.dict(os.environ, {'GIT_DIR': '/not/a/repository', 'GIT_WORK_TREE': '/not/a/worktree'}):
            export_commit(self.checkout, self.commit, self.output)
        self.assertEqual((self.output / 'a.txt').read_text(), 'one\nbefore\n')

    def test_requires_full_commit_identity_not_a_tree_or_ref(self):
        tree = self.git(self.checkout, 'rev-parse', 'HEAD^{tree}').decode().strip()
        for value in ('HEAD', self.commit[:12], tree, 'f' * 40):
            with self.subTest(value=value), self.assertRaises(ValueError):
                export_commit(self.checkout, value, self.output)
            self.assertFalse(self.output.exists())

    def test_checkout_line_endings_are_not_the_commit_contents(self):
        (self.checkout / '.gitattributes').write_text('tool.bat text eol=crlf\n')
        path = self.checkout / 'tool.bat'
        path.write_bytes(b'@echo off\r\necho fixture\r\n')
        commit = self.commit_files(self.checkout)
        blob = self.git(self.checkout, 'show', commit + ':tool.bat')
        self.assertEqual(blob, b'@echo off\necho fixture\n')
        self.assertNotEqual(path.read_bytes(), blob)
        export_commit(self.checkout, commit, self.output)
        self.assertEqual((self.output / 'tool.bat').read_bytes(), blob)


    def test_selection_excludes_uninitialized_unrelated_gitlink(self):
        self.git(self.checkout, 'update-index', '--add', '--cacheinfo', f'160000,{self.commit},unavailable')
        self.git(self.checkout, 'commit', '-qm', 'unavailable unrelated dependency')
        parent = self.git(self.checkout, 'rev-parse', 'HEAD').decode().strip()
        export_commit(self.checkout, parent, self.output, include=('a.txt',), exclude=('unavailable',))
        self.assertEqual([p.name for p in self.output.iterdir()], ['a.txt'])

    def test_invalid_or_missing_selection_fails_before_export(self):
        for include in [(), ('missing',), ('../outside',), ('.git',)]:
            with self.subTest(include=include), self.assertRaises(ValueError):
                export_commit(self.checkout, self.commit, self.output, include=include)
            self.assertFalse(self.output.exists())
