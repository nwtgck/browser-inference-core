"""Real Git object selection and captured patch/runtime ownership, no compiler stub."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from build_tools.runtimes.llama_cpp import inputs
from build_tools.source.checkout import Checkout, checked_sources, git_environment


class CaptureFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='llama-inputs-')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.repository = self.work / 'repository'
        self.repository.mkdir()
        self.initialize(self.repository)
        self.runtime = self.repository / inputs.RUNTIME
        self.upstream = self.repository / inputs.UPSTREAM
        self.upstream.mkdir(parents=True)
        self.initialize(self.upstream)
        (self.upstream / 'source.cpp').write_text('committed upstream bytes\n')
        self.upstream_commit = self.commit(self.upstream)
        for relative, value in {'build_tools/__init__.py': '# fixture\n',
                                'toolchain/config.json': '{}\n',
                                inputs.RUNTIME + '/CMakeLists.txt': '# fixture CMake\n',
                                inputs.RUNTIME + '/' + inputs.PATCH_DIRECTORY + '/accepted.patch': 'committed patch bytes\n',
                                '.gitignore': 'runtimes/*/build/\n**/__pycache__/\n',
                                '.gitmodules': '[submodule "llama"]\n path = ' + inputs.UPSTREAM + '\n url = https://example.invalid/local.git\n'}.items():
            path = self.repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        self.revision = self.commit(self.repository)
        self.checkout = checked_sources(self.repository, {'llama': inputs.UPSTREAM}, revision=self.revision)['llama']
        self.output = self.runtime / 'build/fixture/source-inputs'

    def run_git(self, root, *args):
        return subprocess.check_output(['git', '-C', str(root), *args], env=git_environment(), text=True).strip()

    def initialize(self, root):
        self.run_git(root, 'init', '--quiet')
        self.run_git(root, 'config', 'advice.addEmbeddedRepo', 'false')
        self.run_git(root, 'config', 'user.name', 'Local synthetic fixture')
        self.run_git(root, 'config', 'user.email', 'fixture@example.invalid')

    def commit(self, root):
        self.run_git(root, 'add', '-A')
        self.run_git(root, 'commit', '--quiet', '-m', 'test: local synthetic input fixture')
        return self.run_git(root, 'rev-parse', 'HEAD')

    def capture(self, **kwargs):
        return inputs.capture(self.repository, self.revision, self.checkout, self.output, **kwargs)



class CapturedInputs(CaptureFixture):
    def test_source_and_patch_bytes_come_from_commits_despite_hidden_worktree_changes(self):
        patch_path = self.runtime / inputs.PATCH_DIRECTORY / 'accepted.patch'
        self.run_git(self.repository, 'update-index', '--assume-unchanged', patch_path.relative_to(self.repository).as_posix())
        patch_path.write_text('hidden patch change\n')
        self.run_git(self.upstream, 'update-index', '--assume-unchanged', 'source.cpp')
        (self.upstream / 'source.cpp').write_text('hidden upstream change\n')
        self.assertEqual(self.run_git(self.repository, 'status', '--porcelain'), '')
        captured = self.capture()
        self.assertEqual((captured.upstream / 'source.cpp').read_text(), 'committed upstream bytes\n')
        self.assertEqual((captured.runtime / inputs.PATCH_DIRECTORY / 'accepted.patch').read_text(), 'committed patch bytes\n')
        self.assertFalse((captured.upstream / '.git').exists())
        self.assertEqual(captured.receipt['upstream']['commit'], self.upstream_commit)
        self.assertEqual(captured.build_info, {'commit': self.upstream_commit[:7], 'number': 1})
        inputs.verify(captured)

    def test_original_edits_after_capture_cannot_change_inputs(self):
        captured = self.capture()
        (self.upstream / 'source.cpp').write_text('changed after capture')
        (self.runtime / inputs.PATCH_DIRECTORY / 'accepted.patch').write_text('changed after capture')
        inputs.verify(captured)
        self.assertEqual((captured.upstream / 'source.cpp').read_text(), 'committed upstream bytes\n')
        (captured.upstream / 'source.cpp').write_text('corrupted private input')
        with self.assertRaisesRegex(ValueError, 'Captured upstream inputs changed'):
            inputs.verify(captured)

    def test_dirty_local_inputs_are_explicit_and_ignore_nested_build_caches(self):
        path = self.runtime / inputs.PATCH_DIRECTORY / 'accepted.patch'
        path.write_text('local patch work\n')
        (self.runtime / 'untracked.txt').write_text('local owned work\n')
        cache = self.runtime / 'build/old/cache.txt'
        cache.parent.mkdir(parents=True)
        cache.write_text('must not be captured')
        captured = self.capture(dirty=True)
        self.assertEqual((captured.runtime / inputs.PATCH_DIRECTORY / path.name).read_text(), 'local patch work\n')
        self.assertEqual((captured.runtime / 'untracked.txt').read_text(), 'local owned work\n')
        self.assertFalse((captured.runtime / 'build').exists())
        self.assertEqual(captured.receipt['parentInputMode'], 'dirty-local-worktree')
        inputs.verify(captured)

    def test_noop_capture_preserves_identity_and_failure_keeps_previous_tree(self):
        first = self.capture()
        original = (first.upstream / 'source.cpp').stat()
        second = self.capture()
        after = (second.upstream / 'source.cpp').stat()
        self.assertEqual((original.st_ino, original.st_mtime_ns), (after.st_ino, after.st_mtime_ns))
        self.assertEqual(first.receipt, second.receipt)
        with patch.object(inputs, 'export_commit', side_effect=ValueError('capture failure')), self.assertRaisesRegex(ValueError, 'capture failure'):
            self.capture()
        inputs.verify(first)
        self.assertEqual((first.upstream / 'source.cpp').stat().st_ino, original.st_ino)

    def test_unrelated_gitlink_is_not_materialized(self):
        self.run_git(self.repository, 'update-index', '--add', '--cacheinfo', f'160000,{self.upstream_commit},runtimes/other/vendor/unavailable')
        self.run_git(self.repository, 'commit', '--quiet', '-m', 'test: unavailable unrelated runtime')
        self.revision = self.run_git(self.repository, 'rev-parse', 'HEAD')
        captured = self.capture()
        self.assertFalse((captured.root / 'repository/runtimes/other').exists())
        inputs.verify(captured)

    def test_source_owned_or_linked_destinations_are_rejected(self):
        for output in [self.repository, self.runtime, self.runtime / 'native', self.upstream, self.upstream / 'build']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                inputs.capture(self.repository, self.revision, self.checkout, output)
        linked = self.work / 'linked'
        linked.symlink_to(self.runtime / 'build', target_is_directory=True)
        with self.assertRaises(ValueError):
            inputs.capture(self.repository, self.revision, self.checkout, linked)
