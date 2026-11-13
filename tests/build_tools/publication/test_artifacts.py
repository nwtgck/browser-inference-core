"""Local publication state cannot overwrite its source or acquire Git objects."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from build_tools.publication import artifacts
from build_tools.source.checkout import git_environment
from tests.support.runtime_package import RuntimePackageFixture, package


class PublicationState(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.root / 'package'
        self.package.mkdir()
        self.payload = self.package / 'manifest.json'
        self.payload.write_text('original package bytes')
        self.work = self.root / 'frozen'

    def test_receipt_inside_package_is_rejected_before_validation_or_writes(self):
        for receipt in (self.payload, self.package / 'state.json', self.package / 'nested/state.json'):
            with self.subTest(receipt=receipt), patch.object(artifacts, 'validate') as validate:
                with self.assertRaisesRegex(ValueError, 'outside'):
                    artifacts.freeze(self.package, self.work, receipt)
                validate.assert_not_called()
                self.assertEqual(self.payload.read_text(), 'original package bytes')
                self.assertFalse(self.work.exists())

    def test_parent_links_do_not_hide_overlapping_outputs(self):
        link = self.root / 'linked'; link.symlink_to(self.package, target_is_directory=True)
        for work, receipt in [(self.work, link / 'receipt.json'),
                              (link / 'snapshot', self.root / 'receipt.json'),
                              (self.work, self.work / 'sub/../receipt.json')]:
            with self.subTest(work=work, receipt=receipt):
                with self.assertRaisesRegex(ValueError, 'outside'):
                    artifacts.freeze(self.package, work, receipt)
        self.assertEqual(list(self.package.iterdir()), [self.payload])
        self.assertFalse(self.work.exists())

    def test_receipt_cannot_be_the_future_snapshot_parent(self):
        receipt = self.root / 'receipt.json'
        work = receipt / 'snapshot'
        with patch.object(artifacts, 'validate') as validate, self.assertRaisesRegex(ValueError, 'outside'):
            artifacts.freeze(self.package, work, receipt)
        validate.assert_not_called()
        self.assertFalse(receipt.exists())
        self.assertEqual(self.payload.read_text(), 'original package bytes')

    def test_existing_receipt_or_hardlink_is_not_overwritten(self):
        receipt = self.root / 'receipt.json'
        os.link(self.payload, receipt)
        with self.assertRaisesRegex(ValueError, 'fresh file'):
            artifacts.freeze(self.package, self.work, receipt)
        self.assertEqual(receipt.read_text(), 'original package bytes')
        self.assertFalse(self.work.exists())

    def test_blob_reader_uses_the_same_offline_environment_as_other_git_reads(self):
        subprocess.run(['git', 'init', '-q', str(self.package)], check=True)
        subprocess.run(['git', 'add', 'manifest.json'], cwd=self.package, check=True)
        tree = subprocess.check_output(['git', 'write-tree'], cwd=self.package, text=True).strip()
        data = self.payload.read_bytes()
        identity = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        with patch.dict(os.environ, {'GIT_NO_LAZY_FETCH': '0', 'GIT_ALLOW_PROTOCOL': 'https'}), \
             patch.object(artifacts.subprocess, 'Popen', wraps=subprocess.Popen) as processes:
            artifacts.verify_git_tree(self.package, tree, {'files': []}, identity)
        readers = [call for call in processes.call_args_list if 'cat-file' in call.args[0]]
        self.assertEqual(len(readers), 1)
        self.assertEqual(readers[0].kwargs['env']['GIT_NO_LAZY_FETCH'], '1')
        self.assertEqual(readers[0].kwargs['env']['GIT_ALLOW_PROTOCOL'], '')
        self.assertEqual(readers[0].kwargs['env']['GIT_NO_REPLACE_OBJECTS'], '1')


class PublicationGitContext(RuntimePackageFixture):
    def test_freeze_and_confirm_ignore_an_unrelated_repository_environment(self):
        # Real package validation, npm packing and local Git operations; no Wasm
        # compilation, remote push, or source-provenance validation is simulated.
        package.assemble(self.inputs, self.out, check_npm_pack=False)
        foreign = self.root / 'unrelated'
        foreign.mkdir()
        clean = git_environment()

        def git(directory, *args):
            return subprocess.check_output(['git', *args], cwd=directory,
                                           env=clean, text=True).strip()

        git(foreign, 'init', '-q')
        git(foreign, 'config', 'user.name', 'Unrelated developer')
        git(foreign, 'config', 'user.email', 'unrelated@example.invalid')
        (foreign / 'keep.txt').write_text('unrelated working tree')
        git(foreign, 'add', 'keep.txt')
        git(foreign, 'commit', '-qm', 'unrelated source')
        guarded = [foreign / '.git/config', foreign / '.git/index',
                   foreign / '.git/HEAD', foreign / 'keep.txt']
        before = {path: path.read_bytes() for path in guarded}
        work, receipt = self.root / 'frozen', self.root / 'receipt.json'
        hostile = {
            'GIT_DIR': str(foreign / '.git'), 'GIT_WORK_TREE': str(foreign),
            'GIT_COMMON_DIR': str(foreign / '.git'),
            'GIT_INDEX_FILE': str(foreign / '.git/index'),
            'GIT_OBJECT_DIRECTORY': str(foreign / '.git/objects'),
            'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.worktree',
            'GIT_CONFIG_VALUE_0': str(foreign),
        }
        with patch.dict(os.environ, hostile):
            frozen = artifacts.freeze(self.out, work, receipt)
        self.assertEqual({path: path.read_bytes() for path in guarded}, before)
        self.assertEqual(git(work, 'write-tree'), frozen['tree'])
        git(work, 'commit', '-qm', 'synthetic publication')
        commit = git(work, 'rev-parse', 'HEAD')
        with patch.dict(os.environ, hostile):
            confirmed = artifacts.confirm(work, receipt, commit)
        self.assertEqual(confirmed['commit'], commit)
        self.assertEqual({path: path.read_bytes() for path in guarded}, before)

    def test_environment_config_and_global_files_cannot_override_local_policy(self):
        poison = {'GIT_CONFIG_GLOBAL': str(self.root / 'foreign-config'),
                  'GIT_CONFIG_SYSTEM': str(self.root / 'foreign-system'),
                  'GIT_CONFIG_PARAMETERS': "'core.worktree'='/somewhere-else'",
                  'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'core.worktree',
                  'GIT_CONFIG_VALUE_0': '/elsewhere', 'GIT_DIR': '/elsewhere/.git',
                  'GIT_ALLOW_PROTOCOL': 'https', 'GIT_NO_LAZY_FETCH': '0'}
        with patch.dict(os.environ, poison):
            actual = artifacts.local_environment()
        for key in ('GIT_DIR', 'GIT_CONFIG_SYSTEM', 'GIT_CONFIG_PARAMETERS',
                    'GIT_CONFIG_COUNT', 'GIT_CONFIG_KEY_0', 'GIT_CONFIG_VALUE_0'):
            self.assertNotIn(key, {name: value for name, value in actual.items() if name.startswith('GIT_')})
        self.assertEqual(actual['GIT_CONFIG_GLOBAL'], os.devnull)
        self.assertEqual(actual['GIT_CONFIG_NOSYSTEM'], '1')
        self.assertEqual(actual['GIT_ALLOW_PROTOCOL'], '')
        self.assertEqual(actual['GIT_NO_LAZY_FETCH'], '1')
