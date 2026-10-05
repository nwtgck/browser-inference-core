"""Real local Git checkout with two pins; no GitHub or compiler required."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import checkout_source as checkout
from source_config import get_source


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()


class CheckoutSource(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.temp = Path(tmp.name)
        self.upstream = self.temp / 'upstream'
        self.upstream.mkdir()
        git(self.upstream, 'init', '-q')
        git(self.upstream, 'config', 'user.name', 'Fixture')
        git(self.upstream, 'config', 'user.email', 'fixture@example.invalid')
        (self.upstream / 'fixture').write_text('local checkout test, not llama.cpp')
        git(self.upstream, 'add', '.')
        git(self.upstream, 'commit', '-qm', 'fixture')
        self.commit = git(self.upstream, 'rev-parse', 'HEAD')
        self.repo = self.temp / 'consumer'
        self.root = self.repo / 'llama-cpp'
        self.root.mkdir(parents=True)
        for name in ('config', 'sources', 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'):
            shutil.copytree(ROOT / name, self.root / name)
        git(self.repo, 'init', '-q')
        git(self.repo, 'config', 'user.name', 'Fixture')
        git(self.repo, 'config', 'user.email', 'fixture@example.invalid')
        modules = ''
        for name in ('upstream-stable', 'upstream-nightly'):
            entry = get_source(self.root, name)
            path = self.root / entry['pinFile']
            value = json.loads(path.read_text())
            value['llamaCommit'] = self.commit
            path.write_text(json.dumps(value))
            (self.root / entry['vendorPath']).mkdir(parents=True, exist_ok=True)
            relative = 'llama-cpp/' + entry['vendorPath']
            modules += f'[submodule "{name}"]\n path = {relative}\n url = {self.upstream.as_uri()}\n'
            git(self.repo, 'update-index', '--add', '--cacheinfo', f'160000,{self.commit},{relative}')
        (self.repo / '.gitmodules').write_text(modules)
        git(self.repo, 'add', '.')
        git(self.repo, 'commit', '-qm', 'registered sources')
        # Permit local file transport only inside this test process and children.
        environment = patch.dict(os.environ, {'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'protocol.file.allow', 'GIT_CONFIG_VALUE_0': 'always'})
        environment.start()
        self.addCleanup(environment.stop)

    def test_only_selected_source_is_fetched_and_committed_tree_is_unchanged(self):
        before = git(self.repo, 'rev-parse', 'HEAD^{tree}')
        result = checkout.checkout_source(self.root, 'upstream-nightly')
        self.assertEqual(result['commit'], self.commit)
        self.assertTrue((self.root / 'vendor/llama.cpp-nightly/fixture').is_file())
        self.assertFalse((self.root / 'vendor/llama.cpp/fixture').exists())
        self.assertEqual(git(self.repo, 'rev-parse', 'HEAD^{tree}'), before)
        self.assertEqual(git(self.repo, 'status', '--porcelain'), '')

    def test_missing_registration_stops_before_checkout(self):
        git(self.repo, 'update-index', '--force-remove', 'llama-cpp/vendor/llama.cpp-nightly')
        git(self.repo, 'commit', '-qm', 'reported missing gitlink')
        with self.assertRaisesRegex(ValueError, 'missing'):
            checkout.checkout_source(self.root, 'upstream-nightly')
        self.assertFalse((self.root / 'vendor/llama.cpp-nightly/fixture').exists())

    def test_dirty_selected_checkout_is_not_reset_or_accepted(self):
        checkout.checkout_source(self.root, 'upstream-nightly')
        file = self.root / 'vendor/llama.cpp-nightly/fixture'
        file.write_text('local edit')
        with self.assertRaisesRegex(ValueError, 'dirty'):
            checkout.checkout_source(self.root, 'upstream-nightly')
        self.assertEqual(file.read_text(), 'local edit')


    def test_staged_gitlink_divergence_rejected_before_any_update(self):
        git(self.repo, 'update-index', '--cacheinfo',
            '160000,' + 'f' * 40 + ',llama-cpp/vendor/llama.cpp-nightly')
        real_run = subprocess.run
        with patch.object(checkout.subprocess, 'run', wraps=real_run) as calls:
            with self.assertRaisesRegex(ValueError, 'index'):
                checkout.checkout_source(self.root, 'upstream-nightly')
        self.assertFalse(any('submodule' in call.args[0] for call in calls.call_args_list))
        self.assertFalse((self.root / 'vendor/llama.cpp-nightly/fixture').exists())

    def test_dirty_checkout_at_another_revision_keeps_its_head(self):
        checkout.checkout_source(self.root, 'upstream-nightly')
        directory = self.root / 'vendor/llama.cpp-nightly'
        git(directory, 'config', 'user.name', 'Fixture')
        git(directory, 'config', 'user.email', 'fixture@example.invalid')
        (directory / 'local-commit').write_text('a local commit must not be detached on failure')
        git(directory, 'add', 'local-commit')
        git(directory, 'commit', '-qm', 'local work')
        local_head = git(directory, 'rev-parse', 'HEAD')
        (directory / 'untracked-edit').write_text('do not disturb this checkout')
        with self.assertRaisesRegex(ValueError, 'dirty'):
            checkout.checkout_source(self.root, 'upstream-nightly')
        self.assertEqual(git(directory, 'rev-parse', 'HEAD'), local_head)
        self.assertTrue((directory / 'local-commit').is_file())
        self.assertEqual((directory / 'untracked-edit').read_text(), 'do not disturb this checkout')

    def test_explicit_checkout_overrides_local_update_none(self):
        git(self.repo, 'config', 'submodule.upstream-nightly.update', 'none')
        result = checkout.checkout_source(self.root, 'upstream-nightly')
        self.assertEqual(result['commit'], self.commit)
        self.assertTrue((self.root / 'vendor/llama.cpp-nightly/fixture').is_file())
        self.assertEqual(git(self.repo, 'config', 'submodule.upstream-nightly.update'), 'none')


if __name__ == '__main__':
    unittest.main()
