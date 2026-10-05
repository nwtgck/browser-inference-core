"""Exercise real committed/staged Git states, including file-only patch application."""
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
sys.path.insert(0, str(ROOT / 'llama-cpp/scripts'))
from source_config import get_source, check_gitlink
from source_git import committed_entry
from register_source_gitlinks import stage_missing, index_entries, check_index


def git(root, *args, input=None):
    return subprocess.run(['git', *args], cwd=root, input=input, text=True,
                          check=True, capture_output=True).stdout.strip()


class SourceGitlinks(unittest.TestCase):
    def setUp(self):
        # Test the local migration contract even when this suite runs on Actions.
        environment = patch.dict(os.environ, {'GITHUB_ACTIONS': 'false'})
        environment.start()
        self.addCleanup(environment.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix='bic source git ')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.root = self.repo / 'llama-cpp'
        self.root.mkdir()
        for name in ('config', 'sources', 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'):
            shutil.copytree(ROOT / 'llama-cpp' / name, self.root / name)
        self.sources = [get_source(self.root, name) for name in ('upstream-stable', 'upstream-nightly')]
        text = ''
        for source in self.sources:
            text += (f'[submodule "{source["id"]}"]\n'
                     f'\tpath = llama-cpp/{source["vendorPath"]}\n'
                     f'\turl = https://github.com/{source["repository"]}.git\n')
        (self.repo / '.gitmodules').write_text(text)
        git(self.repo, 'init', '-q')
        git(self.repo, 'config', 'user.name', 'Test')
        git(self.repo, 'config', 'user.email', 'test@example.invalid')
        git(self.repo, 'add', '-A')
        self.add_link(self.sources[0])
        git(self.repo, 'commit', '-qm', 'stable baseline')

    def add_link(self, source, commit=None):
        (self.root / source['vendorPath']).mkdir(parents=True, exist_ok=True)
        git(self.repo, 'update-index', '--add', '--cacheinfo',
            f'160000,{commit or source["commit"]},llama-cpp/{source["vendorPath"]}')

    def index_tree(self):
        return git(self.repo, 'write-tree')

    def test_real_stable_link_passes_from_runtime_subdirectory(self):
        check_gitlink(self.root, self.sources[0])
        path, entry = committed_entry(self.root, self.sources[0]['vendorPath'])
        self.assertEqual(path, 'llama-cpp/vendor/llama.cpp')
        self.assertEqual(entry['commit'], self.sources[0]['commit'])

    def test_missing_link_error_names_source_expected_pin_and_actual_state(self):
        with self.assertRaisesRegex(ValueError, 'upstream-nightly.*HEAD:llama-cpp/vendor/llama.cpp-nightly') as caught:
            check_gitlink(self.root, self.sources[1])
        text = str(caught.exception)
        for value in ('missing', self.sources[1]['commit'], self.sources[1]['pinFile'], '--index'):
            self.assertIn(value, text)

    def test_staging_is_not_a_committed_source_fix(self):
        self.add_link(self.sources[1])
        with self.assertRaisesRegex(ValueError, 'missing'):
            check_gitlink(self.root, self.sources[1])
        git(self.repo, 'commit', '-qm', 'register nightly')
        check_gitlink(self.root, self.sources[1])

    def test_file_only_apply_reproduces_missing_gitlink_and_registration_repairs_it(self):
        source = self.sources[1]
        path = 'llama-cpp/' + source['vendorPath']
        delta = (f'diff --git a/{path} b/{path}\nnew file mode 160000\n'
                 f'index 0000000..{source["commit"][:7]}\n--- /dev/null\n'
                 f'+++ b/{path}\n@@ -0,0 +1 @@\n+Subproject commit {source["commit"]}\n')
        git(self.repo, 'apply', input=delta)
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '--allow-empty', '-qm', 'file-only apply')
        self.assertTrue((self.repo / path).is_dir())
        with self.assertRaisesRegex(ValueError, 'missing'):
            check_gitlink(self.root, source)
        pending = stage_missing(self.root, [s['id'] for s in self.sources])
        self.assertEqual([item['path'] for item in pending], [path])
        self.assertEqual(stage_missing(self.root, [source['id']]), [])
        with self.assertRaisesRegex(ValueError, 'missing'):
            check_gitlink(self.root, source)
        git(self.repo, 'commit', '-qm', 'explicit registration')
        check_gitlink(self.root, source)
        check_gitlink(self.root, self.sources[0])

    def test_wrong_committed_pin_is_diagnosed_not_silently_repaired(self):
        self.add_link(self.sources[1], 'a' * 40)
        git(self.repo, 'commit', '-qm', 'wrong nightly')
        with self.assertRaisesRegex(ValueError, 'actual 160000 commit ' + 'a' * 40):
            check_gitlink(self.root, self.sources[1])
        before = self.index_tree()
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            stage_missing(self.root, [self.sources[1]['id']])
        self.assertEqual(self.index_tree(), before)

    def test_populated_non_git_directory_is_not_turned_into_a_submodule(self):
        path = self.root / self.sources[1]['vendorPath']
        path.mkdir(parents=True)
        (path / 'file.cpp').write_text('// arbitrary checkout\n')
        before = self.index_tree()
        with self.assertRaisesRegex(ValueError, 'not its own Git checkout'):
            stage_missing(self.root, [self.sources[1]['id']])
        self.assertEqual(before, self.index_tree())

    def test_all_sources_are_preflighted_before_any_index_write(self):
        git(self.repo, 'update-index', '--force-remove', 'llama-cpp/' + self.sources[0]['vendorPath'])
        git(self.repo, 'commit', '-qm', 'missing links')
        self.add_link(self.sources[1], 'b' * 40)
        before = self.index_tree()
        with self.assertRaises(ValueError):
            stage_missing(self.root, [s['id'] for s in self.sources])
        self.assertEqual(before, self.index_tree())

    def test_registration_requires_the_declared_repository_url(self):
        path = self.repo / '.gitmodules'
        path.write_text(path.read_text().replace('https://github.com/ggml-org/llama.cpp.git', 'file:///other-repo'))
        git(self.repo, 'add', '.gitmodules')
        before = self.index_tree()
        with self.assertRaisesRegex(ValueError, 'URL'):
            stage_missing(self.root, [self.sources[1]['id']])
        self.assertEqual(before, self.index_tree())

    def test_unstaged_pin_and_deliberate_removal_are_not_repaired(self):
        pin = self.root / self.sources[1]['pinFile']
        original = pin.read_text()
        pin.write_text(original + '\n')
        with self.assertRaisesRegex(ValueError, 'Stage the reviewed'):
            stage_missing(self.root, [self.sources[1]['id']])
        pin.write_text(original)
        git(self.repo, 'update-index', '--force-remove', 'llama-cpp/' + self.sources[0]['vendorPath'])
        with self.assertRaisesRegex(ValueError, 'removal of a committed'):
            stage_missing(self.root, [self.sources[0]['id']])

    def test_non_gitlink_tree_entry_and_symlink_fail_closed(self):
        path = self.root / self.sources[1]['vendorPath']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('not a submodule')
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '-qm', 'wrong file')
        with self.assertRaisesRegex(ValueError, 'actual 100644 blob'):
            check_gitlink(self.root, self.sources[1])
        path.unlink()
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '-qm', 'remove file')
        path.symlink_to(self.repo)
        with self.assertRaisesRegex(ValueError, 'Linked source'):
            stage_missing(self.root, [self.sources[1]['id']])

    def test_unmerged_index_is_never_repaired(self):
        source = self.sources[1]
        path = 'llama-cpp/' + source['vendorPath']
        git(self.repo, 'update-index', '--index-info',
            input=f'160000 {source["commit"]} 2\t{path}\n')
        before = index_entries(self.repo, path)
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            stage_missing(self.root, [source['id']])
        self.assertEqual(index_entries(self.repo, path), before)

    def test_populated_checkout_requires_exact_pin_and_clean_worktree(self):
        source = self.sources[1]
        path = self.root / source['vendorPath']
        path.mkdir(parents=True)
        git(path, 'init', '-q')
        git(path, 'config', 'user.name', 'Test')
        git(path, 'config', 'user.email', 'test@example.invalid')
        (path / 'fixture').write_text('synthetic checkout, not upstream\n')
        git(path, 'add', 'fixture')
        git(path, 'commit', '-qm', 'fixture')
        before = self.index_tree()
        with self.assertRaisesRegex(ValueError, 'differs from its pin'):
            stage_missing(self.root, [source['id']])
        self.assertEqual(self.index_tree(), before)
        pin = self.root / source['pinFile']
        data = json.loads(pin.read_text())
        data['llamaCommit'] = git(path, 'rev-parse', 'HEAD')
        pin.write_text(json.dumps(data))
        git(self.repo, 'add', str(pin.relative_to(self.repo)))
        (path / 'fixture').write_text('dirty\n')
        before = self.index_tree()
        with self.assertRaisesRegex(ValueError, 'Dirty source checkout'):
            stage_missing(self.root, [source['id']])
        self.assertEqual(self.index_tree(), before)
        git(path, 'checkout', '--', 'fixture')
        result = stage_missing(self.root, [source['id']])
        self.assertEqual(result[0]['commit'], data['llamaCommit'])

    def test_stage_missing_is_forbidden_in_actions(self):
        before = self.index_tree()
        with patch.dict(os.environ, {'GITHUB_ACTIONS': 'true'}):
            with self.assertRaisesRegex(ValueError, 'local-only'):
                stage_missing(self.root, [self.sources[1]['id']])
        self.assertEqual(before, self.index_tree())

    def test_check_cli_is_read_only_and_returns_actionable_diagnostic(self):
        before = self.index_tree()
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/register_source_gitlinks.py'),
                                 '--root', str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('upstream-nightly', result.stderr)
        self.assertIn('missing', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        self.assertEqual(before, self.index_tree())

    def test_linked_worktree_uses_its_own_head(self):
        worktree = self.repo.parent / (self.repo.name + '-linked')
        git(self.repo, 'worktree', 'add', '--detach', str(worktree), 'HEAD')
        self.addCleanup(lambda: git(self.repo, 'worktree', 'remove', '--force', str(worktree)))
        check_gitlink(worktree / 'llama-cpp', self.sources[0])
        with self.assertRaisesRegex(ValueError, 'missing'):
            check_gitlink(worktree / 'llama-cpp', self.sources[1])

    def test_gitlink_patch_with_index_needs_no_registration_command(self):
        source = self.sources[1]
        path = 'llama-cpp/' + source['vendorPath']
        delta = (f'diff --git a/{path} b/{path}\nnew file mode 160000\n'
                 f'index 0000000..{source["commit"][:7]}\n--- /dev/null\n'
                 f'+++ b/{path}\n@@ -0,0 +1 @@\n+Subproject commit {source["commit"]}\n')
        git(self.repo, 'apply', '--check', '--index', input=delta)
        git(self.repo, 'apply', '--index', input=delta)
        before = self.index_tree()
        checked = check_index(self.root, [s['id'] for s in self.sources])
        self.assertEqual(self.index_tree(), before)
        self.assertEqual([row['commitRequired'] for row in checked], [False, True])
        with self.assertRaisesRegex(ValueError, 'missing'):
            check_gitlink(self.root, source)
        git(self.repo, 'commit', '-qm', 'gitlink included in patch')
        self.assertTrue(all(not item['commitRequired'] for item in check_index(
            self.root, [s['id'] for s in self.sources])))
        check_gitlink(self.root, source)

    def test_index_preflight_refuses_missing_and_wrong_links_without_writing(self):
        before = self.index_tree()
        with self.assertRaisesRegex(ValueError, 'index must record.*actual missing'):
            check_index(self.root, [s['id'] for s in self.sources])
        self.assertEqual(self.index_tree(), before)
        self.add_link(self.sources[1], 'f' * 40)
        before = self.index_tree()
        with self.assertRaisesRegex(ValueError, 'index must record'):
            check_index(self.root, [s['id'] for s in self.sources])
        self.assertEqual(self.index_tree(), before)

    def test_untracked_pin_cannot_define_a_registration(self):
        registry = self.root / 'config/sources.json'
        data = json.loads(registry.read_text())
        data['sources']['upstream-nightly']['pinFile'] = 'sources/upstream-nightly/untracked-pin.json'
        registry.write_text(json.dumps(data))
        git(self.repo, 'add', str(registry.relative_to(self.repo)))
        (self.root / 'sources/upstream-nightly/untracked-pin.json').write_text(
            json.dumps({'llamaCommit': self.sources[1]['commit']}))
        before = self.index_tree()
        for function in (check_index, stage_missing):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, 'staged regular file'):
                    function(self.root, [s['id'] for s in self.sources])
                self.assertEqual(self.index_tree(), before)

    def test_index_check_cli_is_read_only_and_available_in_actions(self):
        self.add_link(self.sources[1])
        before = self.index_tree()
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/register_source_gitlinks.py'),
                                 '--root', str(self.root), '--index'], capture_output=True, text=True,
                                env={**os.environ, 'GITHUB_ACTIONS': 'true'})
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['checkedRevision'], 'index')
        self.assertTrue(report['valid'])
        self.assertEqual(self.index_tree(), before)
        self.assertEqual(committed_entry(self.root, self.sources[1]['vendorPath'])[1], None)


class RepositorySourceRegistration(unittest.TestCase):
    def test_actual_committed_repository_records_every_configured_source(self):
        # Synthetic fixtures alone did not catch a missing gitlink in a delivered
        # patch. Require the real checkout's source registry to be satisfiable.
        from source_config import load_sources
        root = ROOT / 'llama-cpp'
        for name in load_sources(root)['sources']:
            with self.subTest(source=name):
                check_gitlink(root, get_source(root, name))


if __name__ == '__main__':
    unittest.main()
