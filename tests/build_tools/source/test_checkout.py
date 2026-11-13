import os
from pathlib import Path
import shutil
from unittest.mock import patch

from build_tools.source.checkout import checked_sources, commit_file, selected_gitlinks, tracked_files
from tests.build_tools.source.support import SourceFixture


class LocalCheckout(SourceFixture):
    def setUp(self):
        super().setUp()
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.upstream = self.repository / "vendor/source"
        self.upstream.mkdir(parents=True)
        for where in (self.repository, self.upstream):
            self.git(where, "init", "-q")
            self.git(where, "config", "user.name", "fixture")
            self.git(where, "config", "user.email", "fixture@example.invalid")
        (self.upstream / "source.cpp").write_text("int value = 1;\n")
        self.git(self.upstream, "add", "source.cpp")
        self.git(self.upstream, "commit", "-qm", "upstream fixture")
        self.commit = self.git(self.upstream, "rev-parse", "HEAD").decode().strip()
        self.git(self.repository, "update-index", "--add", "--cacheinfo", f"160000,{self.commit},vendor/source")
        self.git(self.repository, "commit", "-qm", "parent fixture")

    def resolve(self):
        return checked_sources(self.repository, {"upstream": "vendor/source"})

    def test_reads_gitlink_and_independent_clean_checkout(self):
        result = self.resolve()["upstream"]
        self.assertEqual(result.path, self.upstream)
        self.assertEqual(result.commit, self.commit)

    def test_staged_new_pin_is_the_development_input_not_old_head(self):
        (self.upstream / "source.cpp").write_text("int value = 2;\n")
        self.git(self.upstream, "commit", "-qam", "intentional upstream update")
        new = self.git(self.upstream, "rev-parse", "HEAD").decode().strip()
        with self.assertRaisesRegex(ValueError, "differs from the selected gitlink"):
            self.resolve()
        self.git(self.repository, "update-index", "--cacheinfo", f"160000,{new},vendor/source")
        self.assertEqual(self.resolve()["upstream"].commit, new)

    def test_captured_commit_selects_pins_even_after_parent_and_index_switch(self):
        parent = self.git(self.repository, 'rev-parse', 'HEAD').decode().strip()
        (self.upstream / 'source.cpp').write_text('int value = 2;\n')
        self.git(self.upstream, 'commit', '-qam', 'new upstream')
        newer = self.git(self.upstream, 'rev-parse', 'HEAD').decode().strip()
        self.git(self.repository, 'update-index', '--cacheinfo', f'160000,{newer},vendor/source')
        self.git(self.repository, 'commit', '-qm', 'new parent')
        self.assertEqual(selected_gitlinks(self.repository, {'upstream': 'vendor/source'}), {'upstream': newer})
        self.git(self.upstream, 'checkout', '--detach', self.commit)
        selected = checked_sources(self.repository, {'upstream': 'vendor/source'}, revision=parent)
        self.assertEqual(selected['upstream'].commit, self.commit)

    def test_config_reads_captured_blob_instead_of_current_worktree_or_head(self):
        path = self.repository / 'config.json'
        path.write_text('{"selected": "first"}\n')
        self.git(self.repository, 'add', 'config.json')
        self.git(self.repository, 'commit', '-qm', 'first configuration')
        first = self.git(self.repository, 'rev-parse', 'HEAD').decode().strip()
        path.write_text('{"selected": "second"}\n')
        self.git(self.repository, 'commit', '-qam', 'second configuration')
        path.write_text('uncommitted replacement')
        self.assertEqual(commit_file(self.repository, first, 'config.json'), '{"selected": "first"}\n')

    def test_committed_config_must_be_a_regular_blob_at_an_exact_commit(self):
        link = self.repository / 'linked.json'
        link.symlink_to('outside.json')
        self.git(self.repository, 'add', 'linked.json')
        self.git(self.repository, 'commit', '-qm', 'link fixture')
        head = self.git(self.repository, 'rev-parse', 'HEAD').decode().strip()
        for relative in ('linked.json', 'vendor/source'):
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, 'regular committed input'):
                commit_file(self.repository, head, relative)
        with self.assertRaisesRegex(ValueError, 'complete pinned'):
            commit_file(self.repository, 'HEAD', 'linked.json')

    def test_rejects_modified_or_untracked_upstream(self):
        source = self.upstream / "source.cpp"
        source.write_text("int changed;\n")
        with self.assertRaisesRegex(ValueError, "Dirty upstream"):
            self.resolve()
        self.git(self.upstream, "restore", "source.cpp")
        extra = self.upstream / "not-tracked"
        extra.write_text("not an input")
        with self.assertRaisesRegex(ValueError, "Dirty upstream"):
            self.resolve()

    def test_plain_directory_cannot_impersonate_a_gitlink_checkout(self):
        shutil.rmtree(self.upstream / ".git")
        with self.assertRaisesRegex(ValueError, "Initialize"):
            self.resolve()

    def test_non_gitlink_and_missing_path_are_not_accepted(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            checked_sources(self.repository, {"upstream": "vendor/missing"})
        (self.repository / "plain").write_text("not a submodule")
        self.git(self.repository, "add", "plain")
        with self.assertRaisesRegex(ValueError, "stage-zero"):
            checked_sources(self.repository, {"upstream": "plain"})

    def test_environment_cannot_select_another_repository(self):
        with patch.dict(os.environ, {"GIT_DIR": str(self.upstream / ".git"), "GIT_WORK_TREE": str(self.upstream)}):
            self.assertEqual(self.resolve()["upstream"].commit, self.commit)

    def test_rejects_unmerged_gitlink(self):
        self.git(self.repository, "update-index", "--force-remove", "vendor/source")
        self.git(self.repository, "update-index", "--index-info", input=f"160000 {self.commit} 1\tvendor/source\n160000 {self.commit} 2\tvendor/source\n".encode())
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.resolve()

    def test_git_metadata_is_not_an_input_path(self):
        with self.assertRaises(ValueError):
            checked_sources(self.repository, {"upstream": ".git/config"})

    def test_duplicate_source_roles_do_not_alias_one_gitlink(self):
        with self.assertRaisesRegex(ValueError, "distinct gitlinks"):
            checked_sources(self.repository, {"one": "vendor/source", "two": "vendor/source"})

    def test_missing_nested_checkout_is_not_a_complete_source(self):
        # git status alone reports an uninitialized nested gitlink as clean.
        (self.upstream / '.gitmodules').write_text(
            '[submodule "nested"]\n path = nested\n url = https://example.invalid/no-fetch.git\n')
        self.git(self.upstream, 'add', '.gitmodules')
        self.git(self.upstream, 'update-index', '--add', '--cacheinfo', f'160000,{self.commit},nested')
        self.git(self.upstream, 'commit', '-qm', 'declare a required nested source')
        (self.upstream / 'nested').mkdir()
        self.assertEqual(self.git(self.upstream, 'status', '--porcelain'), b'')
        head = self.git(self.upstream, 'rev-parse', 'HEAD').decode().strip()
        self.git(self.repository, 'update-index', '--cacheinfo', f'160000,{head},vendor/source')
        with self.assertRaisesRegex(ValueError, "Initialize the pinned nested"):
            self.resolve()

    def test_ignored_models_and_headers_are_not_tracked_build_inputs(self):
        (self.upstream / '.gitignore').write_text('local-cache/\n')
        self.git(self.upstream, 'add', '.gitignore')
        self.git(self.upstream, 'commit', '-qm', 'ignore local cache')
        commit = self.git(self.upstream, 'rev-parse', 'HEAD').decode().strip()
        self.git(self.repository, 'update-index', '--cacheinfo', f'160000,{commit},vendor/source')
        cache = self.upstream / 'local-cache'
        cache.mkdir()
        (cache / 'header.h').write_text('not part of the selected source')
        selected = self.resolve()['upstream']
        self.assertEqual(set(tracked_files(selected.path)), {'.gitignore', 'source.cpp'})
