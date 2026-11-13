import os
from pathlib import Path
from unittest.mock import patch
import shutil

from build_tools.source import workspace
from tests.build_tools.source.support import SourceFixture, PATCH


class Workspace(SourceFixture):
    def prepare(self):
        return workspace.prepare_sources(self.sources, self.output, self.patches)

    def test_composes_complete_tree_without_mutating_input(self):
        report = self.prepare()
        self.assertFalse(report["vendorCheckoutModified"])
        self.assertEqual((self.output / "stable-diffusion/a.txt").read_text(), "one\nafter\n")
        self.assertEqual((self.sources["stable-diffusion"] / "a.txt").read_text(), "one\nbefore\n")
        self.assertEqual(list(self.root.glob("bicore-output-*")), [])
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)

    def test_no_second_copy_of_the_completed_tree(self):
        with patch.object(workspace.shutil, "copytree", wraps=shutil.copytree) as copying:
            self.prepare()
        self.assertEqual(copying.call_count, 2)

    def test_failure_after_first_patch_does_not_publish_a_partial_tree(self):
        self.inventory["patches"].append({"file": "second.patch", "target": "stable-diffusion", "reason": "bad context", "files": ["a.txt"]})
        (self.patches / "second.patch").write_text(PATCH)
        self.write_inventory()
        with self.assertRaisesRegex(ValueError, "Patch application failed"):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("bicore-output-*")), [])
        self.assertEqual((self.sources["stable-diffusion"] / "a.txt").read_text(), "one\nbefore\n")

    def test_declared_order_and_independent_roles(self):
        (self.sources["ggml"] / "a.txt").write_text("one\nbefore\n")
        (self.patches / "second.patch").write_text(PATCH)
        self.inventory["patches"].append({"file": "second.patch", "target": "ggml", "reason": "same name, separate role", "files": ["a.txt"]})
        self.write_inventory()
        report = self.prepare()
        self.assertEqual([p["target"] for p in report["patches"]], ["stable-diffusion", "ggml"])
        self.assertEqual((self.output / "ggml/a.txt").read_text(), "one\nafter\n")

    def test_preserves_untouched_internal_link_and_executable_mode(self):
        original = self.sources["ggml"] / "tool"
        original.write_text("tool\n")
        original.chmod(0o755)
        (self.sources["ggml"] / "alias").symlink_to("tool")
        self.prepare()
        self.assertEqual(os.readlink(self.output / "ggml/alias"), "tool")
        self.assertEqual((self.output / "ggml/tool").stat().st_mode & 0o777, 0o755)
        self.assertNotEqual(original.stat().st_ino, (self.output / "ggml/tool").stat().st_ino)

    def test_does_not_copy_git_metadata_or_previous_build_state(self):
        for name in (".git", "build", ".github"):
            path = self.sources["ggml"] / name
            path.mkdir()
            (path / "secret").write_text("not a source input")
        self.prepare()
        for name in (".git", "build", ".github"):
            self.assertFalse((self.output / "ggml" / name).exists())

    def test_rejects_output_inside_or_containing_source(self):
        for output in (self.sources["ggml"], self.sources["ggml"] / "out", self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "outside"):
                workspace.prepare_sources(self.sources, output, self.patches)

    def test_rejects_linked_source_or_output_parent(self):
        linked = self.root / "source-link"
        linked.symlink_to(self.sources["ggml"], target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Linked"):
            workspace.prepare_sources({**self.sources, "ggml": linked}, self.output, self.patches)
        with self.assertRaisesRegex(ValueError, "Linked"):
            workspace.prepare_sources(self.sources, linked / "new/path", self.patches)
        self.assertFalse((self.sources["ggml"] / "new").exists())

    def test_does_not_replace_existing_directory_or_dangling_link(self):
        self.output.mkdir()
        (self.output / "keep").write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.prepare()
        shutil.rmtree(self.output)
        self.output.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.prepare()
        self.assertTrue(self.output.is_symlink())

    def test_rejects_nonregular_and_external_link_inputs(self):
        path = self.sources["ggml"] / "fifo"
        os.mkfifo(path)
        with self.assertRaisesRegex(ValueError, "Non-regular"):
            self.prepare()
        path.unlink()
        path.symlink_to(self.root / "outside")
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_empty_invalid_or_aliased_roles(self):
        for sources in ({}, {"../escape": self.sources["ggml"]}, {"ggml": self.sources["ggml"], "other": self.sources["ggml"]}):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                workspace.prepare_sources(sources, self.output, self.patches)

    def test_move_error_cleans_the_private_tree(self):
        with patch.object(Path, "rename", side_effect=OSError("injected move failure")):
            with self.assertRaises(OSError):
                self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("bicore-output-*")), [])

    def test_destination_created_during_preparation_is_not_overwritten(self):
        original = workspace.apply_patch
        def with_concurrent_output(*args):
            result = original(*args)
            self.output.mkdir()
            (self.output / "keep").write_text("keep")
            return result
        with patch.object(workspace, "apply_patch", side_effect=with_concurrent_output):
            with self.assertRaisesRegex(ValueError, "appeared"):
                self.prepare()
        self.assertEqual((self.output / "keep").read_text(), "keep")

    def test_selected_inputs_exclude_local_cache_even_when_it_exists(self):
        (self.sources['stable-diffusion'] / 'ignored-model.bin').write_bytes(b'not an input')
        cache = self.sources['ggml'] / 'generated'
        cache.mkdir()
        (cache / 'shadow.h').write_text('must not reach the compiler')
        report = workspace.prepare_sources(self.sources, self.output, self.patches,
                                          source_files={'stable-diffusion': ('a.txt',), 'ggml': ()})
        self.assertEqual((self.output / 'stable-diffusion/a.txt').read_text(), 'one\nafter\n')
        self.assertFalse((self.output / 'stable-diffusion/ignored-model.bin').exists())
        self.assertFalse((self.output / 'ggml/generated').exists())
        self.assertEqual(len(report['patches']), 1)

    def test_selection_requires_exact_roles_and_declared_patch_input(self):
        for selected in ({'stable-diffusion': ('a.txt',)},
                         {'stable-diffusion': (), 'ggml': ()},
                         {'stable-diffusion': ('a.txt', 'a.txt'), 'ggml': ()},
                         {'stable-diffusion': ('../outside',), 'ggml': ()}):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                workspace.prepare_sources(self.sources, self.output, self.patches, source_files=selected)
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.root.glob('bicore-output-*')), [])

    def test_selected_input_keeps_executable_mode_and_internal_links(self):
        source = self.sources['stable-diffusion']
        (source / 'tool').write_text('fixture executable')
        (source / 'tool').chmod(0o755)
        (source / 'alias').symlink_to('tool')
        workspace.prepare_sources(self.sources, self.output, self.patches,
                                  source_files={'stable-diffusion': ('a.txt', 'tool', 'alias'), 'ggml': ()})
        output = self.output / 'stable-diffusion'
        self.assertEqual((output / 'tool').stat().st_mode & 0o777, 0o755)
        self.assertEqual(os.readlink(output / 'alias'), 'tool')
        self.assertEqual((source / 'a.txt').read_text(), 'one\nbefore\n')

    def test_selected_symlink_cannot_escape_original_source(self):
        source = self.sources['stable-diffusion']
        (self.root / 'outside').write_text('not an input')
        (source / 'alias').symlink_to('../outside')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            workspace.prepare_sources(self.sources, self.output, self.patches,
                                      source_files={'stable-diffusion': ('a.txt', 'alias'), 'ggml': ()})
        self.assertFalse(self.output.exists())

    def test_receipt_is_inside_the_transaction(self):
        import json
        report = self.prepare()
        self.assertEqual(json.loads((self.output / 'patch-provenance.json').read_bytes()), report)

    def test_receipt_write_failure_publishes_nothing(self):
        original = Path.write_text
        def fail_receipt(path, *args, **kwargs):
            if path.name == 'patch-provenance.json':
                raise OSError('injected receipt failure')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'write_text', fail_receipt), self.assertRaisesRegex(OSError, 'receipt failure'):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('bicore-output-*')), [])

    def test_rebuild_failure_preserves_previous_complete_output(self):
        self.prepare()
        old = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        with patch.object(workspace, 'apply_patch', side_effect=ValueError('injected failure')):
            with self.assertRaisesRegex(ValueError, 'injected failure'):
                workspace.prepare_sources(self.sources, self.output, self.patches, replace=True)
        self.assertEqual({p.relative_to(self.output): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}, old)

    def test_activation_failure_restores_previous_tree_and_receipt(self):
        self.prepare()
        old = (self.output / 'patch-provenance.json').read_bytes()
        # A real changed input requires activation; identical trees are reused.
        (self.sources['ggml'] / 'new-input').write_text('new source input')
        original = Path.rename
        def fail_activation(path, target):
            if path.name == 'prepared' and path != self.output:
                raise OSError('injected activation failure')
            return original(path, target)
        with patch.object(Path, 'rename', fail_activation), self.assertRaisesRegex(OSError, 'activation failure'):
            workspace.prepare_sources(self.sources, self.output, self.patches, replace=True)
        self.assertEqual((self.output / 'patch-provenance.json').read_bytes(), old)
        self.assertEqual((self.output / 'stable-diffusion/a.txt').read_text(), 'one\nafter\n')
        self.assertEqual(list(self.root.glob('bicore-output-*')), [])

    def test_successful_rebuild_replaces_stale_outputs_and_keeps_inputs(self):
        self.prepare()
        (self.output / 'old-file').write_text('stale')
        workspace.prepare_sources(self.sources, self.output, self.patches, replace=True)
        self.assertFalse((self.output / 'old-file').exists())
        self.assertEqual((self.sources['stable-diffusion'] / 'a.txt').read_text(), 'one\nbefore\n')

    def test_pinned_inputs_and_receipt_share_commit_identity(self):
        import json
        commits = {}
        for role, source in self.sources.items():
            self.git(source, 'init', '-q')
            self.git(source, 'config', 'user.name', 'fixture')
            self.git(source, 'config', 'user.email', 'fixture@example.invalid')
            self.git(source, 'add', '-A')
            self.git(source, 'commit', '--allow-empty', '-qm', 'source')
            commits[role] = self.git(source, 'rev-parse', 'HEAD').decode().strip()
        (self.sources['stable-diffusion'] / 'a.txt').write_text('mutated after source selection')
        report = workspace.prepare_sources(self.sources, self.output, self.patches, source_commits=commits)
        self.assertEqual(report['upstreams'], {role: {'commit': c} for role, c in commits.items()})
        self.assertEqual(json.loads((self.output / 'patch-provenance.json').read_bytes()), report)
        self.assertEqual((self.output / 'stable-diffusion/a.txt').read_text(), 'one\nafter\n')

    def test_provenance_cannot_claim_a_different_input_commit(self):
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            workspace.prepare_sources(self.sources, self.output, self.patches,
                source_commits={role: 'a' * 40 for role in self.sources},
                upstreams={role: {'commit': 'b' * 40} for role in self.sources})
        self.assertFalse(self.output.exists())

    def test_replace_does_not_overwrite_a_new_output_from_another_invocation(self):
        original = workspace.apply_patch
        def create_output(*args):
            result = original(*args)
            self.output.mkdir()
            (self.output / 'keep').write_text('created independently')
            return result
        with patch.object(workspace, 'apply_patch', side_effect=create_output):
            with self.assertRaisesRegex(ValueError, 'destination changed'):
                workspace.prepare_sources(self.sources, self.output, self.patches, replace=True)
        self.assertEqual((self.output / 'keep').read_text(), 'created independently')
        self.assertEqual(list(self.root.glob('bicore-output-*')), [])

    def test_recovery_failure_keeps_the_previous_complete_tree_for_manual_restore(self):
        self.prepare()
        old = (self.output / 'patch-provenance.json').read_bytes()
        # A real changed input requires activation; identical trees are reused.
        (self.sources['ggml'] / 'new-input').write_text('new source input')
        original = Path.rename
        def fail_switch_and_restore(path, target):
            if path.name in ('prepared', 'previous') and path != self.output:
                raise OSError('injected switch or restore failure')
            return original(path, target)
        with patch.object(Path, 'rename', fail_switch_and_restore):
            with self.assertRaisesRegex(RuntimeError, 'previous output retained at') as caught:
                workspace.prepare_sources(self.sources, self.output, self.patches, replace=True)
        recovery, = self.root.glob('bicore-output-*/previous')
        self.assertIn(str(recovery), str(caught.exception))
        self.assertEqual((recovery / 'patch-provenance.json').read_bytes(), old)
        self.assertEqual((recovery / 'stable-diffusion/a.txt').read_text(), 'one\nafter\n')

    def test_replace_detects_removed_output_before_activation(self):
        self.prepare()
        original = workspace.apply_patch
        def remove_output(*args):
            result = original(*args)
            shutil.rmtree(self.output)
            return result
        with patch.object(workspace, 'apply_patch', side_effect=remove_output):
            with self.assertRaisesRegex(ValueError, 'destination disappeared'):
                workspace.prepare_sources(self.sources, self.output, self.patches, replace=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('bicore-output-*')), [])


class IncrementalWorkspace(SourceFixture):
    def prepare(self):
        return workspace.prepare_sources(self.sources, self.output, self.patches, replace=True)

    def stamps(self):
        return {path.relative_to(self.output).as_posix(): (path.lstat().st_ino, path.lstat().st_mtime_ns)
                for path in self.output.rglob('*')}

    def test_identical_tree_and_receipt_are_not_replaced(self):
        first = self.prepare()
        stamp = self.output.stat().st_ino
        before = self.stamps()
        self.assertEqual(self.prepare(), first)
        self.assertEqual(self.output.stat().st_ino, stamp)
        self.assertEqual(self.stamps(), before)
        self.assertEqual(list(self.root.glob('bicore-output-*')), [])

    def test_only_changed_files_receive_new_timestamps(self):
        source = self.sources['ggml']
        (source / 'constant').write_text('unchanged')
        (source / 'changing').write_text('old bytes')
        self.prepare()
        unchanged = self.output / 'ggml/constant'
        changed = self.output / 'ggml/changing'
        old_time = 1_500_000_000_000_000_000
        for path in (unchanged, changed, source / 'changing'):
            os.utime(path, ns=(old_time, old_time))
        (source / 'changing').write_text('new bytes')
        os.utime(source / 'changing', ns=(old_time, old_time))
        self.prepare()
        self.assertEqual(unchanged.stat().st_mtime_ns, old_time)
        self.assertEqual(changed.read_text(), 'new bytes')
        self.assertNotEqual(changed.stat().st_mtime_ns, old_time)

    def test_stale_files_are_removed_without_touching_equal_inputs(self):
        self.prepare()
        path = self.output / 'stable-diffusion/a.txt'
        before = path.stat().st_mtime_ns
        (self.output / 'stale').write_text('old output')
        self.prepare()
        self.assertFalse((self.output / 'stale').exists())
        self.assertEqual(path.stat().st_mtime_ns, before)

    def test_different_links_and_modes_are_not_treated_as_unchanged(self):
        source = self.sources['ggml']
        (source / 'one').write_text('same')
        (source / 'two').write_text('same')
        link = source / 'alias'
        link.symlink_to('one')
        self.prepare()
        link.unlink()
        link.symlink_to('two')
        (source / 'one').chmod(0o755)
        self.prepare()
        self.assertEqual(os.readlink(self.output / 'ggml/alias'), 'two')
        self.assertEqual((self.output / 'ggml/one').stat().st_mode & 0o777, 0o755)
