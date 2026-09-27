"""Real filesystem restore/rollback probes. No cache server or SDK execution."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cache_admission as admission


class CacheAdmission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / 'toolchain').mkdir()
        self.archive = b'fixture download only, not an SDK'
        (self.root / 'toolchain/config.json').write_text(json.dumps({
            'dawnTag': 'fixture', 'dawnSha256': hashlib.sha256(self.archive).hexdigest()}))
        self.active, self.pristine, self.state = admission.state_paths(self.root)
        (self.active / 'sysroot/include').mkdir(parents=True)
        (self.active / 'sysroot/include/header.h').write_text('sdk original\n')
        (self.active / 'sysroot/lib').mkdir()
        (self.active / 'sysroot/lib/base.a').write_bytes(b'original library')
        (self.active / 'sysroot_install.stamp').write_text('x')
        self.original = admission.inspect_tree(self.active, baseline=True)
        self.emkey = 'bic-v1-em-fixture-partition'
        self.ccrelative = '.cache/ccache/llama-cpp/cpu-wasm32/browser'
        self.cckey = 'bic-v1-cc-fixture-partition-' + 'a' * 40
        self.cc = admission.cc_path(self.root, self.ccrelative)

    def park(self):
        admission.park_em(self.root, self.emkey)
        self.assertTrue(self.pristine.is_dir())
        self.assertFalse(list(self.active.iterdir()))

    def warm_em(self):
        shutil.copytree(self.pristine, self.active, dirs_exist_ok=True, symlinks=True)
        (self.active / 'sysroot/lib/extra.a').write_bytes(b'new trusted-cache library')

    def finish_em(self, *, matched=None, hit='true', outcome='success', mode='enabled'):
        return admission.finish_em(self.root, self.emkey, self.emkey if matched is None else matched,
                                   hit, outcome, mode)

    def assert_original(self):
        self.assertEqual(admission.inspect_tree(self.active, baseline=True), self.original)
        self.assertFalse(self.pristine.exists())

    def test_miss_restores_full_sdk_not_an_empty_directory(self):
        self.park()
        result = self.finish_em(matched='', hit='')
        self.assertFalse(result['accepted']); self.assertTrue(result['saveAllowed'])
        self.assert_original()

    def test_partial_restore_and_silent_restore_failure_are_discarded(self):
        for outcome in ('failure', 'success'):
            with self.subTest(outcome=outcome):
                self.park()
                (self.active / 'incomplete.a').write_bytes(b'partial object')
                result = self.finish_em(matched='', hit='', outcome=outcome)
                self.assertFalse(result['accepted']); self.assert_original()
                self.state.unlink()

    def test_exact_cache_preserves_baseline_and_retains_extra_libraries(self):
        self.park(); self.warm_em()
        (self.active / 'cache.lock').write_text('mutable runtime lock')
        result = self.finish_em()
        self.assertTrue(result['accepted']); self.assertTrue(result['exactHit'])
        self.assertTrue((self.active / 'sysroot/lib/extra.a').exists())
        self.assertFalse(self.pristine.exists())
        self.assertTrue(admission.check_save(self.root, 'em'))

    def test_same_size_sdk_header_change_is_not_admitted(self):
        self.park(); self.warm_em()
        (self.active / 'sysroot/include/header.h').write_text('sdk modified\n')
        result = self.finish_em()
        self.assertFalse(result['accepted']); self.assertFalse(result['saveAllowed'])
        self.assert_original()

    def test_missing_sdk_library_is_not_admitted(self):
        self.park(); self.warm_em(); (self.active / 'sysroot/lib/base.a').unlink()
        self.assertFalse(self.finish_em()['accepted']); self.assert_original()

    def test_em_prefix_or_status_mismatch_cannot_be_used(self):
        for matched, hit, outcome in [(self.emkey+'-extra', 'false', 'success'),
                                      (self.emkey, 'false', 'success'),
                                      (self.emkey, 'true', 'failure')]:
            self.park(); self.warm_em()
            self.assertFalse(self.finish_em(matched=matched, hit=hit, outcome=outcome)['accepted'])
            self.assert_original(); self.state.unlink()

    def test_cold_ignores_even_an_exact_hit_and_cannot_save(self):
        self.park(); self.warm_em()
        result = self.finish_em(mode='cold')
        self.assertFalse(result['accepted']); self.assertFalse(result['saveAllowed'])
        self.assert_original()

    def test_preexisting_admission_state_is_not_silently_reused(self):
        self.park()
        with self.assertRaisesRegex(ValueError, 'Unfinished'):
            admission.park_em(self.root, self.emkey)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            admission.finish_em(self.root, 'other-key', '', '', 'success', 'enabled')
        self.finish_em(matched='', hit=''); self.assert_original()

    def test_restore_root_symlink_is_unlinked_without_touching_target(self):
        outside = self.root / 'outside'; outside.mkdir(); (outside / 'keep').write_text('keep')
        self.park(); self.active.rmdir(); self.active.symlink_to(outside, target_is_directory=True)
        self.assertFalse(self.finish_em()['accepted']); self.assert_original()
        self.assertEqual((outside / 'keep').read_text(), 'keep')

    def test_additional_cache_symlinks_fifo_and_hardlinks_are_rejected(self):
        for kind in ('symlink', 'fifo', 'hardlink'):
            self.park(); self.warm_em()
            extra = self.active / 'bad'
            if kind == 'symlink': extra.symlink_to('/etc/passwd')
            elif kind == 'fifo': os.mkfifo(extra)
            else: os.link(self.pristine / 'sysroot/lib/base.a', extra)
            self.assertFalse(self.finish_em()['accepted']); self.assert_original(); self.state.unlink()

    def test_exact_baseline_symlink_is_preserved_but_cannot_be_redirected(self):
        link = self.active / 'sysroot/include/alias.h'; link.symlink_to('header.h')
        self.original = admission.inspect_tree(self.active, baseline=True)
        self.park(); self.warm_em()
        self.assertTrue(self.finish_em()['accepted'])
        self.assertEqual(os.readlink(link), 'header.h')
        self.state.unlink(); self.park(); self.warm_em()
        link.unlink(); link.symlink_to('/etc/passwd')
        self.assertFalse(self.finish_em()['accepted'])
        self.assertEqual(os.readlink(link), 'header.h')

    def test_sdk_case_sensitive_names_are_allowed(self):
        (self.active / 'sysroot/include/Header.h').write_text('distinct')
        self.park(); self.warm_em()
        self.assertTrue(self.finish_em()['accepted'])

    def test_sdk_metadata_failure_does_not_move_or_destroy_baseline(self):
        with patch.object(Path, 'open', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError): admission.park_em(self.root, self.emkey)
        self.assert_original()

    def test_sdk_restore_cleanup_failure_stops_instead_of_using_partial_state(self):
        self.park(); self.warm_em()
        with patch.object(admission, 'remove_entry', side_effect=OSError('cannot clean')):
            with self.assertRaises(OSError): self.finish_em(matched='', hit='')
        self.assertTrue(self.pristine.is_dir())
        self.finish_em(matched='', hit=''); self.assert_original()

    def test_save_validation_rechecks_live_sdk_after_compilation(self):
        self.park(); self.warm_em(); self.finish_em()
        (self.active / 'sysroot/include/header.h').write_text('sdk changed after admission')
        self.assertFalse(admission.check_save(self.root, 'em'))

    def finish_cc(self, matched=None, hit='true', outcome='success', mode='enabled'):
        return admission.finish_cc(self.root, self.ccrelative, self.cckey,
            self.cckey if matched is None else matched, hit, outcome, mode)

    def test_ccache_exact_and_same_partition_source_suffix_are_allowed(self):
        for matched, hit in [(self.cckey, 'true'), (self.cckey[:-40]+'b'*40, 'false')]:
            admission.clear_cc(self.root, self.ccrelative)
            (self.cc / 'object').write_bytes(b'ccache validates its own data')
            result = self.finish_cc(matched, hit)
            self.assertTrue(result['accepted'])
            self.assertEqual(result['exactHit'], hit == 'true')
            self.assertTrue(admission.check_save(self.root, 'cc', self.ccrelative))

    def test_ccache_wrong_partition_truncated_suffix_and_extra_suffix_are_rejected(self):
        for key in [self.cckey+'x', self.cckey[:-1], 'other-'+self.cckey, self.cckey[:-40]]:
            admission.clear_cc(self.root, self.ccrelative); (self.cc / 'stale').touch()
            self.assertFalse(self.finish_cc(key, 'false')['accepted'])
            self.assertEqual(list(self.cc.iterdir()), [])

    def test_partial_ccache_restore_on_miss_is_cleared(self):
        admission.clear_cc(self.root, self.ccrelative); (self.cc / 'partial').touch()
        self.assertFalse(self.finish_cc('', '', 'failure')['accepted'])
        self.assertEqual(list(self.cc.iterdir()), [])

    def test_ccache_symlink_fifo_and_hardlink_are_not_admitted(self):
        for kind in ('symlink', 'fifo', 'hardlink'):
            admission.clear_cc(self.root, self.ccrelative)
            bad = self.cc / 'bad'
            if kind == 'symlink': bad.symlink_to('/etc/passwd')
            elif kind == 'fifo': os.mkfifo(bad)
            else: os.link(self.active / 'sysroot/lib/base.a', bad)
            self.assertFalse(self.finish_cc()['accepted'])
            self.assertEqual(list(self.cc.iterdir()), [])

    def test_cold_ccache_is_empty_and_not_saved(self):
        admission.clear_cc(self.root, self.ccrelative); (self.cc / 'object').touch()
        result = self.finish_cc(mode='cold')
        self.assertFalse(result['saveAllowed']); self.assertEqual(list(self.cc.iterdir()), [])

    def test_cache_path_cannot_escape_owned_cache_directory(self):
        for path in ('../outside', '.cache/ccache/../../outside', '.cache/ccache/x/p/browser'):
            with self.assertRaises(ValueError): admission.clear_cc(self.root, path)

    def finish_dawn(self, matched='dawn-key', hit='true', outcome='success', mode='enabled'):
        return admission.finish_dawn(self.root, 'dawn-key', matched, hit, outcome, mode)

    def test_dawn_is_verified_independently_of_cache_key(self):
        path = admission.dawn_path(self.root); path.write_bytes(self.archive)
        self.assertTrue(self.finish_dawn()['accepted'])
        path.write_bytes(b'x' * len(self.archive))
        result = self.finish_dawn()
        self.assertFalse(result['accepted']); self.assertFalse(result['saveAllowed'])
        self.assertFalse(path.exists())

    def test_partial_or_miskeyed_dawn_is_removed_before_pinned_download(self):
        for key, hit, status in [('', '', 'failure'), ('dawn-key-suffix', 'false', 'success')]:
            path = admission.dawn_path(self.root); path.write_bytes(self.archive)
            self.assertFalse(self.finish_dawn(key, hit, status)['accepted'])
            self.assertFalse(path.exists())

    def test_save_rejects_modified_dawn(self):
        path = admission.dawn_path(self.root); path.write_bytes(self.archive)
        self.assertTrue(admission.check_save(self.root, 'dawn'))
        path.write_bytes(b'bad'); self.assertFalse(admission.check_save(self.root, 'dawn'))

    def test_tree_limits_reject_but_rollback_is_still_possible(self):
        self.park(); self.warm_em()
        with patch.object(admission, 'MAX_BYTES', 1):
            self.assertFalse(self.finish_em()['accepted'])
        self.assert_original()

    def test_lock_file_may_change_contents_but_not_become_a_directory(self):
        (self.active / 'cache.lock').write_text('unlocked')
        self.park(); self.warm_em()
        (self.active / 'cache.lock').write_text('changed by lock acquisition')
        self.assertTrue(self.finish_em()['accepted'])
        admission.remove_entry(self.state)
        self.park(); self.warm_em()
        (self.active / 'cache.lock').unlink(); (self.active / 'cache.lock').mkdir()
        self.assertFalse(self.finish_em()['accepted'])
        self.assertTrue((self.active / 'cache.lock').is_file())

    def test_parent_link_is_rejected_before_mkdir_can_escape(self):
        admission.remove_entry(self.root / '.cache')
        outside = self.root / 'outside'; outside.mkdir()
        (self.root / '.cache').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError): admission.clear_cc(self.root, self.ccrelative)
        self.assertEqual(list(outside.iterdir()), [])
        (self.root / '.tools/downloads').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError): admission.dawn_path(self.root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_partial_restore_and_missing_outputs_cli_emit_no_raw_values(self):
        self.park(); (self.active / 'partial').write_text('incomplete')
        output = self.root / 'outputs'
        with patch.object(admission, 'ROOT', self.root), patch.dict(os.environ, {'GITHUB_OUTPUT': str(output)}), patch.object(sys, 'argv',
                ['admission', 'finish', '--kind', 'em', '--requested', self.emkey,
                 '--outcome', 'failure', '--mode', 'enabled']):
            admission.main()
        self.assertEqual(output.read_text(), 'accepted=false\nsave-allowed=true\nhit=false\n')
        self.assert_original()

    def test_bad_admission_mode_and_missing_kind_are_not_silently_accepted(self):
        with self.assertRaises(ValueError): admission.decision('em', self.emkey, '', '', 'success', 'invented')
        with patch.object(sys, 'argv', ['admission', 'check-save']), self.assertRaises(SystemExit) as raised:
            admission.main()
        self.assertEqual(raised.exception.code, 2)

    def test_sdk_without_prepopulated_cache_recovers_empty_baseline(self):
        admission.remove_entry(self.active)
        self.park()
        self.assertFalse(self.finish_em(matched='', hit='')['accepted'])
        self.assertTrue(self.active.is_dir())
        self.assertEqual(list(self.active.iterdir()), [])

    def test_malformed_optional_save_state_withholds_cache_without_crashing(self):
        self.park(); self.finish_em(matched='', hit='')
        for bad in ([], None, {'schemaVersion': True, 'phase': 'ready', 'baseline': {}},
                    {'schemaVersion': 1, 'phase': 'ready', 'baseline': {'x': 0}}):
            self.state.write_text(json.dumps(bad))
            self.assertFalse(admission.check_save(self.root, 'em'))


if __name__ == '__main__': unittest.main()
