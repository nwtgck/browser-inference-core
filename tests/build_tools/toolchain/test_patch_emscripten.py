"""Exercise the pinned source patch without installing or building a toolchain."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from build_tools.project import REPOSITORY_ROOT as ROOT
from build_tools.toolchain.patch_emscripten import ASYNCIFY_SOURCE, apply_asyncify_bigint_patch, patched_asyncify_source, verify_asyncify_bigint_patch

SOURCE = b'''\
#if ASYNCIFY == 1 && MEMORY64
    rewindArguments: new Map(),
#endif
#if ASYNCIFY == 1 && MEMORY64
    saveRewindArguments(func, args) { return Asyncify.rewindArguments.set(func, args); },
    restoreRewindArguments(func) { return Asyncify.rewindArguments.get(func); },
#endif
#if MEMORY64
          Asyncify.saveRewindArguments(original, args);
#endif
#if MEMORY64
      func = func.bind(0, ...Asyncify.restoreRewindArguments(original));
#endif
'''
PATCHED = SOURCE.replace(b'&& MEMORY64\n', b'&& (MEMORY64 || WASM_BIGINT)\n').replace(
    b'#if MEMORY64\n', b'#if MEMORY64 || WASM_BIGINT\n')
PATCH = {'sourceSha256': hashlib.sha256(SOURCE).hexdigest(), 'patchedSha256': hashlib.sha256(PATCHED).hexdigest()}


class AsyncifySourcePatch(unittest.TestCase):
    def test_all_four_guards_change_without_replacing_upstream_logic(self):
        result = patched_asyncify_source(SOURCE, PATCH)
        self.assertEqual(result, PATCHED)
        changes = [(before, after) for before, after in zip(SOURCE.splitlines(), result.splitlines()) if before != after]
        self.assertEqual(len(changes), 4)
        self.assertTrue(all(before.startswith(b'#if ') and after.startswith(b'#if ') for before, after in changes))

    def test_application_is_idempotent_and_verified(self):
        with tempfile.TemporaryDirectory(prefix='lcb-asyncify-patch-') as temporary:
            root = Path(temporary)
            file = root / ASYNCIFY_SOURCE
            file.parent.mkdir(parents=True)
            file.write_bytes(SOURCE)
            with self.assertRaisesRegex(RuntimeError, 'prepare the pinned local toolchain'):
                verify_asyncify_bigint_patch(root, PATCH)
            apply_asyncify_bigint_patch(root, PATCH)
            modified = file.stat().st_mtime_ns
            apply_asyncify_bigint_patch(root, PATCH)
            self.assertEqual(file.stat().st_mtime_ns, modified)
            self.assertEqual(file.read_bytes(), PATCHED)
            verify_asyncify_bigint_patch(root, PATCH)

    def test_unknown_source_and_incorrect_output_pin_fail_before_writing(self):
        with tempfile.TemporaryDirectory(prefix='lcb-asyncify-patch-') as temporary:
            root = Path(temporary)
            file = root / ASYNCIFY_SOURCE
            file.parent.mkdir(parents=True)
            for source, patch, message in [
                (SOURCE + b'changed\n', PATCH, 'Unexpected Emscripten'),
                (SOURCE, {**PATCH, 'patchedSha256': '0' * 64}, 'does not match'),
            ]:
                with self.subTest(message=message):
                    file.write_bytes(source)
                    with self.assertRaisesRegex(RuntimeError, message):
                        apply_asyncify_bigint_patch(root, patch)
                    self.assertEqual(file.read_bytes(), source)
                    with self.assertRaisesRegex(RuntimeError, 'prepare the pinned local toolchain'):
                        verify_asyncify_bigint_patch(root, patch)

    def test_missing_runtime_is_not_considered_patched(self):
        with tempfile.TemporaryDirectory(prefix='lcb-asyncify-patch-') as temporary:
            with self.assertRaisesRegex(RuntimeError, 'prepare the pinned local toolchain'):
                verify_asyncify_bigint_patch(Path(temporary), PATCH)


if __name__ == '__main__':
    unittest.main()
