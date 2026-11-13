"""Test the overlay fixture's property snapshots with real, isolated CMake targets."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
SNAPSHOT = runtime_test_root('llama-cpp') / 'mtmd-overlay/snapshot-target-property.cmake'


@unittest.skipUnless(shutil.which('cmake'), 'CMake is required')
class CMakePropertySnapshotTests(unittest.TestCase):
    def configure_snapshots(self, before='', change=''):
        # No llama.cpp checkout or C++ compiler is needed: only CMake's actual
        # target-property semantics and the same helper as the overlay fixture.
        with tempfile.TemporaryDirectory(prefix='lcb-property-snapshot-') as temporary:
            root = Path(temporary)
            source, build = root / 'source', root / 'build'
            source.mkdir()
            (source / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.24)
project(property_snapshot_check LANGUAGES NONE)
include("{SNAPSHOT.as_posix()}")
add_library(mtmd INTERFACE)
{before}
get_target_property(upstream_headers mtmd PRECOMPILE_HEADERS)
file(WRITE "${{CMAKE_BINARY_DIR}}/legacy-before.txt" "${{upstream_headers}}\\n")
lcb_snapshot_target_property(mtmd PRECOMPILE_HEADERS "${{CMAKE_BINARY_DIR}}/before.txt")
{change}
get_target_property(headers mtmd PRECOMPILE_HEADERS)
file(WRITE "${{CMAKE_BINARY_DIR}}/legacy-after.txt" "${{headers}}\\n")
lcb_snapshot_target_property(mtmd PRECOMPILE_HEADERS "${{CMAKE_BINARY_DIR}}/after.txt")
''')
            result = subprocess.run(['cmake', '-S', str(source), '-B', str(build)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return {name: (build / (name + '.txt')).read_bytes()
                    for name in ('before', 'after', 'legacy-before', 'legacy-after')}

    def test_unset_property_does_not_compare_variable_name_sentinels(self):
        result = self.configure_snapshots()
        # This is the exact false failure previously seen with llama.cpp v0.5.0.
        self.assertEqual(result['legacy-before'], b'upstream_headers-NOTFOUND\n')
        self.assertEqual(result['legacy-after'], b'headers-NOTFOUND\n')
        self.assertEqual(result['before'], b'0\n\n')
        self.assertEqual(result['after'], result['before'])

    def test_explicitly_empty_property_is_preserved(self):
        result = self.configure_snapshots('set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS "")')
        self.assertEqual(result['before'], b'1\n\n')
        self.assertEqual(result['after'], result['before'])

    def test_unset_and_explicitly_empty_are_not_conflated(self):
        result = self.configure_snapshots(change='set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS "")')
        self.assertEqual(result['before'], b'0\n\n')
        self.assertEqual(result['after'], b'1\n\n')
        self.assertNotEqual(result['before'], result['after'])

    def test_present_headers_are_preserved_verbatim(self):
        value = '/tmp/path with spaces/models.h;$<$<COMPILE_LANGUAGE:CXX>:<vector>>;other.h'
        result = self.configure_snapshots(f'set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS [=[{value}]=])')
        self.assertEqual(result['before'], ('1\n' + value + '\n').encode())
        self.assertEqual(result['after'], result['before'])

    def test_real_header_changes_are_still_detected(self):
        before = 'set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS "a.h;b.h")'
        for command, expected in (
            ('set_property(TARGET mtmd APPEND PROPERTY PRECOMPILE_HEADERS "c.h")', b'1\na.h;b.h;c.h\n'),
            ('set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS "b.h;a.h")', b'1\nb.h;a.h\n'),
            ('set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS "a.h;c.h")', b'1\na.h;c.h\n'),
            ('set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS "a.h")', b'1\na.h\n'),
            ('set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS "")', b'1\n\n'),
            ('set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS)', b'0\n\n'),
        ):
            with self.subTest(command=command):
                result = self.configure_snapshots(before, command)
                self.assertEqual(result['before'], b'1\na.h;b.h\n')
                self.assertEqual(result['after'], expected)
                self.assertNotEqual(result['before'], result['after'])

    def test_adding_headers_to_an_unset_property_is_detected(self):
        result = self.configure_snapshots(change='set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS "a.h")')
        self.assertEqual(result['before'], b'0\n\n')
        self.assertEqual(result['after'], b'1\na.h\n')
        self.assertNotEqual(result['before'], result['after'])

    def test_false_like_header_values_are_not_normalized_away(self):
        for value in ('OFF', 'FALSE', '0', 'NOTFOUND', 'headers-NOTFOUND', 'upstream_headers-NOTFOUND'):
            with self.subTest(value=value):
                result = self.configure_snapshots(f'set_property(TARGET mtmd PROPERTY PRECOMPILE_HEADERS "{value}")')
                self.assertEqual(result['before'], ('1\n' + value + '\n').encode())
                self.assertEqual(result['after'], result['before'])


if __name__ == '__main__':
    unittest.main()
