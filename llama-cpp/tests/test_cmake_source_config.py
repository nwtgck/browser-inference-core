"""Real CMake source-selection/overlay regression tests without a vendor checkout.

The synthetic translation units contain only old sides of accepted patch hunks.
They test configure-time plumbing, NOT upstream compilation or patch semantics.
The separate real-upstream policy test remains in place and is never replaced.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from source_config import get_source, patch_series


def old_hunks(patch):
    result = []
    offset = None
    for line in patch.read_text().splitlines(keepends=True):
        header = re.match(r'@@ -(\d+)(?:,\d+)? ', line)
        if header:
            offset = int(header[1]) - 1
            while len(result) < offset:
                result.append('// configure-only test gap\n')
        elif offset is not None and line[:1] in (' ', '-'):
            text = line[1:]
            if offset < len(result):
                if result[offset] != text:
                    raise ValueError('Conflicting synthetic old-side hunks')
            else:
                result.append(text)
            offset += 1
    return ''.join(result)


@unittest.skipUnless(shutil.which('cmake'), 'CMake required')
class CMakeSourceSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='bic-cmake-selection-')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.root = self.work / 'llama-cpp'
        self.root.mkdir()
        for name in ('config', 'sources', 'cmake', 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'):
            shutil.copytree(ROOT / name, self.root / name)
        shutil.copytree(ROOT / 'scripts', self.root / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
        self.probe = self.work / 'probe'
        self.probe.mkdir()
        (self.probe / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.24)\nproject(selection NONE)\n'
            f'include("{self.root}/cmake/SourceConfig.cmake")\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/selection.txt" '
            '"${LCB_SOURCE_ID}\\n${LCB_LLAMA_SOURCE}\\n${LCB_PATCH_SERIES_FILE}\\n")\n'
            'get_property(inputs DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS)\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/dependencies.txt" "${inputs}")\n')

    def configure(self, *options, source=None, build=None, success=True):
        source = source or self.probe
        build = build or self.work / 'build'
        result = subprocess.run(['cmake', '-S', str(source), '-B', str(build), *options],
                                capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result, build

    def test_default_and_reconfigure_select_registered_vendor_not_stale_cache(self):
        _, build = self.configure()
        self.assertEqual((build / 'selection.txt').read_text().splitlines()[:2],
                         ['upstream-stable', str(self.root / 'vendor/llama.cpp')])
        self.configure('-DLCB_SOURCE_ID=upstream-nightly')
        self.assertEqual((build / 'selection.txt').read_text().splitlines()[:2],
                         ['upstream-nightly', str(self.root / 'vendor/llama.cpp-nightly')])
        self.configure('-DLCB_SOURCE_ID=upstream-stable')
        self.assertIn(str(self.root / 'vendor/llama.cpp') + '\n', (build / 'selection.txt').read_text())

    def test_explicit_native_source_override_is_preserved(self):
        custom = self.work / 'test upstream'
        _, build = self.configure('-DLCB_SOURCE_ID=upstream-nightly', '-DLCB_LLAMA_SOURCE=' + str(custom))
        self.assertEqual((build / 'selection.txt').read_text().splitlines()[1], str(custom))

    def test_unknown_and_empty_source_fail_before_overlay_python(self):
        for name in ('', 'unknown-fork'):
            with self.subTest(source=name):
                result, _ = self.configure('-DLCB_SOURCE_ID=' + name, success=False)
                self.assertIn('Unknown LCB_SOURCE_ID', result.stderr)
                self.assertNotIn('Traceback', result.stderr)

    def test_registered_plan_path_and_default_are_not_hardcoded(self):
        path = self.root / 'config/sources.json'
        registry = json.loads(path.read_text())
        registry['defaultSource'] = 'upstream-nightly'
        registry['sources']['upstream-nightly']['patchSeries'] = 'sources/separate-plan.json'
        path.write_text(json.dumps(registry))
        shutil.copy2(self.root / 'sources/upstream-nightly/patches.json', self.root / 'sources/separate-plan.json')
        _, build = self.configure()
        self.assertEqual((build / 'selection.txt').read_text().splitlines()[0], 'upstream-nightly')
        dependencies = (build / 'dependencies.txt').read_text().split(';')
        self.assertIn(str(self.root / 'sources/separate-plan.json'), dependencies)
        self.assertIn(str(self.root / 'sources/upstream-nightly/pin.json'), dependencies)
        for item in patch_series(self.root, 'upstream-nightly'):
            self.assertIn(str(self.root / item['file']), dependencies)

    def test_real_fixture_replaces_only_approved_units_for_both_tracks(self):
        # Execute the same CMake fixture that previously passed an empty source ID.
        fixture = self.root / 'tests/mtmd-overlay'
        shutil.copytree(ROOT / 'tests/mtmd-overlay', fixture)
        for name in ('upstream-stable', 'upstream-nightly'):
            source = get_source(self.root, name)
            upstream = self.root / source['vendorPath']
            directory = upstream / 'tools/mtmd'
            directory.mkdir(parents=True)
            (upstream / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.24)\nproject(mock_upstream LANGUAGES C CXX)\n'
                'add_subdirectory(tools/mtmd)\n')
            (directory / 'CMakeLists.txt').write_text('add_library(mtmd STATIC clip.cpp mtmd-audio.cpp retained.cpp)\n')
            (directory / 'retained.cpp').write_text('// not an overlay\n')
            original = {}
            for item in patch_series(self.root, name):
                path = directory / item['translationUnit']
                path.write_text(old_hunks(self.root / item['file']))
                original[path] = path.read_bytes()
            options = [] if name == 'upstream-stable' else ['-DLCB_SOURCE_ID=' + name]
            _, build = self.configure(*options, source=fixture, build=self.work / name)
            before = (build / 'upstream-mtmd-sources.txt').read_text().strip().split(';')
            after = (build / 'mtmd-sources.txt').read_text().strip().split(';')
            self.assertEqual(len(before), len(after))
            self.assertEqual(set(before) - set(after), {'clip.cpp', 'mtmd-audio.cpp'})
            self.assertEqual({Path(p).name for p in set(after) - set(before)}, {'clip.cpp', 'mtmd-audio.cpp'})
            self.assertEqual((build / 'upstream-mtmd-headers.txt').read_bytes(), (build / 'mtmd-headers.txt').read_bytes())
            self.configure(*options, source=fixture, build=build)
            for path, data in original.items():
                self.assertEqual(path.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
