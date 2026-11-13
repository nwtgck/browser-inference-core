"""Real Git-object image input capture; no native compiler is exercised here."""
import json
import os
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.runtimes.stable_diffusion_cpp import inputs
from build_tools.runtimes.stable_diffusion_cpp import build
from build_tools.source.inputs import file_identities, identity
from tests.build_tools.source.support import PATCH


class ImageInputs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='bicore-image-inputs-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.write('.gitignore', '__pycache__/\n/runtimes/stable-diffusion-cpp/build/\n')
        self.write('build_tools/__init__.py', '')
        self.write('build_tools/probe.py', 'print("committed binding module")\n')
        self.write('toolchain/config.json', '{"emsdkVersion":"fixture"}\n')
        self.write(inputs.RUNTIME + '/CMakeLists.txt', '# committed CMake input\n')
        self.write(inputs.RUNTIME + '/config/variants.json', '{"browser":{"assertions":false}}\n')
        self.write(inputs.RUNTIME + '/bindings/module-options.d.ts', '// committed declarations\n')
        self.patch_name = inputs.RUNTIME + '/upstream-patches/fix.patch'
        self.write(self.patch_name, 'committed patch bytes\n')
        self.revision = self.commit()
        self.output = self.repo / inputs.RUNTIME / 'build/browser/source-inputs'

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], stderr=subprocess.STDOUT).decode().strip()

    def write(self, relative, value):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

    def commit(self):
        self.git('add', '-A')
        self.git('commit', '-qm', 'input fixture')
        return self.git('rev-parse', 'HEAD')

    def test_assume_unchanged_cmake_variants_python_and_patches_still_use_commit_objects(self):
        names = ['build_tools/probe.py', inputs.RUNTIME + '/CMakeLists.txt',
                 inputs.RUNTIME + '/config/variants.json', inputs.RUNTIME + '/bindings/module-options.d.ts',
                 self.patch_name]
        original = {name: (self.repo / name).read_bytes() for name in names}
        self.git('update-index', '--assume-unchanged', *names)
        for name in names:
            self.write(name, 'hidden local edit\n')
        self.assertEqual(self.git('status', '--porcelain'), '')
        captured = inputs.capture(self.repo, self.revision, self.output)
        for name in names:
            self.assertEqual((captured.repository / name).read_bytes(), original[name])
        self.assertEqual(captured.receipt['patchFiles'][self.patch_name], identity(captured.repository / self.patch_name))
        env = dict(os.environ, PYTHONPATH=str(self.repo), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-m', 'build_tools.probe'], cwd=captured.repository,
                                env=env, text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), 'committed binding module')
        inputs.verify(captured)
        self.assertEqual(captured.receipt['parentInputMode'], 'commit-objects')

    def test_driver_joins_real_git_capture_and_patch_application_before_fake_compiler(self):
        # Only compiler commands are synthetic. Selection, object export, patch
        # application, input verification, provenance and staging are production code.
        runtime = self.repo / inputs.RUNTIME
        self.write('.gitignore', '/.tools/\n__pycache__/\n/runtimes/stable-diffusion-cpp/build/\n')
        self.write('toolchain/config.json', '{"emsdkVersion":"6.0.9"}\n')
        profile = {'memory64': False, 'jspi': True, 'asyncify': False, 'maximumMemory': 4294967296}
        self.write(inputs.RUNTIME + '/config/profiles.json', json.dumps({'webgpu-wasm32-jspi': profile}))
        declarations = {
            'stable-diffusion': {'path': 'vendor/stable-diffusion.cpp', 'repository': 'fixture/sd'},
            'ggml': {'path': 'vendor/ggml-webgpu-source', 'repository': 'fixture/ggml', 'sourceSubdirectory': 'ggml'},
        }
        self.write(inputs.RUNTIME + '/config/upstreams.json', json.dumps(declarations))
        for role, declaration in declarations.items():
            source = runtime / declaration['path']
            source.mkdir(parents=True)
            for args in [('init', '-q'), ('config', 'user.name', 'fixture'),
                         ('config', 'user.email', 'fixture@example.invalid')]:
                subprocess.run(['git', '-C', str(source), *args], check=True)
            if role == 'stable-diffusion':
                (source / 'a.txt').write_text('one\nbefore\n')
            else:
                (source / 'ggml/include').mkdir(parents=True)
                (source / 'ggml/include/ggml.h').write_text('// pinned upstream fixture\n')
            subprocess.run(['git', '-C', str(source), 'add', '-A'], check=True)
            subprocess.run(['git', '-C', str(source), 'commit', '-qm', 'upstream fixture'], check=True)
        self.write(self.patch_name, PATCH)
        self.write(inputs.RUNTIME + '/upstream-patches/series.json', json.dumps({'formatVersion': 2, 'patches': [{
            'file': 'fix.patch', 'target': 'stable-diffusion', 'reason': 'input boundary fixture', 'files': ['a.txt'],
        }]}))
        revision = self.commit()
        hidden = ['CMakeLists.txt', 'config/variants.json', 'bindings/module-options.d.ts', 'upstream-patches/fix.patch']
        original = {name: (runtime / name).read_bytes() for name in hidden}
        self.git('update-index', '--assume-unchanged', *(inputs.RUNTIME + '/' + name for name in hidden))
        for name in hidden:
            (runtime / name).write_text('hidden live input must not be compiled\n')
        self.write('.tools/emdawnwebgpu_pkg/emdawnwebgpu.port.py', '# offline input fixture\n')
        self.assertEqual(self.git('status', '--porcelain'), '')
        build_dir = runtime / 'build/webgpu-wasm32-jspi/browser'
        run = subprocess.run
        commands = []
        def compiler(command, **options):
            if command[0] == 'emcc':
                return subprocess.CompletedProcess(command, 0, stdout='emcc 6.0.9\n')
            if command[:2] == ['emcmake', 'cmake']:
                selected = Path(command[command.index('-S') + 1])
                self.assertNotEqual(selected, runtime)
                for name, raw in original.items():
                    self.assertEqual((selected / name).read_bytes(), raw)
                self.assertEqual((build_dir / 'prepared/stable-diffusion/a.txt').read_text(), 'one\nafter\n')
                self.assertEqual(options['env']['PYTHONPATH'], str(selected.parents[1]))
                commands.append('configure')
                return subprocess.CompletedProcess(command, 0)
            if command[:2] == ['cmake', '--build']:
                for part, names in (('runtime', ('core.mjs', 'core.wasm', 'core.d.ts')),
                                    ('generated', ('schema.json', 'schema.mjs', 'functions.d.ts', 'exports.json'))):
                    (build_dir / part).mkdir()
                    for name in names:
                        (build_dir / part / name).write_text('synthetic compiler output, never inference\n')
                (runtime / 'CMakeLists.txt').write_text('another hidden mid-build change\n')
                commands.append('build')
                return subprocess.CompletedProcess(command, 0)
            return run(command, **options)
        with patch.object(build, 'ROOT', runtime), patch.object(build, 'REPO', self.repo), \
                patch.object(subprocess, 'run', side_effect=compiler), redirect_stdout(io.StringIO()):
            build.main(['--profile', 'webgpu-wasm32-jspi', '--variant', 'browser'])
        self.assertEqual(commands, ['configure', 'build'])
        stage = runtime / 'build/ci-upload/webgpu-wasm32-jspi/browser'
        provenance = json.loads((stage / 'provenance.json').read_text())
        self.assertEqual(provenance['sourceCommit'], revision)
        self.assertIs(provenance['sourceDirty'], False)
        self.assertEqual(provenance['configuration'], profile)
        self.assertEqual(provenance['variantConfiguration'], json.loads(original['config/variants.json'])['browser'])
        self.assertEqual(provenance['sourceInputs']['parentInputMode'], 'commit-objects')
        self.assertEqual(provenance['sourceInputs']['sourceCommit'], revision)
        self.assertTrue((stage / 'runtime/core.wasm').is_file())

    def test_head_and_index_changes_do_not_change_the_selected_parent(self):
        original = (self.repo / self.patch_name).read_bytes()
        self.write(self.patch_name, 'new commit patch')
        other = self.commit()
        self.assertNotEqual(other, self.revision)
        self.write(self.patch_name, 'new index patch')
        self.git('add', self.patch_name)
        captured = inputs.capture(self.repo, self.revision, self.output)
        self.assertEqual((captured.repository / self.patch_name).read_bytes(), original)
        self.assertEqual(captured.receipt['sourceCommit'], self.revision)

    def test_unselected_vendor_gitlinks_need_no_checkout(self):
        for name in (inputs.RUNTIME + '/vendor/stable-diffusion.cpp', 'runtimes/llama-cpp/vendor/llama.cpp'):
            self.git('update-index', '--add', '--cacheinfo', '160000,' + self.revision + ',' + name)
        self.git('commit', '-qm', 'uninitialized excluded vendors')
        captured = inputs.capture(self.repo, self.git('rev-parse', 'HEAD'), self.output)
        self.assertFalse((captured.runtime / 'vendor').exists())
        self.assertFalse((captured.repository / 'runtimes/llama-cpp').exists())

    def test_dirty_parent_uses_local_selected_bytes_and_records_that_mode(self):
        self.write(self.patch_name, 'local patch')
        self.write(inputs.RUNTIME + '/new.cpp', 'local native addition')
        self.write(inputs.RUNTIME + '/build/ignored', 'not an input')
        captured = inputs.capture(self.repo, self.revision, self.output, dirty=True)
        self.assertEqual((captured.repository / self.patch_name).read_text(), 'local patch')
        self.assertTrue((captured.runtime / 'new.cpp').exists())
        self.assertFalse((captured.runtime / 'build').exists())
        self.assertEqual(captured.receipt['parentInputMode'], 'dirty-local-worktree')

    def test_only_changes_to_the_captured_tree_invalidate_its_identity(self):
        captured = inputs.capture(self.repo, self.revision, self.output)
        self.write(self.patch_name, 'later worktree change')
        inputs.verify(captured)
        (captured.runtime / 'config/variants.json').write_text('changed captured input')
        with self.assertRaisesRegex(ValueError, 'Captured image-runtime inputs changed'):
            inputs.verify(captured)

    def test_failed_recapture_keeps_the_previous_tree_and_identical_capture_reuses_it(self):
        captured = inputs.capture(self.repo, self.revision, self.output)
        before, inode = file_identities(self.output), self.output.stat().st_ino
        inputs.capture(self.repo, self.revision, self.output)
        self.assertEqual(self.output.stat().st_ino, inode)
        def partial(_repository, _revision, destination, **_options):
            destination.mkdir()
            (destination / 'partial').write_text('incomplete')
            raise OSError('Injected object export failure')
        with patch.object(inputs, 'export_commit', side_effect=partial):
            with self.assertRaisesRegex(OSError, 'object export failure'):
                inputs.capture(self.repo, self.revision, self.output)
        self.assertEqual(file_identities(self.output), before)
        self.assertEqual(self.output.stat().st_ino, inode)
        inputs.verify(captured)

    def test_source_owned_destination_is_rejected(self):
        before = (self.repo / 'build_tools/probe.py').read_bytes()
        with self.assertRaisesRegex(ValueError, 'source-owned'):
            inputs.capture(self.repo, self.revision, self.repo / 'build_tools')
        self.assertEqual((self.repo / 'build_tools/probe.py').read_bytes(), before)

    def test_migrating_cmake_source_clears_only_its_old_configuration(self):
        build = self.output.parent
        build.mkdir(parents=True)
        (build / 'CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL=' + str(self.repo / inputs.RUNTIME) + '\n')
        (build / 'CMakeFiles').mkdir()
        (build / 'CMakeFiles/old').write_text('old configuration')
        keep = build / 'runtime/core.wasm'
        keep.parent.mkdir()
        keep.write_bytes(b'previous output')
        selected = self.output / 'repository' / inputs.RUNTIME
        inputs.reset_legacy_cmake_cache(build, selected)
        self.assertFalse((build / 'CMakeCache.txt').exists())
        self.assertFalse((build / 'CMakeFiles').exists())
        self.assertEqual(keep.read_bytes(), b'previous output')
        (build / 'CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL=' + str(selected) + '\n')
        inputs.reset_legacy_cmake_cache(build, selected)
        self.assertTrue((build / 'CMakeCache.txt').exists())


if __name__ == '__main__':
    unittest.main()
