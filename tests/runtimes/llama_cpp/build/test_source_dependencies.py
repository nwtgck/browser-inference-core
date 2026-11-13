"""Real CMake warm reconfiguration with source generation, without a Dawn build."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import REPOSITORY_ROOT, runtime_root
from build_tools.runtimes.llama_cpp import webgpu_plan as plan

ROOT = runtime_root('llama-cpp')


@unittest.skipUnless(shutil.which('cmake') and shutil.which('ninja') and shutil.which('c++'),
                     'CMake, Ninja and a C++ compiler are required')
class SourceReconfiguration(unittest.TestCase):
    def run_cmake(self, *args):
        result = subprocess.run(["cmake", *args], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_normal_recipe_tracks_all_inputs_and_shader_changes(self):
        self.exercise_mode(False)

    def test_profile_recipe_tracks_all_inputs_and_shader_changes(self):
        self.exercise_mode(True)

    def exercise_mode(self, gpu_profile):
        upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        if not (upstream / plan.WGSL / 'embed_wgsl.py').is_file():
            self.skipTest('Set LCB_TEST_LLAMA_SOURCE to a local upstream tree')
        with tempfile.TemporaryDirectory(prefix='source-cmake-') as temp:
            work = Path(temp)
            source = work / 'upstream'
            shutil.copytree(upstream, source, ignore=shutil.ignore_patterns('.git'))
            project, build = work / 'project', work / 'build'
            project.mkdir()
            dawn = work / 'dawn'
            for name in ['webgpu/include/webgpu/webgpu.h', 'webgpu_cpp/include/webgpu/webgpu_cpp.h',
                         'webgpu_cpp/include/webgpu/webgpu_cpp_chained_struct.h',
                         'webgpu_cpp/include/webgpu/webgpu_enum_class_bitmasks.h']:
                path = dawn / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('// not compiled\n')
            launcher = work / 'python-source-only'
            # CMake's Python command is replaced ONLY to omit the unavailable
            # Dawn header gate. The source recipe is the real production module;
            # no C++ source is compiled and no Dawn validation is claimed.
            launcher.write_text('#!' + sys.executable + '\n' + '''import subprocess,sys
args=sys.argv[1:]
assert args[:2] == ['-m', 'build_tools.runtimes.llama_cpp.source']
args[1]='build_tools.runtimes.llama_cpp.webgpu_plan'
i=args.index('--dawn-package');del args[i:i+2]
raise SystemExit(subprocess.run([sys.executable,*args]).returncode)
''')
            launcher.chmod(0o755)
            for filename in ['ggml-webgpu.cpp','ggml-backend.cpp','llama-model-loader.cpp','llama-sampler.cpp','llama-context.cpp']:
                (project / filename).write_text('// configure-only target\n')
            cmake = f'''cmake_minimum_required(VERSION 3.20)
project(SourceGraph LANGUAGES C CXX)
set(LCB_WEBGPU ON)
set(GGML_WEBGPU_GPU_PROFILE {"ON" if gpu_profile else "OFF"})
set(EMSCRIPTEN ON)
set(LCB_LLAMA_SOURCE "{source}")
set(EMDAWNWEBGPU_DIR "{dawn}")
set(Python3_EXECUTABLE "{launcher}")
add_library(ggml-webgpu STATIC EXCLUDE_FROM_ALL ggml-webgpu.cpp)
add_library(ggml-base STATIC EXCLUDE_FROM_ALL ggml-backend.cpp)
add_library(llama STATIC EXCLUDE_FROM_ALL llama-model-loader.cpp llama-sampler.cpp llama-context.cpp)
include("{ROOT / 'cmake/WebgpuSourceOverlay.cmake'}")
get_directory_property(inputs CMAKE_CONFIGURE_DEPENDS)
file(WRITE "${{CMAKE_BINARY_DIR}}/dependencies.txt" "${{inputs}}")
file(APPEND "${{CMAKE_BINARY_DIR}}/configured.txt" "configured\\n")
add_custom_target(observe COMMAND "${{CMAKE_COMMAND}}" -E echo "source graph only")
'''
            (project / 'CMakeLists.txt').write_text(cmake)
            self.run_cmake('-G', 'Ninja', '-S', str(project), '-B', str(build))
            dependencies = set((build / 'dependencies.txt').read_text().split(';'))
            for relative in ['build_tools/runtimes/llama_cpp/webgpu_plan.py','build_tools/source/wgsl.py',
                             'build_tools/source/files.py','build_tools/output.py','build_tools/project.py',
                             'build_tools/toolchain/dawn.py']:
                self.assertIn(str(REPOSITORY_ROOT / relative), dependencies)
            self.assertIn(str(ROOT / 'config/webgpu-patches.json'), dependencies)
            self.assertFalse(any('/tests/' in name or '/sources/' in name for name in dependencies))
            from build_tools.source.patches import read_series
            inventory = read_series(plan.INVENTORY, targets=('llama-webgpu', 'iq3s', 'word-template'))
            selected = set(plan.patch_names(gpu_profile)) | {
                'ggml-webgpu-iq3s-field-loads.patch', 'ggml-webgpu-iq3s-row-split.patch',
                'ggml-webgpu-aligned-word-load.patch'}
            for item in inventory.patches:
                self.assertEqual(str(ROOT / plan.PATCH_DIRECTORY / item.file) in dependencies,
                                 item.file in selected, item.file)
            output = build / 'webgpu-source-overlay'
            def receipt():
                return json.loads((output / plan.RECEIPT).read_text())
            before = receipt()
            # Verify the evaluated graph, not spelling in the CMake source.
            # This replaces historical stage-local hard-coded input lists.
            for relative in before['inputs']:
                if relative.startswith('runtime/'):
                    expected = ROOT / relative.removeprefix('runtime/')
                else:
                    expected = source / relative
                self.assertIn(str(expected), dependencies, relative)
            for patch in before['patches']:
                self.assertIn(str(ROOT / plan.PATCH_DIRECTORY / patch['file']), dependencies)

            new_shader = source / plan.WGSL / 'bicore_dependency_probe.wgsl'
            new_shader.write_text('// New canonical shader fixture.\n')
            self.run_cmake('--build', str(build), '--target', 'observe')
            added = receipt()
            self.assertNotEqual(added['outputs']['ggml-webgpu-aligned-word-load-shaders.hpp'],
                                before['outputs']['ggml-webgpu-aligned-word-load-shaders.hpp'])
            self.assertIn(str(new_shader.relative_to(source)), added['inputs'])
            self.assertIn(str(new_shader), (build / 'dependencies.txt').read_text().split(';'))
            new_shader.write_text('// Changed canonical shader fixture.\n')
            self.run_cmake('--build', str(build), '--target', 'observe')
            changed = receipt()
            self.assertNotEqual(changed['inputs'][str(new_shader.relative_to(source))],
                                added['inputs'][str(new_shader.relative_to(source))])
            new_shader.unlink()
            self.run_cmake('--build', str(build), '--target', 'observe')
            self.assertEqual(receipt(), before)
            self.assertEqual((build / 'configured.txt').read_text().count('configured\n'), 4)
            marker = (output / 'ggml-webgpu.cpp').stat().st_mtime_ns
            self.run_cmake('--build', str(build), '--target', 'observe')
            self.assertEqual((output / 'ggml-webgpu.cpp').stat().st_mtime_ns, marker)
            self.assertEqual((build / 'configured.txt').read_text().count('configured\n'), 4)


if __name__ == '__main__':
    unittest.main()
