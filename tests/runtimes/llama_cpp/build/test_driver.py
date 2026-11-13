"""Exercise build provenance with real Git/CMake and a simulated compiler probe."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from tests.support.llama_toolchain import merged_toolchain, seed_toolchain

from build_tools.project import REPOSITORY_ROOT, runtime_root
ROOT = runtime_root('llama-cpp')


@unittest.skipUnless(shutil.which('cmake') and shutil.which('ninja'), 'CMake and Ninja are required')
class BuildProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='lcb-build-test-')
        self.addCleanup(self.tmp.cleanup)
        self.work=Path(self.tmp.name)
        self.repository=self.work/'source tree'
        self.root=self.repository/'runtimes/llama-cpp'
        self.root.mkdir(parents=True)
        self.env=os.environ.copy()
        for name in list(self.env):
            if name.startswith('GIT_'):
                self.env.pop(name)
        self.env.update({'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':os.devnull})
        self.git(self.repository,'init','--quiet')
        self.upstream=self.root/'vendor/llama.cpp'
        (self.upstream/'include').mkdir(parents=True)
        self.git(self.upstream,'init','--quiet')
        (self.upstream/'include/llama.h').write_text('/* Local Git fixture, not llama.cpp. */\n')
        self.git(self.upstream,'add','include/llama.h')
        self.commit(self.upstream)
        upstream_sha=self.git(self.upstream,'rev-parse','HEAD').strip()
        (self.repository/'.gitmodules').write_text(
            '[submodule "vendor/llama.cpp"]\n\tpath = runtimes/llama-cpp/vendor/llama.cpp\n\turl = https://example.invalid/upstream.git\n')
        self.git(self.repository,'update-index','--add','--cacheinfo',f'160000,{upstream_sha},runtimes/llama-cpp/vendor/llama.cpp')
        shutil.copy2(ROOT/'.gitignore',self.root/'.gitignore')
        # Shared tools now live at the repository root, not in the runtime.
        shutil.copy2(REPOSITORY_ROOT/'.gitignore', self.repository/'.gitignore')
        (self.root/'config').mkdir()
        shutil.copy2(ROOT/'config/profiles.json',self.root/'config/profiles.json')
        shutil.copy2(ROOT/'config/variants.json',self.root/'config/variants.json')
        toolchain=merged_toolchain()
        toolchain['llamaCommit']=upstream_sha
        patched_runtime=b'Synthetic patched Asyncify runtime for the compiler fixture.\n'
        toolchain['emscriptenAsyncifyBigIntPatch']['patchedSha256']=hashlib.sha256(patched_runtime).hexdigest()
        seed_toolchain(self.root, toolchain, scripts=True)
        (self.root/'README.md').write_text('original\n')
        # The real CMake process intentionally runs a probe without WORKING_DIRECTORY,
        # matching the upstream configure-time pattern. No C/C++ or Wasm is compiled.
        (self.root/'CMakeLists.txt').write_text(textwrap.dedent('''\
            cmake_minimum_required(VERSION 3.24)
            project(build_provenance_fixture NONE)
            execute_process(COMMAND "$ENV{LCB_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/probe.py"
                "${CMAKE_CURRENT_SOURCE_DIR}" "${LCB_WEBGPU}" "${GGML_WEBGPU_GPU_PROFILE}" COMMAND_ERROR_IS_FATAL ANY)
            add_custom_target(core
                COMMAND "$ENV{LCB_TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/probe.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}" --build)
        '''))
        (self.root/'probe.py').write_text(textwrap.dedent('''\
            import os
            import json
            from pathlib import Path
            import sys
            root=Path(os.environ['LCB_TEST_ORIGINAL_ROOT'])
            building='--build' in sys.argv
            Path('a.out.wasm' if building else 'a.out.js').write_text('Probe output, not a runtime.\\n')
            if building:
                runtime=Path('runtime')
                runtime.mkdir(exist_ok=True)
                module=runtime/'core.mjs'
                if not module.exists():
                    module.write_text('function findWasmBinary(){if(Module["locateFile"]){return locateFile("core.wasm")}return new URL("core.wasm",import.meta.url).href}\\n')
            if not building:
                macro=' -DGGML_WEBGPU_GPU_PROFILE=1' if sys.argv[3]=='ON' else ''
                commands=[{'file':'ggml-webgpu.cpp','command':'c++ -c ggml-webgpu.cpp'+macro}] if sys.argv[2]=='ON' else []
                if macro:
                    commands += [{'file':name,'command':'c++ -DLCB_WEBGPU_Q8_DIAGNOSTIC=1 -c '+name} for name in ('webgpu-q8-diagnostic.cpp','q8-diagnostic-metadata.cpp')]
                runtime=' -DLCB_MTMD_WEBGPU_BF16_RUNTIME='+('1' if sys.argv[2]=='ON' else '0')
                if os.environ.get('LCB_TEST_MODE')=='bf16-mismatch':
                    runtime=' -DLCB_MTMD_WEBGPU_BF16_RUNTIME=0'
                commands += [{'file':name,'command':'c++ -c '+name+runtime}
                             for name in ('mtmd-model.cpp','mtmd-subcontext.cpp','mtmd.cpp')]
                Path('compile_commands.json').write_text(json.dumps(commands))
                # Compiler/provenance fixture, not an actual model-graph build.
                # build.py requires both CMake preparation outputs to exist.
                overlay=Path('model-graph-chat-overlay')
                overlay.mkdir(exist_ok=True)
                (overlay/'provenance.json').write_text(json.dumps({'fixture': 'model-graph-preparation'}))
                generated=Path('generated')
                generated.mkdir(exist_ok=True)
                if os.environ.get('LCB_TEST_MODE')!='missing-model-graph-contract':
                    (generated/'model-graph-v1.schema.json').write_text(json.dumps({'schemaSha256': 'd'*64}))
                mode=os.environ.get('LCB_TEST_MODE')
                if mode=='modify':
                    (root/'README.md').write_text('modified during configure\\n')
                elif mode=='restore':
                    (root/'README.md').write_text('original\\n')
        '''))
        self.git(self.repository,'add','-A')
        self.commit(self.root)
        self.source_sha=self.git(self.root,'rev-parse','HEAD').strip()
        tools=self.work/'tools'
        tools.mkdir()
        (tools/'src/lib').mkdir(parents=True)
        (tools/'src/lib/libasync.js').write_bytes(patched_runtime)
        emcc=tools/'emcc'
        emcc.write_text(f'#!{sys.executable}\nprint("emcc (test fixture) {toolchain["emsdkVersion"]}")\n')
        emcc.chmod(0o755)
        emcmake=tools/'emcmake'
        emcmake.write_text(f'#!{sys.executable}\nimport os, sys\nos.execvp(sys.argv[1], sys.argv[1:])\n')
        emcmake.chmod(0o755)
        self.env['PATH']=str(tools)+os.pathsep+self.env['PATH']
        self.env['LCB_TEST_PYTHON']=sys.executable
        self.env['LCB_TEST_ORIGINAL_ROOT']=str(self.root)
        self.env['PYTHONPATH']=str(self.repository)
        self.env['PYTHONDONTWRITEBYTECODE']='1'
        self.caller=self.work/'caller'
        self.caller.mkdir()

    def git(self, directory, *args):
        return subprocess.check_output(['git','-C',str(directory),*args],text=True,env=self.env)

    def commit(self, directory):
        self.git(directory,'-c','user.name=Build test','-c','user.email=test@example.invalid',
                 'commit','--quiet','-m','test: initialize local fixture')

    def build(self, profile='cpu-wasm32', *, mode=None, check=True, cwd=None, fresh=False):
        env=self.env.copy()
        if mode is not None:
            env['LCB_TEST_MODE']=mode
        result=subprocess.run([sys.executable,'-m','build_tools.runtimes.llama_cpp.build','--profile',profile,'--variant','browser', *(['--fresh'] if fresh else [])],
                              cwd=cwd or self.root,env=env,text=True,capture_output=True,check=check)
        self.last_result=result
        if result.returncode:
            return result
        return json.loads((self.root/'build'/profile/'browser/provenance.json').read_text())

    def test_all_profiles_keep_probe_outputs_in_the_build_directory(self):
        dawn=self.root.parents[1]/'.tools/emdawnwebgpu_pkg'
        dawn.mkdir(parents=True)
        (dawn/'emdawnwebgpu.port.py').write_text('# ignored tool fixture\n')
        for profile in json.loads((ROOT/'config/profiles.json').read_text()):
            with self.subTest(profile=profile):
                info=self.build(profile)
                self.assertFalse(info['sourceDirty'])
                self.assertEqual(info['modelGraphBoundary'], {'fixture': 'model-graph-preparation', 'schemaSha256': 'd'*64})
                memory64 = json.loads((ROOT / 'config/profiles.json').read_text())[profile]['memory64']
                self.assertIn('-DLCB_FILE_BACKED_PLE=' + ('ON' if memory64 else 'OFF'), info['cmakeCommand'])
                self.assertEqual(info['cmakeCommand'].count('-DLCB_QSA_ALL_POOLS=ON'), 1)
                self.assertEqual(info['sourceCommit'],self.source_sha)
                self.assertEqual(info['sourceStatusBeforeBuild'],[])
                self.assertEqual(info['sourceStatusAfterBuild'],[])
                self.assertFalse(info['browserModulePreparation']['alreadyAnnotated'])
                self.assertEqual(self.git(self.root,'status','--porcelain','--untracked-files=all'),'')
                for name in ('a.out.js','a.out.wasm'):
                    self.assertTrue((self.root/'build'/profile/'browser'/name).is_file())
                    self.assertFalse((self.root/name).exists())
                    self.assertFalse((self.caller/name).exists())

    def test_missing_model_graph_contract_fails_instead_of_inventing_provenance(self):
        result=self.build(mode='missing-model-graph-contract',check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('model-graph-v1.schema.json', result.stderr)

    def test_reusing_a_linked_browser_module_does_not_duplicate_the_annotation(self):
        first=self.build()
        module=self.root/'build/cpu-wasm32/browser/runtime/core.mjs'
        original=module.read_bytes()
        second=self.build()
        self.assertEqual(module.read_bytes(),original)
        self.assertEqual(original.count(b'/* @vite-ignore */'),1)
        self.assertTrue(second['browserModulePreparation']['alreadyAnnotated'])
        self.assertEqual(first['browserModulePreparation']['output'],second['browserModulePreparation']['input'])

    def test_fresh_build_cleans_only_the_selected_profile_variant(self):
        stale=self.root/'build/cpu-wasm32/browser/stale-output'
        other=self.root/'build/cpu-wasm32/test/keep-output'
        for path in (stale, other):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('fixture')
        info=self.build(fresh=True)
        self.assertFalse(stale.exists())
        self.assertTrue(other.is_file())
        self.assertEqual((self.root/'README.md').read_text(),'original\n')
        self.assertFalse(info['sourceDirty'])

    def test_invocation_from_another_directory_does_not_pollute_the_caller(self):
        info=self.build(cwd=self.caller)
        self.assertFalse(info['sourceDirty'])
        self.assertEqual(list(self.caller.iterdir()),[])
        self.assertEqual(self.git(self.root,'status','--porcelain','--untracked-files=all'),'')
        self.assertTrue((self.root/'build/cpu-wasm32/browser/a.out.js').is_file())

    def test_existing_tracked_changes_remain_dirty_and_are_reported(self):
        (self.root/'README.md').write_text('local changes\n')
        info=self.build()
        self.assertTrue(info['sourceDirty'])
        self.assertIn(' M runtimes/llama-cpp/README.md',info['sourceStatusBeforeBuild'])
        self.assertIn(' M runtimes/llama-cpp/README.md',info['sourceStatusAfterBuild'])
        self.assertIn('README.md',self.last_result.stderr)
        self.assertEqual((self.root/'README.md').read_text(),'local changes\n')

    def test_untracked_files_are_not_hidden(self):
        (self.root/'notes').mkdir()
        (self.root/'notes/local.txt').write_text('keep this\n')
        info=self.build()
        self.assertTrue(info['sourceDirty'])
        self.assertIn('?? runtimes/llama-cpp/notes/local.txt',info['sourceStatusBeforeBuild'])
        self.assertIn('?? runtimes/llama-cpp/notes/local.txt',info['sourceStatusAfterBuild'])
        self.assertTrue((self.root/'notes/local.txt').is_file())

    def test_changes_created_during_configure_are_reported(self):
        info=self.build(mode='modify')
        self.assertEqual(info['sourceStatusBeforeBuild'],[])
        self.assertIn(' M runtimes/llama-cpp/README.md',info['sourceStatusAfterBuild'])
        self.assertTrue(info['sourceDirty'])

    def test_restoring_a_dirty_input_does_not_mark_its_build_clean(self):
        (self.root/'README.md').write_text('dirty input\n')
        info=self.build(mode='restore')
        self.assertIn(' M runtimes/llama-cpp/README.md',info['sourceStatusBeforeBuild'])
        self.assertEqual(info['sourceStatusAfterBuild'],[])
        self.assertTrue(info['sourceDirty'])

    def test_shared_runtime_metadata_and_compiler_mismatch(self):
        dawn=self.root.parents[1]/'.tools/emdawnwebgpu_pkg'
        dawn.mkdir(parents=True)
        (dawn/'emdawnwebgpu.port.py').write_text('# local fixture\n')
        profiles=json.loads((ROOT/'config/profiles.json').read_text())
        gpu=next(name for name, info in profiles.items() if info['webgpu'] and not info.get('gpuProfile'))
        for profile in ['cpu-wasm32', gpu]:
            info=self.build(profile)
            self.assertEqual(info['visionBf16Storage'], {'version':1, 'modes':['legacy-expand','retain-checked'], 'default':'legacy-expand'})
            self.assertFalse(any('LCB_WEBGPU_BF16_STORAGE' in arg for arg in info['cmakeCommand']))
            self.assertTrue((self.root/'build'/profile/'browser/runtime/core.mjs').is_file())
            self.assertFalse((self.root/'build'/profile/'browser-bf16-retained').exists())
        result=self.build(gpu, mode='bf16-mismatch', check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('compile definition mismatch', result.stderr)
        self.assertFalse((self.root/'build'/gpu/'browser/provenance.json').exists())

    def test_dirty_submodule_is_rejected_before_configuring(self):
        (self.upstream/'include/llama.h').write_text('modified upstream\n')
        result=self.build(check=False)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Dirty upstream checkout',result.stderr)
        self.assertIn('include/llama.h',result.stderr)
        self.assertFalse((self.root/'build/cpu-wasm32').exists())


if __name__=='__main__':
    unittest.main()
