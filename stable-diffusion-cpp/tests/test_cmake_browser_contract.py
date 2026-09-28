"""Real CMake configuration with mock upstream targets, not a Wasm build."""
from pathlib import Path
import shutil
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('cmake') and shutil.which('c++'), 'CMake and C++ compiler required')
class BrowserLinkContract(unittest.TestCase):
    def configure(self, variant, jspi, memory64=False):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root/'bridge').mkdir(); (root/'config').mkdir()
        (root/'scripts').mkdir(); (root/'tests').mkdir()
        for path in ('config/variants.json','bridge/core.d.ts','scripts/generate_bindings.py','tests/wasm-probes.cpp','tests/qwen-timestep-probe.cpp','tests/bf16-weights-probe.cpp','tests/graph-walk-probe.cpp','tests/conv3d-bias-probe.cpp'):
            shutil.copy2(ROOT/path,root/path)
        (root/'bridge/browser.cpp').write_text('// Fixture, never compiled\n')
        (root/'CMakeLists.txt').write_text((ROOT/'CMakeLists.txt').read_text()+'''
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/link-options.txt"
    CONTENT "$<TARGET_PROPERTY:core,LINK_OPTIONS>")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/defines.txt"
    CONTENT "$<TARGET_PROPERTY:core,COMPILE_DEFINITIONS>")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/link-inputs.txt"
    CONTENT "$<TARGET_PROPERTY:core,LINK_DEPENDS>")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sources.txt"
    CONTENT "$<TARGET_PROPERTY:core,SOURCES>")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/backend-options.txt"
    CONTENT "$<TARGET_PROPERTY:ggml-webgpu,INTERFACE_LINK_OPTIONS>")
''')
        sd, ggml = root/'upstream', root/'ggml'
        for path in (sd,ggml): (path/'include').mkdir(parents=True)
        (sd/'include/stable-diffusion.h').write_text('void sd_ctx_params_init(void *p);');(ggml/'include/ggml.h').touch()
        (sd/'stub.cpp').write_text('// No compilation\n')
        (sd/'CMakeLists.txt').write_text('''
add_library(stable-diffusion STATIC stub.cpp)
add_library(ggml-webgpu INTERFACE)
set_property(TARGET ggml-webgpu PROPERTY INTERFACE_LINK_OPTIONS "-exceptions")
''')
        output = root/'out'
        command = ['cmake','-S',str(root),'-B',str(output),'-DEMSCRIPTEN=ON',
                   '-DSDCB_SOURCE='+str(sd),'-DSDCB_GGML_SOURCE='+str(ggml),
                   '-DSDCB_VARIANT='+variant,'-DSDCB_JSPI='+('ON' if jspi else 'OFF'),
                   '-DSDCB_MEMORY64='+('ON' if memory64 else 'OFF'),
                   '-DSDCB_MAXIMUM_MEMORY='+('17179869184' if memory64 else '4294967296')]
        subprocess.run(command,check=True,capture_output=True,text=True,timeout=60)
        return root, {p.stem:p.read_text() for p in output.glob('*.txt')}

    def assert_probe_boundary(self, root, data, variant):
        # Inspect CMake's evaluated target, not just a spelling of target_sources.
        # Both source files and generated exports must obey the variant boundary.
        count = 1 if variant == 'test' else 0
        sources = data['sources'].split(';')
        for probe in ('tests/wasm-probes.cpp', 'tests/qwen-timestep-probe.cpp', 'tests/bf16-weights-probe.cpp',
                      'tests/graph-walk-probe.cpp', 'tests/conv3d-bias-probe.cpp'):
            self.assertEqual(sources.count(probe), count, probe)
        entries = json.loads((root/'out/generated/exports.json').read_text())
        for probe in ('_sdc_test_gguf_offset', '_sdc_test_gguf_value',
                      '_sdc_test_callbacks', '_sdc_test_safetensors_offset',
                      '_sdc_test_safetensors_value', '_sdc_test_model_tensor_count',
                      '_sdc_test_qwen_timestep', '_sdc_test_bf16_weights', '_sdc_test_graph_walk', '_sdc_test_conv3d_bias'):
            self.assertEqual(entries.count(probe), count, probe)
        suspending = json.loads((root/'out/generated/jspi-exports.json').read_text())
        self.assertEqual(suspending.count('sdc_test_qwen_timestep'), count)
        self.assertEqual(suspending.count('sdc_test_bf16_weights'), count)
        self.assertEqual(suspending.count('sdc_test_conv3d_bias'), count)

    def test_browser_variants_have_no_test_hook_and_keep_both_suspension_modes(self):
        for jspi in (True,False):
            with self.subTest(jspi=jspi):
                root,data = self.configure('browser',jspi)
                self.assert_probe_boundary(root, data, 'browser')
                self.assertNotIn('_sdb_test_callbacks', data['link-options'])
                self.assertNotIn('SDCB_TEST_HOOKS',data['defines'])
                self.assertNotIn('--pre-js',data['link-options'])
                self.assertIn('exports.json',data['link-inputs'])
                self.assertNotIn('_sdc_test_callbacks', json.loads((root/'out/generated/exports.json').read_text()))
                self.assertNotIn('_sdc_test_qwen_timestep', json.loads((root/'out/generated/exports.json').read_text()))
                self.assertIn('-sJSPI=1' if jspi else '-sASYNCIFY=1',data['link-options'])
                incoming=next(part for part in data['link-options'].split(';') if part.startswith('-sINCOMING_MODULE_JS_API'))
                self.assertNotIn('onProgress',incoming);self.assertNotIn('onLog',incoming)
                if not jspi: self.assertNotIn('-exceptions',data['backend-options'])

    def test_memory64_jspi_keeps_large_heap_and_address_width_flags(self):
        for variant in ('browser', 'test'):
            with self.subTest(variant=variant):
                root, data = self.configure(variant, True, memory64=True)
                self.assert_probe_boundary(root, data, variant)
                self.assertIn('-sMEMORY64=1', data['link-options'])
                self.assertIn('-sMAXIMUM_MEMORY=17179869184', data['link-options'])
                self.assertIn('-sJSPI=1', data['link-options'])

    def test_memory64_asyncify_is_rejected(self):
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.configure('browser', False, memory64=True)
        self.assertIn('memory64 profile requires JSPI', failure.exception.stderr)

    def test_test_variants_expose_explicit_synthetic_probes_and_suspending_timestep(self):
        for jspi in (True,False):
            with self.subTest(jspi=jspi):
                root,data = self.configure('test',jspi)
                self.assert_probe_boundary(root, data, 'test')
                self.assertIn('SD_BROWSER_WEBGPU=1',data['defines'])
                self.assertIn('GGML_MAX_NAME=160',data['defines'])
                self.assertIn('SD_USE_UPSTREAM_GGML',data['defines'])
                exports=next(part for part in data['link-options'].split(';') if part.startswith('-sEXPORTED_FUNCTIONS'))
                entries=json.loads((root/'out/generated/exports.json').read_text())
                self.assertEqual(entries.count('_sdc_test_callbacks'),1)
                self.assertIn('_sdc_test_gguf_offset',entries)
                self.assertIn('_sdc_test_qwen_timestep', entries)
                self.assertIn('sdc_test_qwen_timestep', json.loads((root/'out/generated/jspi-exports.json').read_text()))
                self.assertNotIn('_sdb_load',entries)
                self.assertIn('-sASSERTIONS=1',data['link-options'])


if __name__ == '__main__': unittest.main()
