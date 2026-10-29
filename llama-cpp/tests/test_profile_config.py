"""Validate profile selection with mocked subprocesses; never configure or build."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from fixture_toolchain import merged_toolchain, seed_toolchain
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build


class ProfileConfiguration(unittest.TestCase):
    def test_ci_builds_every_packaged_profile(self):
        profiles = json.loads((ROOT / 'config/profiles.json').read_text())
        workflow = (ROOT.parent / '.github/workflows/build.yml').read_text()
        matrix = workflow.split('        profile:\n', 1)[1].split('        variant:', 1)[0]
        self.assertEqual(set(re.findall(r'^          - ([a-z0-9-]+)$', matrix, re.MULTILINE)),
                         set(profiles))
        variants = json.loads((ROOT / 'config/variants.json').read_text())
        variant_matrix = workflow.split('        variant:\n', 1)[1].split('    steps:', 1)[0]
        self.assertEqual(set(re.findall(r'^          - ([a-z0-9-]+)$', variant_matrix, re.MULTILINE)),
                         set(variants))

    def test_each_profile_passes_its_suspension_and_memory_configuration(self):
        profiles = json.loads((ROOT / 'config/profiles.json').read_text())
        expected = {
            'cpu-wasm32': ('OFF', 'OFF', 'OFF', 'OFF', '4294967296'),
            'webgpu-wasm64-jspi-gpu-profile': ('ON', 'ON', 'ON', 'OFF', '17179869184'),
            'webgpu-wasm32-asyncify': ('OFF', 'ON', 'OFF', 'ON', '4294967296'),
            'webgpu-wasm32-jspi': ('OFF', 'ON', 'ON', 'OFF', '4294967296'),
            'webgpu-wasm64-jspi': ('ON', 'ON', 'ON', 'OFF', '17179869184'),
        }
        self.assertEqual(set(profiles), set(expected))
        variants = json.loads((ROOT / 'config/variants.json').read_text())
        self.assertEqual(variants, {
            'browser': {'assertions': 0, 'environment': 'web,worker'},
            'test': {'assertions': 1, 'environment': 'web,worker,node'},
        })
        toolchain = merged_toolchain()
        with tempfile.TemporaryDirectory(prefix='lcb-profile-test-') as temporary:
            root = Path(temporary) / 'llama-cpp'
            root.mkdir()
            (root / 'config').mkdir()
            (root / 'config/profiles.json').write_text(json.dumps(profiles))
            (root / 'config/variants.json').write_text(json.dumps(variants))
            seed_toolchain(root, toolchain)
            (root / 'vendor/llama.cpp/include').mkdir(parents=True)
            (root / 'vendor/llama.cpp/include/llama.h').touch()
            (root.parent / '.tools/emdawnwebgpu_pkg').mkdir(parents=True)
            (root.parent / '.tools/emdawnwebgpu_pkg/emdawnwebgpu.port.py').touch()
            for profile, values in expected.items():
                for variant, settings in variants.items():
                    module = root / 'build' / profile / variant / 'runtime/core.mjs'
                    module.parent.mkdir(parents=True)
                    source = b'function findWasmBinary(){if(Module["locateFile"]){return locateFile("core.wasm")}return new URL("core.wasm",import.meta.url).href}'
                    module.write_bytes(source)
                    macro = ' -DGGML_WEBGPU_GPU_PROFILE=1' if profiles[profile].get('gpuProfile') else ''
                    commands = ([{'file': '/build/generated/ggml-webgpu.cpp',
                                  'command': 'em++ -c /build/generated/ggml-webgpu.cpp' + macro}]
                                if profiles[profile]['webgpu'] else [])
                    if profiles[profile].get('gpuProfile'):
                        commands += [{'file': '/build/generated/' + name,
                                      'command': 'em++ -DLCB_WEBGPU_Q8_DIAGNOSTIC=1 -c /build/generated/' + name}
                                     for name in ('webgpu-q8-diagnostic.cpp', 'q8-diagnostic-metadata.cpp')]
                    commands += [{'file': '/build/generated/' + name,
                                  'command': 'em++ -DLCB_MTMD_WEBGPU_BF16_RUNTIME=' + str(int(profiles[profile]['webgpu'])) + ' -c /build/generated/' + name}
                                 for name in ('clip.cpp', 'mtmd.cpp')]
                    (module.parent.parent / 'compile_commands.json').write_text(json.dumps(commands))
                    with self.subTest(profile=profile, variant=variant), \
                         patch.object(build, 'ROOT', root), \
                         patch.object(build, 'source_status', return_value=[]), \
                         patch.object(build.shutil, 'which', return_value='/test-toolchain/emcc'), \
                         patch.object(build, 'verify_asyncify_bigint_patch') as verify_patch, \
                         patch.object(build, 'output', side_effect=[
                             toolchain['llamaCommit'], f'emcc {toolchain["emsdkVersion"]}', 'a' * 40,
                         ]), \
                         patch.object(build.subprocess, 'run') as run, \
                         patch.object(sys, 'argv', ['build.py', '--profile', profile, '--variant', variant]), \
                         patch('builtins.print'):
                        build.main()
                        command = run.call_args_list[0].args[0]
                        self.assertIn(f'-DLCB_VARIANT={variant}', command)
                        self.assertIn('-DGGML_WEBGPU_GPU_PROFILE=' + ('ON' if profiles[profile].get('gpuProfile') else 'OFF'), command)
                        self.assertFalse(any('LCB_WEBGPU_MOE_DIRECT_SLOT' in arg for arg in command))
                        self.assertIn('-DLCB_WEBGPU_BF16_PROJECTOR='+('ON' if values[1]=='ON' else 'OFF'), command)
                        names = ('MEMORY64', 'WEBGPU', 'JSPI', 'ASYNCIFY', 'MAXIMUM_MEMORY')
                        for name, value in zip(names, values):
                            self.assertIn(f'-DLCB_{name}={value}', command)
                        self.assertEqual(any(arg.startswith('-DEMDAWNWEBGPU_DIR=') for arg in command),
                                         profiles[profile]['webgpu'])
                        provenance = json.loads((root / 'build' / profile / variant / 'provenance.json').read_text())
                        self.assertEqual(provenance['configuration'], profiles[profile])
                        self.assertEqual(provenance['variant'], variant)
                        self.assertEqual(provenance['variantConfiguration'], settings)
                        self.assertEqual('browserModulePreparation' in provenance, variant == 'browser')
                        if variant == 'browser':
                            self.assertEqual(module.read_bytes().replace(b'/* @vite-ignore */', b'', 1), source)
                        else:
                            self.assertEqual(module.read_bytes(), source)
                        if profiles[profile]['asyncify']:
                            verify_patch.assert_called_once_with(Path('/test-toolchain'), toolchain['emscriptenAsyncifyBigIntPatch'])
                        else:
                            verify_patch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
