"""Current complete-header Q8 behavior and explicitly frozen historical guards.

Only the C WebGPU API is mocked; no GPU/model inference is claimed.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
from build_tools.runtimes.llama_cpp import source as source
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy as copy
from tests.support.llama_cpp.frozen_webgpu_stages import q8_short as q8


from tests.support import frozen_webgpu_source as legacy_source
from tests.support.llama_cpp.current_webgpu import compile_recorder, require_recording_inputs


def function(text, marker):
    """Legacy helper for unmigrated tests; the current Q8 recorder does not use it."""
    start = text.index(marker)
    return text[start:text.index('\n}', start) + 2]


def build_host(upstream, prepared, output):
    return compile_recorder(upstream, prepared, output,
                            runtime_test_root('llama-cpp') / 'webgpu-q8-short/host-cases.cpp')


class Q8ShortOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / q8.HEADER_PATH).is_file():
            message = 'Set LCB_TEST_LLAMA_SOURCE for the current Q8 recorder'
            if os.environ.get('CI') == 'true':
                raise RuntimeError(message)
            raise unittest.SkipTest(message)

    def test_preparation_does_not_copy_shaders_or_modify_upstream(self):
        before = {p: copy.digest(self.upstream / p) for p in q8.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source.prepare(self.upstream, out)
            header = out / 'ggml-webgpu-shader-lib.hpp'
            stamp = header.stat().st_mtime_ns
            self.assertTrue(header.is_file())
            content = header.read_bytes()
            self.assertFalse((out / 'ggml-wgsl-shaders.hpp').exists())
            source.prepare(self.upstream, out)
            self.assertEqual(header.stat().st_mtime_ns, stamp)
            self.assertEqual(header.read_bytes(), content)
        self.assertEqual(before, {p: copy.digest(self.upstream / p) for p in before})

    def test_checker_rejects_invalid_node_commands_before_preparing_or_running(self):
        script = runtime_test_root('llama-cpp') / 'webgpu-q8-short/check_gpu.py'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not-created'
            for command in ['[]', 'null', '{}', '"node"', '[""]', '["node", 7]', '[invalid']:
                with self.subTest(command=command):
                    result = subprocess.run([
                        sys.executable, str(script), '--llama-source', str(out),
                        '--work', str(out), '--webgpu-module', str(out),
                        '--node-command-json', command,
                    ], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('--node-command-json must', result.stderr)
                    self.assertFalse(out.exists())

    def test_actual_selector_cache_factory_and_preprocessor(self):
        require_recording_inputs(self)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source.prepare(self.upstream, out / 'prepared')
            exe = build_host(self.upstream, out / 'prepared', out / 'host')
            result = subprocess.run([str(exe), str(out / 'emitted')], check=True,
                                    capture_output=True, text=True)
            self.assertIn('actual selector/cache/factory checks passed', result.stdout)
            for size in (32, 64, 128, 256):
                shader = (out / 'emitted' / f'q8-{size}.wgsl').read_text()
                self.assertIn(f'@workgroup_size({size})', shader)
                self.assertNotIn('#include', shader)
            print(result.stdout.strip())


    def test_current_recorder_accepts_unrelated_input_comments(self):
        require_recording_inputs(self)
        from build_tools.runtimes.llama_cpp import webgpu_plan as plan
        import json
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source.prepare(self.upstream, work / 'original')
            receipt = json.loads((work / 'original' / plan.RECEIPT).read_text())
            private = work / 'source'
            # The recorder needs declarations beyond the recipe's transformed
            # inputs. Copy, never mutate, the current source's header directories.
            for name in ('ggml/include', 'ggml/src'):
                shutil.copytree(self.upstream / name, private / name)
            for name in receipt['inputs']:
                if name.startswith('runtime/') or (private / name).exists():
                    continue
                target = private / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.upstream / name, target)
            header = private / plan.BASE / 'ggml-webgpu-shader-lib.hpp'
            header.write_bytes(header.read_bytes() + b'\n// unrelated current-source comment\n')
            source.prepare(private, work / 'changed')
            executable = build_host(private, work / 'changed', work / 'host')
            result = subprocess.run([str(executable), str(work / 'shaders')],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('actual selector/cache/factory checks passed', result.stdout)

    def test_current_recorder_does_not_hide_invalid_unused_declarations(self):
        require_recording_inputs(self)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source.prepare(self.upstream, work / 'prepared')
            header = work / 'prepared/ggml-webgpu-shader-lib.hpp'
            header.write_bytes(header.read_bytes() + b'\nstruct unused_q8_contract { unknown_q8_type value; };\n')
            with self.assertRaises(subprocess.CalledProcessError) as failure:
                build_host(self.upstream, work / 'prepared', work / 'host')
            self.assertIn('unknown_q8_type', failure.exception.stderr)


class HistoricalQ8ShortGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.support.historical_webgpu import historical_source
        cls.upstream = cls.enterClassContext(historical_source())

    def test_all_preparation_contracts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            for name in q8.CONTRACT_INPUTS:
                with self.subTest(path=name), patch.object(q8, 'CONTRACT_INPUTS', {
                    **q8.CONTRACT_INPUTS, name: '0' * 64,
                }):
                    with self.assertRaisesRegex(ValueError, 'Q8 short matvec contract changed'):
                        legacy_source.prepare(self.upstream, out, patch_root=self.upstream / 'patches')
                self.assertFalse((out / 'ggml-webgpu-shader-lib.hpp').exists())
            for name in ['INPUT_SHA256', 'OUTPUT_SHA256', 'PATCH_SHA256']:
                with self.subTest(guard=name), patch.object(q8, name, '0' * 64):
                    with self.assertRaises(ValueError):
                        legacy_source.prepare(self.upstream, out, patch_root=self.upstream / 'patches')
                self.assertFalse((out / 'ggml-webgpu.cpp').exists())
                self.assertFalse((out / 'ggml-webgpu-shader-lib.hpp').exists())



if __name__ == '__main__':
    unittest.main()
