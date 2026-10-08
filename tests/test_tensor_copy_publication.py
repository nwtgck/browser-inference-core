"""Publication rejection checks, without external Git or network activity."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PUBLISHERS = [load('scripts/publish_artifacts.py', 'copy_root_publisher'),
              load('llama-cpp/scripts/publish_artifacts.py', 'copy_legacy_publisher')]


def manifest(options):
    return {'sourceCommit': 'a' * 40, 'profiles': {'webgpu-wasm32-jspi': {
        'variants': {'browser': {'cmakeCommand': options}}}}}


class TensorCopyPublication(unittest.TestCase):
    def test_off_and_old_absence_remain_eligible(self):
        for publisher in PUBLISHERS:
            for options in ([], ['-DLCB_WEBGPU_TENSOR_COPY=OFF'], ['-DLCB_WEBGPU_PARAM_UPLOAD_BATCHING=OFF'], ['-DLCB_WEBGPU_TENSOR_COPY=OFF', '-DLCB_WEBGPU_PARAM_UPLOAD_BATCHING=OFF']):
                with self.subTest(publisher=publisher.__name__, options=options):
                    publisher.require_publishable_tensor_copy(manifest(options))

    def test_opt_in_or_ambiguous_options_rejected_before_remote_activity(self):
        options_cases = [
            ['-DLCB_WEBGPU_TENSOR_COPY=ON'],
            ['-DLCB_WEBGPU_TENSOR_COPY=OFF', '-DLCB_WEBGPU_TENSOR_COPY=ON'],
            ['-DLCB_WEBGPU_TENSOR_COPY=OFF', '-DLCB_WEBGPU_TENSOR_COPY=OFF'],
            ['-DLCB_WEBGPU_TENSOR_COPY=TRUE'],
            ['-DLCB_WEBGPU_TENSOR_COPY:BOOL=ON'],
            ['-DLCB_WEBGPU_TENSOR_COPY'],
        ]
        options_cases += [[item.replace('TENSOR_COPY', 'PARAM_UPLOAD_BATCHING') for item in options] for options in options_cases]
        options_cases += [['-DLCB_WEBGPU_TENSOR_COPY=OFF', '-DLCB_WEBGPU_PARAM_UPLOAD_BATCHING=ON']]
        for publisher in PUBLISHERS:
            for options in options_cases:
                with self.subTest(publisher=publisher.__name__, options=options), tempfile.TemporaryDirectory() as tmp:
                    package = Path(tmp)
                    inner = manifest(options)
                    if publisher is PUBLISHERS[0]:
                        nested = package / 'llama-cpp-browser-core'
                        nested.mkdir()
                        (nested / 'manifest.json').write_text(json.dumps(inner))
                        outer = {'sourceCommit': 'a' * 40, 'runtimes': {
                            'llama-cpp': {'manifest': 'llama-cpp-browser-core/manifest.json'}}}
                    else:
                        outer = inner
                    (package / 'manifest.json').write_text(json.dumps(outer))
                    # Package validation is independently covered by the package
                    # suites; here isolate the publication authorization gate.
                    with patch.object(publisher, 'validate', return_value=outer), \
                         patch.object(publisher, 'run') as calls, \
                         self.assertRaisesRegex(ValueError, 'tensor-copy.*artifact-only'):
                        publisher.publish(package, 'https://example.invalid/repository.git')
                    self.assertEqual([call.args for call in calls.call_args_list],
                                     [('git', 'check-ref-format', '--branch', 'artifacts')])

    def test_every_variant_is_inspected(self):
        data = manifest(['-DLCB_WEBGPU_TENSOR_COPY=OFF'])
        data['profiles']['webgpu-wasm32-jspi']['variants']['test'] = {
            'cmakeCommand': ['-DLCB_WEBGPU_TENSOR_COPY=ON']}
        for publisher in PUBLISHERS:
            with self.assertRaisesRegex(ValueError, 'artifact-only'):
                publisher.require_publishable_tensor_copy(data)


if __name__ == '__main__':
    unittest.main()
