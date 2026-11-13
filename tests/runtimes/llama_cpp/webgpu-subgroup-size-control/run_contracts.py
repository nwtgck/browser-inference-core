"""Sequential source-bound CPU contracts; no GPU driver or browser is loaded."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from build_tools.project import runtime_root, runtime_test_root
import argparse
import hashlib
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = runtime_root('llama-cpp')
from build_tools.runtimes.llama_cpp import source as source
# Historical byte identities belong to the reference fixture, not production.
from tests.support import frozen_webgpu_source as reference_source
from tests.support.llama_cpp.frozen_prefix_context import without_subgroup_size_control
from tests.support.historical_webgpu import historical_source
from tests.support.llama_cpp.dispatch_ast import current_dispatch


def identity(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
    sdk = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parents[1] / '.tools/emdawnwebgpu_pkg')).resolve()
    reference_source.tensor_copy.verify_dawn(sdk)
    report = {'status': 'running', 'scope': 'Complete current header factories/SDK binding versus owned historical baseline; mocked C GPU API and compiler-AST-selected current host dispatch.',
              'gpu': False, 'commands': []}

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(mode, phase, argv, inputs):
        before = {str(path): identity(path) for path in inputs}
        started = time.monotonic(); result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        (output / (mode + '-' + phase + '.stdout')).write_text(result.stdout)
        (output / (mode + '-' + phase + '.stderr')).write_text(result.stderr)
        report['commands'].append({'mode': mode, 'phase': phase, 'argv': argv, 'exitCode': result.returncode,
                                   'elapsedSeconds': time.monotonic() - started, 'sourceFiles': before})
        save(); print(json.dumps({'mode': mode, 'phase': phase, 'exitCode': result.returncode}), flush=True)
        assert result.returncode == 0, result.stderr[-6000:]
        assert before == {str(path): identity(path) for path in inputs}

    try:
        for mode, profile in [('normal', False), ('profile', True)]:
            work = output / mode; work.mkdir()
            baseline, candidate = work / 'baseline', work / 'candidate'
            # Only the before/after reference owns historical byte identities.
            # The candidate below continues to use the actual current upstream.
            with historical_source() as frozen, without_subgroup_size_control(reference_source):
                reference_source.prepare(frozen, baseline, patch_root=frozen / 'patches', gpu_profile=profile)
            source.prepare(upstream, candidate, gpu_profile=profile)
            fragment = work / 'host-dispatch.inc'
            fragment.write_text(current_dispatch(upstream, candidate, namespace='candidate'))
            inputs = [HERE / 'recording.cpp', HERE.parent / 'webgpu-native-recording/api.hpp', fragment, *[p for directory in [baseline, candidate] for p in directory.iterdir() if p.is_file()]]
            argv = [os.environ.get('CXX', 'g++'), '-std=c++20', '-O1', '-D__wasi__', '-pthread',
                    '-DBASE_HEADER="' + str(baseline / 'ggml-webgpu-shader-lib.hpp') + '"',
                    '-DCANDIDATE_HEADER="' + str(candidate / 'ggml-webgpu-shader-lib.hpp') + '"',
                    '-DOPTIONAL_HEADER="' + str(candidate / 'ggml-webgpu-optional-pipeline.hpp') + '"',
                    '-DHOST_FRAGMENTS="' + str(fragment) + '"',
                    '-DDIAGNOSTIC_HEADER="' + str(ROOT / 'native/diagnostics/webgpu-q8-diagnostic.h') + '"',
                    '-DRECORD_PROFILE=' + ('1' if profile else '0')]
            argv += ['-I' + str(path) for path in [candidate, upstream / 'ggml/src/ggml-webgpu', upstream / 'ggml/include', upstream / 'ggml/src']]
            argv += ['-isystem', str(sdk / 'webgpu/include'), '-isystem', str(sdk / 'webgpu_cpp/include'),
                     str(HERE / 'recording.cpp'), '-o', str(work / 'recording')]
            run(mode, 'compile', argv, inputs)
            run(mode, 'execute', [str(work / 'recording'), str(work / 'emissions')], inputs)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        save()


if __name__ == '__main__': main()
