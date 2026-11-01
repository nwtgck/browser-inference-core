"""Sequential source-bound CPU contracts; no native runtime or GPU is loaded."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
sys.path.insert(0, str(ROOT / 'tests'))
from webgpu_prefix_context import without_subgroup_size_control


def identity(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
    sdk = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg')).resolve()
    source.tensor_copy.verify_dawn(sdk)
    report = {'status': 'running', 'scope': 'Complete actual header factories/SDK binding with mocked C GPU API; source-extracted host dispatch.',
              'gpu': False, 'commands': []}

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(mode, phase, argv, inputs):
        before = {str(path): identity(path) for path in inputs}
        started = time.monotonic(); result = subprocess.run(argv, capture_output=True, text=True)
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
            with without_subgroup_size_control(source):
                source.prepare(upstream, baseline, gpu_profile=profile)
            source.prepare(upstream, candidate, gpu_profile=profile)
            backend = (candidate / 'ggml-webgpu.cpp').read_text()
            start = backend.index('static inline void compute_2d_workgroups(')
            compute = backend[start:backend.index('\n}', start) + 2]
            start = backend.index('    // Calculate workgroup dimensions', backend.index('ggml_webgpu_mul_mat'))
            start = backend.index('    if (use_mat_vec) {', start) + len('    if (use_mat_vec) {\n')
            dispatch = backend[start:backend.index('    } else {', start)]
            fragment = work / 'host-dispatch.inc'
            fragment.write_text(compute + '\nstatic std::array<uint32_t,2> production_dispatch(candidate::webgpu_pipeline pipeline, ggml_tensor* dst, uint32_t max_wg_per_dim) {\n'
                                'using ggml_webgpu_mul_mat_vec_shader_decisions = candidate::ggml_webgpu_mul_mat_vec_shader_decisions;\n'
                                'uint32_t wg_x=1,wg_y=1;\n' + dispatch + '\nreturn {wg_x,wg_y};\n}\n')
            inputs = [HERE / 'recording.cpp', fragment, *[p for directory in [baseline, candidate] for p in directory.iterdir() if p.is_file()]]
            argv = [os.environ.get('CXX', 'g++'), '-std=c++20', '-O1', '-D__wasi__', '-pthread',
                    '-DBASE_HEADER="' + str(baseline / 'ggml-webgpu-shader-lib.hpp') + '"',
                    '-DCANDIDATE_HEADER="' + str(candidate / 'ggml-webgpu-shader-lib.hpp') + '"',
                    '-DOPTIONAL_HEADER="' + str(candidate / 'ggml-webgpu-optional-pipeline.hpp') + '"',
                    '-DHOST_FRAGMENTS="' + str(fragment) + '"',
                    '-DDIAGNOSTIC_HEADER="' + str(ROOT / 'bridge/webgpu-q8-diagnostic.h') + '"',
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
