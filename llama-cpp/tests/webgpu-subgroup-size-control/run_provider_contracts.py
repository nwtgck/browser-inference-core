"""Compile only newly connected provider guards extracted from actual sources."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

HERE = Path(__file__).resolve().parent


def identity(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); prepared = args.prepared.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cpp = prepared / 'ggml-webgpu.cpp'; header = prepared / 'ggml-webgpu-batch-readback.hpp'
    backend, provider = cpp.read_text(), header.read_text()
    start = backend.index('static bool ggml_webgpu_device_is_healthy(')
    helpers = backend[start:backend.index('#include "ggml-webgpu-batch-readback.hpp"', start)]
    start = provider.index('        if (!source || !source->global_ctx || !source->buffer)')
    first = provider[start:provider.index('        const auto * root', start)]
    assert provider.index('ggml_webgpu_require_healthy_global_device(global);') < provider.index('source->buffer.GetUsage()') < provider.index('source->buffer.GetSize()')
    start = provider.index('        auto commands = encoder.Finish();')
    submit = provider[start:provider.index('        ggml_backend_webgpu_map_buffer(', start)]
    start = provider.index('        if (!ggml_webgpu_device_is_healthy(global))')
    post_map = provider[start:provider.index('        bool okay = true;', start)]
    start = provider.index('    if (count == 0) return true;')
    empty_request = provider[start:provider.index('    std::array<range', start)]
    empty_global = '    if (!global) return true;\n'
    assert provider.count(empty_global) == 1
    at = provider.index('        if (!current.count) break;')
    line = provider[provider.index('\n', at) + 1:provider.index('\n', provider.index('\n', at) + 1)]
    assert line.strip() == 'ggml_webgpu_require_healthy_global_device(global);'
    fragments = {'GLOBAL_HELPERS': helpers, 'FIRST_API_GUARD': first, 'SUBMIT_GUARD': submit,
                 'POST_MAP_GUARD': post_map, 'EMPTY_REQUEST_GUARD': empty_request,
                 'EMPTY_GLOBAL_GUARD': empty_global, 'LOOP_GUARD': line + '\n'}
    paths = {}
    for macro, text in fragments.items():
        path = output / (macro.lower() + '.inc'); path.write_text(text); paths[macro] = path
    inputs = [cpp, header, HERE / 'provider.cpp', *paths.values()]
    pins = {str(path): identity(path) for path in inputs}
    command = [os.environ.get('CXX', 'g++'), '-std=c++20', '-O1']
    command += ['-D' + macro + '="' + str(path) + '"' for macro, path in paths.items()]
    command += [str(HERE / 'provider.cpp'), '-o', str(output / 'provider')]
    report = {'status': 'running', 'scope': 'Only new source-bound terminal boundaries; CPU API counters, not runtime readback proof.',
              'gpu': False, 'sourceFiles': pins, 'commands': []}
    try:
        for phase, argv in [('compile', command), ('execute', [str(output / 'provider')])]:
            result = subprocess.run(argv, capture_output=True, text=True)
            (output / (phase + '.stdout')).write_text(result.stdout); (output / (phase + '.stderr')).write_text(result.stderr)
            report['commands'].append({'phase': phase, 'argv': argv, 'exitCode': result.returncode})
            assert result.returncode == 0, result.stderr[-6000:]
        assert pins == {str(path): identity(path) for path in inputs}
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error)); raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
