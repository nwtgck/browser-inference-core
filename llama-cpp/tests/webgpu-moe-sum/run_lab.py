"""Reproduce an offline, ordered expert-sum checks. This does not build a runtime."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--lab', type=Path, required=True, help='Extracted qwen-next-hc-lab directory')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--benchmarks', action='store_true')
    args = parser.parse_args()
    source, lab, work = (getattr(args, k).resolve() for k in ('llama_source', 'lab', 'work'))
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[1] / 'scripts'))
    import prepare_webgpu_source
    from emit_factory import emit
    from check_dispatch import check as check_dispatch
    required = [source/'CMakeLists.txt', source/'ggml/src/ggml-webgpu/wgsl-shaders/binary.wgsl',
                lab/'bin/node', lab/'lib/ld-linux-x86-64.so.2', lab/'node_modules/webgpu/index.js']
    for path in required:
        if not path.is_file(): parser.error('Missing input: ' + str(path))
    for root in (source, lab, here):
        if work == root or root in work.parents or work in root.parents:
            parser.error('--work must not overlap source, lab, or the test directory')
    work.mkdir(parents=True, exist_ok=True)
    build, shaders = work/'native', work/'shaders'
    shaders.mkdir(exist_ok=True)
    prepared = work / 'prepared'
    prepare_webgpu_source.prepare(source, prepared)
    factory = emit(source, prepared, work / 'factory')
    commands = []
    def run(label, command, env=None, expect=0):
        with (work/(label+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=600)
        commands.append({'label': label, 'command': command, 'returncode': result.returncode, 'expected': expect})
        (work/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        if result.returncode != expect: raise RuntimeError(label+' failed; inspect '+str(work/(label+'.log')))
    run('production-factory', [str(factory), str(shaders)])
    run('configure', ['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF',
        '-DGGML_CCACHE=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
        '-DLLAMA_BUILD_TOOLS=OFF', '-DLLAMA_BUILD_COMMON=OFF', '-DLLAMA_CURL=OFF'])
    run('build', ['cmake', '--build', str(build), '--target', 'ggml-base', '-j', '4'])
    run('admission-build', ['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
        '-isystem', str(source/'ggml/include'), '-isystem', str(source/'ggml/src'),
        '-I'+str(prepared), str(here/'admission.cpp'), str(build/'ggml/src/libggml-base.a'), '-pthread', '-ldl', '-lm', '-o', str(work/'admission')])
    run('admission', [str(work/'admission')])
    dispatch_result = check_dispatch(source, prepared, build / 'ggml/src/libggml-base.a', work / 'dispatch')
    (work / 'dispatch.log').write_text(dispatch_result)
    run('emitter-build', ['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-isystem',
        str(source/'ggml/src/ggml-webgpu'), str(here/'emit_reference.cpp'), '-o', str(work/'emit-reference')])
    run('emitter', [str(work/'emit-reference'), str(required[1]), str(shaders)])
    icd = json.loads((lab/'icd-template.json').read_text())
    icd['ICD']['library_path'] = str(lab/'lib/libvulkan_lvp.so')
    (work/'icd.json').write_text(json.dumps(icd)+'\n')
    env = {**os.environ, 'VK_ICD_FILENAMES': str(work/'icd.json'), 'XDG_RUNTIME_DIR': str(work),
           'LCB_TEST_WEBGPU_MODULE': str(lab/'node_modules/webgpu/index.js'),
           'MOE_SUM_BENCHMARKS': '1' if args.benchmarks else '0', 'MOE_SUM_NEGATIVE': '0'}
    node = [str(lab/'lib/ld-linux-x86-64.so.2'), '--library-path', str(lab/'lib'), str(lab/'bin/node'), '--expose-gc', str(here/'check_gpu.mjs'), str(shaders)]
    run('gpu', node+[str(work/'gpu.json')], env)
    run('negative-control', node+[str(work/'negative.json')], {**env, 'MOE_SUM_NEGATIVE': '1'}, expect=1)
    result = json.loads((work/'gpu.json').read_text())
    negative = json.loads((work/'negative.json').read_text())
    if not result['ok'] or not negative['cases'] or not negative['cases'][0]['mismatch']:
        raise RuntimeError('GPU result or negative-control mismatch evidence missing')
    identified = list(here.glob('*')) + list(prepared.glob('*')) + [source/'ggml/src/ggml-impl.h', required[1]] + list(shaders.glob('*.wgsl'))
    (work/'source-identity.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in identified if p.is_file()}, indent=2)+'\n')
    print('Production-source graph admission and shader checks passed; not a compiled-runtime/model test.')


if __name__ == '__main__':
    main()
