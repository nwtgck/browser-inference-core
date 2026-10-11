#!/usr/bin/env python3
"""Compile actual prepared dispatch builders and shader index guards on CPU.

The API double records uniforms, bindings and grid dimensions. WGSL arithmetic
beyond the dispatch index is unchanged and is not emulated here. This is not a
WGSL compiler or GPU test; the optional real-backend probe executes large rows.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib import import_module
function = import_module('webgpu-wait-probe').function


def shader_guard(source, name):
    main = source.split('fn main(', 1)[1]
    guard = main.split('{', 1)[1].split('    let i3 =', 1)[0]
    if name != 'glu' and ('lid.' in guard or 'workgroupBarrier' in guard or 'src[' in guard):
        raise ValueError('Row guard must be uniform and precede memory/barriers')
    guard = re.sub(r'\bvar i =', 'uint32_t i =', guard)
    guard = re.sub(r'\blet group =', 'const uint32_t group =', guard)
    guard = guard.replace('return;', 'return false;')
    return ('bool shader_' + name + '(uint32_t count, uint32_t WG_SIZE, U3 wid, U3 num_wg, U3 lid, uint32_t& index) {\n'
            'struct {uint32_t ne, nrows;} params{count,count};\n' + guard + '\nindex=i; return true;\n}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cxx', required=True)
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    source = args.source.read_text()
    builders = [function(source, prefix) for prefix in [
        'static void ggml_webgpu_linear_workgroups(',
        'static inline uint32_t ggml_webgpu_u32_from_f32(',
        'static webgpu_encoded_op ggml_webgpu_row_norm(',
        'static std::optional<webgpu_encoded_op> ggml_webgpu_rms_norm_mul(',
        'static webgpu_encoded_op ggml_webgpu_soft_max(',
        'static webgpu_encoded_op ggml_webgpu_glu(',
    ]]
    root = Path(__file__).parent
    code = (root / 'prefix.cpp').read_text() + '\n'.join(builders)
    for name in ['row_norm', 'rms_norm_mul', 'soft_max', 'glu']:
        shader = (args.source.parent / 'wgsl-shaders' / (name + '.wgsl')).read_text()
        uniform = function(shader, 'struct Params {')
        fields = re.findall(r'^\s*(\w+): (u32|f32)', uniform, re.M)
        code += '\nconstexpr size_t ' + name + '_uniform_words = ' + str(len(fields)) + ';\n'
        if name != 'glu' and fields[-1] != ('nrows', 'u32'):
            raise ValueError('Logical row count must be the last scalar')
        code += shader_guard(shader, name)
    code += (root / 'cases.cpp').read_text()
    with tempfile.TemporaryDirectory(prefix='sdc-grid-') as tmp:
        cpp, exe = Path(tmp) / 'grid.cpp', Path(tmp) / 'grid'
        cpp.write_text(code)
        flags = ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'] if args.sanitize else ['-O2']
        subprocess.run([args.cxx, '-std=c++17', *flags, str(cpp), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)

if __name__ == '__main__':
    main()
