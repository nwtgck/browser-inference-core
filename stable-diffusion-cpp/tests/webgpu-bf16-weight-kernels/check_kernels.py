#!/usr/bin/env python3
"""Check prepared BF16 source and execute extracted WGSL scalar work on CPU.

This does not compile WGSL or execute a GPU. The fixture extracts the actual
unpack, row-copy, matvec and matrix-tile loading expressions. Matrix accumulation
and robust storage reads are modeled explicitly, separate from GPU validation.
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


def between(source, start, end):
    if source.count(start) != 1 or end not in source.split(start, 1)[1]:
        raise ValueError('Reviewed BF16 source anchor changed: ' + start)
    return source.split(start, 1)[1].split(end, 1)[0]


def wgsl_scalar(source):
    source = re.sub(r'fn (\w+)\((.*?)\)(?: -> ([^{]+))? \{',
                    lambda m: (m[3].strip() if m[3] else 'void') + ' ' + m[1] + '(' +
                    ', '.join(p.strip().split(':')[1].strip() + ' ' + p.strip().split(':')[0]
                              for p in m[2].split(',') if p.strip()) + ') {', source)
    source = source.replace('array<array<f32, OUTPUTS_PER_WG>, NUM_COLS>',
                            'std::array<std::array<float, OUTPUTS_PER_WG>, NUM_COLS>')
    source = source.replace('var acc: std::array<std::array<float, OUTPUTS_PER_WG>, NUM_COLS>;',
                            'std::array<std::array<float, OUTPUTS_PER_WG>, NUM_COLS> acc{};')
    source = re.sub(r'var (\w+): array<([^,]+), ([^>]+)>;', r'std::array<\2, \3> \1{};', source)
    source = re.sub(r'\bvar (\w+) =', r'auto \1 =', source)
    source = re.sub(r'\blet (\w+) =', r'const auto \1 =', source)
    source = source.replace('bitcast<f32>', 'bits_to_float')
    return re.sub(r'\bf16\b', 'half', re.sub(r'\bf32\b', 'float', re.sub(r'\bu32\b', 'uint32_t', source)))


def check_factory(source, name, expected):
    body = function(source, 'webgpu_pipeline ' + name + '(')
    branch = between(body, '            case GGML_TYPE_BF16:', '            default:')
    for token in expected:
        if token not in branch:
            raise ValueError(name + ' missing BF16 definition: ' + token)
    if 'TYPE_F16' in branch or 'SRC0_INNER_TYPE=f16' in branch:
        raise ValueError('BF16 storage incorrectly treated as F16')
    if 'key.vectorized =' in body:
        vectorization = body.split('key.vectorized =', 1)[1].split(';', 1)[0]
        if 'GGML_TYPE_BF16' in vectorization:
            raise ValueError('BF16 unexpectedly enabled the vectorized storage path')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Prepared ggml-webgpu.cpp')
    parser.add_argument('--cxx', required=True)
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    folder = args.source.parent
    shaders = folder / 'wgsl-shaders'
    backend = args.source.read_text()
    factory = (folder / 'ggml-webgpu-shader-lib.hpp').read_text()
    supports = function(backend, 'static bool ggml_backend_webgpu_device_supports_op(')
    rows = between(supports, '        case GGML_OP_GET_ROWS:', '        case GGML_OP_MUL_MAT:')
    mul = between(supports, '        case GGML_OP_MUL_MAT:', '        case GGML_OP_MUL_MAT_ID:')
    mul_id = between(supports, '        case GGML_OP_MUL_MAT_ID:', '        case GGML_OP_FLASH_ATTN_EXT:')
    for name, body in [('GET_ROWS', rows), ('MUL_MAT', mul), ('MUL_MAT_ID', mul_id)]:
        if 'GGML_TYPE_BF16' not in body:
            raise ValueError(name + ' does not admit BF16')
    if 'src_address_unit' not in rows or 'ggml_webgpu_tensor_misalignment' not in rows:
        raise ValueError('Existing GET_ROWS alignment admission was lost')
    if 'maxStorageBufferBindingSize' not in supports:
        raise ValueError('Existing binding-size admission was lost')
    # These kernels remain one-dimensional; do not silently classify BF16 as the
    # float-parallel path that launches one invocation for each element.
    row_builder = function(backend, 'static webgpu_encoded_op ggml_webgpu_get_rows(')
    parallel = re.search(r'const bool float_parallel = ([^;]+);', row_builder)
    if not parallel or 'GGML_TYPE_BF16' in parallel[1]:
        raise ValueError('BF16 GET_ROWS dispatch contract changed')
    check_factory(factory, 'get_get_rows_pipeline', ['"BF16"', '"SRC_TYPE=u32"', '"DST_TYPE=f32"'])
    for name in ['get_mul_mat_vec_pipeline', 'get_mul_mat_id_vec_pipeline',
                 'get_mul_mat_fast_pipeline', 'get_mul_mat_id_pipeline']:
        loading = '"MUL_ACC_FLOAT"' if 'vec' in name else '"INIT_SRC0_SHMEM_FLOAT"'
        check_factory(factory, name, ['"TYPE_BF16"', '"SRC0_INNER_TYPE=u32"', loading])

    common = (shaders / 'common_decls.tmpl').read_text()
    unpack = function(common, 'fn bf16_word_to_f32(')
    row_shader = (shaders / 'get_rows.wgsl').read_text()
    copy = function(between(row_shader, '#ifdef BF16\n', '#endif'), 'fn copy_elements(')
    matvec = (shaders / 'mul_mat_vec_acc.tmpl').read_text()
    inner = function(matvec, 'fn inner_dot(src0_val: f32,')
    accumulation = function(matvec, 'fn accumulate_vec_dot(')
    matrix = (shaders / 'mul_mat_decls.tmpl').read_text()
    tile = function(matrix, 'fn init_shmem_src0(')
    root = Path(__file__).parent
    program = (root / 'prefix.cpp').read_text() + '\n' + '\n'.join(map(wgsl_scalar, [unpack, copy, inner, accumulation, tile]))
    # Execute the actual capability case bodies with small tensor/API doubles.
    case_body = ('case GGML_OP_GET_ROWS:' + rows + 'case GGML_OP_MUL_MAT:' + mul +
                 'case GGML_OP_MUL_MAT_ID:' + mul_id)
    enum_types = sorted(set(re.findall(r'\bGGML_TYPE_\w+', case_body)))
    program += '\nenum ggml_type {' + ','.join(enum_types) + '};\n'
    program += (root / 'admission.cpp.in').read_text().replace('@CASES@', case_body)
    program += '\n' + (root / 'cases.cpp').read_text()
    with tempfile.TemporaryDirectory(prefix='sdc-bf16-kernels-') as tmp:
        cpp, exe = Path(tmp) / 'bf16.cpp', Path(tmp) / 'bf16'
        cpp.write_text(program)
        flags = ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'] if args.sanitize else ['-O2']
        subprocess.run([args.cxx, '-std=c++17', *flags, str(cpp), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()
