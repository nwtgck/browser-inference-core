#!/usr/bin/env python3
"""Execute the prepared WGSL lane loops on a checked CPU model, not a GPU.

Only declarations differ in the translation; memory expressions, cache guards,
loop bounds and arithmetic come from the shader under test. Workgroup reductions
and storage helpers are modeled explicitly. The optional real-backend smoke
probe covers shader compilation/execution separately.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def between(text, start, end):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('Reviewed shader anchors changed: ' + start)
    return text.split(start, 1)[1].split(end, 1)[0]


def cpp(body):
    body = re.sub(r'var (\w+): u32', r'uint32_t \1', body)
    body = re.sub(r'var (\w+): f32;', r'float \1{};', body)
    body = re.sub(r'var (\w+) =', r'auto \1 =', body)
    body = re.sub(r'let (\w+) =', r'const auto \1 =', body)
    body = body.replace('max(max_val, val)', 'std::max(max_val, val)')
    body = body.replace('exp(val - row_max)', 'std::exp(val - row_max)')
    # WGSL infers f32 from row_sum; avoid an unintended C++ double operation.
    body = body.replace('1.0 / row_sum', '1.0f / row_sum')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cxx', required=True)
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    source = args.source.read_text()
    cache = re.search(r'const CACHE_SIZE: u32 = (\d+);', source)
    if not cache:
        raise ValueError('Missing cache bound')
    first = between(source, '    var max_val =', '    scratch[lid.x] = max_val;')
    first = '    var max_val =' + first + '\n return max_val;'
    second = between(source, '    var sum =', '    scratch[lid.x] = sum;')
    second = '    var sum =' + second + '\n return sum;'
    third = source.split('    let sum_recip =', 1)[1]
    if not third.endswith('}\n'):
        raise ValueError('Unexpected shader entry-point suffix')
    third = '    let sum_recip =' + third[:-2]
    template = Path(__file__).with_name('cache-test.cpp.in').read_text()
    for key, value in {'CACHE_SIZE': cache[1], 'FIRST': cpp(first), 'SECOND': cpp(second), 'THIRD': cpp(third)}.items():
        template = template.replace('@' + key + '@', value)
    with tempfile.TemporaryDirectory(prefix='sdc-softmax-cache-') as tmp:
        root = Path(tmp); code = root / 'cache.cpp'; exe = root / 'cache'
        code.write_text(template)
        flags = ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'] if args.sanitize else ['-O2']
        subprocess.run([args.cxx, '-std=c++17', *flags, str(code), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    main()
