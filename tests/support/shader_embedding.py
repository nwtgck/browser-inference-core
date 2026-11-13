"""Read generated shader constants through a real C++ consumer, not a C++ parser."""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Mapping, Sequence


def compiled_shaders(headers: Mapping[str, Path], symbols: Sequence[str], work: Path) -> dict[str, dict[str, str]]:
    compiler = shutil.which('c++')
    if compiler is None:
        raise RuntimeError('A native C++ compiler is required to check shader embeddings')
    if not headers or len(set(symbols)) != len(symbols):
        raise ValueError('Expected distinct shader symbols and at least one header')
    for name in (*headers, *symbols):
        if not re.fullmatch(r'[A-Za-z_]\w*', name):
            raise ValueError('Invalid C++ test identifier')
    work.mkdir(parents=True, exist_ok=True)
    source = ['#include <iostream>', '#include <string>', '#include <cstring>']
    for namespace, header in headers.items():
        source += ['namespace ' + namespace + ' {', '#include ' + json.dumps(str(header.resolve())), '}']
    source += ['int main() {']
    for namespace in headers:
        for symbol in symbols:
            source += [f'{{ const char *value = {namespace}::wgsl_{symbol};',
                       'const auto size = std::strlen(value);',
                       'std::cout << size << "\\n";',
                       'std::cout.write(value, size); }']
    source += ['return std::cout.good() ? 0 : 1;', '}']
    cpp, executable = work / 'read.cpp', work / 'read'
    cpp.write_text('\n'.join(source) + '\n')
    subprocess.run([compiler, '-std=c++17', str(cpp), '-o', str(executable)], check=True, capture_output=True)
    raw = io.BytesIO(subprocess.check_output([str(executable)]))
    result = {}
    for namespace in headers:
        result[namespace] = {}
        for symbol in symbols:
            line = raw.readline()
            if not re.fullmatch(rb'[0-9]+\n', line):
                raise ValueError('Truncated shader emission length')
            length = int(line)
            shader = raw.read(length)
            if len(shader) != length:
                raise ValueError('Truncated shader emission')
            result[namespace][symbol] = shader.decode('utf-8')
    if raw.read(1):
        raise ValueError('Unexpected trailing shader emission')
    return result
