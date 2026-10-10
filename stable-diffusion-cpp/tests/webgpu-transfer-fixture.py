"""Extract prepared copy callbacks for byte/queue/ownership API contract tests.

The generated C++ links the real RunnerCache implementation. WGPU handles are
API doubles; neither this test nor its counters represent GPU measurements.
"""
import argparse
import importlib.util
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
tests = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('probe', tests / 'webgpu-wait-probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
source = args.source.read_text()
for field, function in [('cpy_tensor_async', 'ggml_backend_webgpu_cpy_tensor_async'),
                        ('cpy_tensor', 'ggml_backend_webgpu_buffer_cpy_tensor')]:
    if not re.search(r'/\* \.?' + field + r'\s*= \*/\s*' + function, source):
        raise ValueError('Missing live WebGPU vtable entry: ' + field)
functions = '\n'.join(probe.function(source, prefix) for prefix in [
    'static size_t ggml_webgpu_tensor_offset(',
    'static void * ggml_backend_webgpu_buffer_get_base(',
    '// Never stage a same-device aligned copy',
    'static bool ggml_backend_webgpu_buffer_cpy_tensor(',
    'static bool ggml_backend_webgpu_cpy_tensor_async(',
])
template = (tests / 'webgpu-transfer-fixture.cpp.in').read_text()
if template.count('@COPY_FUNCTIONS@') != 1:
    raise ValueError('Expected one source insertion point')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(template.replace('@COPY_FUNCTIONS@', functions))
