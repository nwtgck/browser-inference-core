#!/usr/bin/env python3
"""Run prepared registry/discovery/device functions with owned API handles.

Tests browser/native compile branches, concurrent/repeated lookup, initial
failure and retained-buffer device identity. Not a Dawn or browser GPU test.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from importlib import import_module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
function = import_module('webgpu-wait-probe').function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cxx', required=True)
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    source = args.source.read_text()
    bodies = '\n'.join(function(source, prefix) for prefix in [
        'static void ggml_backend_webgpu_request_adapter(',
        'static void create_webgpu_device(',
        'static size_t ggml_backend_webgpu_reg_get_device_count(',
        'static ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(',
        'ggml_backend_reg_t ggml_backend_webgpu_reg()',
    ])
    root = Path(__file__).parent
    code = (root / 'prefix.cpp').read_text() + bodies + (root / 'cases.cpp').read_text()
    cases = ['normal','null-instance','null-adapter','no-f16','no-subgroups',
             'no-implicit-sync','device-failure','reordered','concurrent','held-buffers','device-lost',
             'adapter-wait-error','device-wait-error','missing-adapter-callback','missing-device-callback']
    with tempfile.TemporaryDirectory(prefix='sdc-registry-') as tmp:
        cpp = Path(tmp) / 'registry.cpp'; cpp.write_text(code)
        flags = ['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'] if args.sanitize else ['-O2']
        for mode in ['browser','native']:
            exe = Path(tmp) / mode
            subprocess.run([args.cxx,'-std=c++17','-pthread',*flags,*(['-D__EMSCRIPTEN__=1'] if mode=='browser' else []),str(cpp),'-o',str(exe)],check=True)
            for case in cases:
                subprocess.run([str(exe),case],check=True)

if __name__ == '__main__':
    main()
