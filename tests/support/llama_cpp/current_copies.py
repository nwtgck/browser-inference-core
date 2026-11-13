"""Current snapshot/scheduler functions executed with an owned deferred queue.

Source selections are typed by Clang; only resource/queue bookkeeping is modeled.
This is not backend, WebGPU device, Emscripten, or browser execution.
"""
import json
import os
from pathlib import Path
import subprocess

from build_tools.project import runtime_test_root
from build_tools.runtimes.llama_cpp import webgpu_plan
from build_tools.toolchain.dawn import verify
from tests.support.llama_cpp.copy_ast import Declarations
from tests.support.llama_cpp.current_webgpu import dawn_package
from tests.support.llama_cpp.dispatch_ast import macro_definition

HERE = runtime_test_root('llama-cpp') / 'webgpu-copy-recording'
SNAPSHOT = runtime_test_root('llama-cpp') / 'webgpu-host-input-snapshot'

# Markers are owned model slots, not search strings in upstream C++.
GROUPS = {
    'ACTUAL_OFFSET': ('provider', ['ggml_webgpu_tensor_offset']),
    'ACTUAL_BUFFER_BASE': ('provider', ['ggml_backend_webgpu_buffer_get_base']),
    'ACTUAL_GENERIC_BASE': ('backend', ['ggml_backend_buffer_get_base']),
    'ACTUAL_DEVICE_ACCESS': ('backend', ['ggml_backend_get_device', 'ggml_backend_dev_type']),
    'ACTUAL_WEBGPU_SYNCHRONIZE': ('provider', ['ggml_backend_webgpu_synchronize']),
    'ACTUAL_SYNCHRONIZE': ('backend', ['ggml_backend_synchronize']),
    'ACTUAL_WEBGPU_SET': ('provider', ['ggml_backend_webgpu_buffer_set_tensor']),
    'ACTUAL_GENERIC_SET_GET': ('backend', ['ggml_backend_tensor_set', 'ggml_backend_tensor_get']),
    'ACTUAL_BUFFER_COPY': ('provider', ['ggml_backend_webgpu_buffer_cpy_tensor']),
    'ACTUAL_GENERIC_BUFFER_COPY': ('backend', ['ggml_backend_buffer_copy_tensor']),
    'ACTUAL_HOOK': ('provider', ['ggml_backend_webgpu_cpy_tensor_async']),
    'ACTUAL_GENERIC_COPY': ('backend', ['ggml_backend_tensor_copy']),
    'ACTUAL_GENERIC_ASYNC_COPY': ('backend', ['ggml_backend_tensor_copy_async']),
    'ACTUAL_GENERIC_USAGE': ('backend', ['ggml_backend_buffer_get_usage']),
    'ACTUAL_HOST_WEIGHT': ('backend', ['ggml_backend_sched_is_host_weight']),
    'ACTUAL_SCHEDULER_COPY': ('backend', ['ggml_backend_sched_copy_input']),
    'ACTUAL_CPU_REGISTRY': ('cpu', ['ggml_backend_cpu_get_proc_address']),
    'ACTUAL_SNAPSHOT_PROVIDER': ('provider', ['ggml_backend_webgpu_copy_host_input_snapshot_v1']),
    'ACTUAL_WEBGPU_REGISTRY': ('provider', ['ggml_backend_webgpu_reg_get_proc_address']),
    'ACTUAL_GET_ASYNC': ('provider', ['ggml_backend_webgpu_get_tensor_async']),
    'ACTUAL_REGISTRY_ACCESS': ('backend', ['ggml_backend_dev_backend_reg', 'ggml_backend_reg_get_proc_address']),
}


class CurrentCopies:
    def __init__(self, upstream: Path, prepared: Path, *, batch: bool):
        upstream, prepared = upstream.resolve(), prepared.resolve()
        sdk = dawn_package()
        verify(sdk)
        receipt = json.loads((prepared / webgpu_plan.RECEIPT).read_text())
        self.macros = ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'] if batch else []
        if receipt['gpuProfile']:
            self.macros += ['-DGGML_WEBGPU_GPU_PROFILE', '-DGGML_WEBGPU_CPU_PROFILE']
        self.receipt = receipt
        contract = prepared / 'ggml-backend-host-snapshot.h'
        interface = upstream / 'ggml/src/ggml-backend-impl.h'
        diagnostic = prepared / 'ggml-webgpu-q8-diagnostic.h'
        headers = (interface, contract, diagnostic) if receipt['gpuProfile'] else (interface, contract)
        includes = [prepared, upstream / 'ggml/include', upstream / 'ggml/src',
                    upstream / webgpu_plan.BASE, upstream / 'ggml/src/ggml-cpu']
        common = ['-std=c++20', '-pthread', '-DGGML_SCHED_MAX_COPIES=4']
        common += ['-I' + str(p) for p in includes]
        provider_args = common + self.macros + ['-D__wasi__', '-isystem', str(sdk / 'webgpu/include'),
                                                '-isystem', str(sdk / 'webgpu_cpp/include')]
        requests = {kind: {name for group, names in GROUPS.values() if group == kind for name in names}
                    for kind in ('provider', 'backend', 'cpu')}
        requests['provider'].update({'ggml_backend_webgpu_i', 'ggml_backend_i',
                                     'ggml_backend_webgpu_reg_i', 'ggml_backend_reg_i'})
        if receipt['gpuProfile']:
            requests['provider'].add('ggml_backend_webgpu_q8_diagnostic_v1_t')
        requests['backend'].update({'ggml_backend_sched_new', 'ggml_backend_sched',
                                    'ggml_backend_copy_host_input_snapshot_v1_t'})
        self.units = {
            'provider': Declarations(prepared / 'ggml-webgpu.cpp', provider_args, requests['provider'],
                                     headers=headers),
            'backend': Declarations(prepared / 'ggml-backend.cpp', common, requests['backend'],
                                    headers=(interface, contract)),
            'cpu': Declarations(upstream / 'ggml/src/ggml-cpu/ggml-cpu.cpp', common, requests['cpu']),
        }
        values = {marker: '\n'.join(self.units[kind].function(name) for name in names)
                  for marker, (kind, names) in GROUPS.items()}
        provider = self.units['provider']
        registered = provider.registered_function('ggml_backend_webgpu_i', 'ggml_backend_i',
                                                  'cpy_tensor_async', interface)
        if registered != 'ggml_backend_webgpu_cpy_tensor_async':
            raise ValueError('Unexpected current async-copy registration: ' + registered)
        registry = provider.registered_function('ggml_backend_webgpu_reg_i', 'ggml_backend_reg_i',
                                                'get_proc_address', interface)
        if registry != 'ggml_backend_webgpu_reg_get_proc_address':
            raise ValueError('Unexpected current WebGPU registry: ' + registry)
        values['REGISTERED_HOOK'] = registered + ','
        values['ACTUAL_DIAGNOSTIC_CONTRACT'] = ''
        if receipt['gpuProfile']:
            typedef = provider.definition('ggml_backend_webgpu_q8_diagnostic_v1_t', 'TypedefDecl',
                                          owner=diagnostic)
            values['ACTUAL_DIAGNOSTIC_CONTRACT'] = (
                provider.text(typedef, owner=diagnostic) + ';\n'
                + macro_definition(prepared / 'ggml-webgpu.cpp', 'GGML_BACKEND_WEBGPU_Q8_DIAGNOSTIC_V1',
                                   provider_args))
        backend = self.units['backend']
        values['ACTUAL_CONSTRUCTOR_DISCOVERY'] = backend.constructor_discovery()
        typedef = backend.definition('ggml_backend_copy_host_input_snapshot_v1_t', 'TypedefDecl', owner=contract)
        values['ACTUAL_SNAPSHOT_CONTRACT'] = (
            backend.text(typedef, owner=contract) + ';\n'
            + macro_definition(prepared / 'ggml-backend.cpp', 'GGML_BACKEND_HOST_INPUT_SNAPSHOT_V1', common))
        code = (HERE / 'model.cpp.in').read_text()
        for marker, value in values.items():
            slot = '// @' + marker + '@'
            if code.count(slot) != 1:
                raise ValueError('Expected one owned copy-model slot: ' + marker)
            code = code.replace(slot, value)
        if '// @' in code:
            raise ValueError('Unexpanded copy-model slot')
        self.code = code
        self.assert_unchanged()

    def assert_unchanged(self):
        for unit in self.units.values():
            unit.assert_unchanged()

    def build(self, output: Path, *, snapshot: bool) -> Path:
        self.assert_unchanged()
        output.mkdir(parents=True, exist_ok=True)
        main = SNAPSHOT / 'cases.cpp' if snapshot else HERE / 'ordinary-main.cpp'
        unit = output / 'copy.cpp'
        unit.write_text(self.code + '\n#include "' + str(main) + '"\n')
        binary = output / 'copy'
        command = [os.environ.get('CXX', 'c++'), '-std=c++20', '-Wall', '-Wextra', '-Werror', '-O0',
                   *self.macros, *(['-DBICORE_RECORD_SNAPSHOTS'] if snapshot else []),
                   str(unit), '-o', str(binary)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        except subprocess.CalledProcessError as error:
            error.add_note(error.stderr)
            raise
        self.assert_unchanged()
        return binary
