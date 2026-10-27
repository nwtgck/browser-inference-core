"""Compose real snapshot/scheduler bodies with the existing deferred CPU queue.

Only registry/event/allocation bookkeeping is modeled. No GPU, runtime build or
copy of the production decision logic is needed to exercise the new contract.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
from test_webgpu_async_tensor_copy import function


def block(text: str, token: str) -> str:
    start = text.index(token)
    brace = text.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end] + '\n'


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError('Inherited async harness needs review: ' + old)
    return text.replace(old, new)


def emit(upstream: Path, prepared: Path) -> str:
    actual = (prepared / 'ggml-webgpu.cpp').read_text()
    generic = (prepared / 'ggml-backend.cpp').read_text()
    cpu = (upstream / 'ggml/src/ggml-cpu/ggml-cpu.cpp').read_text()
    code = (ROOT / 'tests/webgpu-async-tensor-copy/mock-harness.cpp').read_text()
    code = code[:code.index('int main(){')]
    contract = (prepared / 'ggml-backend-host-snapshot.h').read_text()
    contract = contract.replace('#pragma once\n', '').replace('#include "ggml-backend.h"\n', '')
    code = replace_once(code, 'struct ggml_backend;using ggml_backend_t=ggml_backend*;',
        'struct ggml_backend;using ggml_backend_t=ggml_backend*;\n'
        'struct ggml_backend_reg;using ggml_backend_reg_t=ggml_backend_reg*;\n' + contract)
    code = replace_once(code, 'GGML_BACKEND_DEVICE_TYPE_CPU,GGML_BACKEND_DEVICE_TYPE_GPU',
        'GGML_BACKEND_DEVICE_TYPE_CPU,GGML_BACKEND_DEVICE_TYPE_GPU,GGML_BACKEND_DEVICE_TYPE_ACCEL')
    code = replace_once(code, 'device_iface iface;enum ggml_backend_dev_type type;',
        'device_iface iface;enum ggml_backend_dev_type type;ggml_backend_reg_t reg=nullptr;')
    code = replace_once(code,
        'struct ggml_backend_sched_split { int backend_id=0;ggml_cgraph graph; };',
        'struct ggml_backend_sched_split { int backend_id=0;ggml_cgraph graph;int n_inputs=0;ggml_tensor* inputs[4]{}; };')
    # Observe retained host snapshots in the existing queue model. Counts are
    # independent of the scheduler guard and decrease only when work executes.
    code = replace_once(code, 'std::vector<std::function<void()>> pending;',
        'std::vector<std::function<void()>> pending;size_t pending_snapshot_bytes=0,peak_snapshot_bytes=0;')
    code = replace_once(code,
        'pending.push_back([=]{std::copy(snapshot.begin(),snapshot.end(),b.p->bytes.begin()+offset);});',
        'pending_snapshot_bytes+=n;peak_snapshot_bytes=std::max(peak_snapshot_bytes,pending_snapshot_bytes);\n'
        '        pending.push_back([=]{std::copy(snapshot.begin(),snapshot.end(),b.p->bytes.begin()+offset);pending_snapshot_bytes-=n;});')
    code = replace_once(code, 'ggml_backend_t backends[1]{};ggml_backend_event_t events[1][1]{};int cur_copy=0;',
        'ggml_backend_t backends[8]{};ggml_backend_event_t events[8][4]{};int cur_copy=0;\n'
        '    int n_copies=1;ggml_backend_copy_host_input_snapshot_v1_t host_input_snapshot[8]{};\n'
        '    ggml_backend_buffer_type_t bufts[8]{};')
    code = replace_once(code,
        'bool host_copy(){return ggml_backend_webgpu_cpy_tensor_async(&cpu,&gpu,&src,&dst);}',
        '''bool host_copy(){
            const auto before_storage=b.buffer.p;
            const auto before_bytes=before_storage?before_storage->bytes:std::vector<uint8_t>{};
            const auto pending=g->queue.pending.size();
            const int waits=g->queue.waits,writes=g->queue.writes,submits=g->queue.submissions,cpu_waits=cpu_state.waits;
            const bool accepted=ggml_backend_webgpu_copy_host_input_snapshot_v1(&cpu,&gpu,&src,&dst);
            if(!accepted){
                CHECK(g->queue.pending.size()==pending&&g->queue.waits==waits&&g->queue.writes==writes&&g->queue.submissions==submits);
                CHECK(cpu_state.waits==cpu_waits&&b.buffer.p==before_storage);
                CHECK(!before_storage||before_storage->bytes==before_bytes);
            }
            return accepted;
        }''')
    # The existing deferred read stub is the private buffer accessor recognized
    # by the real async readback helper. It still performs no GPU emulation.
    code = code.replace('gpu_get', 'ggml_backend_webgpu_buffer_get_tensor')
    here = Path(__file__).parent
    code = replace_once(code, '// @ACTUAL_GENERIC_ASYNC_COPY@',
        '// @ACTUAL_GENERIC_ASYNC_COPY@\n' + (here / 'registry-support.cpp').read_text())
    code = replace_once(code, '// @ACTUAL_SCHEDULER_COPY@',
        '// @ACTUAL_SCHEDULER_COPY@\n' + (here / 'discovery-support.cpp').read_text())
    groups = {
        'ACTUAL_OFFSET': (actual, ['ggml_webgpu_tensor_offset']),
        'ACTUAL_BUFFER_BASE': (actual, ['ggml_backend_webgpu_buffer_get_base']),
        'ACTUAL_GENERIC_BASE': (generic, ['ggml_backend_buffer_get_base']),
        'ACTUAL_DEVICE_ACCESS': (generic, ['ggml_backend_get_device', 'ggml_backend_dev_type']),
        'ACTUAL_WEBGPU_SYNCHRONIZE': (actual, ['ggml_backend_webgpu_synchronize']),
        'ACTUAL_SYNCHRONIZE': (generic, ['ggml_backend_synchronize']),
        'ACTUAL_WEBGPU_SET': (actual, ['ggml_backend_webgpu_buffer_set_tensor']),
        'ACTUAL_GENERIC_SET_GET': (generic, ['ggml_backend_tensor_set', 'ggml_backend_tensor_get']),
        'ACTUAL_BUFFER_COPY': (actual, ['ggml_backend_webgpu_buffer_cpy_tensor']),
        'ACTUAL_GENERIC_BUFFER_COPY': (generic, ['ggml_backend_buffer_copy_tensor']),
        'ACTUAL_HOOK': (actual, ['ggml_backend_webgpu_cpy_tensor_async']),
        'ACTUAL_GENERIC_COPY': (generic, ['ggml_backend_tensor_copy']),
        'ACTUAL_GENERIC_ASYNC_COPY': (generic, ['ggml_backend_tensor_copy_async']),
        'ACTUAL_GENERIC_USAGE': (generic, ['ggml_backend_buffer_get_usage']),
        'ACTUAL_HOST_WEIGHT': (generic, ['ggml_backend_sched_is_host_weight']),
        'ACTUAL_SCHEDULER_COPY': (generic, ['ggml_backend_sched_copy_input']),
        'ACTUAL_CPU_REGISTRY': (cpu, ['ggml_backend_cpu_get_proc_address']),
        'ACTUAL_SNAPSHOT_PROVIDER': (actual, ['ggml_backend_webgpu_copy_host_input_snapshot_v1']),
        'ACTUAL_WEBGPU_REGISTRY': (actual, ['ggml_backend_webgpu_reg_get_proc_address']),
        'ACTUAL_GET_ASYNC': (actual, ['ggml_backend_webgpu_get_tensor_async']),
    }
    for marker, (text, names) in groups.items():
        code = replace_once(code, '// @' + marker + '@', '\n'.join(function(text, name) for name in names))
    registry = block(generic, 'ggml_backend_reg_t ggml_backend_dev_backend_reg(')
    registry += function(generic, 'ggml_backend_reg_get_proc_address')
    code = replace_once(code, '// @ACTUAL_REGISTRY_ACCESS@', registry)
    constructor = generic[generic.index('ggml_backend_sched_t ggml_backend_sched_new('):]
    discovery = block(constructor, '    for (int b = 0; b < n_backends; b++) {')
    code = replace_once(code, '// @ACTUAL_CONSTRUCTOR_DISCOVERY@', discovery)
    registration = re.search(r'/\* \.cpy_tensor_async\s+= \*/\s*(\w+)\s*,', actual)
    if not registration:
        raise ValueError('Missing registered async hook')
    code = replace_once(code, '// @REGISTERED_HOOK@', registration.group(1) + ',')
    if '// @' in code:
        raise ValueError('Unexpanded production helper placeholder')
    return code + '\n' + (here / 'cases.cpp').read_text()
