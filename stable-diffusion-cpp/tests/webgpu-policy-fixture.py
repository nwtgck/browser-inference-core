"""Extract the pinned WebGPU admission predicate for native policy tests.

This replays capabilities/shape decisions only: no GPU, driver allocation or
shader execution is emulated. An upstream signature change must fail generation.
"""
from pathlib import Path
import sys, re, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec=importlib.util.spec_from_file_location('probe',Path(sys.path[0])/'webgpu-wait-probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.source.parent
s=(root/'ggml-webgpu.cpp').read_text(); h=(root/'ggml-webgpu-shader-lib.hpp').read_text()
out='''// Generated from pinned prepared WebGPU supports_op and pure policy helpers.
// This is a source-policy replay, NOT a GPU backend, allocation or execution.
#include "webgpu-policy-fixture.h"
#include <algorithm>
#include <memory>
#define WEBGPU_LOG_DEBUG(...) ((void)0)
#define WEBGPU_STORAGE_BUF_BINDING_MULT 4
#define ROUNDUP_POW2(x, y) (((x) + (y) - 1) & ~((y) - 1))
struct Limits { size_t maxStorageBufferBindingSize = 0, minStorageBufferOffsetAlignment = 256,
maxComputeWorkgroupStorageSize = 0; uint32_t maxComputeWorkgroupsPerDimension = 65535, maxComputeInvocationsPerWorkgroup = 256; };
struct Capabilities { Limits limits; bool supports_subgroups = true, supports_subgroup_matrix = false;
uint32_t sg_mat_k = 8, sg_mat_m = 8, sg_mat_n = 8, max_subgroup_size = 32; };
struct Global { Capabilities capabilities; };
using webgpu_global_context = std::shared_ptr<Global>;
struct ggml_backend_webgpu_device_context { webgpu_global_context webgpu_global_ctx; };
static void* const webgpu_ptr_base = (void*)(uintptr_t)0x1000;
'''
for name in ('GGML_WEBGPU_FLASH_ATTN_TILE_KV_VEC_WIDTH','GGML_WEBGPU_FLASH_ATTN_TILE_Q_TILE','GGML_WEBGPU_KV_SEQ_PAD','GGML_WEBGPU_F16_SIZE_BYTES','GGML_WEBGPU_F32_SIZE_BYTES'):
    line=next(l for l in h.splitlines() if (l.startswith('#define '+name+' ') or l.startswith('inline constexpr') and name+' ' in l))
    out+=line+'\n'
for pre in ['static size_t ggml_webgpu_tensor_offset(', 'static size_t ggml_webgpu_tensor_align_offset(const ggml_tensor * t', 'static size_t ggml_webgpu_tensor_misalignment(const ggml_tensor * t', 'static size_t ggml_webgpu_tensor_binding_size(const ggml_tensor * t','static bool ggml_webgpu_tensor_binding_overlap(']:
    out+=m.function(s,pre)+'\n'
for pre in ['inline size_t ggml_webgpu_flash_attn_tensor_offset(', 'inline bool ggml_webgpu_flash_attn_float_vec4_aligned(const ggml_tensor * K, size_t', 'inline bool ggml_webgpu_flash_attn_k_direct(', 'inline bool ggml_webgpu_flash_attn_v_direct(', 'inline size_t ggml_webgpu_flash_attn_wg_mem_bytes(', 'inline uint32_t ggml_webgpu_flash_attn_max_kv_tile(', 'inline bool ggml_webgpu_flash_attn_can_use_subgroup_matrix_path(']:
    out+=m.function(h,pre)+'\n'
# This function's name is pinned with the rest of the source.
prefix=next(line.split('{')[0].rstrip() for line in s.splitlines() if line.startswith('static bool ggml_webgpu_supported_qtype('))
out+=m.function(s,prefix)+'\n'+m.function(s,'static bool ggml_backend_webgpu_device_supports_op(')+'\n'
out+='''bool webgpu_test_supports(const ggml_tensor* op, const WebgpuTestLimits& limits) {
    auto global = std::make_shared<Global>();
    global->capabilities.limits.maxStorageBufferBindingSize = limits.storage;
    global->capabilities.limits.maxComputeWorkgroupStorageSize = limits.shared;
    global->capabilities.supports_subgroups = limits.subgroups;
    global->capabilities.supports_subgroup_matrix = limits.subgroup_matrix;
    ggml_backend_webgpu_device_context ctx{global};
    ggml_backend_device dev{}; dev.context = &ctx;
    return ggml_backend_webgpu_device_supports_op(&dev, op);
}
'''
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(out)
