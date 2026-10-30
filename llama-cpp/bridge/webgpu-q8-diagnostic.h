#ifndef GGML_WEBGPU_Q8_DIAGNOSTIC_V1_H
#define GGML_WEBGPU_Q8_DIAGNOSTIC_V1_H
// Optional GPU-profile-only contract. Records use generated layout queries.
#include "ggml-backend.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define GGML_BACKEND_WEBGPU_Q8_DIAGNOSTIC_V1 "ggml_backend_webgpu_q8_diagnostic_v1"
enum ggml_webgpu_q8_diagnostic_variant_v1 {
    GGML_WEBGPU_Q8_DIAGNOSTIC_AUTO_V1 = 0,
    GGML_WEBGPU_Q8_DIAGNOSTIC_COOPERATIVE_WG256_R4_V1 = 1,
    GGML_WEBGPU_Q8_DIAGNOSTIC_INDEPENDENT_WG128_R4_V1 = 2,
    GGML_WEBGPU_Q8_DIAGNOSTIC_INDEPENDENT_WG128_R1_V1 = 3,
    GGML_WEBGPU_Q8_DIAGNOSTIC_INDEPENDENT_WG128_R2_V1 = 4,
    GGML_WEBGPU_Q8_DIAGNOSTIC_COOPERATIVE_WG128_R2_V1 = 5
};
enum ggml_webgpu_q8_diagnostic_phase_v1 {
    GGML_WEBGPU_Q8_DIAGNOSTIC_PREPARE_V1 = 1,
    GGML_WEBGPU_Q8_DIAGNOSTIC_EXECUTE_V1 = 2
};
enum ggml_webgpu_q8_diagnostic_status_v1 {
    GGML_WEBGPU_Q8_DIAGNOSTIC_OK_V1 = 0,
    GGML_WEBGPU_Q8_DIAGNOSTIC_UNAVAILABLE_V1 = 1,
    GGML_WEBGPU_Q8_DIAGNOSTIC_BAD_REQUEST_V1 = 2,
    GGML_WEBGPU_Q8_DIAGNOSTIC_UNSUPPORTED_V1 = 3,
    GGML_WEBGPU_Q8_DIAGNOSTIC_NOT_PREPARED_V1 = 4,
    GGML_WEBGPU_Q8_DIAGNOSTIC_BUSY_V1 = 5
};
enum ggml_webgpu_q8_diagnostic_algorithm_v1 {
    GGML_WEBGPU_Q8_DIAGNOSTIC_ALGORITHM_NONE_V1 = 0,
    GGML_WEBGPU_Q8_DIAGNOSTIC_ALGORITHM_COOPERATIVE_V1 = 1,
    GGML_WEBGPU_Q8_DIAGNOSTIC_ALGORITHM_INDEPENDENT_V1 = 2
};
enum ggml_webgpu_q8_diagnostic_bound_v1 {
    GGML_WEBGPU_Q8_DIAGNOSTIC_VERSION_V1 = 1,
    GGML_WEBGPU_Q8_DIAGNOSTIC_K_MIN_V1 = 32,
    GGML_WEBGPU_Q8_DIAGNOSTIC_K_MAX_V1 = 16384,
    GGML_WEBGPU_Q8_DIAGNOSTIC_M_MIN_V1 = 1,
    GGML_WEBGPU_Q8_DIAGNOSTIC_M_MAX_V1 = 65536,
    GGML_WEBGPU_Q8_DIAGNOSTIC_WEIGHT_BYTES_MAX_V1 = 67108864,
    GGML_WEBGPU_Q8_DIAGNOSTIC_REPETITIONS_MAX_V1 = 128
};
struct ggml_webgpu_q8_diagnostic_request_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t variant;
    uint32_t phase;
    uint32_t repetitions;
    uint32_t reserved;
};
struct ggml_webgpu_q8_diagnostic_report_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t status;
    uint32_t requested_variant;
    uint32_t phase;
    uint32_t actual_algorithm;
    uint32_t workgroup_size;
    uint32_t logical_subgroup_size;
    uint32_t real_min_subgroup_size;
    uint32_t real_max_subgroup_size;
    uint32_t rows_per_tile;
    uint32_t outputs_per_workgroup;
    uint32_t cooperative_subgroup_reduction;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t dispatches;
    uint32_t codegen_revision;
    uint32_t shader_identity_id; // Reserved in v1; actual WGSL SHA-256 follows.
    uint32_t reserved;
    uint8_t shader_sha256[32];
};
// Borrowed pointers; copy request before the first suspension. No retained tensor
// pointers. Existing bind-group buffer refs follow their generation invalidation.
// report is pure output. An undersized capacity causes no writes to report.
typedef int32_t (*ggml_backend_webgpu_q8_diagnostic_v1_t)(ggml_backend_t backend, struct ggml_tensor * dst,
                                                          const struct ggml_webgpu_q8_diagnostic_request_v1 * request,
                                                          struct ggml_webgpu_q8_diagnostic_report_v1 * report,
                                                          uint32_t report_capacity);
// Fixed-width bridge surface. Only run is a JSPI export; PREPARE may suspend.
// Hosts must share one idle/busy gate with the ordinary native API, keep borrowed
// objects alive until completion, and use metadata leaves only while idle.
// Native guards cover same-context graph/diagnostic reentry, not raw frees or
// direct heap mutation. After unexpected rejection/device loss/timeout, poison
// and discard the runtime; do not individually free objects still in use.
uint32_t lcb_webgpu_q8_diagnostic_version(uint64_t backend);
int32_t lcb_webgpu_q8_diagnostic_run_v1(uint64_t backend, uint64_t dst, uint64_t request, uint64_t report,
                                        uint32_t report_capacity);
#ifdef __cplusplus
}
#endif
#endif
