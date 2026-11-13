static unsigned checks = 0, calls = 0;
static const ggml_webgpu_q8_diagnostic_request_v1 * caller_request = nullptr;
static const char * registry_name = "WebGPU";
static enum ggml_backend_dev_type device_type = GGML_BACKEND_DEVICE_TYPE_GPU;
static const char * test_reg_name(ggml_backend_reg_t) {
    return registry_name;
}
static enum ggml_backend_dev_type test_dev_type(ggml_backend_dev_t) {
    return device_type;
}
static void check(bool value) {
    ++checks;
    if (!value)
        throw std::runtime_error("bridge check " + std::to_string(checks));
}
static bool ggml_backend_webgpu_copy_host_input_snapshot_v1(ggml_backend_t, ggml_backend_t, const ggml_tensor *,
                                                            ggml_tensor *) {
    return false;
}
static int32_t ggml_backend_webgpu_q8_diagnostic_v1(ggml_backend_t, ggml_tensor *,
                                                    const ggml_webgpu_q8_diagnostic_request_v1 * request,
                                                    ggml_webgpu_q8_diagnostic_report_v1 * out, uint32_t capacity) {
    ++calls;
    check(request != caller_request && capacity >= sizeof(*out));
    const uint32_t variant = request->variant;
    const_cast<ggml_webgpu_q8_diagnostic_request_v1 *>(caller_request)->variant = 99;
    check(request->variant == variant);
    *out = {};
    out->abi_version = 1;
    out->struct_size = sizeof(*out);
    out->requested_variant = variant;
    return 0;
}
// @ACTUAL_REGISTRY@
#include "webgpu-q8-diagnostic.cpp"
extern "C" uint64_t lcb_webgpu_q8_diagnostic_schema_hash();
extern "C" uint64_t lcb_webgpu_q8_diagnostic_sizeof_record(uint32_t);
extern "C" uint64_t lcb_webgpu_q8_diagnostic_alignof_record(uint32_t);
extern "C" uint64_t lcb_webgpu_q8_diagnostic_offsetof_field(uint32_t, uint32_t);
extern "C" uint64_t lcb_webgpu_q8_diagnostic_sizeof_field(uint32_t, uint32_t);
int main(int argc, char ** argv) {
    try {
        check(argc == 2);
        check(std::strcmp(reinterpret_cast<const char *>(uintptr_t(lcb_webgpu_q8_diagnostic_schema_hash())), argv[1]) ==
              0);
        check(lcb_webgpu_q8_diagnostic_sizeof_record(UINT32_MAX) == UINT64_MAX);
        check(lcb_webgpu_q8_diagnostic_offsetof_field(0, UINT32_MAX) == UINT64_MAX);
        // @GENERATED_LAYOUT_CHECKS@
        ggml_backend backend{};
        ggml_backend_device device{};
        ggml_backend_reg reg{};
        ggml_tensor tensor{};
        const auto b = uint64_t(uintptr_t(&backend)), d = uint64_t(uintptr_t(&tensor));
        check(lcb_webgpu_q8_diagnostic_version(0) == 0);
        check(lcb_webgpu_q8_diagnostic_version(b) == 0);
        backend.device = &device;
        check(lcb_webgpu_q8_diagnostic_version(b) == 0);
        device.iface.get_type = test_dev_type;
        reg.iface.get_name = test_reg_name;
        device.reg = &reg;
        check(lcb_webgpu_q8_diagnostic_version(b) == 0);
        reg.iface.get_proc_address = ggml_backend_webgpu_reg_get_proc_address;
        check(lcb_webgpu_q8_diagnostic_version(b) == 1);
        device_type = GGML_BACKEND_DEVICE_TYPE_CPU;
        check(lcb_webgpu_q8_diagnostic_version(b) == 0);
        device_type = GGML_BACKEND_DEVICE_TYPE_GPU;
        registry_name = "CPU";
        check(lcb_webgpu_q8_diagnostic_version(b) == 0);
        registry_name = "WebGPU";
        check(ggml_backend_webgpu_reg_get_proc_address(&reg, "unknown_v1") == nullptr);
        check(ggml_backend_webgpu_reg_get_proc_address(&reg, nullptr) == nullptr);
        check(ggml_backend_webgpu_reg_get_proc_address(&reg, GGML_BACKEND_HOST_INPUT_SNAPSHOT_V1) != nullptr);
        ggml_webgpu_q8_diagnostic_request_v1 request{1, sizeof(request), 2, 1, 0, 0};
        caller_request = &request;
        ggml_webgpu_q8_diagnostic_report_v1 report;
        std::memset(&report, 0xa5, sizeof(report));
        const auto before = report;
        const auto in = uint64_t(uintptr_t(&request)), out = uint64_t(uintptr_t(&report));
        check(lcb_webgpu_q8_diagnostic_run_v1(b, d, in, out, sizeof(report) - 1) == 2 && calls == 0 &&
              std::memcmp(&before, &report, sizeof(report)) == 0);
        check(lcb_webgpu_q8_diagnostic_run_v1(b, d, in + 1, out, sizeof(report)) == 2 && calls == 0);
        check(lcb_webgpu_q8_diagnostic_run_v1(b, d, in, out + 1, sizeof(report)) == 2 && calls == 0);
        check(lcb_webgpu_q8_diagnostic_run_v1(b, d, 0, out, sizeof(report)) == 2 && calls == 0);
        check(lcb_webgpu_q8_diagnostic_run_v1(0, d, in, out, sizeof(report)) == 1 && report.status == 1 && calls == 0);
        check(lcb_webgpu_q8_diagnostic_run_v1(b, d, in, out, sizeof(report)) == 0 && calls == 1 &&
              report.requested_variant == 2 && request.variant == 99);
        reg.iface.get_proc_address = nullptr;
        request.variant = 0;
        check(lcb_webgpu_q8_diagnostic_run_v1(b, d, in, out, sizeof(report)) == 1 && report.status == 1 && calls == 1);
        std::cout << checks << " typed registry/bridge/capacity/copy/compiled-schema/layout checks\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
