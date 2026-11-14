#pragma once

namespace q8_diagnostic_cases {
using Request = ggml_webgpu_q8_diagnostic_request_v1;
using Report = ggml_webgpu_q8_diagnostic_report_v1;

static uint64_t address(const void * value) {
    return reinterpret_cast<uintptr_t>(value);
}

static void algorithm_vectors() {
    // Algorithm outputs, not acceptance hashes for source or shader files.
    const auto check = [](const std::string & input, const char * expected) {
        for (const size_t chunk : {input.size() + 1, size_t(7)}) {
            ggml_webgpu_q8_digest::sha256_ctx state;
            ggml_webgpu_q8_digest::sha256_init(state);
            for (size_t offset = 0; offset < input.size(); offset += chunk)
                ggml_webgpu_q8_digest::sha256_update(state, input.data() + offset,
                                                     std::min(chunk, input.size() - offset));
            std::array<uint8_t, 32> digest;
            ggml_webgpu_q8_digest::sha256_final(state, digest.data());
            std::string hex;
            for (const auto byte : digest) {
                hex += "0123456789abcdef"[byte >> 4];
                hex += "0123456789abcdef"[byte & 15];
            }
            BICORE_CHECK(hex == expected);
        }
    };
    check("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    for (const auto [size, expected] :
         {std::pair{55u, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
          {56u, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
          {64u, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
          {129u, "c12cb024a2e5551cca0e08fce8f1c5e314555cc3fef6329ee994a3db752166ae"}})
        check(std::string(size, 'a'), expected);
}

static void boundary() {
    transfer_cases::Backend gpu(transfer_cases::BackendKind::webgpu), cpu(transfer_cases::BackendKind::host);
    gpu.backend.guid = ggml_backend_webgpu_guid();
    gpu.registry.iface.get_name = [](ggml_backend_reg_t) { return "WebGPU"; };
    const auto memory = transfer_cases::Memory(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    BICORE_CHECK(memory);
    auto * a = ggml_new_tensor_2d(memory.get(), GGML_TYPE_Q8_0, 64, 16);
    auto * x = ggml_new_tensor_1d(memory.get(), GGML_TYPE_F32, 64);
    auto * dst = ggml_mul_mat(memory.get(), a, x);
    std::vector<transfer_cases::Buffer> buffers;
    for (auto * tensor : {a, x, dst}) {
        const auto size = ggml_backend_buft_get_alloc_size(&gpu.type, tensor);
        buffers.emplace_back(ggml_backend_buft_alloc_buffer(&gpu.type, size), ggml_backend_buffer_free);
        BICORE_CHECK(buffers.back());
        BICORE_CHECK(ggml_backend_tensor_alloc(buffers.back().get(), tensor,
                                               ggml_backend_buffer_get_base(buffers.back().get())) ==
                     GGML_STATUS_SUCCESS);
    }
    Request request{
        1, sizeof(Request), GGML_WEBGPU_Q8_DIAGNOSTIC_COOPERATIVE_WG256_R4_V1, GGML_WEBGPU_Q8_DIAGNOSTIC_EXECUTE_V1, 3,
        0};
    Report report{};
    const auto run = [&] {
        return lcb_webgpu_q8_diagnostic_run_v1(address(&gpu.backend), address(dst), address(&request), address(&report),
                                               sizeof(report));
    };
    BICORE_CHECK(lcb_webgpu_q8_diagnostic_version(address(&gpu.backend)) == 1);
    const auto cold = created;
    commands = {};
    readback = {};
    BICORE_CHECK(run() == GGML_WEBGPU_Q8_DIAGNOSTIC_NOT_PREPARED_V1 && created == cold && commands.dispatches.empty());
    std::array<uint8_t, sizeof(Report)> sentinel;
    sentinel.fill(0xa5);
    for (const bool bridge : {false, true}) {
        std::memcpy(&report, sentinel.data(), sentinel.size());
        const auto status =
            bridge ? lcb_webgpu_q8_diagnostic_run_v1(address(&gpu.backend), address(dst), address(&request),
                                                     address(&report), sizeof(report) - 1)
                   : ggml_backend_webgpu_q8_diagnostic_v1(&gpu.backend, dst, &request, &report, sizeof(report) - 1);
        BICORE_CHECK(status == GGML_WEBGPU_Q8_DIAGNOSTIC_BAD_REQUEST_V1 &&
                     std::memcmp(&report, sentinel.data(), sentinel.size()) == 0);
    }
    request.phase = GGML_WEBGPU_Q8_DIAGNOSTIC_PREPARE_V1;
    request.repetitions = 0;
    BICORE_CHECK(run() == GGML_WEBGPU_Q8_DIAGNOSTIC_OK_V1 && commands.dispatches.empty());
    const auto prepared = report;
    const auto warm = created;
    BICORE_CHECK(warm == cold + 1 && run() == GGML_WEBGPU_Q8_DIAGNOSTIC_OK_V1 && created == warm);
    request.phase = GGML_WEBGPU_Q8_DIAGNOSTIC_EXECUTE_V1;
    request.repetitions = 3;
    unsigned reentries = 0;
    readback.before_map = [&] {
        Report nested{};
        BICORE_CHECK(ggml_backend_webgpu_q8_diagnostic_v1(&gpu.backend, dst, &request, &nested, sizeof(nested)) ==
                     GGML_WEBGPU_Q8_DIAGNOSTIC_BUSY_V1);
        ++reentries;
    };
    BICORE_CHECK(run() == GGML_WEBGPU_Q8_DIAGNOSTIC_OK_V1 && report.dispatches == request.repetitions &&
                 commands.dispatches.size() == request.repetitions && created == warm && reentries == 1);
    BICORE_CHECK(std::memcmp(prepared.shader_sha256, report.shader_sha256, sizeof(report.shader_sha256)) == 0);
    readback.before_map = {};
    BICORE_CHECK(run() == GGML_WEBGPU_Q8_DIAGNOSTIC_OK_V1 && created == warm);

    // Observe the actual bridge's typed provider boundary, before a provider
    // could suspend. Its request must be a copy, even if caller storage changes.
    static Request * caller;
    caller = &request;
    const auto actual_provider_lookup = gpu.registry.iface.get_proc_address;
    gpu.registry.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
        BICORE_CHECK(std::strcmp(name, GGML_BACKEND_WEBGPU_Q8_DIAGNOSTIC_V1) == 0);
        const ggml_backend_webgpu_q8_diagnostic_v1_t observed =
            [](ggml_backend_t backend, ggml_tensor * tensor, const Request * copy, Report * out, uint32_t capacity) {
                BICORE_CHECK(copy != caller);
                const auto expected = *caller;
                *caller = {};
                BICORE_CHECK(std::memcmp(copy, &expected, sizeof(expected)) == 0);
                return ggml_backend_webgpu_q8_diagnostic_v1(backend, tensor, copy, out, capacity);
            };
        return reinterpret_cast<void *>(observed);
    };
    BICORE_CHECK(run() == GGML_WEBGPU_Q8_DIAGNOSTIC_OK_V1 && report.dispatches == 3);
    gpu.registry.iface.get_proc_address = actual_provider_lookup;
    caller = nullptr;
    request = {
        1, sizeof(Request), GGML_WEBGPU_Q8_DIAGNOSTIC_COOPERATIVE_WG256_R4_V1, GGML_WEBGPU_Q8_DIAGNOSTIC_EXECUTE_V1, 3,
        0};
    const auto lookups = gpu.lookups;
    gpu.registry.iface.get_name = [](ggml_backend_reg_t) { return "foreign-gpu"; };
    for (auto * rejected : {&cpu.backend, &gpu.backend}) {
        BICORE_CHECK(lcb_webgpu_q8_diagnostic_version(address(rejected)) == 0);
        BICORE_CHECK(lcb_webgpu_q8_diagnostic_run_v1(address(rejected), address(dst), address(&request),
                                                     address(&report),
                                                     sizeof(report)) == GGML_WEBGPU_Q8_DIAGNOSTIC_UNAVAILABLE_V1);
        BICORE_CHECK(report.status == GGML_WEBGPU_Q8_DIAGNOSTIC_UNAVAILABLE_V1 && report.dispatches == 0);
    }
    BICORE_CHECK(gpu.lookups == lookups);
}

static void all() {
    algorithm_vectors();
    boundary();
}
} // namespace q8_diagnostic_cases
