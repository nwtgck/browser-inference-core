// Tiny self-authored GGUFs route through the production ModelLoader. The fake
// device is CPU memory with a nonhost WebGPU buffer contract; it runs no GPU API.
#include "ggml-backend-impl.h"
#include "model_io/browser_weight_reader.h"
#include "model_loader.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * reason) {
    if (!ok)
        throw std::runtime_error(reason);
}

struct Temporary {
    std::filesystem::path directory;
    Temporary() {
        auto prefix =
            "sd-bounded-reader-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        for (int i = 0; i < 100; ++i) {
            directory = std::filesystem::temp_directory_path() / (prefix + "-" + std::to_string(i));
            if (std::filesystem::create_directory(directory))
                return;
        }
        throw std::runtime_error("cannot create reader fixture directory");
    }
    ~Temporary() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }
};

static void integer(std::ostream & out, uint64_t value, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i)
        out.put(char(value >> (i * 8)));
}

static size_t fixture(const std::filesystem::path & path, ggml_type type, uint64_t elements,
                      const std::vector<uint8_t> & payload) {
    std::ofstream out(path, std::ios::binary);
    integer(out, 0x46554747, 4);
    integer(out, 3, 4);
    integer(out, 1, 8);
    integer(out, 0, 8);
    integer(out, 6, 8);
    out.write("weight", 6);
    integer(out, 1, 4);
    integer(out, elements, 8);
    integer(out, type, 4);
    integer(out, 0, 8);
    const size_t start = (size_t(out.tellp()) + 31) / 32 * 32;
    while (size_t(out.tellp()) < start)
        out.put(0);
    out.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    check(bool(out), "cannot write GGUF fixture");
    return start;
}

struct Storage {
    std::vector<uint8_t> bytes;
    std::vector<size_t> writes;
    std::filesystem::path truncate_path;
    size_t truncate_bytes = 0;
    bool throw_on_second = false;
};

struct FakeDevice {
    ggml_backend_reg registry{};
    ggml_backend_device device{};
    ggml_backend_buffer_type type{};
    bool host = false;
    const char * registry_name = "WebGPU";

    explicit FakeDevice(bool host = false, const char * name = "WebGPU") : host(host), registry_name(name) {
        registry.api_version = GGML_BACKEND_API_VERSION;
        registry.context = this;
        registry.iface.get_name = [](ggml_backend_reg_t reg) {
            return static_cast<FakeDevice *>(reg->context)->registry_name;
        };
        device.reg = &registry;
        type.device = &device;
        type.context = this;
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "CPU fake upload buffer"; };
        type.iface.get_alignment = [](ggml_backend_buffer_type_t) { return size_t(4); };
        type.iface.is_host = [](ggml_backend_buffer_type_t type) {
            return static_cast<FakeDevice *>(type->context)->host;
        };
    }

    ggml_backend_buffer_t buffer(Storage & storage, size_t size) {
        storage.bytes.resize(size);
        ggml_backend_buffer_i iface{};
        iface.get_base = [](ggml_backend_buffer_t buffer) {
            return static_cast<void *>(static_cast<Storage *>(buffer->context)->bytes.data());
        };
        iface.set_tensor = [](ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                              size_t size) {
            auto & storage = *static_cast<Storage *>(buffer->context);
            const auto base =
                reinterpret_cast<uintptr_t>(tensor->data) - reinterpret_cast<uintptr_t>(storage.bytes.data());
            check(base <= storage.bytes.size() && offset <= storage.bytes.size() - base &&
                      size <= storage.bytes.size() - base - offset,
                  "fake backend tensor range");
            storage.writes.push_back(size);
            if (storage.throw_on_second && storage.writes.size() == 2)
                throw std::runtime_error("injected upload failure");
            std::memcpy(storage.bytes.data() + base + offset, data, size);
            if (!storage.truncate_path.empty() && storage.writes.size() == 1)
                std::filesystem::resize_file(storage.truncate_path, storage.truncate_bytes);
        };
        return ggml_backend_buffer_init(&type, iface, &storage, size);
    }
};

static void load_bf16(ggml_type destination_type, bool host, bool other_registry, int failure) {
    Temporary temporary;
    const auto path = temporary.directory / "weight.gguf";
    const size_t elements = sd::browser_weights::Reader::source_chunk_bytes + 3;
    std::vector<ggml_bf16_t> values(elements);
    for (size_t i = 0; i < elements; ++i)
        values[i] = ggml_fp32_to_bf16(float(int(i % 29) - 14) / 8);
    std::vector<uint8_t> payload(elements * 2);
    std::memcpy(payload.data(), values.data(), payload.size());
    const auto start = fixture(path, GGML_TYPE_BF16, elements, payload);
    ModelLoader loader;
    loader.set_n_threads(1);
    check(loader.init_from_file_and_convert_name(path.string()), "production BF16 GGUF init");
    const auto before = loader.get_tensor_storage_map().at("weight");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}),
                                                                ggml_free);
    check(bool(context), "production metadata context");
    auto tensor = ggml_new_tensor_1d(context.get(), destination_type, elements);
    FakeDevice device(host, other_registry ? "Other fake GPU" : "WebGPU");
    Storage storage;
    if (failure == 1) {
        storage.truncate_path = path;
        storage.truncate_bytes = start + 512 * 1024;
    }
    if (failure == 2)
        storage.throw_on_second = true;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        device.buffer(storage, ggml_nbytes(tensor)), ggml_backend_buffer_free);
    check(ggml_backend_tensor_alloc(buffer.get(), tensor, storage.bytes.data()) == GGML_STATUS_SUCCESS,
          "fake tensor allocation");
    const bool loaded = loader.load_tensors(
        [&](const TensorStorage &, ggml_tensor ** out) {
            *out = tensor;
            return true;
        },
        false, nullptr, false);
    if (failure) {
        check(!loaded && !storage.writes.empty(), "production reader failure was published as loaded");
        return;
    }
    check(loaded, "production BF16 load");
    const auto & after = loader.get_tensor_storage_map().at("weight");
    check(before.type == GGML_TYPE_BF16 && after.type == before.type && after.offset == before.offset &&
              after.n_dims == before.n_dims && after.nelements() == before.nelements(),
          "source metadata mutated");
    if (host)
        check(storage.writes.empty(), "host destination intercepted by GPU streaming");
    else if (other_registry)
        check(storage.writes.size() == 1, "other GPU legacy path changed");
    else
        check(storage.writes.size() > 1 &&
                  *std::max_element(storage.writes.begin(), storage.writes.end()) <= 1024 * 1024,
              "production loader did not bound WebGPU writes");
    for (size_t i = 0; i < elements; ++i) {
        if (destination_type == GGML_TYPE_F32) {
            float observed;
            std::memcpy(&observed, storage.bytes.data() + i * 4, 4);
            check(observed == ggml_bf16_to_fp32(values[i]), "production BF16->F32 value");
        } else {
            uint16_t observed;
            std::memcpy(&observed, storage.bytes.data() + i * 2, 2);
            const auto expected =
                destination_type == GGML_TYPE_BF16 ? values[i].bits : ggml_fp32_to_fp16(ggml_bf16_to_fp32(values[i]));
            check(observed == expected, "production raw BF16 / F16 value");
        }
    }
}

static void raw_quantized() {
    Temporary temporary;
    const auto path = temporary.directory / "quantized.gguf";
    const size_t blocks = 4001, elements = blocks * ggml_blck_size(GGML_TYPE_Q4_K);
    std::vector<uint8_t> payload(blocks * ggml_type_size(GGML_TYPE_Q4_K));
    for (size_t i = 0; i < payload.size(); ++i)
        payload[i] = static_cast<uint8_t>(i * 13 + 3);
    fixture(path, GGML_TYPE_Q4_K, elements, payload);
    ModelLoader loader;
    loader.set_n_threads(1);
    check(loader.init_from_file_and_convert_name(path.string()), "production quantized init");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}),
                                                                ggml_free);
    auto tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_Q4_K, elements);
    FakeDevice device;
    Storage storage;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        device.buffer(storage, ggml_nbytes(tensor)), ggml_backend_buffer_free);
    check(ggml_backend_tensor_alloc(buffer.get(), tensor, storage.bytes.data()) == GGML_STATUS_SUCCESS,
          "quant tensor alloc");
    check(loader.load_tensors(
              [&](const TensorStorage &, ggml_tensor ** out) {
                  *out = tensor;
                  return true;
              },
              false, nullptr, false),
          "production raw quantized load");
    check(storage.bytes == payload && storage.writes.size() > 1 &&
              *std::max_element(storage.writes.begin(), storage.writes.end()) <= 512 * 1024,
          "quantized raw bytes / chunk limit changed");
}

int main() {
    try {
        for (auto type : {GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_F16})
            load_bf16(type, false, false, 0);
        load_bf16(GGML_TYPE_BF16, true, false, 0);
        load_bf16(GGML_TYPE_F32, true, false, 0);
        load_bf16(GGML_TYPE_F16, false, true, 0);
        load_bf16(GGML_TYPE_F32, false, false, 1);
        load_bf16(GGML_TYPE_F32, false, false, 2);
        raw_quantized();
        std::cout << "production bounded browser weight reader passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
