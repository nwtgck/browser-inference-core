static std::vector<uint32_t> pack(const std::vector<uint16_t> & input) {
    std::vector<uint32_t> words((input.size() + 1) / 2, 0xaced0000u);
    for (size_t i = 0; i < input.size(); ++i) {
        const uint32_t shift = uint32_t(i % 2) * 16;
        words[i / 2] = (words[i / 2] & ~(0xffffu << shift)) | (uint32_t(input[i]) << shift);
    }
    return words;
}

static float reference(uint16_t bits) {
    return bits_to_float(uint32_t(bits) << 16);
}

static void unpack_all_bits() {
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        // Includes signed zeros, subnormals, infinities and NaN payloads. Do not
        // perform floating-point arithmetic before comparing decoded bit fields.
        for (uint32_t odd = 0; odd < 2; ++odd) {
            const uint32_t word = odd ? (bits << 16) | 0x1234u : 0xabcd0000u | bits;
            require(float_bits(bf16_word_to_f32(word, odd)) == bits << 16, "BF16 bit recovery");
        }
    }
}

static void gather_rows() {
    for (uint32_t width : {1u, 3u, 7u, 16u}) {
        for (uint32_t base : {0u, 1u, 127u, 129u}) {
            constexpr uint32_t rows = 5;
            std::vector<uint16_t> bytes(base + width * rows, 0x7f81);
            for (uint32_t i = 0; i < width * rows; ++i)
                bytes[base + i] = uint16_t(0x3e00 + (i % 51) * 7);
            src.data = pack(bytes);
            dst.assign(2 + 4 * width + 3, -1234.0f);
            const std::array<uint32_t, 4> selected{4, 0, 3, 4};
            for (uint32_t i = 0; i < selected.size(); ++i) {
                for (uint32_t column = 0; column < width; ++column)
                    copy_elements(base + selected[i] * width, 2 + i * width, column);
            }
            for (uint32_t i = 0; i < selected.size(); ++i) {
                for (uint32_t column = 0; column < width; ++column)
                    require(float_bits(dst[2 + i * width + column]) ==
                                float_bits(reference(bytes[base + selected[i] * width + column])),
                            "GET_ROWS selected row/odd offset");
            }
            require(dst.front() == -1234.0f && dst[1] == -1234.0f && dst.back() == -1234.0f,
                    "GET_ROWS destination guard");
        }
    }
}

static void matvec_rows() {
    for (uint32_t k : {1u, 3u, 7u, 16u}) {
        for (uint32_t base : {0u, 1u, 127u}) {
            constexpr uint32_t rows = 5, columns = 2;
            std::vector<uint16_t> weight(base + k * rows, 0x7f81);
            for (uint32_t i = 0; i < k * rows; ++i)
                weight[base + i] = uint16_t((i % 3 ? 0x3e00 : 0xbe00) + (i % 29) * 5);
            src0.data = pack(weight);
            src1.data.resize(k * columns);
            for (uint32_t i = 0; i < src1.data.size(); ++i)
                src1.data[i] = float(int(i % 7) - 3) / 8;
            params = {k, rows, columns, k, k};
            for (uint32_t start : {0u, 3u}) {
                std::array<std::array<float, OUTPUTS_PER_WG>, NUM_COLS> result{};
                for (uint32_t thread = 0; thread < WG_SIZE; ++thread) {
                    auto partial = accumulate_vec_dot(thread, start, base, 0);
                    for (uint32_t c = 0; c < columns; ++c)
                        for (uint32_t r = 0; r < OUTPUTS_PER_WG; ++r)
                            result[c][r] += partial[c][r];
                }
                for (uint32_t c = 0; c < columns; ++c) {
                    for (uint32_t r = 0; r < OUTPUTS_PER_WG; ++r) {
                        double expected = 0;
                        if (start + r < rows)
                            for (uint32_t i = 0; i < k; ++i)
                                expected +=
                                    double(reference(weight[base + (start + r) * k + i])) * src1.data[c * k + i];
                        require(std::abs(double(result[c][r]) - expected) < 2e-6, "BF16 matvec dot/tail");
                    }
                }
            }
        }
    }
}

static void matrix_tile_loads() {
    for (uint32_t k : {1u, 3u, 7u, 16u}) {
        for (uint32_t base : {0u, 1u, 127u}) {
            constexpr uint32_t rows = 5;
            std::vector<uint16_t> weight(base + k * rows, 0x7f81);
            for (uint32_t i = 0; i < k * rows; ++i)
                weight[base + i] = uint16_t(0x3d00 + (i % 53) * 11);
            src0.data = pack(weight);
            params = {k, rows, 1, k, k};
            for (uint32_t start : {0u, 4u}) {
                for (uint32_t outer = 0; outer < k; outer += TILE_K) {
                    shmem.fill(half(-1234));
                    for (uint32_t thread = 0; thread < TOTAL_WORKGROUP_SIZE; ++thread)
                        init_shmem_src0(thread, base, start, outer);
                    for (uint32_t i = 0; i < TILE_SRC0_SHMEM; ++i) {
                        uint32_t r = start + i / TILE_K, column = outer + i % TILE_K;
                        const half expected =
                            r < rows && column < k ? half(reference(weight[base + r * k + column])) : half(0);
                        require(float(shmem[i]) == float(expected), "BF16 fast matrix tile/zero padding");
                    }
                }
            }
        }
    }
}

static void capability_cases() {
    tensor weight{GGML_TYPE_BF16}, input{GGML_TYPE_F32}, output{GGML_TYPE_F32};
    for (size_t offset : {size_t(0), size_t(2), size_t(254), size_t(256), size_t(258)}) {
        weight.offset = offset;
        require(admitted(GGML_OP_GET_ROWS, &weight, &input, &output), "BF16 aligned GET_ROWS rejected");
    }
    weight.offset = 1;
    require(!admitted(GGML_OP_GET_ROWS, &weight, &input, &output), "Misaligned BF16 GET_ROWS admitted");
    weight.offset = 0;
    output.type = GGML_TYPE_I32;
    require(!admitted(GGML_OP_GET_ROWS, &weight, &input, &output), "BF16 non-F32 lookup output admitted");
    output.type = GGML_TYPE_F32;
    for (auto operation : {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID}) {
        require(admitted(operation, &weight, &input, &output), "BF16/F32 matrix rejected");
        input.type = GGML_TYPE_F16;
        require(!admitted(operation, &weight, &input, &output), "BF16/F16 matrix broadened");
        input.type = GGML_TYPE_BF16;
        require(!admitted(operation, &weight, &input, &output), "BF16 right operand admitted");
        input.type = GGML_TYPE_F32;
    }
}

static void intermediate_range() {
    // Storage and scalar matvec retain the BF16 exponent range; the upstream
    // fast matrix path explicitly narrows its shared tile to half precision.
    src0.data = pack({0x4780}); // 65536: finite BF16/F32, outside finite F16.
    src1.data = {1.0f, 1.0f};
    params = {1, 1, 2, 1, 1};
    auto result = accumulate_vec_dot(0, 0, 0, 0);
    require(result[0][0] == 65536.0f && std::isfinite(result[0][0]), "BF16 matvec range lost");
    shmem.fill(half(0));
    for (uint32_t thread = 0; thread < TOTAL_WORKGROUP_SIZE; ++thread)
        init_shmem_src0(thread, 0, 0, 0);
    require(std::isinf(float(shmem[0])), "Fast matrix F16 intermediate contract changed");
}

int main() {
    unpack_all_bits();
    gather_rows();
    matvec_rows();
    matrix_tile_loads();
    capability_cases();
    intermediate_range();
    std::cout << "PASS: BF16 unpack 131072 bit patterns, odd/offset row lookup, matvec tails, fast matrix tile loads, "
                 "capability/alignment contracts (CPU extraction; no GPU)\n";
}
