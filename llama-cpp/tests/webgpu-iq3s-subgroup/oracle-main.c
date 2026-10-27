// Synthetic fixture oracle. The pinned upstream dequantizer is prepended at build.
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void die(const char * message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static void * load(const char * path, size_t * bytes) {
    FILE * file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) die("Cannot read oracle input");
    const long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET)) die("Invalid oracle input size");
    *bytes = (size_t) length;
    void * data = malloc(*bytes);
    if (!data || fread(data, 1, *bytes, file) != *bytes) die("Cannot load oracle input");
    fclose(file);
    return data;
}

static void self_test(void) {
    block_iq3_s block;
    float values[256];
    memset(&block, 0, sizeof(block));
    block.d = 0x3c00;
    dequantize_row_iq3_s(&block, values, 256);
    for (int i = 0; i < 256; ++i) if (values[i] != 1.0f) die("CPU grid-zero oracle self-test failed");
    memset(block.qs, 255, sizeof(block.qs));
    memset(block.qh, 255, sizeof(block.qh));
    memset(block.signs, 255, sizeof(block.signs));
    memset(block.scales, 255, sizeof(block.scales));
    dequantize_row_iq3_s(&block, values, 256);
    for (int i = 0; i < 256; ++i) {
        const float expected = i % 4 < 2 ? -31.0f : -465.0f;
        if (values[i] != expected) die("CPU grid-511/sign/scale oracle self-test failed");
    }
    block.d = 0xbc00;
    dequantize_row_iq3_s(&block, values, 256);
    for (int i = 0; i < 256; ++i) {
        const float expected = i % 4 < 2 ? 31.0f : 465.0f;
        if (values[i] != expected) die("CPU negative-half oracle self-test failed");
    }
    block.d = 0;
    dequantize_row_iq3_s(&block, values, 256);
    for (int i = 0; i < 256; ++i) if (values[i] != 0.0f) die("CPU zero-half oracle self-test failed");
}

int main(int argc, char ** argv) {
    const uint32_t endian = 1;
    if (*(const uint8_t *) &endian != 1 || sizeof(float) != 4 || sizeof(double) != 8 || sizeof(block_iq3_s) != 110) {
        die("Oracle requires LE Float32/Float64 and pinned 110-byte IQ3_S blocks");
    }
    self_test();
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        puts("actual upstream IQ3_S dequantizer self-tests passed");
        return 0;
    }
    if (argc != 6) die("Usage: oracle WEIGHTS INPUT IDS PARAMS REFERENCE");
    size_t weight_bytes, input_bytes, ids_bytes, params_bytes;
    uint8_t * weights = load(argv[1], &weight_bytes);
    float * input = load(argv[2], &input_bytes);
    uint32_t * ids = load(argv[3], &ids_bytes);
    uint32_t * params = load(argv[4], &params_bytes);
    if (params_bytes != 13 * sizeof(uint32_t) || input_bytes % 4 || ids_bytes % 4) die("Invalid oracle parameter/input layout");
    const uint32_t offset_a = params[0], offset_x = params[1], offset_ids = params[2];
    const uint32_t k = params[4], m = params[5], experts = params[6], slots = params[7], input_rows = params[8];
    const uint32_t stride_01 = params[9], stride_11 = params[10], stride_02 = params[11];
    if (!k || k % 256 || !m || !experts || !slots || !input_rows || input_rows > slots) die("Invalid oracle dimensions");
    if ((size_t) offset_ids + slots > ids_bytes / 4) die("Oracle IDs outside input");
    if ((size_t) slots * m > SIZE_MAX / (2 * sizeof(double))) die("Oracle output too large");
    const size_t outputs = (size_t) slots * m;
    double * reference = malloc(outputs * 2 * sizeof(double));
    if (!reference) die("Cannot allocate oracle output");
    float dequantized[256];
    for (uint32_t slot = 0; slot < slots; ++slot) {
        const uint32_t expert = ids[offset_ids + slot];
        if (expert >= experts) die("Oracle selected expert outside stored payload");
        const size_t input_base = (size_t) offset_x + (slot % input_rows) * (size_t) stride_11;
        if (input_base + k > input_bytes / 4) die("Oracle input row outside payload");
        for (uint32_t row = 0; row < m; ++row) {
            const size_t block_base = (size_t) offset_a + expert * (size_t) stride_02 + row * (size_t) stride_01;
            if ((block_base + k / 256) * 110 > weight_bytes) die("Oracle weight row outside payload");
            double sum = 0.0, sum_abs = 0.0;
            for (uint32_t block = 0; block < k / 256; ++block) {
                const block_iq3_s * packed = (const block_iq3_s *) (weights + (block_base + block) * 110);
                dequantize_row_iq3_s(packed, dequantized, 256);
                for (uint32_t j = 0; j < 256; ++j) {
                    const double term = (double) dequantized[j] * (double) input[input_base + block * 256 + j];
                    if (!isfinite(term)) die("Nonfinite oracle term");
                    sum += term;
                    sum_abs += fabs(term);
                }
            }
            const size_t index = (size_t) slot * m + row;
            reference[index * 2] = sum;
            reference[index * 2 + 1] = sum_abs;
        }
    }
    FILE * output = fopen(argv[5], "wb");
    if (!output || fwrite(reference, sizeof(double), outputs * 2, output) != outputs * 2 || fclose(output)) {
        die("Cannot write oracle output");
    }
    free(reference); free(params); free(ids); free(input); free(weights);
    printf("%zu outputs referenced with actual IQ3_S CPU dequantization and Float64 dot\n", outputs);
    return 0;
}
