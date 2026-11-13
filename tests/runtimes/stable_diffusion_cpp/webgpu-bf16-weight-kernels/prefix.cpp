#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using half = _Float16;

template <class T> T select(T left, T right, bool condition) {
    return condition ? right : left;
}

static float bits_to_float(uint32_t bits) {
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void require(bool value, const char * message) {
    if (!value)
        throw std::runtime_error(message);
}

// WGSL storage reads outside the bound are modeled as zero. Destinations use
// vector::at so an unexpected write is an explicit failure in this CPU fixture.
template <class T> struct storage {
    std::vector<T> data;
    T operator[](size_t index) const {
        return index < data.size() ? data[index] : T{};
    }
};

static storage<uint32_t> src, src0;
static storage<float> src1;
struct checked_output : std::vector<float> {
    float & operator[](size_t index) {
        return at(index);
    }
};
static checked_output dst;
static std::array<half, 16> shmem;
static constexpr uint32_t WG_SIZE = 4, NUM_COLS = 2, OUTPUTS_PER_WG = 3;
static constexpr uint32_t TOTAL_WORKGROUP_SIZE = 4, TILE_K = 4, TILE_SRC0_SHMEM = 16;
struct {
    uint32_t k, m, n, stride_01, stride_11;
} params;

static void store_shmem(half value, uint32_t index) {
    shmem.at(index) = value;
}

#define SRC0 src0
#define SRC1 src1
#define SRC1_TYPE float
#define SHMEM_TYPE half
#define VEC_SIZE 1u
#define TYPE_BF16
#define MUL_ACC_FLOAT
