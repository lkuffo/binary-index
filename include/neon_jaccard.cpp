#include <cstdint>
#include <cstddef>
#include <string.h>
#include <iostream>
#include <arm_neon.h>
#include <cassert>

#include "jaccard_byte_luts.h"
#include "jaccard_nibble_luts.h"


struct KNNCandidate {
    uint32_t index;
    float distance;
};

struct VectorComparator {
    bool operator() (const KNNCandidate& a, const KNNCandidate& b) {
        return a.distance < b.distance;
    }
};

enum JaccardKernel {
    // 128
    JACCARD_B128_VPOPCNTQ,
    JACCARD_B128_SERIAL_LUT,
    JACCARD_B128_SERIAL_BUILTINPOPCNT,
    JACCARD_B128_VPOPCNTQ_WORDBYWORD,
    JACCARD_B128_SERIAL_LUT_NIBBLE_PDX,
    JACCARD_B128_SERIAL_LUT_BYTE_PDX,
    JACCARD_B128_VPOPCNTQ_PDX,
    JACCARD_B256_VPOPCNTQ_PDX,
    // 1536
    JACCARD_B1536_SERIAL,
    JACCARD_B1536_VPOPCNTQ_WORDBYWORD
};

float jaccard_b128_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector) {
    uint8x16_t a_vec = vld1q_u8(first_vector);
    uint8x16_t b_vec = vld1q_u8(second_vector);
    uint8x16_t and_count_vec = vcntq_u8(vandq_u8(a_vec, b_vec));
    uint8x16_t or_count_vec = vcntq_u8(vorrq_u8(a_vec, b_vec));
    auto intersection = vaddvq_u8(and_count_vec);
    auto union_ = vaddvq_u8(or_count_vec);
    return (union_ != 0) ? 1 - (float)intersection / (float)union_ : 1;
}

static uint8_t intersections_tmp[256];
static uint8_t unions_tmp[256];
static float distances_tmp[256];

// 1-to-256 vectors
// second_vector is a 256*128 matrix in a column-major layout
void jaccard_b128_serial_lut_nibble_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    uint32_t intersection = 0, union_ = 0;
    for (size_t dim = 0; dim != 16; dim++){
        uint8_t nibble_first_a = (first_vector[dim] & 0xF0) >> 4;
        uint8_t nibble_first_b = first_vector[dim] & 0x0F;
        const uint8_t* intersection_lookup_a = intersection_lookup_tables[nibble_first_a];
        const uint8_t* intersection_lookup_b = intersection_lookup_tables[nibble_first_b];
        const uint8_t* union_lookup_a = union_lookup_tables[nibble_first_a];
        const uint8_t* union_lookup_b = union_lookup_tables[nibble_first_b];
        for (size_t i = 0; i < 256; i++){
            uint8_t nibble_second_a = (second_vector[i] & 0xF0) >> 4;
            uint8_t nibble_second_b = second_vector[i] & 0x0F;
            intersections_tmp[i] += intersection_lookup_a[nibble_second_a] + intersection_lookup_b[nibble_second_b];
            unions_tmp[i] += union_lookup_a[nibble_second_a] + union_lookup_b[nibble_second_b];
        }
        second_vector += 256;
    }
    // TODO: Ugly
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = (unions_tmp[i] != 0) ? 1 - (float)intersections_tmp[i] / (float)unions_tmp[i] : 1.0f;
    }
}


// 1-to-256 vectors
// second_vector is a 256*128 matrix in a column-major layout
void jaccard_b128_serial_lut_byte_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    uint32_t intersection = 0, union_ = 0;
    for (size_t dim = 0; dim != 16; dim++){
        uint8_t byte_first = first_vector[dim];
        const uint8_t* intersection_lookup = intersection_byte_lookup_tables[byte_first];
        const uint8_t* union_lookup = union_byte_lookup_tables[byte_first];
        for (size_t i = 0; i < 256; i++){
            uint8_t byte_second = second_vector[i];
            intersections_tmp[i] += intersection_lookup[byte_second];
            unions_tmp[i] += union_lookup[byte_second];
        }
        second_vector += 256;
    }
    // TODO: Ugly
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = (unions_tmp[i] != 0) ? 1 - (float)intersections_tmp[i] / (float)unions_tmp[i] : 1.0f;
    }
}

// 1-to-256 vectors
// second_vector is a 256*128 matrix in a column-major layout
void jaccard_b128_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    uint8x16_t intersections_result[16];
    uint8x16_t unions_result[16];
    // Load initial values
    for (size_t i = 0; i < 16; ++i) { // 256 vectors at a time (probably overflows the registers)
        intersections_result[i] = vdupq_n_u32(0);
        unions_result[i] = vdupq_n_u32(0);
    }
    for (size_t dim = 0; dim != 16; dim++){
        uint8x16_t a_vec = vdupq_n_u8(first_vector[dim]);
        for (size_t i = 0; i < 16; i++){
            uint8x16_t b_vec = vld1q_u8(second_vector);
            uint8x16_t and_count_vec = vcntq_u8(vandq_u8(a_vec, b_vec));
            uint8x16_t or_count_vec = vcntq_u8(vorrq_u8(a_vec, b_vec));
            intersections_result[i] = vaddq_u8(intersections_result[i], and_count_vec);
            unions_result[i] = vaddq_u8(unions_result[i], or_count_vec);
            second_vector += 16; // 256 values (16 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 16; i++) {
        vst1q_u8(&intersections_tmp[i * 16], intersections_result[i]);
        vst1q_u8(&unions_tmp[i * 16], unions_result[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = (unions_tmp[i] != 0) ? 1 - (float)intersections_tmp[i] / (float)unions_tmp[i] : 1.0f;
    }
}


// 1-to-256 vectors
// second_vector is a 256*256 matrix in a column-major layout
void jaccard_b256_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    uint8x16_t intersections_result[16];
    uint8x16_t unions_result[16];
    // Load initial values
    for (size_t i = 0; i < 16; ++i) { // 256 vectors at a time (probably overflows the registers)
        intersections_result[i] = vdupq_n_u32(0);
        unions_result[i] = vdupq_n_u32(0);
    }
    for (size_t dim = 0; dim != 32; dim++){
        uint8x16_t a_vec = vdupq_n_u8(first_vector[dim]);
        for (size_t i = 0; i < 16; i++){
            uint8x16_t b_vec = vld1q_u8(second_vector);
            uint8x16_t and_count_vec = vcntq_u8(vandq_u8(a_vec, b_vec));
            uint8x16_t or_count_vec = vcntq_u8(vorrq_u8(a_vec, b_vec));
            intersections_result[i] = vaddq_u8(intersections_result[i], and_count_vec);
            unions_result[i] = vaddq_u8(unions_result[i], or_count_vec);
            second_vector += 16; // 256 values (16 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 16; i++) {
        vst1q_u8(&intersections_tmp[i * 16], intersections_result[i]);
        vst1q_u8(&unions_tmp[i * 16], unions_result[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = (unions_tmp[i] != 0) ? 1 - (float)intersections_tmp[i] / (float)unions_tmp[i] : 1.0f;
    }
}

float jaccard_b128_serial_lut(uint8_t const *first_vector, uint8_t const *second_vector) {
    static uint8_t lookup_table[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, //
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};
    uint32_t intersection = 0, union_ = 0;
    for (size_t i = 0; i != 16; ++i)
        intersection += lookup_table[first_vector[i] & second_vector[i]], union_ += lookup_table[first_vector[i] | second_vector[i]];
    return (union_ != 0) ? 1 - (float)intersection / (float)union_ : 1.0f;
}

float jaccard_b128_serial_builtinpopcnt(uint8_t const *first_vector, uint8_t const *second_vector) {
    uint32_t intersection = 0, union_ = 0;
    for (size_t i = 0; i != 16; ++i)
        intersection += __builtin_popcount(first_vector[i] & second_vector[i]), union_ += __builtin_popcount(first_vector[i] | second_vector[i]);
    return (union_ != 0) ? 1 - (float)intersection / (float)union_ : 1.0f;
}


float jaccard_b1536_serial(uint8_t const *first_vector, uint8_t const *second_vector) {
    static uint8_t lookup_table[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, //
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};
    uint32_t intersection = 0, union_ = 0;
    for (size_t i = 0; i != 192; ++i)
        intersection += lookup_table[first_vector[i] & second_vector[i]], union_ += lookup_table[first_vector[i] | second_vector[i]];
    return (union_ != 0) ? 1 - (float)intersection / (float)union_ : 1.0f;
}

static unsigned char simsimd_popcount_b8(unsigned char x) {
    static unsigned char lookup_table[] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, //
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};
    return lookup_table[x];
}

uint32_t _simsimd_reduce_u8x16_neon(uint8x16_t vec) {
    // Split the vector into two halves and widen to `uint16x8_t`
    uint16x8_t low_half = vmovl_u8(vget_low_u8(vec));   // widen lower 8 elements
    uint16x8_t high_half = vmovl_u8(vget_high_u8(vec)); // widen upper 8 elements

    // Sum the widened halves
    uint16x8_t sum16 = vaddq_u16(low_half, high_half);

    // Now reduce the `uint16x8_t` to a single `uint32_t`
    uint32x4_t sum32 = vpaddlq_u16(sum16);       // pairwise add into 32-bit integers
    uint64x2_t sum64 = vpaddlq_u32(sum32);       // pairwise add into 64-bit integers
    uint32_t final_sum = vaddvq_u64(sum64); // final horizontal add to 32-bit result
    return final_sum;
}

float jaccard_b128_vpopcntq_wordbyword(uint8_t const *first_vector, uint8_t const *second_vector) {
    int32_t intersection = 0, union_ = 0;
    size_t i = 0;
    // In each 8-bit word we may have up to 8 intersections/unions.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the intersections/unions into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= 16) {
        uint8x16_t intersections_cycle_vec = vdupq_n_u8(0);
        uint8x16_t unions_cycle_vec = vdupq_n_u8(0);
        for (size_t cycle = 0; cycle < 31 && i + 16 <= 16; ++cycle, i += 16) {
            uint8x16_t a_vec = vld1q_u8(first_vector + i);
            uint8x16_t b_vec = vld1q_u8(second_vector + i);
            uint8x16_t and_count_vec = vcntq_u8(vandq_u8(a_vec, b_vec));
            uint8x16_t or_count_vec = vcntq_u8(vorrq_u8(a_vec, b_vec));
            intersections_cycle_vec = vaddq_u8(intersections_cycle_vec, and_count_vec);
            unions_cycle_vec = vaddq_u8(unions_cycle_vec, or_count_vec);
        }
        intersection += _simsimd_reduce_u8x16_neon(intersections_cycle_vec);
        union_ += _simsimd_reduce_u8x16_neon(unions_cycle_vec);
    }
    // Handle the tail
    for (; i != 16; ++i)
        intersection += simsimd_popcount_b8(first_vector[i] & second_vector[i]), union_ += simsimd_popcount_b8(first_vector[i] | second_vector[i]);
    return (union_ != 0) ? 1 - (float)intersection / (float)union_ : 1;
}

float jaccard_b1536_vpopcntq_wordbyword(uint8_t const *first_vector, uint8_t const *second_vector) {
    int32_t intersection = 0, union_ = 0;
    size_t i = 0;
    // In each 8-bit word we may have up to 8 intersections/unions.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the intersections/unions into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= 192) {
        uint8x16_t intersections_cycle_vec = vdupq_n_u8(0);
        uint8x16_t unions_cycle_vec = vdupq_n_u8(0);
        for (size_t cycle = 0; cycle < 31 && i + 16 <= 192; ++cycle, i += 16) {
            uint8x16_t a_vec = vld1q_u8(first_vector + i);
            uint8x16_t b_vec = vld1q_u8(second_vector + i);
            uint8x16_t and_count_vec = vcntq_u8(vandq_u8(a_vec, b_vec));
            uint8x16_t or_count_vec = vcntq_u8(vorrq_u8(a_vec, b_vec));
            intersections_cycle_vec = vaddq_u8(intersections_cycle_vec, and_count_vec);
            unions_cycle_vec = vaddq_u8(unions_cycle_vec, or_count_vec);
        }
        intersection += _simsimd_reduce_u8x16_neon(intersections_cycle_vec);
        union_ += _simsimd_reduce_u8x16_neon(unions_cycle_vec);
    }
    // Handle the tail
    for (; i != 192; ++i)
        intersection += simsimd_popcount_b8(first_vector[i] & second_vector[i]), union_ += simsimd_popcount_b8(first_vector[i] | second_vector[i]);
    return (union_ != 0) ? 1 - (float)intersection / (float)union_ : 1;
}

template <JaccardKernel kernel=JACCARD_B128_VPOPCNTQ, int N_WORDS=16>
std::vector<KNNCandidate> jaccard_standalone_partial_sort(
    uint8_t const *first_vector,
    uint8_t const *second_vector,
    size_t num_queries,
    size_t num_vectors,
    size_t knn
) {
    std::vector<KNNCandidate> result(knn * num_queries);
    std::vector<KNNCandidate> all_distances(num_vectors);
    const uint8_t* query = first_vector;
    for (size_t i = 0; i < num_queries; ++i) {
        const uint8_t* data = second_vector;
        // Fill all_distances by direct indexing
        for (size_t j = 0; j < num_vectors; ++j) {
            float current_distance;
            if constexpr (kernel == JACCARD_B128_VPOPCNTQ){
                current_distance = jaccard_b128_vpopcntq(query, data);
            } else if constexpr (kernel == JACCARD_B128_SERIAL_LUT) {
                current_distance = jaccard_b128_serial_lut(query, data);
            } else if constexpr (kernel == JACCARD_B128_VPOPCNTQ_WORDBYWORD) {
                current_distance = jaccard_b128_vpopcntq_wordbyword(query, data);
            } else if constexpr (kernel == JACCARD_B128_SERIAL_BUILTINPOPCNT) {
                current_distance = jaccard_b128_serial_builtinpopcnt(query, data);
            } else if constexpr (kernel == JACCARD_B1536_SERIAL) {
                current_distance = jaccard_b1536_serial(query, data);
            } else if constexpr (kernel == JACCARD_B1536_VPOPCNTQ_WORDBYWORD) {
                current_distance = jaccard_b1536_vpopcntq_wordbyword(query, data);
            } else {
                uint8x16_t c0 = vcntq_u8(veorq_u8(vld1q_u8(query), vld1q_u8(data)));
                current_distance = vaddvq_u8(c0);
            }

            all_distances[j].index = static_cast<uint32_t>(j);
            all_distances[j].distance = current_distance;

            data += N_WORDS;
        }

        // Partial sort to get top-k
        std::partial_sort(
            all_distances.begin(),
            all_distances.begin() + knn,
            all_distances.end(),
            VectorComparator()
        );
        // Copy top-k results to result vector
        for (size_t k = 0; k < knn; ++k) {
            result[i * knn + k] = all_distances[k];
        }
        query += N_WORDS;
    }
    return result;
}


template <JaccardKernel kernel=JACCARD_B128_SERIAL_LUT_NIBBLE_PDX, int N_WORDS=16, int PDX_BLOCK_SIZE=256>
std::vector<KNNCandidate> jaccard_pdx_standalone_partial_sort(
    uint8_t const *first_vector,
    uint8_t const *second_vector,
    size_t num_queries,
    size_t num_vectors,
    size_t knn
) {
    std::vector<KNNCandidate> result(knn * num_queries);
    std::vector<KNNCandidate> all_distances(num_vectors);
    const uint8_t* query = second_vector;
    for (size_t i = 0; i < num_queries; ++i) {
        const uint8_t* data = first_vector;
        // Fill all_distances by direct indexing
        size_t global_offset = 0;
        for (size_t j = 0; j < num_vectors; j+=PDX_BLOCK_SIZE) {
            // TODO: Ugly
            memset((void*) distances_tmp, 0, PDX_BLOCK_SIZE * sizeof(float));
            memset((void*) intersections_tmp, 0, PDX_BLOCK_SIZE * sizeof(uint8_t));
            memset((void*) unions_tmp, 0, PDX_BLOCK_SIZE * sizeof(uint8_t));
            if constexpr (kernel == JACCARD_B128_SERIAL_LUT_NIBBLE_PDX){
                jaccard_b128_serial_lut_nibble_pdx(query, data);
            } else if constexpr (kernel == JACCARD_B128_SERIAL_LUT_BYTE_PDX){
                jaccard_b128_serial_lut_byte_pdx(query, data);
            } else if constexpr (kernel == JACCARD_B128_VPOPCNTQ_PDX){
                jaccard_b128_vpopcntq_pdx(query, data);
            } else if constexpr (kernel == JACCARD_B256_VPOPCNTQ_PDX){
                jaccard_b256_vpopcntq_pdx(query, data);
            }
            // TODO: Ugly
            for (uint32_t z = 0; z < PDX_BLOCK_SIZE; ++z) {
                all_distances[global_offset].index = global_offset;
                all_distances[global_offset].distance = distances_tmp[z];
                global_offset++;
            }
            data += N_WORDS * PDX_BLOCK_SIZE;
        }

        // Partial sort to get top-k
        std::partial_sort(
            all_distances.begin(),
            all_distances.begin() + knn,
            all_distances.end(),
            VectorComparator()
        );
        // Copy top-k results to result vector
        for (size_t k = 0; k < knn; ++k) {
            result[i * knn + k] = all_distances[k];
        }
        query += N_WORDS;
    }
    return result;
}

std::vector<KNNCandidate> jaccard_standalone(
    JaccardKernel kernel,
    uint8_t const *first_vector,
    uint8_t const *second_vector,
    size_t num_queries,
    size_t num_vectors,
    size_t knn
) {
    switch (kernel) {
        case JACCARD_B128_VPOPCNTQ:
            return jaccard_standalone_partial_sort<JACCARD_B128_VPOPCNTQ, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B128_SERIAL_LUT:
            return jaccard_standalone_partial_sort<JACCARD_B128_SERIAL_LUT, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B128_VPOPCNTQ_WORDBYWORD:
            return jaccard_standalone_partial_sort<JACCARD_B128_VPOPCNTQ_WORDBYWORD, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B128_SERIAL_BUILTINPOPCNT:
            return jaccard_standalone_partial_sort<JACCARD_B128_SERIAL_BUILTINPOPCNT, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B1536_SERIAL:
            return jaccard_standalone_partial_sort<JACCARD_B1536_SERIAL, 192>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B1536_VPOPCNTQ_WORDBYWORD:
            return jaccard_standalone_partial_sort<JACCARD_B1536_VPOPCNTQ_WORDBYWORD, 192>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B128_SERIAL_LUT_NIBBLE_PDX:
            return jaccard_pdx_standalone_partial_sort<JACCARD_B128_SERIAL_LUT_NIBBLE_PDX, 16, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B128_SERIAL_LUT_BYTE_PDX:
            return jaccard_pdx_standalone_partial_sort<JACCARD_B128_SERIAL_LUT_BYTE_PDX, 16, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B128_VPOPCNTQ_PDX:
            return jaccard_pdx_standalone_partial_sort<JACCARD_B128_VPOPCNTQ_PDX, 16, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B256_VPOPCNTQ_PDX:
            return jaccard_pdx_standalone_partial_sort<JACCARD_B256_VPOPCNTQ_PDX, 32, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
        default:
            return jaccard_standalone_partial_sort<JACCARD_B128_VPOPCNTQ, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
    }
}