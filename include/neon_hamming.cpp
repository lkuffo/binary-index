#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <vector>
#include <arm_neon.h>


struct KNNCandidate {
    uint32_t index;
    uint32_t distance;
};

struct VectorComparator {
    bool operator() (const KNNCandidate& a, const KNNCandidate& b) {
        return a.distance < b.distance;
    }
};

enum HammingKernel {
    // 128
    HAMMING_B128_VPOPCNTQ,
    HAMMING_B128_SERIAL_LUT,
    HAMMING_B128_SERIAL_BUILTINPOPCNT,
    HAMMING_B128_VPOPCNTQ_WORDBYWORD,
    HAMMING_B128_VPOPCNTQ_PDX,
    // 256
    HAMMING_B256_VPOPCNTQ_PDX,
    // 1536
    HAMMING_B1536_SERIAL,
    HAMMING_B1536_VPOPCNTQ_WORDBYWORD
};

int32_t hamming_b128_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector) {
    uint8x16_t c0 = vcntq_u8(veorq_u8(vld1q_u8(first_vector), vld1q_u8(second_vector)));
    auto dis = vaddvq_u8(c0);
    return dis;
}


int32_t hamming_b128_serial_lut(uint8_t const *first_vector, uint8_t const *second_vector) {
    static uint8_t lookup_table[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, //
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};
    int32_t differences = 0;
    for (size_t i = 0; i != 16; ++i) {
       differences += lookup_table[first_vector[i] ^ second_vector[i]];
    }
    return static_cast<int32_t>(differences);
}

int32_t hamming_b128_serial_builtinpopcnt(uint8_t const *first_vector, uint8_t const *second_vector) {
    int32_t differences = 0;
    for (size_t i = 0; i != 16; ++i) {
       differences += __builtin_popcount(first_vector[i] ^ second_vector[i]);
    }
    return differences;
}


int32_t hamming_b1536_serial(uint8_t const *first_vector, uint8_t const *second_vector) {
    int32_t differences = 0;
    for (size_t i = 0; i != 192; ++i) {
       differences += __builtin_popcount(first_vector[i] ^ second_vector[i]);
    }
    return differences;
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

static uint8_t popcnt_tmp[256];

// 1-to-256 vectors
// second_vector is a 256*128 matrix in a column-major layout
void hamming_b128_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    uint8x16_t popcnt_result[16];
    // Load initial values
    for (size_t i = 0; i < 16; ++i) { // 256 vectors at a time (probably overflows the registers)
        popcnt_result[i] = vdupq_n_u32(0);
    }
    for (size_t dim = 0; dim != 16; dim++){
        uint8x16_t a_vec = vdupq_n_u8(first_vector[dim]);
        for (size_t i = 0; i < 16; i++){
            uint8x16_t b_vec = vld1q_u8(second_vector);
            uint8x16_t xor_count_vec = vcntq_u8(veorq_u8(a_vec, b_vec));
            popcnt_result[i] = vaddq_u8(popcnt_result[i], xor_count_vec);
            second_vector += 16; // 256 values (16 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 16; i++) {
        vst1q_u8(&popcnt_tmp[i * 16], popcnt_result[i]);
    }
}

// 1-to-256 vectors
// second_vector is a 256*256 matrix in a column-major layout
void hamming_b256_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    uint8x16_t popcnt_result[16];
    // Load initial values
    for (size_t i = 0; i < 16; ++i) { // 256 vectors at a time (probably overflows the registers)
        popcnt_result[i] = vdupq_n_u32(0);
    }
    for (size_t dim = 0; dim != 32; dim++){
        uint8x16_t a_vec = vdupq_n_u8(first_vector[dim]);
        for (size_t i = 0; i < 16; i++){
            uint8x16_t b_vec = vld1q_u8(second_vector);
            uint8x16_t xor_count_vec = vcntq_u8(veorq_u8(a_vec, b_vec));
            popcnt_result[i] = vaddq_u8(popcnt_result[i], xor_count_vec);
            second_vector += 16; // 256 values (16 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 16; i++) {
        vst1q_u8(&popcnt_tmp[i * 16], popcnt_result[i]);
    }
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

int32_t hamming_b128_vpopcntq_wordbyword(uint8_t const *first_vector, uint8_t const *second_vector) {
    int32_t differences = 0;
    size_t i = 0;
    // In each 8-bit word we may have up to 8 differences.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the differences into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= 16) {
        uint8x16_t differences_cycle_vec = vdupq_n_u8(0);
        for (size_t cycle = 0; cycle < 31 && i + 16 <= 16; ++cycle, i += 16) {
            uint8x16_t a_vec = vld1q_u8(first_vector + i);
            uint8x16_t b_vec = vld1q_u8(second_vector + i);
            uint8x16_t xor_count_vec = vcntq_u8(veorq_u8(a_vec, b_vec));
            differences_cycle_vec = vaddq_u8(differences_cycle_vec, xor_count_vec);
        }
        differences += _simsimd_reduce_u8x16_neon(differences_cycle_vec);
    }
    // Handle the tail
    for (; i != 16; ++i) differences += simsimd_popcount_b8(first_vector[i] ^ second_vector[i]);
    return differences;
}

int32_t hamming_b1536_vpopcntq_wordbyword(uint8_t const *first_vector, uint8_t const *second_vector) {
    int32_t differences = 0;
    size_t i = 0;
    // In each 8-bit word we may have up to 8 differences.
    // So for up-to 31 cycles (31 * 16 = 496 word-dimensions = 3968 bits)
    // we can aggregate the differences into a `uint8x16_t` vector,
    // where each component will be up-to 255.
    while (i + 16 <= 192) {
        uint8x16_t differences_cycle_vec = vdupq_n_u8(0);
        for (size_t cycle = 0; cycle < 31 && i + 16 <= 192; ++cycle, i += 16) {
            uint8x16_t a_vec = vld1q_u8(first_vector + i);
            uint8x16_t b_vec = vld1q_u8(second_vector + i);
            uint8x16_t xor_count_vec = vcntq_u8(veorq_u8(a_vec, b_vec));
            differences_cycle_vec = vaddq_u8(differences_cycle_vec, xor_count_vec);
        }
        differences += _simsimd_reduce_u8x16_neon(differences_cycle_vec);
    }
    // Handle the tail
    for (; i != 192; ++i) differences += __builtin_popcount(first_vector[i] ^ second_vector[i]);
    return differences;
}

template <HammingKernel kernel=HAMMING_B128_VPOPCNTQ>
std::vector<KNNCandidate> hamming_standalone_heap(
    uint8_t const *first_vector,
    uint8_t const *second_vector,
    size_t num_queries,
    size_t num_vectors,
    size_t knn
) {

    uint8_t const * query = first_vector;
    std::priority_queue<KNNCandidate, std::vector<KNNCandidate>, VectorComparator> best_k;
    std::vector<KNNCandidate> result;
    result.resize(knn * num_queries);
    for (size_t i = 0; i < num_queries; i++) {
        uint8_t const * data = second_vector;

        for (size_t j = 0; j < num_vectors; j++) {
            uint32_t current_distance;
            if constexpr (kernel == HAMMING_B128_VPOPCNTQ){
                current_distance = hamming_b128_vpopcntq(query, data);
            } else if constexpr (kernel == HAMMING_B128_SERIAL_LUT) {
                current_distance = hamming_b128_serial_lut(query, data);
            } else {
                uint8x16_t c0 = vcntq_u8(veorq_u8(vld1q_u8(query), vld1q_u8(data)));
                current_distance = vaddvq_u8(c0);
            }

            if (best_k.size() < knn || best_k.top().distance > current_distance) {
                KNNCandidate e{};
                e.index = j;
                e.distance = current_distance;
                if (best_k.size() == knn) {
                    best_k.pop();
                }
                best_k.emplace(e);
            }

            data += 16;
        }
        for (size_t k = 0; k < knn && !best_k.empty(); ++k) {
            result[(i * 10) + (knn - k - 1)] = best_k.top();
            best_k.pop();
        }
        query += 16;
    }
    return result;
}

template <HammingKernel kernel=HAMMING_B128_VPOPCNTQ, int N_WORDS>
std::vector<KNNCandidate> hamming_standalone_partial_sort(
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
            uint32_t current_distance;
            if constexpr (kernel == HAMMING_B128_VPOPCNTQ){ // 128
                current_distance = hamming_b128_vpopcntq(query, data);
            } else if constexpr (kernel == HAMMING_B128_SERIAL_LUT) {
                current_distance = hamming_b128_serial_lut(query, data);
            } else if constexpr (kernel == HAMMING_B128_VPOPCNTQ_WORDBYWORD) {
                current_distance = hamming_b128_vpopcntq_wordbyword(query, data);
            } else if constexpr (kernel == HAMMING_B128_SERIAL_BUILTINPOPCNT) {
                current_distance = hamming_b128_serial_builtinpopcnt(query, data);
            } else if constexpr (kernel == HAMMING_B1536_SERIAL) { // 1536
                current_distance = hamming_b1536_serial(query, data);
            } else if constexpr (kernel == HAMMING_B1536_VPOPCNTQ_WORDBYWORD) {
                current_distance = hamming_b1536_vpopcntq_wordbyword(query, data);
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

template <HammingKernel kernel=HAMMING_B128_VPOPCNTQ_PDX, int N_WORDS=16, int PDX_BLOCK_SIZE=256>
std::vector<KNNCandidate> hamming_pdx_standalone_partial_sort(
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
            memset((void*) popcnt_tmp, 0, PDX_BLOCK_SIZE * sizeof(uint8_t));
            if constexpr (kernel == HAMMING_B128_VPOPCNTQ_PDX){
                hamming_b128_vpopcntq_pdx(query, data);
            } else if constexpr (kernel == HAMMING_B256_VPOPCNTQ_PDX){
                hamming_b256_vpopcntq_pdx(query, data);
            }
            // TODO: Ugly
            for (uint32_t z = 0; z < PDX_BLOCK_SIZE; ++z) {
                all_distances[global_offset].index = global_offset;
                all_distances[global_offset].distance = popcnt_tmp[z];
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

std::vector<KNNCandidate> hamming_standalone(
    HammingKernel kernel,
    uint8_t const *first_vector,
    uint8_t const *second_vector,
    size_t num_queries,
    size_t num_vectors,
    size_t knn
) {
    switch (kernel) {
    case HAMMING_B128_VPOPCNTQ:
        return hamming_standalone_partial_sort<HAMMING_B128_VPOPCNTQ, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
    case HAMMING_B128_SERIAL_LUT:
        return hamming_standalone_partial_sort<HAMMING_B128_SERIAL_LUT, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
    case HAMMING_B128_VPOPCNTQ_WORDBYWORD:
        return hamming_standalone_partial_sort<HAMMING_B128_VPOPCNTQ_WORDBYWORD, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
    case HAMMING_B128_SERIAL_BUILTINPOPCNT:
        return hamming_standalone_partial_sort<HAMMING_B128_SERIAL_BUILTINPOPCNT, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
    case HAMMING_B1536_SERIAL:
        return hamming_standalone_partial_sort<HAMMING_B1536_SERIAL, 192>(first_vector, second_vector, num_queries, num_vectors, knn);
    case HAMMING_B1536_VPOPCNTQ_WORDBYWORD:
        return hamming_standalone_partial_sort<HAMMING_B1536_VPOPCNTQ_WORDBYWORD, 192>(first_vector, second_vector, num_queries, num_vectors, knn);
    case HAMMING_B128_VPOPCNTQ_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B128_VPOPCNTQ_PDX, 16, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
    case HAMMING_B256_VPOPCNTQ_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B256_VPOPCNTQ_PDX, 32, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
    default:
        return hamming_standalone_partial_sort<HAMMING_B128_VPOPCNTQ, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
    }
}