#include <cstdint>
#include <cstddef>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <immintrin.h>

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
    // 256
    JACCARD_U64X4_C,
    JACCARD_B256_VPSHUFB_SAD,
    JACCARD_B256_VPOPCNTQ,
    JACCARD_B256_VPOPCNTQ_PDX,
    JACCARD_B256_VPSHUFB_PDX,
    // 1024
    JACCARD_U8X128_C,
    JACCARD_U64X16_C,
    JACCARD_B1024_VPOPCNTQ,
    JACCARD_B1024_VPSHUFB_SAD,
    JACCARD_B1024_VPSHUFB_DPB,
    JACCARD_U64X16_CSA3_C,
    JACCARD_U64X16_CSA15_CPP,
    JACCARD_B1024_VPOPCNTQ_PDX,
    // 1536
    JACCARD_U64X24_C,
    JACCARD_B1536_VPOPCNTQ,
    JACCARD_B1536_VPOPCNTQ_3CSA
};


///////////////////////////////
///////////////////////////////
// region: 256d kernels ///////
///////////////////////////////
///////////////////////////////

static uint8_t intersections_tmp[256];
static uint8_t unions_tmp[256];
static float distances_tmp[256];
// 1-to-256 vectors
// second_vector is a 256*256 matrix in a column-major layout
void jaccard_b256_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m256i intersections_result[8];
    __m256i unions_result[8];
    // Load initial values
    for (size_t i = 0; i < 8; ++i) { // 256 vectors at a time (using 8 registers)
        intersections_result[i] = _mm256_set1_epi8(0);
        unions_result[i] = _mm256_set1_epi8(0);
    }
    for (size_t dim = 0; dim != 32; dim++){
        __m256i first = _mm256_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 8; i++){
            __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));
            __m256i intersection = _mm256_popcnt_epi8(_mm256_and_epi64(first, second));
            __m256i union_ = _mm256_popcnt_epi8(_mm256_or_epi64(first, second));
            intersections_result[i] = _mm256_add_epi8(intersections_result[i], intersection);
            unions_result[i] = _mm256_add_epi8(unions_result[i], union_);
            second_vector += 32; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 8; i++) {
        _mm256_storeu_si256((__m256i *)(intersections_tmp + (i * 32)), intersections_result[i]);
        _mm256_storeu_si256((__m256i *)(unions_tmp + (i * 32)), unions_result[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = (unions_tmp[i] != 0) ? 1 - (float)intersections_tmp[i] / (float)unions_tmp[i] : 1.0f;
    }
}


// 1-to-256 vectors
// second_vector is a 256*256 matrix in a column-major layout
void jaccard_b256_vpshufb_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m256i intersections_result[8];
    __m256i unions_result[8];
    // Load initial values
    for (size_t i = 0; i < 8; ++i) { // 256 vectors at a time (using 8 registers)
        intersections_result[i] = _mm256_set1_epi8(0);
        unions_result[i] = _mm256_set1_epi8(0);
    }
    for (size_t dim = 0; dim != 32; dim++){
        __m256i first = _mm256_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 8; i++){
            __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));
            __m256i intersection = _mm256_popcnt_epi8(_mm256_and_epi64(first, second));
            __m256i union_ = _mm256_popcnt_epi8(_mm256_or_epi64(first, second));
            intersections_result[i] = _mm256_add_epi8(intersections_result[i], intersection);
            unions_result[i] = _mm256_add_epi8(unions_result[i], union_);
            second_vector += 32; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 8; i++) {
        _mm256_storeu_si256((__m256i *)(intersections_tmp + (i * 32)), intersections_result[i]);
        _mm256_storeu_si256((__m256i *)(unions_tmp + (i * 32)), unions_result[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = (unions_tmp[i] != 0) ? 1 - (float)intersections_tmp[i] / (float)unions_tmp[i] : 1.0f;
    }
}


float jaccard_u64x4_c(uint8_t const *a, uint8_t const *b) {
    uint32_t intersection = 0, union_ = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 4; ++i)
        intersection += __builtin_popcountll(a64[i] & b64[i]),
        union_ += __builtin_popcountll(a64[i] | b64[i]);
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}

inline uint64_t _mm256_reduce_add_epi64(__m256i vec) {
    __m128i lo128 = _mm256_castsi256_si128(vec);
    __m128i hi128 = _mm256_extracti128_si256(vec, 1);
    __m128i sum128 = _mm_add_epi64(lo128, hi128);
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i total = _mm_add_epi64(sum128, hi64);
    return uint64_t(_mm_cvtsi128_si64(total));
}

/*
 * Define the AVX2 variant using the `vpshufb` and `vpsadbw` instruction.
 * It resorts to cheaper byte-shuffling instructions, than population counts.
 * Source: https://github.com/CountOnes/hamming_weight/blob/1dd7554c0fc39e01c9d7fa54372fd4eccf458875/src/sse_jaccard_index.c#L17
 */
__attribute__((target("avx2,bmi2,avx")))
float jaccard_b256_vpshufb_sad(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m256i first = _mm256_loadu_epi8((__m256i const*)(first_vector));
    __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));

    __m256i intersection = _mm256_and_epi64(first, second);
    __m256i union_ = _mm256_or_epi64(first, second);

    __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i lookup = _mm256_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

    __m256i intersection_low = _mm256_and_si256(intersection, low_mask);
    __m256i intersection_high = _mm256_and_si256(_mm256_srli_epi16(intersection, 4), low_mask);
    __m256i union_low = _mm256_and_si256(union_, low_mask);
    __m256i union_high = _mm256_and_si256(_mm256_srli_epi16(union_, 4), low_mask);

    __m256i intersection_popcount = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, intersection_low),
        _mm256_shuffle_epi8(lookup, intersection_high));
    __m256i union_popcount = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, union_low),
        _mm256_shuffle_epi8(lookup, union_high));

    __m256i intersection_counts = _mm256_sad_epu8(intersection_popcount, _mm256_setzero_si256());
    __m256i union_counts = _mm256_sad_epu8(union_popcount, _mm256_setzero_si256());
    return 1.f - (_mm256_reduce_add_epi64(intersection_counts) + 1.f) / (_mm256_reduce_add_epi64(union_counts) + 1.f);
}

// Define the AVX-512 variant using the `vpopcntq` instruction.
// It's known to over-rely on port 5 on x86 CPUs, so the next `vpshufb` variant should be faster.
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b256_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m256i first = _mm256_loadu_epi8((__m256i const*)(first_vector));
    __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));

    __m256i intersection = _mm256_popcnt_epi64(_mm256_and_epi64(first, second));
    __m256i union_ = _mm256_popcnt_epi64(_mm256_or_epi64(first, second));
    return 1.f - (_mm256_reduce_add_epi64(intersection) + 1.f) / (_mm256_reduce_add_epi64(union_) + 1.f);
}



///////////////////////////////
///////////////////////////////
// region: 1024d kernels //////
///////////////////////////////
///////////////////////////////

static uint8_t intersections_tmp_1024_a[256];
static uint8_t intersections_tmp_1024_b[256];
static uint8_t intersections_tmp_1024_c[256];
static uint8_t intersections_tmp_1024_d[256];

static uint8_t unions_tmp_1024_a[256];
static uint8_t unions_tmp_1024_b[256];
static uint8_t unions_tmp_1024_c[256];
static uint8_t unions_tmp_1024_d[256];

// 1-to-256 vectors
// second_vector is a 256*1024 matrix in a column-major layout
// Processing the 1024 dimensions in 4 groups of 32 words each to not overflow the uint8_t accumulators
void jaccard_b1024_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m256i intersections_result_a[8];
    __m256i intersections_result_b[8];
    __m256i intersections_result_c[8];
    __m256i intersections_result_d[8];
    __m256i unions_result_a[8];
    __m256i unions_result_b[8];
    __m256i unions_result_c[8];
    __m256i unions_result_d[8];
    for (size_t i = 0; i < 8; ++i) { // 256 vectors at a time (using 8 registers)
        intersections_result_a[i] = _mm256_set1_epi8(0);
        intersections_result_b[i] = _mm256_set1_epi8(0);
        intersections_result_c[i] = _mm256_set1_epi8(0);
        intersections_result_d[i] = _mm256_set1_epi8(0);
        unions_result_a[i] = _mm256_set1_epi8(0);
        unions_result_b[i] = _mm256_set1_epi8(0);
        unions_result_c[i] = _mm256_set1_epi8(0);
        unions_result_d[i] = _mm256_set1_epi8(0);
    }
    // Word 0 to 31
    for (size_t dim = 0; dim != 32; dim++){
        __m256i first = _mm256_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 8; i++){
            __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));
            __m256i intersection = _mm256_popcnt_epi8(_mm256_and_epi64(first, second));
            __m256i union_ = _mm256_popcnt_epi8(_mm256_or_epi64(first, second));
            intersections_result_a[i] = _mm256_add_epi8(intersections_result_a[i], intersection);
            unions_result_a[i] = _mm256_add_epi8(unions_result_a[i], union_);
            second_vector += 32; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // Word 32 to 63
    for (size_t dim = 32; dim != 64; dim++){
        __m256i first = _mm256_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 8; i++){
            __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));
            __m256i intersection = _mm256_popcnt_epi8(_mm256_and_epi64(first, second));
            __m256i union_ = _mm256_popcnt_epi8(_mm256_or_epi64(first, second));
            intersections_result_b[i] = _mm256_add_epi8(intersections_result_b[i], intersection);
            unions_result_b[i] = _mm256_add_epi8(unions_result_b[i], union_);
            second_vector += 32; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // Word 64 to 95
    for (size_t dim = 64; dim != 96; dim++){
        __m256i first = _mm256_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 8; i++){
            __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));
            __m256i intersection = _mm256_popcnt_epi8(_mm256_and_epi64(first, second));
            __m256i union_ = _mm256_popcnt_epi8(_mm256_or_epi64(first, second));
            intersections_result_c[i] = _mm256_add_epi8(intersections_result_c[i], intersection);
            unions_result_c[i] = _mm256_add_epi8(unions_result_c[i], union_);
            second_vector += 32; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // Word 96 to 127
    for (size_t dim = 96; dim != 128; dim++){
        __m256i first = _mm256_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 8; i++){
            __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));
            __m256i intersection = _mm256_popcnt_epi8(_mm256_and_epi64(first, second));
            __m256i union_ = _mm256_popcnt_epi8(_mm256_or_epi64(first, second));
            intersections_result_d[i] = _mm256_add_epi8(intersections_result_d[i], intersection);
            unions_result_d[i] = _mm256_add_epi8(unions_result_d[i], union_);
            second_vector += 32; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 8; i++) {
        _mm256_storeu_si256((__m256i *)(intersections_tmp_1024_a + (i * 32)), intersections_result_a[i]);
        _mm256_storeu_si256((__m256i *)(unions_tmp_1024_a + (i * 32)), unions_result_a[i]);
        _mm256_storeu_si256((__m256i *)(intersections_tmp_1024_b + (i * 32)), intersections_result_b[i]);
        _mm256_storeu_si256((__m256i *)(unions_tmp_1024_b + (i * 32)), unions_result_b[i]);
        _mm256_storeu_si256((__m256i *)(intersections_tmp_1024_c + (i * 32)), intersections_result_c[i]);
        _mm256_storeu_si256((__m256i *)(unions_tmp_1024_c + (i * 32)), unions_result_c[i]);
        _mm256_storeu_si256((__m256i *)(intersections_tmp_1024_d + (i * 32)), intersections_result_d[i]);
        _mm256_storeu_si256((__m256i *)(unions_tmp_1024_d + (i * 32)), unions_result_d[i]);
    }
    // TODO: Probably can use SIMD for the pairwise sum of the 4 groups
    for (size_t i = 0; i < 256; i++){
        float intersection = intersections_tmp_1024_a[i] + intersections_tmp_1024_b[i] + intersections_tmp_1024_c[i] + intersections_tmp_1024_d[i];
        float union_ = unions_tmp_1024_a[i] + unions_tmp_1024_b[i] + unions_tmp_1024_c[i] + unions_tmp_1024_d[i];
        distances_tmp[i] = (union_ != 0) ? 1 - intersection / union_ : 1.0f;
    }
}

float jaccard_u8x128_c(uint8_t const *a, uint8_t const *b) {
    uint32_t intersection = 0, union_ = 0;
#pragma unroll
    for (size_t i = 0; i != 128; ++i)
        intersection += __builtin_popcount(a[i] & b[i]),
        union_ += __builtin_popcount(a[i] | b[i]);
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}

float jaccard_u64x16_c(uint8_t const *a, uint8_t const *b) {
    uint32_t intersection = 0, union_ = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 16; ++i)
        intersection += __builtin_popcountll(a64[i] & b64[i]),
        union_ += __builtin_popcountll(a64[i] | b64[i]);
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}

// Define the AVX-512 variant using the `vpopcntq` instruction.
// It's known to over-rely on port 5 on x86 CPUs, so the next `vpshufb` variant should be faster.
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b1024_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i first_end = _mm512_loadu_si512((__m512i const*)(first_vector + 64));
    __m512i second_start = _mm512_loadu_si512((__m512i const*)(second_vector));
    __m512i second_end = _mm512_loadu_si512((__m512i const*)(second_vector + 64));

    __m512i intersection_start = _mm512_popcnt_epi64(_mm512_and_epi64(first_start, second_start));
    __m512i intersection_end = _mm512_popcnt_epi64(_mm512_and_epi64(first_end, second_end));
    __m512i union_start = _mm512_popcnt_epi64(_mm512_or_epi64(first_start, second_start));
    __m512i union_end = _mm512_popcnt_epi64(_mm512_or_epi64(first_end, second_end));

    __m512i intersection = _mm512_add_epi64(intersection_start, intersection_end);
    __m512i union_ = _mm512_add_epi64(union_start, union_end);
    return 1.f - (_mm512_reduce_add_epi64(intersection) + 1.f) / (_mm512_reduce_add_epi64(union_) + 1.f);
}

// Define the AVX-512 variant using the `vpshufb` and `vpsadbw` instruction.
// It resorts to cheaper byte-shuffling instructions, than population counts.
// Source: https://github.com/CountOnes/hamming_weight/blob/1dd7554c0fc39e01c9d7fa54372fd4eccf458875/src/sse_jaccard_index.c#L17
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b1024_vpshufb_sad(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i first_end = _mm512_loadu_si512((__m512i const*)(first_vector + 64));
    __m512i second_start = _mm512_loadu_si512((__m512i const*)(second_vector));
    __m512i second_end = _mm512_loadu_si512((__m512i const*)(second_vector + 64));

    __m512i intersection_start = _mm512_and_epi64(first_start, second_start);
    __m512i intersection_end = _mm512_and_epi64(first_end, second_end);
    __m512i union_start = _mm512_or_epi64(first_start, second_start);
    __m512i union_end = _mm512_or_epi64(first_end, second_end);

    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

    __m512i intersection_start_low = _mm512_and_si512(intersection_start, low_mask);
    __m512i intersection_start_high = _mm512_and_si512(_mm512_srli_epi16(intersection_start, 4), low_mask);
    __m512i intersection_end_low = _mm512_and_si512(intersection_end, low_mask);
    __m512i intersection_end_high = _mm512_and_si512(_mm512_srli_epi16(intersection_end, 4), low_mask);

    __m512i union_start_low = _mm512_and_si512(union_start, low_mask);
    __m512i union_start_high = _mm512_and_si512(_mm512_srli_epi16(union_start, 4), low_mask);
    __m512i union_end_low = _mm512_and_si512(union_end, low_mask);
    __m512i union_end_high = _mm512_and_si512(_mm512_srli_epi16(union_end, 4), low_mask);

    __m512i intersection_start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, intersection_start_low),
        _mm512_shuffle_epi8(lookup, intersection_start_high));
    __m512i intersection_end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, intersection_end_low),
        _mm512_shuffle_epi8(lookup, intersection_end_high));
    __m512i union_start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, union_start_low),
        _mm512_shuffle_epi8(lookup, union_start_high));
    __m512i union_end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, union_end_low),
        _mm512_shuffle_epi8(lookup, union_end_high));

    __m512i intersection = _mm512_add_epi64(
        _mm512_sad_epu8(intersection_start_popcount, _mm512_setzero_si512()),
        _mm512_sad_epu8(intersection_end_popcount, _mm512_setzero_si512()));
    __m512i union_ = _mm512_add_epi64(
        _mm512_sad_epu8(union_start_popcount, _mm512_setzero_si512()),
        _mm512_sad_epu8(union_end_popcount, _mm512_setzero_si512()));

    return 1.f - (_mm512_reduce_add_epi64(intersection) + 1.f) / (_mm512_reduce_add_epi64(union_) + 1.f);
}

// # Define the AVX-512 variant using the `vpshufb` and `vpdpbusd` instruction.
// # It replaces the horizontal addition with a dot-product.
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b1024_vpshufb_dpb(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i first_end = _mm512_loadu_si512((__m512i const*)(first_vector + 64));
    __m512i second_start = _mm512_loadu_si512((__m512i const*)(second_vector));
    __m512i second_end = _mm512_loadu_si512((__m512i const*)(second_vector + 64));

    __m512i intersection_start = _mm512_and_epi64(first_start, second_start);
    __m512i intersection_end = _mm512_and_epi64(first_end, second_end);
    __m512i union_start = _mm512_or_epi64(first_start, second_start);
    __m512i union_end = _mm512_or_epi64(first_end, second_end);

    __m512i ones = _mm512_set1_epi8(1);
    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

    __m512i intersection_start_low = _mm512_and_si512(intersection_start, low_mask);
    __m512i intersection_start_high = _mm512_and_si512(_mm512_srli_epi16(intersection_start, 4), low_mask);
    __m512i intersection_end_low = _mm512_and_si512(intersection_end, low_mask);
    __m512i intersection_end_high = _mm512_and_si512(_mm512_srli_epi16(intersection_end, 4), low_mask);

    __m512i union_start_low = _mm512_and_si512(union_start, low_mask);
    __m512i union_start_high = _mm512_and_si512(_mm512_srli_epi16(union_start, 4), low_mask);
    __m512i union_end_low = _mm512_and_si512(union_end, low_mask);
    __m512i union_end_high = _mm512_and_si512(_mm512_srli_epi16(union_end, 4), low_mask);

    __m512i intersection_start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, intersection_start_low),
        _mm512_shuffle_epi8(lookup, intersection_start_high));
    __m512i intersection_end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, intersection_end_low),
        _mm512_shuffle_epi8(lookup, intersection_end_high));
    __m512i union_start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, union_start_low),
        _mm512_shuffle_epi8(lookup, union_start_high));
    __m512i union_end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, union_end_low),
        _mm512_shuffle_epi8(lookup, union_end_high));

    __m512i intersection = _mm512_dpbusd_epi32(_mm512_setzero_si512(), intersection_start_popcount, ones);
    intersection = _mm512_dpbusd_epi32(intersection, intersection_end_popcount, ones);

    __m512i union_ = _mm512_dpbusd_epi32(_mm512_setzero_si512(), union_start_popcount, ones);
    union_ = _mm512_dpbusd_epi32(union_, union_end_popcount, ones);

    return 1.f - (_mm512_reduce_add_epi32(intersection) + 1.f) / (_mm512_reduce_add_epi32(union_) + 1.f);
}

// Harley-Seal transformation and Odd-Major-style Carry-Save-Adders can be used to replace
// several population counts with a few bitwise operations and one `popcount`, which can help
// lift the pressure on the CPU ports.
inline int popcount_csa3(uint64_t x, uint64_t y, uint64_t z) {
    uint64_t odd  = (x ^ y) ^ z;
    uint64_t major = ((x ^ y) & z) | (x & y);
    return 2 * __builtin_popcountll(major) + __builtin_popcountll(odd);
}

float jaccard_u64x16_csa3_c(uint8_t const *a, uint8_t const *b) {
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;

    int intersection =
        popcount_csa3(a64[0] & b64[0], a64[1] & b64[1], a64[2] & b64[2]) +
        popcount_csa3(a64[3] & b64[3], a64[4] & b64[4], a64[5] & b64[5]) +
        popcount_csa3(a64[6] & b64[6], a64[7] & b64[7], a64[8] & b64[8]) +
        popcount_csa3(a64[9] & b64[9], a64[10] & b64[10], a64[11] & b64[11]) +
        popcount_csa3(a64[12] & b64[12], a64[13] & b64[13], a64[14] & b64[14]) +
        __builtin_popcountll(a64[15] & b64[15]);


    int union_ =
        popcount_csa3(a64[0] | b64[0], a64[1] | b64[1], a64[2] | b64[2]) +
        popcount_csa3(a64[3] | b64[3], a64[4] | b64[4], a64[5] | b64[5]) +
        popcount_csa3(a64[6] | b64[6], a64[7] | b64[7], a64[8] | b64[8]) +
        popcount_csa3(a64[9] | b64[9], a64[10] | b64[10], a64[11] | b64[11]) +
        popcount_csa3(a64[12] | b64[12], a64[13] | b64[13], a64[14] | b64[14]) +
        __builtin_popcountll(a64[15] | b64[15]);

    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}

// That CSA can be scaled further to fold 15 population counts into 4.
// It's a bit more complex and for readability we will use C++ tuple unpacking:
struct uint64_csa_t {
   uint64_t ones;
   uint64_t twos;
};

constexpr uint64_csa_t csa(uint64_t x, uint64_t y, uint64_t z) {
    uint64_t odd  = (x ^ y) ^ z;
    uint64_t major = ((x ^ y) & z) | (x & y);
    return {odd, major};
}

constexpr int popcount_csa15(
    uint64_t x1, uint64_t x2, uint64_t x3,
    uint64_t x4, uint64_t x5, uint64_t x6, uint64_t x7,
    uint64_t x8, uint64_t x9, uint64_t x10, uint64_t x11,
    uint64_t x12, uint64_t x13, uint64_t x14, uint64_t x15) {

    auto [one1, two1] = csa(x1,  x2,  x3);
    auto [one2, two2] = csa(x4,  x5,  x6);
    auto [one3, two3] = csa(x7,  x8,  x9);
    auto [one4, two4] = csa(x10, x11, x12);
    auto [one5, two5] = csa(x13, x14, x15);

    // Level‐2: fold the five “one” terms down to two + a final “ones”
    auto [one6, two6] = csa(one1, one2, one3);
    auto [ones, two7] = csa(one4, one5, one6);

    // Level‐2: fold the five “two” terms down to two + a “twos”
    auto [two8, four1] = csa(two1, two2, two3);
    auto [two9, four2] = csa(two4, two5, two6);
    auto [twos, four3] = csa(two7, two8, two9);

    // Level‐3: fold the three “four” terms down to one “four” + one “eight”
    auto [four, eight] = csa(four1, four2, four3);

    // Now you have a full 4-bit per-bit‐position counter in (ones, twos, four, eight).
    int count_ones  = __builtin_popcountll(ones);
    int count_twos  = __builtin_popcountll(twos);
    int count_four  = __builtin_popcountll(four);
    int count_eight = __builtin_popcountll(eight);
    return count_ones + 2 * count_twos + 4 * count_four + 8 * count_eight;
}

float jaccard_u64x16_csa15_cpp(uint8_t const *a, uint8_t const *b) {
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;

    int intersection = popcount_csa15(
        a64[0] & b64[0], a64[1] & b64[1], a64[2] & b64[2], a64[3] & b64[3],
        a64[4] & b64[4], a64[5] & b64[5], a64[6] & b64[6], a64[7] & b64[7],
        a64[8] & b64[8], a64[9] & b64[9], a64[10] & b64[10], a64[11] & b64[11],
        a64[12] & b64[12], a64[13] & b64[13], a64[14] & b64[14]) +
        __builtin_popcountll(a64[15] & b64[15]);

    int union_ = popcount_csa15(
        a64[0] | b64[0], a64[1] | b64[1], a64[2] | b64[2], a64[3] | b64[3],
        a64[4] | b64[4], a64[5] | b64[5], a64[6] | b64[6], a64[7] | b64[7],
        a64[8] | b64[8], a64[9] | b64[9], a64[10] | b64[10], a64[11] | b64[11],
        a64[12] | b64[12], a64[13] | b64[13], a64[14] | b64[14]) +
        __builtin_popcountll(a64[15] | b64[15]);

    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}


///////////////////////////////
///////////////////////////////
// region: 1536d kernels //////
///////////////////////////////
///////////////////////////////

float jaccard_u64x24_c(uint8_t const *a, uint8_t const *b) {
    uint32_t intersection = 0, union_ = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 24; ++i)
        intersection += __builtin_popcountll(a64[i] & b64[i]),
        union_ += __builtin_popcountll(a64[i] | b64[i]);
    return 1.f - (intersection + 1.f) / (union_ + 1.f); // ! Avoid division by zero
}


// Define the AVX-512 variant using the `vpopcntq` instruction for 1536d vectors
// It's known to over-rely on port 5 on x86 CPUs, so the next `vpshufb` variant should be faster.
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b1536_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i first0 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 0));
    __m512i first1 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 1));
    __m512i first2 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 2));
    __m512i second0 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 0));
    __m512i second1 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 1));
    __m512i second2 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 2));

    __m512i intersection0 = _mm512_popcnt_epi64(_mm512_and_epi64(first0, second0));
    __m512i intersection1 = _mm512_popcnt_epi64(_mm512_and_epi64(first1, second1));
    __m512i intersection2 = _mm512_popcnt_epi64(_mm512_and_epi64(first2, second2));
    __m512i union0 = _mm512_popcnt_epi64(_mm512_or_epi64(first0, second0));
    __m512i union1 = _mm512_popcnt_epi64(_mm512_or_epi64(first1, second1));
    __m512i union2 = _mm512_popcnt_epi64(_mm512_or_epi64(first2, second2));

    __m512i intersection = _mm512_add_epi64(_mm512_add_epi64(intersection0, intersection1), intersection2);
    __m512i union_ = _mm512_add_epi64(_mm512_add_epi64(union0, union1), union2);
    return 1.f - (_mm512_reduce_add_epi64(intersection) + 1.f) / (_mm512_reduce_add_epi64(union_) + 1.f);
}

// Define the AVX-512 variant, combining Harley-Seal transform to reduce the number
// of population counts for the 1536-dimensional case to the 1024-dimensional case,
// at the cost of several ternary bitwise operations.
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b1536_vpopcntq_3csa(uint8_t const *first_vector, uint8_t const *second_vector) {

    __m512i first0 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 0));
    __m512i first1 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 1));
    __m512i first2 = _mm512_loadu_si512((__m512i const*)(first_vector + 64 * 2));
    __m512i second0 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 0));
    __m512i second1 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 1));
    __m512i second2 = _mm512_loadu_si512((__m512i const*)(second_vector + 64 * 2));

    __m512i intersection0 = _mm512_and_epi64(first0, second0);
    __m512i intersection1 = _mm512_and_epi64(first1, second1);
    __m512i intersection2 = _mm512_and_epi64(first2, second2);
    __m512i union0 = _mm512_or_epi64(first0, second0);
    __m512i union1 = _mm512_or_epi64(first1, second1);
    __m512i union2 = _mm512_or_epi64(first2, second2);

    __m512i intersection_odd = _mm512_ternarylogic_epi64(
        intersection0, intersection1, intersection2,
        (_MM_TERNLOG_A ^ _MM_TERNLOG_B ^ _MM_TERNLOG_C));
    __m512i intersection_major = _mm512_ternarylogic_epi64(
        intersection0, intersection1, intersection2,
        ((_MM_TERNLOG_A ^ _MM_TERNLOG_B) & _MM_TERNLOG_C) | (_MM_TERNLOG_A & _MM_TERNLOG_B));
    __m512i union_odd = _mm512_ternarylogic_epi64(
        union0, union1, union2,
        (_MM_TERNLOG_A ^ _MM_TERNLOG_B ^ _MM_TERNLOG_C));
    __m512i union_major = _mm512_ternarylogic_epi64(
        union0, union1, union2,
        ((_MM_TERNLOG_A ^ _MM_TERNLOG_B) & _MM_TERNLOG_C) | (_MM_TERNLOG_A & _MM_TERNLOG_B));

    __m512i intersection_odd_count = _mm512_popcnt_epi64(intersection_odd);
    __m512i intersection_major_count = _mm512_popcnt_epi64(intersection_major);
    __m512i union_odd_count = _mm512_popcnt_epi64(union_odd);
    __m512i union_major_count = _mm512_popcnt_epi64(union_major);

    // Shift left the majors by 1 to multiply by 2
    __m512i intersection = _mm512_add_epi64(_mm512_slli_epi64(intersection_major_count, 1), intersection_odd_count);
    __m512i union_ = _mm512_add_epi64(_mm512_slli_epi64(union_major_count, 1), union_odd_count);
    return 1.f - (_mm512_reduce_add_epi64(intersection) + 1.f) / (_mm512_reduce_add_epi64(union_) + 1.f);
}


template <JaccardKernel kernel=JACCARD_B1024_VPOPCNTQ, int N_WORDS=16>
std::vector<KNNCandidate> jaccard_standalone_partial_sort(
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
        for (size_t j = 0; j < num_vectors; ++j) {

            float current_distance;
            if constexpr (kernel == JACCARD_U64X4_C){ // 256
                current_distance = jaccard_u64x4_c(query, data);
            } else if constexpr (kernel == JACCARD_B256_VPSHUFB_SAD) {
                current_distance = jaccard_b256_vpshufb_sad(query, data);
            } else if constexpr (kernel == JACCARD_B256_VPOPCNTQ) {
                current_distance = jaccard_b256_vpopcntq(query, data);

            } else if constexpr (kernel == JACCARD_U8X128_C) { // 1024
                current_distance = jaccard_u8x128_c(query, data);
            } else if constexpr (kernel == JACCARD_U64X16_C) {
                current_distance = jaccard_u64x16_c(query, data);
            } else if constexpr (kernel == JACCARD_B1024_VPOPCNTQ) {
                current_distance = jaccard_b1024_vpopcntq(query, data);
            } else if constexpr (kernel == JACCARD_B1024_VPSHUFB_SAD) {
                current_distance = jaccard_b1024_vpshufb_sad(query, data);
            } else if constexpr (kernel == JACCARD_B1024_VPSHUFB_DPB) {
                current_distance = jaccard_b1024_vpshufb_dpb(query, data);
            } else if constexpr (kernel == JACCARD_U64X16_CSA3_C) {
                current_distance = jaccard_u64x16_csa3_c(query, data);
            } else if constexpr (kernel == JACCARD_U64X16_CSA15_CPP) {
                current_distance = jaccard_u64x16_csa15_cpp(query, data);

            } else if constexpr (kernel == JACCARD_U64X24_C) { // 1536
                current_distance = jaccard_u64x24_c(query, data);
            } else if constexpr (kernel == JACCARD_B1536_VPOPCNTQ) {
                current_distance = jaccard_b1536_vpopcntq(query, data);
            } else if constexpr (kernel == JACCARD_B1536_VPOPCNTQ_3CSA) {
                current_distance = jaccard_b1536_vpopcntq_3csa(query, data);
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

template <JaccardKernel kernel=JACCARD_B256_VPOPCNTQ_PDX, int N_WORDS=32, int PDX_BLOCK_SIZE=256>
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
            if constexpr (kernel == JACCARD_B256_VPOPCNTQ_PDX){
                jaccard_b256_vpopcntq_pdx(query, data);
            } else if constexpr (kernel == JACCARD_B1024_VPOPCNTQ_PDX){
                jaccard_b1024_vpopcntq_pdx(query, data);
            } else if constexpr (kernel == JACCARD_B256_VPSHUFB_PDX){
                current_distance = jaccard_b256_vpshufb_pdx(query, data);
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
        case JACCARD_U64X4_C: // 256
            return jaccard_standalone_partial_sort<JACCARD_U64X4_C, 32>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B256_VPSHUFB_SAD:
            return jaccard_standalone_partial_sort<JACCARD_B256_VPSHUFB_SAD, 32>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B256_VPOPCNTQ:
            return jaccard_standalone_partial_sort<JACCARD_B256_VPOPCNTQ, 32>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B256_VPOPCNTQ_PDX:
            return jaccard_pdx_standalone_partial_sort<JACCARD_B256_VPOPCNTQ_PDX, 32, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B256_VPSHUFB_PDX:
            return jaccard_pdx_standalone_partial_sort<JACCARD_B256_VPSHUFB_PDX, 32, 256>(first_vector, second_vector, num_queries, num_vectors, knn);

        case JACCARD_U8X128_C: // 1024
            return jaccard_standalone_partial_sort<JACCARD_U8X128_C, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_U64X16_C:
            return jaccard_standalone_partial_sort<JACCARD_U64X16_C, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B1024_VPOPCNTQ:
            return jaccard_standalone_partial_sort<JACCARD_B1024_VPOPCNTQ, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B1024_VPSHUFB_SAD:
            return jaccard_standalone_partial_sort<JACCARD_B1024_VPSHUFB_SAD, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B1024_VPSHUFB_DPB:
            return jaccard_standalone_partial_sort<JACCARD_B1024_VPSHUFB_DPB, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_U64X16_CSA3_C:
            return jaccard_standalone_partial_sort<JACCARD_U64X16_CSA3_C, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_U64X16_CSA15_CPP:
            return jaccard_standalone_partial_sort<JACCARD_U64X16_CSA15_CPP, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B1024_VPOPCNTQ_PDX:
            return jaccard_pdx_standalone_partial_sort<JACCARD_B1024_VPOPCNTQ_PDX, 128, 256>(first_vector, second_vector, num_queries, num_vectors, knn);

        case JACCARD_U64X24_C: // 1536
            return jaccard_standalone_partial_sort<JACCARD_U64X24_C, 192>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B1536_VPOPCNTQ:
            return jaccard_standalone_partial_sort<JACCARD_B1536_VPOPCNTQ, 192>(first_vector, second_vector, num_queries, num_vectors, knn);
        case JACCARD_B1536_VPOPCNTQ_3CSA:
            return jaccard_standalone_partial_sort<JACCARD_B1536_VPOPCNTQ_3CSA, 192>(first_vector, second_vector, num_queries, num_vectors, knn);
        default:
            return jaccard_standalone_partial_sort<JACCARD_U64X4_C, 32>(first_vector, second_vector, num_queries, num_vectors, knn);
    }
}