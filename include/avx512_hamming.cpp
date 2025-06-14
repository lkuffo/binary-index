#include <cstdint>
#include <cstddef>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <immintrin.h>

#include "jaccard_nibble_luts.h"
#include "jaccard_nibble_luts_avx2.h"
#include "jaccard_nibble_luts_avx512.h"

struct KNNCandidate {
    uint32_t index;
    float distance;
};

struct VectorComparator {
    bool operator() (const KNNCandidate& a, const KNNCandidate& b) {
        return a.distance < b.distance;
    }
};

enum HammingKernel {
    // 128
    HAMMING_U64X2_C,
    HAMMING_B128_VPOPCNTQ,
    HAMMING_B128_VPSHUFB_SAD,
    HAMMING_B128_VPOPCNTQ_PDX,
    HAMMING_B128_VPSHUFB_PDX,
    // 256
    HAMMING_U64X4_C,
    HAMMING_B256_VPOPCNTQ,
    HAMMING_B256_VPSHUFB_SAD,
    HAMMING_B256_VPOPCNTQ_PDX,
    HAMMING_B256_VPSHUFB_PDX,
    HAMMING_B256_XORLUT_PDX,
    // 512
    HAMMING_U64X8_C,
    HAMMING_B512_VPSHUFB_SAD,
    HAMMING_B512_VPOPCNTQ,
    HAMMING_B512_VPOPCNTQ_PDX,
    HAMMING_B512_VPSHUFB_PDX,
    // 1024
    HAMMING_U8X128_C,
    HAMMING_U64X16_C,
    HAMMING_B1024_VPOPCNTQ,
    HAMMING_B1024_VPSHUFB_SAD,
    HAMMING_U64X16_CSA3_C,
    HAMMING_U64X16_CSA15_CPP,
    HAMMING_B1024_VPOPCNTQ_PDX,
    HAMMING_B1024_VPSHUFB_PDX,
};


///////////////////////////////
///////////////////////////////
// region: 128d kernels ///////
///////////////////////////////
///////////////////////////////

static uint8_t popcnt_tmp[256];
static uint32_t distances_tmp[256];

// 1-to-256 vectors
// second_vector is a 256*256 matrix in a column-major layout
inline void hamming_b128_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i popcnt_result[4];
    // Load initial values
    for (size_t i = 0; i < 4; ++i) { // 256 vectors at a time (using 8 registers)
        popcnt_result[i] = _mm512_set1_epi8(0);
    }
    for (size_t dim = 0; dim != 16; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 4; i++){
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i popcnt_ = _mm512_popcnt_epi8(_mm512_xor_epi64(first, second));
            popcnt_result[i] = _mm512_add_epi8(popcnt_result[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 4; i++) {
        _mm512_storeu_si512(popcnt_tmp + (i * 64), popcnt_result[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = popcnt_tmp[i];
    }
}

// 1-to-256 vectors
// second_vector is a 256*256 matrix in a column-major layout
inline void hamming_b128_vpshufb_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i popcnt_result[4];
    __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    // Load initial values
    for (size_t i = 0; i < 4; ++i) { // 256 vectors at a time (using 8 registers)
        popcnt_result[i] = _mm512_setzero_si512();
    }
    for (size_t dim = 0; dim != 16; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);

        for (size_t i = 0; i < 4; i++){ // 256 uint8_t values
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i xor_ = _mm512_xor_epi64(first, second);

            // Getting nibbles from data
            __m512i second_low = _mm512_and_si512(xor_, low_mask);
            __m512i second_high = _mm512_and_si512(_mm512_srli_epi16(xor_, 4), low_mask);

            __m512i popcnt_ = _mm512_add_epi8(
                _mm512_shuffle_epi8(lookup, second_low),
                _mm512_shuffle_epi8(lookup, second_low)
            );

            popcnt_result[i] = _mm512_add_epi8(popcnt_result[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 4; i++) {
        _mm512_storeu_si512(popcnt_tmp + (i * 64), popcnt_result[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = popcnt_tmp[i];
    }
}

inline float hamming_u64x2_c(uint8_t const *a, uint8_t const *b) {
    uint32_t popcnt = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 2; ++i)
        popcnt += __builtin_popcountll(a64[i] ^ b64[i]);
    return popcnt;
}

inline uint64_t _mm256_reduce_add_epi64(__m256i vec) {
    __m128i lo128 = _mm256_castsi256_si128(vec);
    __m128i hi128 = _mm256_extracti128_si256(vec, 1);
    __m128i sum128 = _mm_add_epi64(lo128, hi128);
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i total = _mm_add_epi64(sum128, hi64);
    return uint64_t(_mm_cvtsi128_si64(total));
}

inline uint64_t _mm128_reduce_add_epi64(__m128i vec) {
    __m128i hi64 = _mm_unpackhi_epi64(vec, vec);
    __m128i sum = _mm_add_epi64(vec, hi64);
    return static_cast<uint64_t>(_mm_cvtsi128_si64(sum));
}

/*
 * Define the AVX2 variant using the `vpshufb` and `vpsadbw` instruction.
 * It resorts to cheaper byte-shuffling instructions, than population counts.
 * Source: https://github.com/CountOnes/hamming_weight/blob/1dd7554c0fc39e01c9d7fa54372fd4eccf458875/src/sse_jaccard_index.c#L17
 */
__attribute__((target("avx2,bmi2,avx")))
inline float hamming_b128_vpshufb_sad(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m128i first = _mm_loadu_epi8((__m128i const*)(first_vector));
    __m128i second = _mm_loadu_epi8((__m128i const*)(second_vector));

    __m128i xor_ = _mm_xor_epi64(first, second);

    __m128i low_mask = _mm_set1_epi8(0x0f);
    __m128i lookup = _mm_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

    __m128i xor_low = _mm_and_si128(xor_, low_mask);
    __m128i xor_high = _mm_and_si128(_mm_srli_epi16(xor_, 4), low_mask);

    __m128i popcnt_ = _mm_add_epi8(
        _mm_shuffle_epi8(lookup, xor_low),
        _mm_shuffle_epi8(lookup, xor_high));

    __m128i popcnt = _mm_sad_epu8(popcnt_, _mm_setzero_si128());
    return _mm128_reduce_add_epi64(popcnt);
}


// Define the AVX-512 variant using the `vpopcntq` instruction.
// It's known to over-rely on port 5 on x86 CPUs, so the next `vpshufb` variant should be faster.
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
inline float hamming_b128_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m128i first = _mm_loadu_epi8((__m128i const*)(first_vector));
    __m128i second = _mm_loadu_epi8((__m128i const*)(second_vector));

    __m128i popcnt = _mm_popcnt_epi64(_mm_xor_epi64(first, second));
    return _mm128_reduce_add_epi64(popcnt);
}


///////////////////////////////
///////////////////////////////
// region: 256d kernels ///////
///////////////////////////////
///////////////////////////////

// 1-to-256 vectors
// second_vector is a 256*256 matrix in a column-major layout
inline void hamming_b256_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i popcnt_result[4];
    // Load initial values
    for (size_t i = 0; i < 4; ++i) { // 256 vectors at a time (using 8 registers)
        popcnt_result[i] = _mm512_set1_epi8(0);
    }
    for (size_t dim = 0; dim != 32; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 4; i++){
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i popcnt_ = _mm512_popcnt_epi8(_mm512_xor_epi64(first, second));
            popcnt_result[i] = _mm512_add_epi8(popcnt_result[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 4; i++) {
        _mm512_storeu_si512(popcnt_tmp + (i * 64), popcnt_result[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = popcnt_tmp[i];
    }
}

// 1-to-256 vectors
// second_vector is a 256*256 matrix in a column-major layout
inline void hamming_b256_vpshufb_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i popcnt_result[4];
    __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    // Load initial values
    for (size_t i = 0; i < 4; ++i) { // 256 vectors at a time (using 8 registers)
        popcnt_result[i] = _mm512_setzero_si512();
    }
    for (size_t dim = 0; dim != 32; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);

        for (size_t i = 0; i < 4; i++){ // 256 uint8_t values
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i xor_ = _mm512_xor_epi64(first, second);

            // Getting nibbles from data
            __m512i second_low = _mm512_and_si512(xor_, low_mask);
            __m512i second_high = _mm512_and_si512(_mm512_srli_epi16(xor_, 4), low_mask);

            __m512i popcnt_ = _mm512_add_epi8(
                _mm512_shuffle_epi8(lookup, second_low),
                _mm512_shuffle_epi8(lookup, second_low)
            );

            popcnt_result[i] = _mm512_add_epi8(popcnt_result[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 4; i++) {
        _mm512_storeu_si512(popcnt_tmp + (i * 64), popcnt_result[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = popcnt_tmp[i];
    }
}

// 256 bits -> 32 words -> 64 nibbles
// Each nibble is a LUT of 64 bytes (to fit on the AVX512 register)
// 64 x 64 = 4096 bytes needed
static uint8_t* query_aware_b256_xorluts_avx512[4096];
// Or just enough: Each nibble is a LUT of 16 bytes: 32 x 16 x 2 (high/low):
static uint8_t* query_aware_b256_xorluts_high[1024];
static uint8_t* query_aware_b256_xorluts_low[1024];

// 1-to-256 vectors
// second_vector is a 256*256 matrix in a column-major layout
inline void hamming_b256_xorlut_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i popcnt_result[4];
    // Load initial values
    for (size_t i = 0; i < 4; ++i) { // 256 vectors at a time (using 8 registers)
        popcnt_result[i] = _mm512_setzero_si512();
    }

    for (size_t dim = 0; dim != 32; dim++){
//        uint8_t first_high = (first_vector[dim] & 0xF0) >> 4;
//        uint8_t first_low = first_vector[dim] & 0x0F;
//        __m512i lut_xor_high = m512_xor_lookup_tables[first_high];
//        __m512i lut_xor_low = m512_xor_lookup_tables[first_low];

//        __m512i lut_xor_high = _mm512_loadu_epi8(query_aware_b256_xorluts_avx512 + ((dim * 2) * 64));
//        __m512i lut_xor_low = _mm512_loadu_epi8(query_aware_b256_xorluts_avx512 + (((dim * 2) + 1) * 64));

        __m512i lut_xor_high = _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(query_aware_b256_xorluts_high + (dim * 16))));
        __m512i lut_xor_low = _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(query_aware_b256_xorluts_low + (dim * 16))));

        for (size_t i = 0; i < 4; i++){ // 256 uint8_t values
            __m512i second = _mm512_loadu_epi8(second_vector);

            // Getting nibbles from data
            __m512i second_low = _mm512_and_si512(second, low_mask);
            __m512i second_high = _mm512_and_si512(_mm512_srli_epi16(second, 4), low_mask);

            __m512i popcnt_ = _mm512_add_epi8(
                _mm512_shuffle_epi8(lut_xor_high, second_high),
                _mm512_shuffle_epi8(lut_xor_low, second_low)
            );

            popcnt_result[i] = _mm512_add_epi8(popcnt_result[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 4; i++) {
        _mm512_storeu_si512(popcnt_tmp + (i * 64), popcnt_result[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = popcnt_tmp[i];
    }
}

inline float hamming_u64x4_c(uint8_t const *a, uint8_t const *b) {
    uint32_t popcnt = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 4; ++i)
        popcnt += __builtin_popcountll(a64[i] ^ b64[i]);
    return popcnt;
}

/*
 * Define the AVX2 variant using the `vpshufb` and `vpsadbw` instruction.
 * It resorts to cheaper byte-shuffling instructions, than population counts.
 * Source: https://github.com/CountOnes/hamming_weight/blob/1dd7554c0fc39e01c9d7fa54372fd4eccf458875/src/sse_jaccard_index.c#L17
 */
__attribute__((target("avx2,bmi2,avx")))
inline float hamming_b256_vpshufb_sad(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m256i first = _mm256_loadu_epi8((__m256i const*)(first_vector));
    __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));

    __m256i xor_ = _mm256_xor_epi64(first, second);

    __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i lookup = _mm256_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

    __m256i xor_low = _mm256_and_si256(xor_, low_mask);
    __m256i xor_high = _mm256_and_si256(_mm256_srli_epi16(xor_, 4), low_mask);

    __m256i popcnt_ = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, xor_low),
        _mm256_shuffle_epi8(lookup, xor_high));

    __m256i popcnt = _mm256_sad_epu8(popcnt_, _mm256_setzero_si256());
    return _mm256_reduce_add_epi64(popcnt);
}


// Define the AVX-512 variant using the `vpopcntq` instruction.
// It's known to over-rely on port 5 on x86 CPUs, so the next `vpshufb` variant should be faster.
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
inline float hamming_b256_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m256i first = _mm256_loadu_epi8((__m256i const*)(first_vector));
    __m256i second = _mm256_loadu_epi8((__m256i const*)(second_vector));

    __m256i popcnt = _mm256_popcnt_epi64(_mm256_xor_epi64(first, second));
    return _mm256_reduce_add_epi64(popcnt);
}



///////////////////////////////
///////////////////////////////
// region: 512d kernels ///////
///////////////////////////////
///////////////////////////////

inline float hamming_u64x8_c(uint8_t const *a, uint8_t const *b) {
    uint32_t popcnt = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 8; ++i)
        popcnt += __builtin_popcountll(a64[i] ^ b64[i]);
    return popcnt;
}

static uint8_t popcnt_tmp_512_a[256];
static uint8_t popcnt_tmp_512_b[256];

inline float hamming_b512_vpshufb_sad(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i second_start = _mm512_loadu_si512((__m512i const*)(second_vector));

    __m512i xor_ = _mm512_xor_epi64(first_start, second_start);

    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

    __m512i xor_low = _mm512_and_si512(xor_, low_mask);
    __m512i xor_high = _mm512_and_si512(_mm512_srli_epi16(xor_, 4), low_mask);

    __m512i popcnt_ = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, xor_low),
        _mm512_shuffle_epi8(lookup, xor_high));

    __m512i popcnt = _mm512_sad_epu8(popcnt_, _mm512_setzero_si512());

    return _mm512_reduce_add_epi64(popcnt);
}




// Define the AVX-512 variant using the `vpopcntq` instruction.
// It's known to over-rely on port 5 on x86 CPUs, so the next `vpshufb` variant should be faster.
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
inline float hamming_b512_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i second_start = _mm512_loadu_si512((__m512i const*)(second_vector));

    __m512i popcnt = _mm512_popcnt_epi64(_mm512_xor_epi64(first_start, second_start));

    return _mm512_reduce_add_epi64(popcnt);
}


inline void hamming_b512_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i popcnt_result_a[4];
    __m512i popcnt_result_b[4];
    for (size_t i = 0; i < 4; ++i) { // 256 vectors at a time (using 4 _m512i registers)
        popcnt_result_a[i] = _mm512_setzero_si512();
        popcnt_result_b[i] = _mm512_setzero_si512();
    }
    // Word 0 to 31
    for (size_t dim = 0; dim != 32; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 4; i++){
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i popcnt_ = _mm512_popcnt_epi8(_mm512_xor_epi64(first, second));
            popcnt_result_a[i] = _mm512_add_epi8(popcnt_result_a[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // Word 32 to 63
    for (size_t dim = 32; dim != 64; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 4; i++){
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i popcnt_ = _mm512_popcnt_epi8(_mm512_xor_epi64(first, second));
            popcnt_result_b[i] = _mm512_add_epi8(popcnt_result_b[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 4; i++) {
        _mm512_storeu_si512((__m512i *)(popcnt_tmp_512_a + (i * 64)), popcnt_result_a[i]);
        _mm512_storeu_si512((__m512i *)(popcnt_tmp_512_b + (i * 64)), popcnt_result_b[i]);
    }
    // TODO: Probably can use SIMD for the pairwise sum of the 4 groups
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = popcnt_tmp_512_a[i] + popcnt_tmp_512_b[i];
    }
}


// 1-to-256 vectors
// second_vector is a 256*256 matrix in a column-major layout
inline void hamming_b512_vpshufb_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i popcnt_result_a[4];
    __m512i popcnt_result_b[4];
    __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    // Load initial values
    for (size_t i = 0; i < 4; ++i) { // 256 vectors at a time (using 8 registers)
        popcnt_result_a[i] = _mm512_setzero_si512();
        popcnt_result_b[i] = _mm512_setzero_si512();
    }
    for (size_t dim = 0; dim != 32; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 4; i++){ // 256 uint8_t values
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i xor_ = _mm512_xor_epi64(first, second);

            // Getting nibbles from data
            __m512i second_low = _mm512_and_si512(xor_, low_mask);
            __m512i second_high = _mm512_and_si512(_mm512_srli_epi16(xor_, 4), low_mask);

            __m512i popcnt_ = _mm512_add_epi8(
                _mm512_shuffle_epi8(lookup, second_low),
                _mm512_shuffle_epi8(lookup, second_high)
            );
            popcnt_result_a[i] = _mm512_add_epi8(popcnt_result_a[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    for (size_t dim = 32; dim != 64; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);

        for (size_t i = 0; i < 4; i++){ // 256 uint8_t values
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i xor_ = _mm512_xor_epi64(first, second);

            // Getting nibbles from data
            __m512i second_low = _mm512_and_si512(xor_, low_mask);
            __m512i second_high = _mm512_and_si512(_mm512_srli_epi16(xor_, 4), low_mask);

            __m512i popcnt_ = _mm512_add_epi8(
                _mm512_shuffle_epi8(lookup, second_low),
                _mm512_shuffle_epi8(lookup, second_high)
            );
            popcnt_result_b[i] = _mm512_add_epi8(popcnt_result_b[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 4; i++) {
        _mm512_storeu_si512(popcnt_tmp_512_a + (i * 64), popcnt_result_a[i]);
        _mm512_storeu_si512(popcnt_tmp_512_b + (i * 64), popcnt_result_b[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = popcnt_tmp_512_a[i] + popcnt_tmp_512_b[i];
    }
}



///////////////////////////////
///////////////////////////////
// region: 1024d kernels //////
///////////////////////////////
///////////////////////////////

static uint8_t popcnt_tmp_1024_a[256];
static uint8_t popcnt_tmp_1024_b[256];
static uint8_t popcnt_tmp_1024_c[256];
static uint8_t popcnt_tmp_1024_d[256];

// 1-to-256 vectors
// second_vector is a 256*1024 matrix in a column-major layout
// Processing the 1024 dimensions in 4 groups of 32 words each to not overflow the uint8_t accumulators
inline void hamming_b1024_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i popcnt_result_a[4];
    __m512i popcnt_result_b[4];
    __m512i popcnt_result_c[4];
    __m512i popcnt_result_d[4];
    for (size_t i = 0; i < 4; ++i) { // 256 vectors at a time (using 4 _m512i registers)
        popcnt_result_a[i] = _mm512_setzero_si512();
        popcnt_result_b[i] = _mm512_setzero_si512();
        popcnt_result_c[i] = _mm512_setzero_si512();
        popcnt_result_d[i] = _mm512_setzero_si512();
    }
    // Word 0 to 31
    for (size_t dim = 0; dim != 32; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 4; i++){
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i popcnt_ = _mm512_popcnt_epi8(_mm512_xor_epi64(first, second));
            popcnt_result_a[i] = _mm512_add_epi8(popcnt_result_a[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // Word 32 to 63
    for (size_t dim = 32; dim != 64; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 4; i++){
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i popcnt_ = _mm512_popcnt_epi8(_mm512_xor_epi64(first, second));
            popcnt_result_b[i] = _mm512_add_epi8(popcnt_result_b[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // Word 64 to 95
    for (size_t dim = 64; dim != 96; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 4; i++){
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i popcnt_ = _mm512_popcnt_epi8(_mm512_and_epi64(first, second));
            popcnt_result_c[i] = _mm512_add_epi8(popcnt_result_c[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // Word 96 to 127
    for (size_t dim = 96; dim != 128; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 4; i++){
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i popcnt_ = _mm512_popcnt_epi8(_mm512_and_epi64(first, second));
            popcnt_result_d[i] = _mm512_add_epi8(popcnt_result_d[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 4; i++) {
        _mm512_storeu_si512((__m512i *)(popcnt_tmp_1024_a + (i * 64)), popcnt_result_a[i]);
        _mm512_storeu_si512((__m512i *)(popcnt_tmp_1024_b + (i * 64)), popcnt_result_b[i]);
        _mm512_storeu_si512((__m512i *)(popcnt_tmp_1024_c + (i * 64)), popcnt_result_c[i]);
        _mm512_storeu_si512((__m512i *)(popcnt_tmp_1024_d + (i * 64)), popcnt_result_d[i]);
    }
    // TODO: Probably can use SIMD for the pairwise sum of the 4 groups
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = popcnt_tmp_1024_a[i] + popcnt_tmp_1024_b[i] + popcnt_tmp_1024_c[i] + popcnt_tmp_1024_d[i];
    }
}

inline void hamming_b1024_vpshufb_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i popcnt_result_a[4];
    __m512i popcnt_result_b[4];
    __m512i popcnt_result_c[4];
    __m512i popcnt_result_d[4];
    __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    // Load initial values
    for (size_t i = 0; i < 4; ++i) { // 256 vectors at a time (using 8 registers)
        popcnt_result_a[i] = _mm512_setzero_si512();
        popcnt_result_b[i] = _mm512_setzero_si512();
        popcnt_result_c[i] = _mm512_setzero_si512();
        popcnt_result_d[i] = _mm512_setzero_si512();
    }
    for (size_t dim = 0; dim != 32; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        for (size_t i = 0; i < 4; i++){ // 256 uint8_t values
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i xor_ = _mm512_xor_epi64(first, second);

            // Getting nibbles from data
            __m512i second_low = _mm512_and_si512(xor_, low_mask);
            __m512i second_high = _mm512_and_si512(_mm512_srli_epi16(xor_, 4), low_mask);

            __m512i popcnt_ = _mm512_add_epi8(
                _mm512_shuffle_epi8(lookup, second_low),
                _mm512_shuffle_epi8(lookup, second_high)
            );
            popcnt_result_a[i] = _mm512_add_epi8(popcnt_result_a[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    for (size_t dim = 32; dim != 64; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);

        for (size_t i = 0; i < 4; i++){ // 256 uint8_t values
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i xor_ = _mm512_xor_epi64(first, second);

            // Getting nibbles from data
            __m512i second_low = _mm512_and_si512(xor_, low_mask);
            __m512i second_high = _mm512_and_si512(_mm512_srli_epi16(xor_, 4), low_mask);

            __m512i popcnt_ = _mm512_add_epi8(
                _mm512_shuffle_epi8(lookup, second_low),
                _mm512_shuffle_epi8(lookup, second_high)
            );
            popcnt_result_b[i] = _mm512_add_epi8(popcnt_result_b[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    for (size_t dim = 64; dim != 96; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);

        for (size_t i = 0; i < 4; i++){ // 256 uint8_t values
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i xor_ = _mm512_xor_epi64(first, second);

            // Getting nibbles from data
            __m512i second_low = _mm512_and_si512(xor_, low_mask);
            __m512i second_high = _mm512_and_si512(_mm512_srli_epi16(xor_, 4), low_mask);

            __m512i popcnt_ = _mm512_add_epi8(
                _mm512_shuffle_epi8(lookup, second_low),
                _mm512_shuffle_epi8(lookup, second_high)
            );
            popcnt_result_c[i] = _mm512_add_epi8(popcnt_result_c[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    for (size_t dim = 96; dim != 128; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);

        for (size_t i = 0; i < 4; i++){ // 256 uint8_t values
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i xor_ = _mm512_xor_epi64(first, second);

            // Getting nibbles from data
            __m512i second_low = _mm512_and_si512(xor_, low_mask);
            __m512i second_high = _mm512_and_si512(_mm512_srli_epi16(xor_, 4), low_mask);

            __m512i popcnt_ = _mm512_add_epi8(
                _mm512_shuffle_epi8(lookup, second_low),
                _mm512_shuffle_epi8(lookup, second_high)
            );
            popcnt_result_d[i] = _mm512_add_epi8(popcnt_result_d[i], popcnt_);
            second_vector += 64; // 256x8-bit values (using 8 registers at a time)
        }
    }
    // TODO: Ugly
    for (size_t i = 0; i < 4; i++) {
        _mm512_storeu_si512(popcnt_tmp_1024_a + (i * 64), popcnt_result_a[i]);
        _mm512_storeu_si512(popcnt_tmp_1024_b + (i * 64), popcnt_result_b[i]);
        _mm512_storeu_si512(popcnt_tmp_1024_c + (i * 64), popcnt_result_c[i]);
        _mm512_storeu_si512(popcnt_tmp_1024_d + (i * 64), popcnt_result_d[i]);
    }
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = popcnt_tmp_1024_a[i] + popcnt_tmp_1024_b[i] + popcnt_tmp_1024_c[i] + popcnt_tmp_1024_d[i];
    }
}


inline float hamming_u8x128_c(uint8_t const *a, uint8_t const *b) {
    uint32_t popcnt = 0;
#pragma unroll
    for (size_t i = 0; i != 128; ++i)
        popcnt += __builtin_popcount(a[i] & b[i]);
    return popcnt;
}

inline float hamming_u64x16_c(uint8_t const *a, uint8_t const *b) {
    uint32_t popcnt = 0;
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;
#pragma unroll
    for (size_t i = 0; i != 16; ++i)
        popcnt += __builtin_popcountll(a64[i] ^ b64[i]);
    return popcnt;
}

// Define the AVX-512 variant using the `vpopcntq` instruction.
// It's known to over-rely on port 5 on x86 CPUs, so the next `vpshufb` variant should be faster.
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
inline float hamming_b1024_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i first_end = _mm512_loadu_si512((__m512i const*)(first_vector + 64));
    __m512i second_start = _mm512_loadu_si512((__m512i const*)(second_vector));
    __m512i second_end = _mm512_loadu_si512((__m512i const*)(second_vector + 64));

    __m512i popcnt_start = _mm512_popcnt_epi64(_mm512_xor_epi64(first_start, second_start));
    __m512i popcnt_end = _mm512_popcnt_epi64(_mm512_xor_epi64(first_end, second_end));

    return _mm512_reduce_add_epi64(popcnt_start) + _mm512_reduce_add_epi64(popcnt_end);
}

inline float hamming_b1024_vpshufb_sad(uint8_t const *first_vector, uint8_t const *second_vector) {
    __m512i first_start = _mm512_loadu_si512((__m512i const*)(first_vector));
    __m512i first_end = _mm512_loadu_si512((__m512i const*)(first_vector + 64));
    __m512i second_start = _mm512_loadu_si512((__m512i const*)(second_vector));
    __m512i second_end = _mm512_loadu_si512((__m512i const*)(second_vector + 64));

    __m512i xor_start = _mm512_xor_epi64(first_start, second_start);
    __m512i xor_end = _mm512_xor_epi64(first_end, second_end);

    __m512i low_mask = _mm512_set1_epi8(0x0f);
    __m512i lookup = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);

    __m512i xor_start_low = _mm512_and_si512(xor_start, low_mask);
    __m512i xor_start_high = _mm512_and_si512(_mm512_srli_epi16(xor_start, 4), low_mask);
    __m512i xor_end_low = _mm512_and_si512(xor_end, low_mask);
    __m512i xor_end_high = _mm512_and_si512(_mm512_srli_epi16(xor_end, 4), low_mask);


    __m512i start_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, xor_start_low),
        _mm512_shuffle_epi8(lookup, xor_start_high));
    __m512i end_popcount = _mm512_add_epi8(
        _mm512_shuffle_epi8(lookup, xor_end_low),
        _mm512_shuffle_epi8(lookup, xor_end_high));

    __m512i popcnt = _mm512_add_epi64(
        _mm512_sad_epu8(start_popcount, _mm512_setzero_si512()),
        _mm512_sad_epu8(end_popcount, _mm512_setzero_si512()));

    return _mm512_reduce_add_epi64(popcnt);
}

// Harley-Seal transformation and Odd-Major-style Carry-Save-Adders can be used to replace
// several population counts with a few bitwise operations and one `popcount`, which can help
// lift the pressure on the CPU ports.
inline int popcount_csa3(uint64_t x, uint64_t y, uint64_t z) {
    uint64_t odd  = (x ^ y) ^ z;
    uint64_t major = ((x ^ y) & z) | (x & y);
    return 2 * __builtin_popcountll(major) + __builtin_popcountll(odd);
}

inline float hamming_u64x16_csa3_c(uint8_t const *a, uint8_t const *b) {
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;

    int popcnt =
        popcount_csa3(a64[0] ^ b64[0], a64[1] ^ b64[1], a64[2] ^ b64[2]) +
        popcount_csa3(a64[3] ^ b64[3], a64[4] ^ b64[4], a64[5] ^ b64[5]) +
        popcount_csa3(a64[6] ^ b64[6], a64[7] ^ b64[7], a64[8] ^ b64[8]) +
        popcount_csa3(a64[9] ^ b64[9], a64[10] ^ b64[10], a64[11] ^ b64[11]) +
        popcount_csa3(a64[12] ^ b64[12], a64[13] ^ b64[13], a64[14] ^ b64[14]) +
        __builtin_popcountll(a64[15] ^ b64[15]);

    return popcnt;
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

inline float hamming_u64x16_csa15_cpp(uint8_t const *a, uint8_t const *b) {
    uint64_t const *a64 = (uint64_t const *)a;
    uint64_t const *b64 = (uint64_t const *)b;

    int popcnt = popcount_csa15(
        a64[0] ^ b64[0], a64[1] ^ b64[1], a64[2] ^ b64[2], a64[3] ^ b64[3],
        a64[4] ^ b64[4], a64[5] ^ b64[5], a64[6] ^ b64[6], a64[7] ^ b64[7],
        a64[8] ^ b64[8], a64[9] ^ b64[9], a64[10] ^ b64[10], a64[11] ^ b64[11],
        a64[12] ^ b64[12], a64[13] ^ b64[13], a64[14] ^ b64[14]) +
        __builtin_popcountll(a64[15] ^ b64[15]);

    return popcnt; // ! Avoid division by zero
}


template <HammingKernel kernel=HAMMING_B1024_VPOPCNTQ, int N_WORDS=16>
std::vector<KNNCandidate> hamming_standalone_partial_sort(
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
            if constexpr (kernel == HAMMING_U64X2_C){ // 128
                current_distance = hamming_u64x2_c(query, data);
            } else if constexpr (kernel == HAMMING_B128_VPSHUFB_SAD) {
                current_distance = hamming_b128_vpshufb_sad(query, data);
            } else if constexpr (kernel == HAMMING_B128_VPOPCNTQ) {
                current_distance = hamming_b128_vpopcntq(query, data);


            } else if constexpr (kernel == HAMMING_U64X4_C){ // 256
                current_distance = hamming_u64x4_c(query, data);
            } else if constexpr (kernel == HAMMING_B256_VPSHUFB_SAD) {
                current_distance = hamming_b256_vpshufb_sad(query, data);
            } else if constexpr (kernel == HAMMING_B256_VPOPCNTQ) {
                current_distance = hamming_b256_vpopcntq(query, data);

            } else if constexpr (kernel == HAMMING_U64X8_C){ // 256
                current_distance = hamming_u64x8_c(query, data);
            } else if constexpr (kernel == HAMMING_B512_VPSHUFB_SAD) { // 512
                current_distance = hamming_b512_vpshufb_sad(query, data);
            } else if constexpr (kernel == HAMMING_B512_VPOPCNTQ) {
                current_distance = hamming_b512_vpopcntq(query, data);


            } else if constexpr (kernel == HAMMING_U8X128_C) { // 1024
                current_distance = hamming_u8x128_c(query, data);
            } else if constexpr (kernel == HAMMING_U64X16_C) {
                current_distance = hamming_u64x16_c(query, data);
            } else if constexpr (kernel == HAMMING_B1024_VPOPCNTQ) {
                current_distance = hamming_b1024_vpopcntq(query, data);
            } else if constexpr (kernel == HAMMING_B1024_VPSHUFB_SAD) {
                current_distance = hamming_b1024_vpshufb_sad(query, data);
            } else if constexpr (kernel == HAMMING_U64X16_CSA3_C) {
                current_distance = hamming_u64x16_csa3_c(query, data);
            } else if constexpr (kernel == HAMMING_U64X16_CSA15_CPP) {
                current_distance = hamming_u64x16_csa15_cpp(query, data);
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

inline void fill_b256_xorluts(const uint8_t *query){
    for (size_t d = 0; d < 32; ++d){
        uint8_t first_high = (query[d] & 0xF0) >> 4;
        uint8_t first_low = query[d] & 0x0F;

//        __m512i lut_xor_high = m512_xor_lookup_tables[first_high];
//        __m512i lut_xor_low = m512_xor_lookup_tables[first_low];
//        _mm512_storeu_si512(query_aware_b256_xorluts_avx512 + ((d * 2) * 64), lut_xor_high);
//        _mm512_storeu_si512(query_aware_b256_xorluts_avx512 + (((d * 2) + 1) * 64), lut_xor_low);

        memcpy(query_aware_b256_xorluts_high + (d * 16), xor_lookup_tables[first_high], 16);
        memcpy(query_aware_b256_xorluts_low + (d * 16), xor_lookup_tables[first_low], 16);

    }
};

template <HammingKernel kernel=HAMMING_B256_VPOPCNTQ_PDX, int N_WORDS=32, int PDX_BLOCK_SIZE=256>
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
        if constexpr (kernel == HAMMING_B256_XORLUT_PDX){
            fill_b256_xorluts(query);
        }
        const uint8_t* data = first_vector;
        // Fill all_distances by direct indexing
        size_t global_offset = 0;
        for (size_t j = 0; j < num_vectors; j+=PDX_BLOCK_SIZE) {
            if constexpr (kernel == HAMMING_B256_VPOPCNTQ_PDX){
                hamming_b256_vpopcntq_pdx(query, data);
            } else if constexpr (kernel == HAMMING_B1024_VPOPCNTQ_PDX){
                hamming_b1024_vpopcntq_pdx(query, data);
            } else if constexpr (kernel == HAMMING_B512_VPOPCNTQ_PDX){
                hamming_b512_vpopcntq_pdx(query, data);
            } else if constexpr (kernel == HAMMING_B256_VPSHUFB_PDX){
                hamming_b256_vpshufb_pdx(query, data);
            } else if constexpr (kernel == HAMMING_B256_XORLUT_PDX){
                hamming_b256_xorlut_pdx(query, data);
            } else if constexpr (kernel == HAMMING_B512_VPSHUFB_PDX){
                hamming_b512_vpshufb_pdx(query, data);
            } else if constexpr (kernel == HAMMING_B1024_VPSHUFB_PDX){
                hamming_b1024_vpshufb_pdx(query, data);
            } else if constexpr (kernel == HAMMING_B128_VPSHUFB_PDX){
                hamming_b128_vpshufb_pdx(query, data);
            } else if constexpr (kernel == HAMMING_B128_VPOPCNTQ_PDX){
                hamming_b128_vpopcntq_pdx(query, data);
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

std::vector<KNNCandidate> hamming_standalone(
    HammingKernel kernel,
    uint8_t const *first_vector,
    uint8_t const *second_vector,
    size_t num_queries,
    size_t num_vectors,
    size_t knn
) {
    switch (kernel) {
        case HAMMING_U64X2_C: // 128
            return hamming_standalone_partial_sort<HAMMING_U64X2_C, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B128_VPSHUFB_SAD:
            return hamming_standalone_partial_sort<HAMMING_B128_VPSHUFB_SAD, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B128_VPOPCNTQ:
            return hamming_standalone_partial_sort<HAMMING_B128_VPOPCNTQ, 16>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B128_VPOPCNTQ_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B128_VPOPCNTQ_PDX, 16, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B128_VPSHUFB_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B128_VPSHUFB_PDX, 16, 256>(first_vector, second_vector, num_queries, num_vectors, knn);


        case HAMMING_U64X4_C: // 256
            return hamming_standalone_partial_sort<HAMMING_U64X4_C, 32>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B256_VPSHUFB_SAD:
            return hamming_standalone_partial_sort<HAMMING_B256_VPSHUFB_SAD, 32>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B256_VPOPCNTQ:
            return hamming_standalone_partial_sort<HAMMING_B256_VPOPCNTQ, 32>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B256_VPOPCNTQ_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B256_VPOPCNTQ_PDX, 32, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B256_VPSHUFB_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B256_VPSHUFB_PDX, 32, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B256_XORLUT_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B256_XORLUT_PDX, 32, 256>(first_vector, second_vector, num_queries, num_vectors, knn);


        case HAMMING_U64X8_C: // 512
            return hamming_standalone_partial_sort<HAMMING_U64X8_C, 64>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B512_VPSHUFB_SAD:
            return hamming_standalone_partial_sort<HAMMING_B512_VPSHUFB_SAD, 64>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B512_VPOPCNTQ:
            return hamming_standalone_partial_sort<HAMMING_B512_VPOPCNTQ, 64>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B512_VPOPCNTQ_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B512_VPOPCNTQ_PDX, 64, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B512_VPSHUFB_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B512_VPSHUFB_PDX, 64, 256>(first_vector, second_vector, num_queries, num_vectors, knn);


        case HAMMING_U8X128_C: // 1024
            return hamming_standalone_partial_sort<HAMMING_U8X128_C, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_U64X16_C:
            return hamming_standalone_partial_sort<HAMMING_U64X16_C, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B1024_VPOPCNTQ:
            return hamming_standalone_partial_sort<HAMMING_B1024_VPOPCNTQ, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B1024_VPSHUFB_SAD:
            return hamming_standalone_partial_sort<HAMMING_B1024_VPSHUFB_SAD, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_U64X16_CSA3_C:
            return hamming_standalone_partial_sort<HAMMING_U64X16_CSA3_C, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_U64X16_CSA15_CPP:
            return hamming_standalone_partial_sort<HAMMING_U64X16_CSA15_CPP, 128>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B1024_VPOPCNTQ_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B1024_VPOPCNTQ_PDX, 128, 256>(first_vector, second_vector, num_queries, num_vectors, knn);
        case HAMMING_B1024_VPSHUFB_PDX:
            return hamming_pdx_standalone_partial_sort<HAMMING_B1024_VPSHUFB_PDX, 128, 256>(first_vector, second_vector, num_queries, num_vectors, knn);


        default:
            return hamming_standalone_partial_sort<HAMMING_U64X4_C, 32>(first_vector, second_vector, num_queries, num_vectors, knn);
    }
}