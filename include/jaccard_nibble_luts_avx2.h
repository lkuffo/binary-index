///////////////////////////////
///////////////////////////////
/// Intersection and Union  ///
/// LUTs for nibbles [AVX2] ///
///////////////////////////////
///////////////////////////////

/*
x = range(0, 16)
y = range(15, -1, -1)
for i in x:
     print(f"alignas(64) static const __m256i m256_xor_lut_{i} = _mm256_set_epi8(")
     print("\t", end = "")
     for j in y:
         print(bin(i ^ j).count("1"), end=", ")
     print("\n\t", end = "")
     for j in y:
         print(bin(i ^ j).count("1"), end=", ")
     print()
     print(");")
     print()
*/

alignas(64) static const __m256i m256_intersection_lut_0  = _mm256_set_epi8(
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
);

alignas(64) static const __m256i m256_intersection_lut_1  = _mm256_set_epi8(
    1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
    1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0
);

alignas(64) static const __m256i m256_intersection_lut_2  = _mm256_set_epi8(
    1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,
    1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0
);

alignas(64) static const __m256i m256_intersection_lut_3  = _mm256_set_epi8(
    2,1,1,0,2,1,1,0,2,1,1,0,2,1,1,0,
    2,1,1,0,2,1,1,0,2,1,1,0,2,1,1,0
);

alignas(64) static const __m256i m256_intersection_lut_4  = _mm256_set_epi8(
    1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,
    1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0
);

alignas(64) static const __m256i m256_intersection_lut_5  = _mm256_set_epi8(
    2,1,2,1,1,0,1,0,2,1,2,1,1,0,1,0,
    2,1,2,1,1,0,1,0,2,1,2,1,1,0,1,0
);

alignas(64) static const __m256i m256_intersection_lut_6  = _mm256_set_epi8(
    2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,
    2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1
);

alignas(64) static const __m256i m256_intersection_lut_7  = _mm256_set_epi8(
    3,2,2,1,3,2,2,1,3,2,2,1,3,2,2,1,
    3,2,2,1,3,2,2,1,3,2,2,1,3,2,2,1
);

alignas(64) static const __m256i m256_intersection_lut_8  = _mm256_set_epi8(
    1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0
);

alignas(64) static const __m256i m256_intersection_lut_9  = _mm256_set_epi8(
    2,1,2,1,2,1,2,1,1,0,1,0,1,0,1,0,
    2,1,2,1,2,1,2,1,1,0,1,0,1,0,1,0
);

alignas(64) static const __m256i m256_intersection_lut_10 = _mm256_set_epi8(
    2,2,1,1,2,2,1,1,1,1,0,0,1,1,0,0,
    2,2,1,1,2,2,1,1,1,1,0,0,1,1,0,0
);

alignas(64) static const __m256i m256_intersection_lut_11 = _mm256_set_epi8(
    3,2,2,1,3,2,2,1,2,1,1,0,3,2,2,1,
    3,2,2,1,3,2,2,1,2,1,1,0,3,2,2,1
);

alignas(64) static const __m256i m256_intersection_lut_12 = _mm256_set_epi8(
    2,2,2,2,1,1,1,1,1,1,1,1,0,0,0,0,
    2,2,2,2,1,1,1,1,1,1,1,1,0,0,0,0
);

alignas(64) static const __m256i m256_intersection_lut_13 = _mm256_set_epi8(
    3,2,3,2,2,1,2,1,2,1,2,1,0,1,0,1,
    3,2,3,2,2,1,2,1,2,1,2,1,0,1,0,1
);

alignas(64) static const __m256i m256_intersection_lut_14 = _mm256_set_epi8(
    3,3,2,2,2,2,1,1,2,2,1,1,1,1,0,0,
    3,3,2,2,2,2,1,1,2,2,1,1,1,1,0,0
);

alignas(64) static const __m256i m256_intersection_lut_15 = _mm256_set_epi8(
    4,3,3,2,3,2,2,1,3,2,2,1,1,0,1,0,
    4,3,3,2,3,2,2,1,3,2,2,1,1,0,1,0
);


alignas(64) static const __m256i m256_union_lut_0  = _mm256_set_epi8(
    4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
    4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
);

alignas(64) static const __m256i m256_union_lut_1  = _mm256_set_epi8(
    4, 4, 3, 3, 3, 3, 2, 2, 4, 4, 3, 3, 3, 3, 1, 1,
    4, 4, 3, 3, 3, 3, 2, 2, 4, 4, 3, 3, 3, 3, 1, 1
);

alignas(64) static const __m256i m256_union_lut_2  = _mm256_set_epi8(
    4, 3, 4, 3, 3, 2, 3, 2, 3, 2, 3, 2, 2, 1, 2, 1,
    4, 3, 4, 3, 3, 2, 3, 2, 3, 2, 3, 2, 2, 1, 2, 1
);

alignas(64) static const __m256i m256_union_lut_3  = _mm256_set_epi8(
    4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
    4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2
);

alignas(64) static const __m256i m256_union_lut_4  = _mm256_set_epi8(
    4, 3, 3, 2, 4, 3, 3, 2, 3, 4, 3, 2, 3, 2, 2, 1,
    4, 3, 3, 2, 4, 3, 3, 2, 3, 4, 3, 2, 3, 2, 2, 1
);

alignas(64) static const __m256i m256_union_lut_5  = _mm256_set_epi8(
    4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2,
    4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2
);

alignas(64) static const __m256i m256_union_lut_6  = _mm256_set_epi8(
    4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 2, 3, 2,
    4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 2, 3, 2
);

alignas(64) static const __m256i m256_union_lut_7  = _mm256_set_epi8(
    4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3
);

alignas(64) static const __m256i m256_union_lut_8  = _mm256_set_epi8(
    4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1,
    4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1
);

alignas(64) static const __m256i m256_union_lut_9  = _mm256_set_epi8(
    4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2,
    4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2
);

alignas(64) static const __m256i m256_union_lut_10 = _mm256_set_epi8(
    4, 3, 4, 3, 4, 3, 3, 2, 3, 2, 3, 2, 4, 3, 3, 2,
    4, 3, 4, 3, 4, 3, 3, 2, 3, 2, 3, 2, 4, 3, 3, 2
);

alignas(64) static const __m256i m256_union_lut_11 = _mm256_set_epi8(
    4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3,
    4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3
);

alignas(64) static const __m256i m256_union_lut_12 = _mm256_set_epi8(
    4, 3, 3, 2, 4, 3, 3, 2, 4, 3, 3, 2, 4, 3, 3, 2,
    4, 3, 3, 2, 4, 3, 3, 2, 4, 3, 3, 2, 4, 3, 3, 2
);

alignas(64) static const __m256i m256_union_lut_13 = _mm256_set_epi8(
    4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3,
    4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3
);

alignas(64) static const __m256i m256_union_lut_14 = _mm256_set_epi8(
    4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
    4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3
);

alignas(64) static const __m256i m256_union_lut_15 = _mm256_set_epi8(
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
);

alignas(64) static const __m256i m256_xor_lut_0 = _mm256_set_epi8(
	4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
	4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
);

alignas(64) static const __m256i m256_xor_lut_1 = _mm256_set_epi8(
	3, 4, 2, 3, 2, 3, 1, 2, 2, 3, 1, 2, 1, 2, 0, 1,
	3, 4, 2, 3, 2, 3, 1, 2, 2, 3, 1, 2, 1, 2, 0, 1
);

alignas(64) static const __m256i m256_xor_lut_2 = _mm256_set_epi8(
	3, 2, 4, 3, 2, 1, 3, 2, 2, 1, 3, 2, 1, 0, 2, 1,
	3, 2, 4, 3, 2, 1, 3, 2, 2, 1, 3, 2, 1, 0, 2, 1
);

alignas(64) static const __m256i m256_xor_lut_3 = _mm256_set_epi8(
	2, 3, 3, 4, 1, 2, 2, 3, 1, 2, 2, 3, 0, 1, 1, 2,
	2, 3, 3, 4, 1, 2, 2, 3, 1, 2, 2, 3, 0, 1, 1, 2
);

alignas(64) static const __m256i m256_xor_lut_4 = _mm256_set_epi8(
	3, 2, 2, 1, 4, 3, 3, 2, 2, 1, 1, 0, 3, 2, 2, 1,
	3, 2, 2, 1, 4, 3, 3, 2, 2, 1, 1, 0, 3, 2, 2, 1
);

alignas(64) static const __m256i m256_xor_lut_5 = _mm256_set_epi8(
	2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 0, 1, 2, 3, 1, 2,
	2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 0, 1, 2, 3, 1, 2
);

alignas(64) static const __m256i m256_xor_lut_6 = _mm256_set_epi8(
	2, 1, 3, 2, 3, 2, 4, 3, 1, 0, 2, 1, 2, 1, 3, 2,
	2, 1, 3, 2, 3, 2, 4, 3, 1, 0, 2, 1, 2, 1, 3, 2
);

alignas(64) static const __m256i m256_xor_lut_7 = _mm256_set_epi8(
	1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3,
	1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3
);

alignas(64) static const __m256i m256_xor_lut_8 = _mm256_set_epi8(
	3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1,
	3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1
);

alignas(64) static const __m256i m256_xor_lut_9 = _mm256_set_epi8(
	2, 3, 1, 2, 1, 2, 0, 1, 3, 4, 2, 3, 2, 3, 1, 2,
	2, 3, 1, 2, 1, 2, 0, 1, 3, 4, 2, 3, 2, 3, 1, 2
);

alignas(64) static const __m256i m256_xor_lut_10 = _mm256_set_epi8(
	2, 1, 3, 2, 1, 0, 2, 1, 3, 2, 4, 3, 2, 1, 3, 2,
	2, 1, 3, 2, 1, 0, 2, 1, 3, 2, 4, 3, 2, 1, 3, 2
);

alignas(64) static const __m256i m256_xor_lut_11 = _mm256_set_epi8(
	1, 2, 2, 3, 0, 1, 1, 2, 2, 3, 3, 4, 1, 2, 2, 3,
	1, 2, 2, 3, 0, 1, 1, 2, 2, 3, 3, 4, 1, 2, 2, 3
);

alignas(64) static const __m256i m256_xor_lut_12 = _mm256_set_epi8(
	2, 1, 1, 0, 3, 2, 2, 1, 3, 2, 2, 1, 4, 3, 3, 2,
	2, 1, 1, 0, 3, 2, 2, 1, 3, 2, 2, 1, 4, 3, 3, 2
);

alignas(64) static const __m256i m256_xor_lut_13 = _mm256_set_epi8(
	1, 2, 0, 1, 2, 3, 1, 2, 2, 3, 1, 2, 3, 4, 2, 3,
	1, 2, 0, 1, 2, 3, 1, 2, 2, 3, 1, 2, 3, 4, 2, 3
);

alignas(64) static const __m256i m256_xor_lut_14 = _mm256_set_epi8(
	1, 0, 2, 1, 2, 1, 3, 2, 2, 1, 3, 2, 3, 2, 4, 3,
	1, 0, 2, 1, 2, 1, 3, 2, 2, 1, 3, 2, 3, 2, 4, 3
);

alignas(64) static const __m256i m256_xor_lut_15 = _mm256_set_epi8(
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
);


alignas(64) static const __m256i m256_intersection_lookup_tables[16] = {
    m256_intersection_lut_0, m256_intersection_lut_1, m256_intersection_lut_2, m256_intersection_lut_3,
    m256_intersection_lut_4, m256_intersection_lut_5, m256_intersection_lut_6, m256_intersection_lut_7,
    m256_intersection_lut_8, m256_intersection_lut_9, m256_intersection_lut_10, m256_intersection_lut_11,
    m256_intersection_lut_12, m256_intersection_lut_13, m256_intersection_lut_14, m256_intersection_lut_15
};

alignas(64) static const __m256i m256_union_lookup_tables[16] = {
    m256_union_lut_0, m256_union_lut_1, m256_union_lut_2, m256_union_lut_3,
    m256_union_lut_4, m256_union_lut_5, m256_union_lut_6, m256_union_lut_7,
    m256_union_lut_8, m256_union_lut_9, m256_union_lut_10, m256_union_lut_11,
    m256_union_lut_12, m256_union_lut_13, m256_union_lut_14, m256_union_lut_15
};

alignas(64) static const __m256i m256_xor_lookup_tables[16] = {
    m256_xor_lut_0, m256_xor_lut_1, m256_xor_lut_2, m256_xor_lut_3,
    m256_xor_lut_4, m256_xor_lut_5, m256_xor_lut_6, m256_xor_lut_7,
    m256_xor_lut_8, m256_xor_lut_9, m256_xor_lut_10, m256_xor_lut_11,
    m256_xor_lut_12, m256_xor_lut_13, m256_xor_lut_14, m256_xor_lut_15
};