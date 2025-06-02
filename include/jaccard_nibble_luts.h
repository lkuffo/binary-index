static const uint8_t intersection_lut_0[16]  = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static const uint8_t intersection_lut_1[16]  = {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
static const uint8_t intersection_lut_2[16]  = {0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1};
static const uint8_t intersection_lut_3[16]  = {0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2};
static const uint8_t intersection_lut_4[16]  = {0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1};
static const uint8_t intersection_lut_5[16]  = {0,1,0,1,1,2,1,2,0,1,0,1,1,2,1,2};
static const uint8_t intersection_lut_6[16]  = {0,0,1,1,1,1,2,2,0,0,1,1,1,1,2,2};
static const uint8_t intersection_lut_7[16]  = {0,1,1,2,1,2,2,3,0,1,1,2,1,2,2,3};
static const uint8_t intersection_lut_8[16]  = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1};
static const uint8_t intersection_lut_9[16]  = {0,1,0,1,0,1,0,1,1,2,1,2,1,2,1,2};
static const uint8_t intersection_lut_10[16] = {0,0,1,1,0,0,1,1,1,1,2,2,1,1,2,2};
static const uint8_t intersection_lut_11[16] = {0,1,1,2,0,1,1,2,1,2,2,3,1,2,2,3};
static const uint8_t intersection_lut_12[16] = {0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2};
static const uint8_t intersection_lut_13[16] = {0,1,0,1,1,2,1,2,1,2,1,2,2,3,2,3};
static const uint8_t intersection_lut_14[16] = {0,0,1,1,1,1,2,2,1,1,2,2,2,2,3,3};
static const uint8_t intersection_lut_15[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};

static const uint8_t union_lut_0[16]  = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
static const uint8_t union_lut_1[16]  = {1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3, 4, 4};
static const uint8_t union_lut_2[16]  = {1, 2, 1, 2, 2, 3, 2, 3, 2, 3, 2, 3, 3, 4, 3, 4};
static const uint8_t union_lut_3[16]  = {2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4};
static const uint8_t union_lut_4[16]  = {1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4};
static const uint8_t union_lut_5[16]  = {2, 2, 3, 3, 2, 2, 3, 3, 3, 3, 4, 4, 3, 3, 4, 4};
static const uint8_t union_lut_6[16]  = {2, 3, 2, 3, 2, 3, 2, 3, 3, 4, 3, 4, 3, 4, 3, 4};
static const uint8_t union_lut_7[16]  = {3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
static const uint8_t union_lut_8[16]  = {1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4};
static const uint8_t union_lut_9[16]  = {2, 2, 3, 3, 3, 3, 4, 4, 2, 2, 3, 3, 3, 3, 4, 4};
static const uint8_t union_lut_10[16] = {2, 3, 2, 3, 3, 4, 3, 4, 2, 3, 2, 3, 3, 4, 3, 4};
static const uint8_t union_lut_11[16] = {3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4};
static const uint8_t union_lut_12[16] = {2, 3, 3, 4, 2, 3, 3, 4, 2, 3, 3, 4, 2, 3, 3, 4};
static const uint8_t union_lut_13[16] = {3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 4, 4};
static const uint8_t union_lut_14[16] = {3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4};
static const uint8_t union_lut_15[16] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

alignas(64) static const uint8_t xor_lut_0[16] = {
	4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
};

alignas(64) static const uint8_t xor_lut_1[16] = {
	3, 4, 2, 3, 2, 3, 1, 2, 2, 3, 1, 2, 1, 2, 0, 1
};

alignas(64) static const uint8_t xor_lut_2[16] = {
	3, 2, 4, 3, 2, 1, 3, 2, 2, 1, 3, 2, 1, 0, 2, 1
};

alignas(64) static const uint8_t xor_lut_3[16] = {
	2, 3, 3, 4, 1, 2, 2, 3, 1, 2, 2, 3, 0, 1, 1, 2
};

alignas(64) static const uint8_t xor_lut_4[16] = {
	3, 2, 2, 1, 4, 3, 3, 2, 2, 1, 1, 0, 3, 2, 2, 1
};

alignas(64) static const uint8_t xor_lut_5[16] = {
	2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 0, 1, 2, 3, 1, 2
};

alignas(64) static const uint8_t xor_lut_6[16] = {
	2, 1, 3, 2, 3, 2, 4, 3, 1, 0, 2, 1, 2, 1, 3, 2
};

alignas(64) static const uint8_t xor_lut_7[16] = {
	1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3
};

alignas(64) static const uint8_t xor_lut_8[16] = {
	3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1
};

alignas(64) static const uint8_t xor_lut_9[16] = {
	2, 3, 1, 2, 1, 2, 0, 1, 3, 4, 2, 3, 2, 3, 1, 2
};

alignas(64) static const uint8_t xor_lut_10[16] = {
	2, 1, 3, 2, 1, 0, 2, 1, 3, 2, 4, 3, 2, 1, 3, 2
};

alignas(64) static const uint8_t xor_lut_11[16] = {
	1, 2, 2, 3, 0, 1, 1, 2, 2, 3, 3, 4, 1, 2, 2, 3
};

alignas(64) static const uint8_t xor_lut_12[16] = {
	2, 1, 1, 0, 3, 2, 2, 1, 3, 2, 2, 1, 4, 3, 3, 2
};

alignas(64) static const uint8_t xor_lut_13[16] = {
	1, 2, 0, 1, 2, 3, 1, 2, 2, 3, 1, 2, 3, 4, 2, 3
};

alignas(64) static const uint8_t xor_lut_14[16] = {
	1, 0, 2, 1, 2, 1, 3, 2, 2, 1, 3, 2, 3, 2, 4, 3
};

alignas(64) static const uint8_t xor_lut_15[16] = {
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
};

static const uint8_t* intersection_lookup_tables[16] = {
    intersection_lut_0, intersection_lut_1, intersection_lut_2, intersection_lut_3,
    intersection_lut_4, intersection_lut_5, intersection_lut_6, intersection_lut_7,
    intersection_lut_8, intersection_lut_9, intersection_lut_10, intersection_lut_11,
    intersection_lut_12, intersection_lut_13, intersection_lut_14, intersection_lut_15
};

static const uint8_t* union_lookup_tables[16] = {
    union_lut_0, union_lut_1, union_lut_2, union_lut_3,
    union_lut_4, union_lut_5, union_lut_6, union_lut_7,
    union_lut_8, union_lut_9, union_lut_10, union_lut_11,
    union_lut_12, union_lut_13, union_lut_14, union_lut_15
};

static const uint8_t* xor_lookup_tables[16] = {
    xor_lut_0, xor_lut_1, xor_lut_2, xor_lut_3,
    xor_lut_4, xor_lut_5, xor_lut_6, xor_lut_7,
    xor_lut_8, xor_lut_9, xor_lut_10, xor_lut_11,
    xor_lut_12, xor_lut_13, xor_lut_14, xor_lut_15
};