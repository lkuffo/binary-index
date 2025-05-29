#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "numba",
#   "cppyy",
#   "usearch",
#   "faiss-cpu",
# ]
# ///

# ? Uses USearch native functionality for exact search with custom metrics
# ? to benchmark the efficiency of various Jaccard similarity implementations.
# ? Usage examples:
# ?
# ?   uv run --script kernels_jaccard_avx.py
# ?   uv run --script kernels_jaccard_avx.py --count 100 --ndims 1024
# ?   uv run --script kernels_jaccard_avx.py --count 10000 --ndims "512,1024,1536" --k 1
# ?
# ? The last example will compute 10K by 10K meaning 100M distances for 512-bit,
# ? 1024-bit and 1536-bit vectors. For each, only the top-1 nearest neighbor will
# ? be fetched.
from typing import List, Literal
import time
import os
os.environ["EXTRA_CLING_ARGS"] = "-O3 -march=native"

import cppyy
import cppyy.ll
import numpy as np
from numba import cfunc, types, carray
from faiss import (
    METRIC_Jaccard as FAISS_METRIC_JACCARD,
    omp_set_num_threads as faiss_set_threads,
)
from faiss.contrib.exhaustive_search import knn as faiss_knn
from usearch.index import (
    Index,
    CompiledMetric,
    MetricKind,
    MetricSignature,
    ScalarKind,
    BatchMatches,
    search,
)


def popcount_reduce_numpy(a: np.ndarray) -> np.ndarray:
    return np.unpackbits(a).astype(np.uint16).sum()


def jaccard_numpy(a: np.ndarray, b: np.ndarray) -> float:
    intersection = popcount_reduce_numpy(np.bitwise_and(a, b))
    union = popcount_reduce_numpy(np.bitwise_or(a, b))
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero


@cfunc(types.uint64(types.uint64))
def popcount_u64_numba(v):
    v -= (v >> 1) & 0x5555555555555555
    v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v * 0x0101010101010101) >> 56
    return v


@cfunc(types.float32(types.CPointer(types.uint64), types.CPointer(types.uint64)))
def jaccard_u64x2_numba(a, b):
    a_array = carray(a, 4)
    b_array = carray(b, 4)
    intersection = 0
    union = 0
    for i in range(2):
        intersection += popcount_u64_numba(a_array[i] & b_array[i])
        union += popcount_u64_numba(a_array[i] | b_array[i])
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero

# region: 256d kernels


@cfunc(types.float32(types.CPointer(types.uint64), types.CPointer(types.uint64)))
def jaccard_u64x4_numba(a, b):
    a_array = carray(a, 4)
    b_array = carray(b, 4)
    intersection = 0
    union = 0
    for i in range(4):
        intersection += popcount_u64_numba(a_array[i] & b_array[i])
        union += popcount_u64_numba(a_array[i] | b_array[i])
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero


# endregion


# region: 1024d kernels


@cfunc(types.float32(types.CPointer(types.uint8), types.CPointer(types.uint8)))
def jaccard_u8x128_numba(a, b):
    a_array = carray(a, 128)
    b_array = carray(b, 128)
    intersection = 0
    union = 0
    for i in range(128):
        intersection += popcount_u64_numba(a_array[i] & b_array[i])
        union += popcount_u64_numba(a_array[i] | b_array[i])
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero


@cfunc(types.float32(types.CPointer(types.uint64), types.CPointer(types.uint64)))
def jaccard_u64x16_numba(a, b):
    a_array = carray(a, 16)
    b_array = carray(b, 16)
    intersection = 0
    union = 0
    for i in range(16):
        intersection += popcount_u64_numba(a_array[i] & b_array[i])
        union += popcount_u64_numba(a_array[i] | b_array[i])
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero


# endregion

# region: 1536d kernels

@cfunc(types.float32(types.CPointer(types.uint64), types.CPointer(types.uint64)))
def jaccard_u64x24_numba(a, b):
    a_array = carray(a, 24)
    b_array = carray(b, 24)
    intersection = 0
    union = 0
    for i in range(24):
        intersection += popcount_u64_numba(a_array[i] & b_array[i])
        union += popcount_u64_numba(a_array[i] | b_array[i])
    return 1.0 - (intersection + 1.0) / (union + 1.0)  # ! Avoid division by zero


# endregion

cppyy.load_library('./include/avx512_jaccard.dylib')

cppyy.cppdef("""
struct KNNCandidate {
    uint32_t index;
    uint32_t distance;
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
    JACCARD_B256_VPSHUFB_SAD_PRECOMPUTED,
    JACCARD_B256_VPOPCNTQ,
    JACCARD_B256_VPOPCNTQ_PRECOMPUTED, // TODO
    JACCARD_B256_VPOPCNTQ_PDX,
    JACCARD_B256_VPOPCNTQ_PRECOMPUTED_PDX, // TODO
    JACCARD_B256_VPSHUFB_PDX,
    JACCARD_B256_VPSHUFB_PRECOMPUTED_PDX, // TODO
    JACCARD_B256_VPOPCNTQ_VPSHUFB_PDX,
    // 512
    JACCARD_B512_VPSHUFB_SAD, // TODO
    JACCARD_B512_VPSHUFB_SAD_PRECOMPUTED, // TODO
    JACCARD_B512_VPOPCNTQ, // TODO
    JACCARD_B512_VPOPCNTQ_PRECOMPUTED, // TODO
    JACCARD_B512_VPOPCNTQ_PDX, // TODO
    JACCARD_B512_VPOPCNTQ_PRECOMPUTED_PDX, // TODO
    JACCARD_B512_VPSHUFB_PDX, // TODO
    JACCARD_B512_VPSHUFB_PRECOMPUTED_PDX, // TODO
    JACCARD_B512_VPOPCNTQ_VPSHUFB_PDX, // TODO
    // 1024
    JACCARD_U8X128_C,
    JACCARD_U64X16_C,
    JACCARD_B1024_VPOPCNTQ,
    JACCARD_B1024_VPOPCNTQ_PRECOMPUTED,
    JACCARD_B1024_VPSHUFB_SAD,
    JACCARD_B1024_VPSHUFB_SAD_PRECOMPUTED, // TODO
    JACCARD_B1024_VPSHUFB_DPB,
    JACCARD_U64X16_CSA3_C,
    JACCARD_U64X16_CSA15_CPP,
    JACCARD_B1024_VPOPCNTQ_PDX,
    JACCARD_B1024_VPOPCNTQ_PRECOMPUTED_PDX,
    JACCARD_B1024_VPOPCNTQ_VPSHUFB_PDX, // TODO
    JACCARD_B1024_VPSHUFB_PRECOMPUTED_PDX, // TODO
    // 1536
    JACCARD_U64X24_C,
    JACCARD_B1536_VPOPCNTQ,
    JACCARD_B1536_VPOPCNTQ_3CSA
};

//
// 256 region
//
float jaccard_u64x4_c(uint8_t const *a, uint8_t const *b);
__attribute__((target("avx2,bmi2,avx")))
float jaccard_b256_vpshufb_sad(uint8_t const *first_vector, uint8_t const *second_vector);
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b256_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector);
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b1024_vpopcntq_precomputed(
    uint8_t const *first_vector, uint8_t const *second_vector,
    uint32_t const popcount_first, uint32_t const popcount_second
);
__attribute__((target("avx2,bmi2,avx")))
float jaccard_b256_vpshufb_sad_precomputed(
    uint8_t const *first_vector, uint8_t const *second_vector,
    uint32_t const first_popcount, uint32_t const second_popcount
);
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b256_vpopcntq_precomputed(
    uint8_t const *first_vector, uint8_t const *second_vector,
    uint32_t const first_popcount, uint32_t const second_popcount
);

void jaccard_b256_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector);
void jaccard_b256_vpshufb_pdx(uint8_t const *first_vector, uint8_t const *second_vector);
void jaccard_b1024_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector);
void jaccard_b256_vpopcntq_vpshufb_pdx(uint8_t const *first_vector, uint8_t const *second_vector);
void jaccard_b1024_vpopcntq_precomputed_pdx(uint8_t const *first_vector, uint8_t const *second_vector, uint32_t const first_popcount, uint32_t const *second_popcounts);


//
// 1024 region
//
float jaccard_u8x128_c(uint8_t const *a, uint8_t const *b);
float jaccard_u64x16_c(uint8_t const *a, uint8_t const *b);
float jaccard_b1024_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector);
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b1024_vpshufb_sad(uint8_t const *first_vector, uint8_t const *second_vector);
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b1024_vpshufb_dpb(uint8_t const *first_vector, uint8_t const *second_vector);
float jaccard_u64x16_csa3_c(uint8_t const *a, uint8_t const *b);
float jaccard_u64x16_csa15_cpp(uint8_t const *a, uint8_t const *b);

//
// 1536 region
//
float jaccard_u64x24_c(uint8_t const *a, uint8_t const *b);
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b1536_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector);
__attribute__((target("avx512f,avx512vl,bmi2,avx512bw,avx512dq")))
float jaccard_b1536_vpopcntq_3csa(uint8_t const *first_vector, uint8_t const *second_vector);

std::vector<KNNCandidate> jaccard_standalone(
    JaccardKernel kernel,
    uint8_t const *first_vector,
    uint8_t const *second_vector,
    size_t num_queries,
    size_t num_vectors,
    size_t knn,
    uint32_t const *precomputed_popcnts = nullptr
);

""")


def generate_random_vectors(count: int, bits_per_vector: int) -> np.ndarray:
    bools = np.random.randint(0, 2, size=(count, bits_per_vector), dtype=np.uint8)
    bits = np.packbits(bools, axis=1)
    return bits


def bench_faiss(
    vectors: np.ndarray,
    k: int,
    threads: int,
    query_count: int = 1000
) -> dict:

    faiss_set_threads(threads)
    n = vectors.shape[0]
    queries = vectors.copy()[:query_count]
    start = time.perf_counter()
    _, matches = faiss_knn(vectors, queries, k, metric=FAISS_METRIC_JACCARD)
    elapsed = time.perf_counter() - start

    computed_distances = n * len(queries)
    recalled_top_match = int((matches[:, 0] == np.arange(n)).sum())
    bits_per_vector = vectors.shape[1] * 8
    bit_ops_per_distance = bits_per_vector * 2
    return {
        "elapsed_s": elapsed,
        "computed_distances": computed_distances,
        "visited_members": computed_distances,
        "bit_ops_per_s": computed_distances * bit_ops_per_distance / elapsed,
        "recalled_top_match": recalled_top_match,
    }


def bench_kernel(
    kernel_pointer: int,
    vectors: np.ndarray,
    k: int,
    threads: int,
    approximate: bool,
    query_count: int = 1000
) -> dict:

    keys: np.ndarray = np.arange(vectors.shape[0], dtype=np.uint64)
    compiled_metric = CompiledMetric(
        pointer=kernel_pointer,
        kind=MetricKind.Tanimoto,
        signature=MetricSignature.ArrayArray,
    )
    queries = vectors.copy()[:query_count]

    bits_per_vector = vectors.shape[1] * 8
    start = time.perf_counter()
    matches = None
    if not approximate:
        matches: BatchMatches = search(
            metric=compiled_metric,
            dataset=vectors,
            query=queries,
            count=k,  # ? Matches wanted per query
            exact=True,
            threads=threads,
        )
    else:
        index = Index(
            ndim=bits_per_vector,
            dtype=ScalarKind.B1,
            metric=compiled_metric,
        )
        index.add(keys, vectors, log=False)
        matches: BatchMatches = index.search(queries, k, log=False)

    # Reduce stats
    elapsed_s = time.perf_counter() - start
    bit_ops_per_distance = bits_per_vector * 2
    recalled_top_match: int = matches.count_matches(keys, count=1)
    return {
        "visited_members": matches.visited_members,
        "computed_distances": matches.computed_distances,
        "elapsed_s": elapsed_s,
        "bit_ops_per_s": matches.computed_distances * bit_ops_per_distance / elapsed_s,
        "recalled_top_match": recalled_top_match,
    }


def bench_standalone(
        vectors: np.ndarray,
        k: int,
        kernel=cppyy.gbl.JaccardKernel.JACCARD_B1024_VPOPCNTQ,
        query_count: int = 1000,
        kernel_name: str = "",
) -> dict:
    queries = vectors.copy()[:query_count]
    bits_per_vector = vectors.shape[1] * 8

    start = time.perf_counter()
    if "PRECOMPUTED" in kernel_name:
        data_popcounts = np.bitwise_count(vectors).sum(axis=1).astype(np.uint32)
        assert len(data_popcounts) == len(vectors)
        start = time.perf_counter()
        result = cppyy.gbl.jaccard_standalone(
            kernel,
            vectors, queries,
            len(queries), len(vectors), k, data_popcounts)
    else:
        result = cppyy.gbl.jaccard_standalone(
            kernel,
            vectors, queries,
            len(queries), len(vectors), k)
    elapsed_s = time.perf_counter() - start
    matches = []
    for i in range(0, (len(queries) * k), k):
        knn_candidate = result[i]
        matches.append(knn_candidate.index)

    # Reduce stats
    computed_distances = len(queries) * len(vectors)
    recalled_top_match = int((matches == np.arange(len(queries))).sum())
    bit_ops_per_distance = bits_per_vector * 2
    return {
        "visited_members": computed_distances,
        "computed_distances": computed_distances,
        "elapsed_s": elapsed_s,
        "bit_ops_per_s": computed_distances * bit_ops_per_distance / elapsed_s,
        "recalled_top_match": recalled_top_match,
    }


def row_major_to_pdx(vectors, block_size=256) -> np.ndarray:
    V, dims = vectors.shape  # V must be multiple of block_size
    chunks = V // block_size
    total_size = V * dims
    result = np.empty(total_size, dtype=vectors.dtype)
    cur_offset = 0
    for i in range(chunks):
        chunk = vectors[cur_offset: cur_offset + block_size, :]
        # Flatten chunk in Fortran order and put into result
        result[i * block_size * dims: (i + 1) * block_size * dims] = chunk.flatten(order='F')
        cur_offset += block_size
    return result

def bench_standalone_pdx(
        vectors: np.ndarray,
        k: int,
        kernel,
        query_count: int = 1000,
        kernel_name: str = ""
) -> dict:
    queries = vectors.copy()[:query_count]
    bits_per_vector = vectors.shape[1] * 8

    if len(vectors) % 256 != 0:
        raise Exception('Number of vectors must be divisible by 256')

    vectors_pdx = row_major_to_pdx(vectors, 256)

    start = time.perf_counter()
    if "PRECOMPUTED" in kernel_name:
        data_popcounts = np.bitwise_count(vectors).sum(axis=1).astype(np.uint32)
        assert len(data_popcounts) == len(vectors)
        start = time.perf_counter()
        result = cppyy.gbl.jaccard_standalone(
            kernel,
            vectors_pdx, queries,
            len(queries), len(vectors), k, data_popcounts)
    else:
        result = cppyy.gbl.jaccard_standalone(
            kernel,
            vectors_pdx, queries,
            len(queries), len(vectors), k)
    elapsed_s = time.perf_counter() - start
    matches = []
    for i in range(0, (len(queries) * k), k):
        knn_candidate = result[i]
        matches.append(knn_candidate.index)
    # Reduce stats
    # print(matches)
    computed_distances = len(queries) * len(vectors)
    recalled_top_match = int((matches == np.arange(len(queries))).sum())
    bit_ops_per_distance = bits_per_vector * 2
    return {
        "visited_members": computed_distances,
        "computed_distances": computed_distances,
        "elapsed_s": elapsed_s,
        "bit_ops_per_s": computed_distances * bit_ops_per_distance / elapsed_s,
        "recalled_top_match": recalled_top_match,
    }


def main(
    count: int,
    k: int = 1,
    ndims: List[int] = [256, 1024, 1536],
    approximate: bool = True,
    threads: int = 1,
    query_count: int = -1
):

    kernels_cpp_256d = [
        # C++:
        (
            'JACCARD_U64X4_C',
            cppyy.gbl.jaccard_u64x4_c,
            cppyy.gbl.JaccardKernel.JACCARD_U64X4_C
        ),
        (
            "JACCARD_B256_VPSHUFB_SAD",
            cppyy.gbl.jaccard_b256_vpshufb_sad,
            cppyy.gbl.JaccardKernel.JACCARD_B256_VPSHUFB_SAD
        ),
        (
            "JACCARD_B256_VPOPCNTQ",
            cppyy.gbl.jaccard_b256_vpopcntq,
            cppyy.gbl.JaccardKernel.JACCARD_B256_VPOPCNTQ
        ),
        (
            "JACCARD_B256_VPOPCNTQ_PRECOMPUTED",
            cppyy.gbl.jaccard_b256_vpopcntq_precomputed,
            cppyy.gbl.JaccardKernel.JACCARD_B256_VPOPCNTQ_PRECOMPUTED
        ),
        (
            "JACCARD_B256_VPSHUFB_SAD_PRECOMPUTED",
            cppyy.gbl.jaccard_b256_vpshufb_sad_precomputed,
            cppyy.gbl.JaccardKernel.JACCARD_B256_VPSHUFB_SAD_PRECOMPUTED
        )
    ]
    kernels_numba_256d = [
        # Baselines:
        (
            "jaccard_u64x4_numba",
            jaccard_u64x4_numba,
            jaccard_u64x4_numba.address,
        ),
    ]

    kernels_cpp_1024d = [
        # C++:
        # (
        #     "JACCARD_U64X16_C",
        #     cppyy.gbl.jaccard_u64x16_c,
        #     cppyy.gbl.JaccardKernel.JACCARD_U64X16_C
        # ),
        # (
        #     "JACCARD_U8X128_C",
        #     cppyy.gbl.jaccard_u8x128_c,
        #     cppyy.gbl.JaccardKernel.JACCARD_U8X128_C
        # ),
        # (
        #     "JACCARD_U64X16_CSA3_C",
        #     cppyy.gbl.jaccard_u64x16_csa3_c,
        #     cppyy.gbl.JaccardKernel.JACCARD_U64X16_CSA3_C
        # ),
        # (
        #     "JACCARD_U64X16_CSA15_CPP",
        #     cppyy.gbl.jaccard_u64x16_csa15_cpp,
        #     cppyy.gbl.JaccardKernel.JACCARD_U64X16_CSA15_CPP
        # ),
        # SIMD:
        (
            "JACCARD_B1024_VPOPCNTQ",
            cppyy.gbl.jaccard_b1024_vpopcntq,
            cppyy.gbl.JaccardKernel.JACCARD_B1024_VPOPCNTQ
        ),
        (
            "JACCARD_B1024_VPOPCNTQ_PRECOMPUTED",
            cppyy.gbl.jaccard_b1024_vpopcntq_precomputed,
            cppyy.gbl.JaccardKernel.JACCARD_B1024_VPOPCNTQ_PRECOMPUTED
        ),
        # (
        #     "JACCARD_B1024_VPSHUFB_SAD",
        #     cppyy.gbl.jaccard_b1024_vpshufb_sad,
        #     cppyy.gbl.JaccardKernel.JACCARD_B1024_VPSHUFB_SAD
        # ),
        # (
        #     "JACCARD_B1024_VPSHUFB_DPB",
        #     cppyy.gbl.jaccard_b1024_vpshufb_dpb,
        #     cppyy.gbl.JaccardKernel.JACCARD_B1024_VPSHUFB_DPB
        # ),
    ]
    kernels_numba_1024d = [
        # Baselines:
        (
            "jaccard_u64x16_numba",
            jaccard_u64x16_numba,
            jaccard_u64x16_numba.address,
        ),
        # ! Slower irrelevant kernels in the end if someone has the patience:
        (
            "jaccard_u8x128_numba",
            jaccard_u8x128_numba,
            jaccard_u8x128_numba.address,
        ),
    ]
    kernels_cpp_1536d = [
        # C++:
        (
            "JACCARD_U64X24_C",
            cppyy.gbl.jaccard_u64x24_c,
            cppyy.gbl.JaccardKernel.JACCARD_U64X24_C
        ),
        # SIMD:
        (
            "JACCARD_B1536_VPOPCNTQ",
            cppyy.gbl.jaccard_b1536_vpopcntq,
            cppyy.gbl.JaccardKernel.JACCARD_B1536_VPOPCNTQ
        ),
        (
            "JACCARD_B1536_VPOPCNTQ_3CSA",
            cppyy.gbl.jaccard_b1536_vpopcntq_3csa,
            cppyy.gbl.JaccardKernel.JACCARD_B1536_VPOPCNTQ_3CSA
        ),
    ]
    kernels_numba_1536d = [
        # Baselines:
        (
            "jaccard_u64x24_numba",
            jaccard_u64x24_numba,
            jaccard_u64x24_numba.address,
        ),
    ]
    standalone_kernels_cpp_pdx_256d = [
        (
            "JACCARD_B256_VPOPCNTQ_PDX",
            cppyy.gbl.jaccard_b256_vpopcntq_pdx,
            cppyy.gbl.JaccardKernel.JACCARD_B256_VPOPCNTQ_PDX
        ),
        (
            "JACCARD_B256_VPSHUFB_PDX",
            cppyy.gbl.jaccard_b256_vpshufb_pdx,
            cppyy.gbl.JaccardKernel.JACCARD_B256_VPSHUFB_PDX
        ),
        (
            "JACCARD_B256_VPOPCNTQ_VPSHUFB_PDX",
            cppyy.gbl.jaccard_b256_vpopcntq_vpshufb_pdx,
            cppyy.gbl.JaccardKernel.JACCARD_B256_VPOPCNTQ_VPSHUFB_PDX
        ),
    ]
    standalone_kernels_cpp_pdx_1024d = [
        (
            "JACCARD_B1024_VPOPCNTQ_PDX",
            cppyy.gbl.jaccard_b1024_vpopcntq_pdx,
            cppyy.gbl.JaccardKernel.JACCARD_B1024_VPOPCNTQ_PDX
        ),
        (
            "JACCARD_B1024_VPOPCNTQ_PRECOMPUTED_PDX",
            cppyy.gbl.jaccard_b1024_vpopcntq_precomputed_pdx,
            cppyy.gbl.JaccardKernel.JACCARD_B1024_VPOPCNTQ_PRECOMPUTED_PDX
        ),
    ]

    # Group kernels by dimension:
    kernels_cpp_per_dimension = {
        256: kernels_cpp_256d,
        1024: kernels_cpp_1024d,
        1536: kernels_cpp_1536d,
    }
    kernels_numba_per_dimension = {
        256: kernels_numba_256d,
        1024: kernels_numba_1024d,
        1536: kernels_numba_1536d,
    }
    kernels_cpp_pdx = {
        256: standalone_kernels_cpp_pdx_256d,
        1024: standalone_kernels_cpp_pdx_1024d,
    }

    if query_count == -1:
        query_count = count

    # Check which dimensions should be covered:
    for ndim in ndims:
        print("-" * 80)
        print(f"Testing {ndim:,}d kernels")
        kernels_cpp = kernels_cpp_per_dimension.get(ndim, [])
        kernels_numba = kernels_numba_per_dimension.get(ndim, [])
        kernels_cpp_pdx = kernels_cpp_pdx.get(ndim, [])
        vectors = generate_random_vectors(count, ndim)

        # Run a few tests on this data:
        # tests_per_kernel = 10
        # for name, accelerated_kernel, _ in kernels_cpp:
        #     for _ in range(tests_per_kernel):
        #         first_vector_index = np.random.randint(0, count)
        #         second_vector_index = np.random.randint(0, count)
        #         first_vector, second_vector = (
        #             vectors[first_vector_index],
        #             vectors[second_vector_index],
        #         )
        #         baseline_distance = jaccard_numpy(first_vector, second_vector)
        #         accelerated_distance = accelerated_kernel(first_vector, second_vector)
        #         assert (
        #             abs(baseline_distance - accelerated_distance) < 1e-5
        #         ), f"Distance mismatch for {name} kernel: {baseline_distance} vs {accelerated_distance}"
        #
        # print("- passed!")

        # Provide FAISS benchmarking baselines:
        # print(f"Profiling FAISS over {count:,} vectors and {query_count} queries with Jaccard metric")
        # stats = bench_faiss(
        #     vectors=vectors,
        #     k=k,
        #     threads=threads,
        #     query_count = query_count
        # )
        # print(f"- BOP/S: {stats['bit_ops_per_s'] / 1e9:,.2f} G")
        # print(f"- Recall@1: {stats['recalled_top_match'] / query_count:.2%}")

        # Analyze all the kernels:
        for name, _, kernel_id in kernels_cpp:
            print(f"Profiling `{name}` in standalone c++ over {count:,} vectors and {query_count} queries")
            stats = bench_standalone(vectors=vectors, k=k, kernel=kernel_id, query_count=query_count, kernel_name=name)
            print(f"- BOP/S: {stats['bit_ops_per_s'] / 1e9:,.2f} G")
            print(f"- Elapsed: {stats['elapsed_s']:,.4f} s")
            print(f"- Recall@1: {stats['recalled_top_match'] / query_count:.2%}")

        for name, _, kernel_id in kernels_cpp_pdx:
            print(f"Profiling `{name}` in standalone c++ with the PDX layout over {count:,} vectors and {query_count} queries")
            stats = bench_standalone_pdx(vectors=vectors, k=k, kernel=kernel_id, query_count=query_count, kernel_name=name)
            print(f"- BOP/S: {stats['bit_ops_per_s'] / 1e9:,.2f} G")
            print(f"- Elapsed: {stats['elapsed_s']:,.4f} s")
            print(f"- Recall@1: {stats['recalled_top_match'] / query_count:.2%}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser(
        description="Comparing HPC kernels for Jaccard distance"
    )
    arg_parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of vectors to generate for the benchmark",
    )
    arg_parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of nearest neighbors to search for",
    )
    arg_parser.add_argument(
        "--ndims",
        type=int,
        nargs="+",
        default=[256, 1024, 1536],
        help="List of dimensions to test (e.g., 256, 1024, 1536)",
    )
    arg_parser.add_argument(
        "--approximate",
        action="store_true",
        help="Use approximate search instead of exact search",
    )
    arg_parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for the benchmark",
    )
    arg_parser.add_argument(
        "--query_count",
        type=int,
        default=-1,
        help="Number of queries to use for the benchmark",
    )
    args = arg_parser.parse_args()
    main(
        count=args.count,
        k=args.k,
        ndims=args.ndims,
        approximate=args.approximate,
        threads=args.threads,
        query_count=args.query_count
    )
