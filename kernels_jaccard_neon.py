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
    return 1.0 - intersection / union  # TODO: Avoid division by zero without returning wrong results

@cfunc(types.uint64(types.uint64))
def popcount_u64_numba(v):
    v -= (v >> 1) & 0x5555555555555555
    v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v * 0x0101010101010101) >> 56
    return v

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


# TODO: Dynamically load based on CPU
cppyy.load_library('./include/neon_jaccard.dylib')

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

float jaccard_b128_vpopcntq(uint8_t const *first_vector, uint8_t const *second_vector);
float jaccard_b128_serial_lut(uint8_t const *first_vector, uint8_t const *second_vector);
float jaccard_b128_serial_builtinpopcnt(uint8_t const *first_vector, uint8_t const *second_vector);
float jaccard_b1536_serial(uint8_t const *first_vector, uint8_t const *second_vector);
float jaccard_b128_vpopcntq_wordbyword(uint8_t const *first_vector, uint8_t const *second_vector);
float jaccard_b1536_vpopcntq_wordbyword(uint8_t const *first_vector, uint8_t const *second_vector);

void jaccard_b128_serial_lut_nibble_pdx(uint8_t const *first_vector, uint8_t const *second_vector);
void jaccard_b128_serial_lut_byte_pdx(uint8_t const *first_vector, uint8_t const *second_vector);
void jaccard_b128_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector);
void jaccard_b256_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector);

enum JaccardKernel {
    JACCARD_B128_VPOPCNTQ,
    JACCARD_B128_SERIAL_LUT,
    JACCARD_B128_SERIAL_BUILTINPOPCNT,
    JACCARD_B128_VPOPCNTQ_WORDBYWORD,
    JACCARD_B128_SERIAL_LUT_NIBBLE_PDX,
    JACCARD_B128_SERIAL_LUT_BYTE_PDX,
    JACCARD_B128_VPOPCNTQ_PDX,
    JACCARD_B256_VPOPCNTQ_PDX,
    JACCARD_B1536_SERIAL,
    JACCARD_B1536_VPOPCNTQ_WORDBYWORD
};

std::vector<KNNCandidate> jaccard_standalone(
    JaccardKernel kernel,
    uint8_t const *first_vector,
    uint8_t const *second_vector,
    size_t num_queries,
    size_t num_vectors,
    size_t knn);

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
        kind=MetricKind.Jaccard,
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
        kernel=cppyy.gbl.JaccardKernel.JACCARD_B128_VPOPCNTQ,
        query_count: int = 1000
) -> dict:
    queries = vectors.copy()[:query_count]
    bits_per_vector = vectors.shape[1] * 8

    start = time.perf_counter()
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
        query_count: int = 1000
) -> dict:
    queries = vectors.copy()[:query_count]
    bits_per_vector = vectors.shape[1] * 8

    if len(vectors) % 256 != 0:
        raise Exception('Number of vectors must be divisible by 256')

    vectors_pdx = row_major_to_pdx(vectors, 256)

    start = time.perf_counter()
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
    standalone_kernels_cpp_128d = [
        (
            'JACCARD_B128_VPOPCNTQ',
            cppyy.gbl.jaccard_b128_vpopcntq,
            cppyy.gbl.JaccardKernel.JACCARD_B128_VPOPCNTQ
        ),
        (
            'JACCARD_B128_SERIAL_LUT',
            cppyy.gbl.jaccard_b128_serial_lut,
            cppyy.gbl.JaccardKernel.JACCARD_B128_SERIAL_LUT
        ),
        (
            'JACCARD_B128_VPOPCNTQ_WORDBYWORD',
            cppyy.gbl.jaccard_b128_vpopcntq_wordbyword,
            cppyy.gbl.JaccardKernel.JACCARD_B128_VPOPCNTQ_WORDBYWORD
        ),
        (
            'JACCARD_B128_SERIAL_BUILTINPOPCNT',
            cppyy.gbl.jaccard_b128_serial_builtinpopcnt,
            cppyy.gbl.JaccardKernel.JACCARD_B128_SERIAL_BUILTINPOPCNT
        ),
    ]
    standalone_kernels_cpp_1536d = [
        (
            'JACCARD_B1536_SERIAL',
            cppyy.gbl.jaccard_b1536_serial,
            cppyy.gbl.JaccardKernel.JACCARD_B1536_SERIAL
        ),
        (
            'JACCARD_B1536_VPOPCNTQ_WORDBYWORD',
            cppyy.gbl.jaccard_b1536_vpopcntq_wordbyword,
            cppyy.gbl.JaccardKernel.JACCARD_B1536_VPOPCNTQ_WORDBYWORD
        ),
    ]
    kernels_numba_128d = [
        # Baselines:
        (
            "jaccard_u64x2_numba",
            jaccard_u64x2_numba,
            jaccard_u64x2_numba.address,
        ),
    ]
    standalone_kernels_cpp_pdx_128d = [
        (
            "JACCARD_B128_SERIAL_LUT_NIBBLE_PDX",
            cppyy.gbl.jaccard_b128_serial_lut_nibble_pdx,
            cppyy.gbl.JaccardKernel.JACCARD_B128_SERIAL_LUT_NIBBLE_PDX
        ),
        (
            "JACCARD_B128_SERIAL_LUT_BYTE_PDX",
            cppyy.gbl.jaccard_b128_serial_lut_byte_pdx,
            cppyy.gbl.JaccardKernel.JACCARD_B128_SERIAL_LUT_BYTE_PDX
        ),
        (
            "JACCARD_B128_VPOPCNTQ_PDX",
            cppyy.gbl.jaccard_b128_vpopcntq_pdx,
            cppyy.gbl.JaccardKernel.JACCARD_B128_VPOPCNTQ_PDX
        )
    ]
    standalone_kernels_cpp_pdx_256d = [
        (
            "JACCARD_B256_VPOPCNTQ_PDX",
            cppyy.gbl.jaccard_b256_vpopcntq_pdx,
            cppyy.gbl.JaccardKernel.JACCARD_B256_VPOPCNTQ_PDX
        )
    ]

    # Group kernels by dimension:
    kernels_cpp_per_dimension = {
        128: standalone_kernels_cpp_128d,
        1536: standalone_kernels_cpp_1536d,
    }
    kernels_numba_per_dimension = {
        128: kernels_numba_128d,
    }
    kernels_cpp_pdx = {
        128: standalone_kernels_cpp_pdx_128d,
        256: standalone_kernels_cpp_pdx_256d
    }

    if query_count == -1:
        query_count = count

    # Check which dimensions should be covered:
    for ndim in ndims:
        print("-" * 80)
        print(f"Testing {ndim:,}d kernels, {count} queries, {query_count}, queries")
        kernels_cpp = kernels_cpp_per_dimension.get(ndim, [])
        kernels_numba = kernels_numba_per_dimension.get(ndim, [])
        kernels_cpp_pdx = kernels_cpp_pdx.get(ndim, [])
        vectors = generate_random_vectors(count, ndim)

        # Run a few tests on this data:
        tests_per_kernel = 10
        for name, accelerated_kernel, _ in kernels_cpp:
            for _ in range(tests_per_kernel):
                first_vector_index = np.random.randint(0, count)
                second_vector_index = np.random.randint(0, count)
                first_vector, second_vector = (
                    vectors[first_vector_index],
                    vectors[second_vector_index],
                )
                baseline_distance = jaccard_numpy(first_vector, second_vector)
                accelerated_distance = accelerated_kernel(first_vector, second_vector)
                assert (
                        abs(baseline_distance - accelerated_distance) < 1e-5
                ), f"Distance mismatch for {name} kernel: {baseline_distance} vs {accelerated_distance}"

        print("- passed!")

        # Provide FAISS benchmarking baselines:
        print(f"Profiling FAISS over {count:,} vectors and {query_count} queries with Jaccard metric")
        stats = bench_faiss(
            vectors=vectors,
            k=k,
            threads=threads,
            query_count=query_count
        )
        print(f"- BOP/S: {stats['bit_ops_per_s'] / 1e9:,.2f} G")
        print(f"- Elapsed: {stats['elapsed_s']:,.4f} s")
        print(f"- Recall@1: {stats['recalled_top_match'] / query_count:.2%}")

        for name, _, kernel_id in kernels_cpp:
            print(f"Profiling `{name}` in standalone c++ over {count:,} vectors and {query_count} queries")
            stats = bench_standalone(vectors=vectors, k=k, kernel=kernel_id, query_count=query_count)
            print(f"- BOP/S: {stats['bit_ops_per_s'] / 1e9:,.2f} G")
            print(f"- Elapsed: {stats['elapsed_s']:,.4f} s")
            print(f"- Recall@1: {stats['recalled_top_match'] / query_count:.2%}")

        for name, _, kernel_id in kernels_cpp_pdx:
            print(f"Profiling `{name}` in standalone c++ with the PDX layout over {count:,} vectors and {query_count} queries")
            stats = bench_standalone_pdx(vectors=vectors, k=k, kernel=kernel_id, query_count=query_count)
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
        default=10,
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
