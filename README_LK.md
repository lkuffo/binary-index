This README goes through the optimizations tried on this fork to improve Jaccard and Hamming calculations for vector similarity search.
The code was changed to bypass USearch and always use a standalone implementation of KNN search with a `partial_sort`. Benchmarks were carried out on r7a.xlarge (Zen 4) and r7iz.xlarge (Sapphire Rapids). The full benchmarks results can be found [here](https://docs.google.com/spreadsheets/d/1NqGJHJUozaXX6Zdt_9gavu875oLaNzTXbh5ddq8vxq0/edit?usp=sharing). 

## Optimizations tried:

- Hybrid kernel: `VPOPCNTQ` + `VPSHUFB`
- Precomputed population counts
- Column-major layout
- Precomputed Jaccard JUTs

## Hybrid kernel: `VPOPCNTQ` + `VPSHUFB`
In Sapphire Rapids, both `VPOPCNTQ` and `VPSHUFB` are dispatched through port 5. However, in Zen4, port 0 is exclusive to `VPOPCNTQ`, and port 2 is exclusive to `VPSHUFB`, with both sharing port 1. Therefore, we implemented a LUT-based kernel that uses `VPSHUFB` for INTERSECTION and `VPOPCNTQ` for UNION.

```sh
# Zen 4
Profiling `JACCARD_B1024_VPOPCNTQ` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 438.16
- Elapsed (ms): 4901.1959
Profiling `JACCARD_B1024_VPOPCNTQ_VPSHUFB` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 268.52
- Elapsed (ms): 7997.3436
Profiling `JACCARD_B1024_VPSHUFB_SAD` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 223.92
- Elapsed (ms): 9590.215

Profiling `JACCARD_B1024_VPOPCNTQ` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 383.04
- Elapsed (ms): 5.475
Profiling `JACCARD_B1024_VPOPCNTQ_VPSHUFB` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 233.8
- Elapsed (ms): 8.9697
Profiling `JACCARD_B1024_VPSHUFB_SAD` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 198.35
- Elapsed (ms): 10.5731
```

```sh
# Sapphire Rapids
Profiling `JACCARD_B1024_VPOPCNTQ` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 182.6
- Elapsed (ms): 11760.7875
Profiling `JACCARD_B1024_VPOPCNTQ_VPSHUFB` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 163.68
- Elapsed (ms): 13120.1021
Profiling `JACCARD_B1024_VPSHUFB_SAD` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 153.33
- Elapsed (ms): 14005.9874
```

While being better than the `VPSHUFB` kernel, it is never better than the `VPOPCNTQ` kernel. Perhaps we are overlooking that the `VPSHUFB` kernel comes at the expense of additional instructions: 1 `VPADDB` (to be executed after the 2 `VPSHUFB`), 2 `VPANDQ` (one of them waiting for 1 `PSRLW`), 1 `PSRLW`, and 1 `VPSADBW`.


## Precomputed population counts
Given the Jaccard metric as:

$$
\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

We can avoid the calculation of ${|A \cup B|}$ if the popcounts of the vector collection are stored as metadata. This saves 1 `VPORQ` and 1 `VPOPCNTQ`, reducing the Jaccard kernel to just 1 `VPANDQ` and 1 `VPOPCNTQ`. In the LUT-based kernel, it saves even more instructions: 2 `VPSHUFB`, 1 `VPADDQ`, 2 `VPSADBW`, 2 `VPANDQ`, and 1 `VPSRLQ`. 

The tradeoff is 1 POPCOUNT for every query vector, which is amortized by all the distance calculations. There is also a storage overhead to store the popcounts of every vector in the collection. The latter is 3% for 1024-bit vectors and 6% for 512-bit vectors if we store them in 32 bits. However, storing them on 16 bits should be possible.

**One can get up to 50% performance improvement**, depending on whether the data fits in cache. 

```sh
# Zen 4
# 1024d
Profiling `JACCARD_B1024_VPOPCNTQ` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 383.04
- Elapsed (ms): 5.475
Profiling `JACCARD_B1024_VPOPCNTQ_PRECOMPUTED` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 500.92
- Elapsed (ms): 4.1866

Profiling `JACCARD_B1024_VPOPCNTQ` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 438.16
- Elapsed (ms): 4901.1959
Profiling `JACCARD_B1024_VPOPCNTQ_PRECOMPUTED` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 433.15
- Elapsed (ms): 4957.8258

# 512d
Profiling `JACCARD_B512_VPOPCNTQ` over 1048576 vectors and 1000 queries of 512d
- BOP/S: 297.06
- Elapsed (ms): 3614.5316
Profiling `JACCARD_B512_VPOPCNTQ_PRECOMPUTED` over 1048576 vectors and 1000 queries of 512d
- BOP/S: 335.33
- Elapsed (ms): 3202.0707
```

```sh
# Sapphire Rapids
# 1024d
Profiling `JACCARD_B1024_VPOPCNTQ` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 365.18
- Elapsed (ms): 5.7428
Profiling `JACCARD_B1024_VPOPCNTQ_PRECOMPUTED` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 487.58
- Elapsed (ms): 4.3012

Profiling `JACCARD_B1024_VPOPCNTQ` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 182.6
- Elapsed (ms): 11760.7875
Profiling `JACCARD_B1024_VPOPCNTQ_PRECOMPUTED` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 222.22
- Elapsed (ms): 9663.9672

#512d
Profiling `JACCARD_B512_VPOPCNTQ` over 524288 vectors and 1000 queries of 512d
- BOP/S: 253.27
- Elapsed (ms): 2119.7363
Profiling `JACCARD_B512_VPOPCNTQ_PRECOMPUTED` over 524288 vectors and 1000 queries of 512d
- BOP/S: 329.51
- Elapsed (ms): 1629.2936
```


## Column-major layout
Transposing the vector collection has some interesting effects on the kernels. In these kernels, distances are computed between a single query dimension (a dimension being 1 byte or 1 nibble) and multiple vectors at a time. The latter lets us seamlessly accumulate the metric into different SIMD lanes. 

In the `VPOPCNTQ` kernel, this eliminates the `_mm256_reduce_add_epi64` step. In the LUT-based kernel, this also eliminates the `VPSADBW` steps.

The transposition is done in blocks of 256 vectors to avoid intermediate LOAD/STORE instructions of the accumulated distances. 

A kernel on this layout looks like this:

```cpp
void jaccard_b256_vpopcntq_pdx(uint8_t const *first_vector, uint8_t const *second_vector) {
    // Init accumulators that fit 256 uint8_t values
    __m512i intersections_result[4];
    __m512i unions_result[4];
    for (size_t i = 0; i < 4; ++i) {
        intersections_result[i] = _mm512_setzero_si512();
        unions_result[i] = _mm512_setzero_si512();
    }
    
    // For every dimension
    for (size_t dim = 0; dim != 32; dim++){
        __m512i first = _mm512_set1_epi8(first_vector[dim]);
        // We accumulate the distance between that dimension and 256 vectors also in that dimension
        for (size_t i = 0; i < 4; i++){ 
            __m512i second = _mm512_loadu_epi8(second_vector);
            __m512i intersection = _mm512_popcnt_epi8(_mm512_and_epi64(first, second));
            __m512i union_ = _mm512_popcnt_epi8(_mm512_or_epi64(first, second));
            intersections_result[i] = _mm512_add_epi8(intersections_result[i], intersection);
            unions_result[i] = _mm512_add_epi8(unions_result[i], union_);
            second_vector += 64; 
        }
    }
    
    // Move the accumulated distances to main memory
    for (size_t i = 0; i < 4; i++) {
        _mm512_storeu_si512(intersections_tmp + (i * 64), intersections_result[i]);
        _mm512_storeu_si512(unions_tmp + (i * 64), unions_result[i]);
    }
    // Calculate the Jaccard index
    for (size_t i = 0; i < 256; i++){
        distances_tmp[i] = (unions_tmp[i] != 0) ? 1 - (float)intersections_tmp[i] / (float)unions_tmp[i] : 1.0f;
    }
}
```

**This kernels deliver interesting performance improvements** over their non-transposed counterpart when data does not spill to main memory. In the following benchmarks the suffix `_PDX` represents the kernels in the column-major layout.

```sh
# Zen 4
# 1024d
Profiling `JACCARD_B1024_VPOPCNTQ` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 383.04
- Elapsed (ms): 5.475
Profiling `JACCARD_B1024_VPOPCNTQ_PDX` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 524.38
- Elapsed (ms): 3.9993

Profiling `JACCARD_B1024_VPOPCNTQ` over 65536 vectors and 1000 queries of 1024d
- BOP/S: 502.9
- Elapsed (ms): 266.8891
Profiling `JACCARD_B1024_VPOPCNTQ_PDX` over 65536 vectors and 1000 queries of 1024d
- BOP/S: 756.64
- Elapsed (ms): 177.3861

Profiling `JACCARD_B1024_VPOPCNTQ` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 438.16
- Elapsed (ms): 4901.1959
Profiling `JACCARD_B1024_VPOPCNTQ_PDX` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 420.87
- Elapsed (ms): 5102.5275

# 256d
Profiling `JACCARD_U64X4_C` over 131072 vectors and 1000 queries of 256d
- BOP/S: 300.04
- Elapsed (ms): 223.6676
Profiling `JACCARD_B256_VPOPCNTQ` over 131072 vectors and 1000 queries of 256d
- BOP/S: 219.38
- Elapsed (ms): 305.8999
Profiling `JACCARD_B256_VPOPCNTQ_PDX` over 131072 vectors and 1000 queries of 256d
- BOP/S: 422.37
- Elapsed (ms): 158.8864
```

```sh
# Sapphire Rapids
# 1024d
Profiling `JACCARD_B1024_VPOPCNTQ` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 365.18
- Elapsed (ms): 5.7428
Profiling `JACCARD_B1024_VPOPCNTQ_PDX` over 1024 vectors and 1000 queries of 1024d
- BOP/S: 486.36
- Elapsed (ms): 4.3119

Profiling `JACCARD_B1024_VPOPCNTQ` over 65536 vectors and 1000 queries of 1024d
- BOP/S: 393.35
- Elapsed (ms): 341.2187
Profiling `JACCARD_B1024_VPOPCNTQ_PDX` over 65536 vectors and 1000 queries of 1024d
- BOP/S: 425.77
- Elapsed (ms): 315.238

Profiling `JACCARD_B1024_VPOPCNTQ` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 182.6
- Elapsed (ms): 11760.7875
Profiling `JACCARD_B1024_VPOPCNTQ_PDX` over 1048576 vectors and 1000 queries of 1024d
- BOP/S: 197.23
- Elapsed (ms): 10888.1483

# 256d
Profiling `JACCARD_U64X4_C` over 131072 vectors and 1000 queries of 256d
- BOP/S: 282.35
- Elapsed (ms): 237.6803
Profiling `JACCARD_B256_VPOPCNTQ` over 131072 vectors and 1000 queries of 256d
- BOP/S: 169.5
- Elapsed (ms): 395.9201
Profiling `JACCARD_B256_VPOPCNTQ_PDX` over 131072 vectors and 1000 queries of 256d
- BOP/S: 314.36
- Elapsed (ms): 213.4752
```

- 30% to 75% faster when data fits in cache.
- 3% to 10% faster when data does not fit in cache.
- Finds more benefits on Zen4 than in Sapphire Rapids.
- Performance degrades when data spills to main memory.
- Precomputed popcounts still speedup these kernels, being these the fastest versions so far:

```sh
# Zen 4
Profiling `JACCARD_B1024_VPOPCNTQ` over 4096 vectors and 1000 queries of 1024d
- BOP/S: 471.16
- Elapsed (ms): 17.8043
Profiling `JACCARD_B1024_VPOPCNTQ_PRECOMPUTED` over 4096 vectors and 1000 queries of 1024d
- BOP/S: 681.39
- Elapsed (ms): 12.311
Profiling `JACCARD_B1024_VPOPCNTQ_PDX` over 4096 vectors and 1000 queries of 1024d
- BOP/S: 688.14
- Elapsed (ms): 12.1902
Profiling `JACCARD_B1024_VPOPCNTQ_PRECOMPUTED_PDX` over 4096 vectors and 1000 queries of 1024d
- BOP/S: 838.34
- Elapsed (ms): 10.0063

# Sapphire Rapids
Profiling `JACCARD_B1024_VPOPCNTQ` over 4096 vectors and 1000 queries of 1024d
- BOP/S: 438.77
- Elapsed (ms): 19.1183
Profiling `JACCARD_B1024_VPOPCNTQ_PRECOMPUTED` over 4096 vectors and 1000 queries of 1024d
- BOP/S: 633.89
- Elapsed (ms): 13.2334
Profiling `JACCARD_B1024_VPOPCNTQ_PDX` over 4096 vectors and 1000 queries of 1024d
- BOP/S: 641.6
- Elapsed (ms): 13.0746
Profiling `JACCARD_B1024_VPOPCNTQ_PRECOMPUTED_PDX` over 4096 vectors and 1000 queries of 1024d
- BOP/S: 790.62
- Elapsed (ms): 10.6101
```

The degradation when data spills to main memory could be due to the last ugly part of the kernel that moves the distances from SIMD to main memory. Outside of the kernel there is another movement of these 256 distances to another container where the `partial_sort` is performed at the end of the search.


Transposing the vector collection comes with a data management overhead. It is yet to be seen if it is possible to use it on HNSW indexes search.

This data layout for KNN search is currently being researched for 32-bit floats and other quantized types in a side project called [PDX](https://github.com/cwida/PDX).



## Precomputed Jaccard JUTs.

Kernels in the column-major layout calculate distances dimensions-at-a-time (a dimension being 1 byte or 1 nibble). Knowing this, we can precompute INTERSECTION, UNION, and HAMMING LUTs that already have the result of the `AND+POPCNT`, `OR+POPCNT`, and `XOR+POPCNT`, respectively, of every nibble. 

This results in 16 lookup tables (one for every nibble). Each lookup table with a size of 16 bytes (1 byte for every pairing nibble), for a total of 256 bytes. Adding both INTERSECTION and UNION, we end up with 512 bytes in lookup tables for the Jaccard kernel. For Hamming, we only need 256 bytes. Similarly, if we have precomputed popcounts, we only need 256 bytes for the INTERSECTION LUTs. **The resolution of which lookup table to use is done in the outer loop of the kernels.** 

Unfortunately, this does not surpass the efficiency of the `VPOPCNTQ` kernels. 

```sh
# Zen 4
Profiling `JACCARD_B512_VPOPCNTQ_PRECOMPUTED_PDX` over 4096 vectors and 1000 queries of 512d
- BOP/S: 592.19
- Elapsed (ms): 7.0827
Profiling `JACCARD_B512_VPSHUFB_PRECOMPUTED_PDX` over 4096 vectors and 1000 queries of 512d
- BOP/S: 519.13
- Elapsed (ms): 8.0794

# Sapphire Rapids
Profiling `JACCARD_B512_VPOPCNTQ_PRECOMPUTED_PDX` over 4096 vectors and 1000 queries of 512d
- BOP/S: 558.09
- Elapsed (ms): 7.5155
Profiling `JACCARD_B512_VPSHUFB_PRECOMPUTED_PDX` over 4096 vectors and 1000 queries of 512d
- BOP/S: 494.18
- Elapsed (ms): 8.4874
```

We think this is due to:
- A LUT bigger than 64 bytes already affects performance and starts falling back in the cache hierarchy.
- Access to the LUTs is non-uniform and unpredictable. The latter also prevents efficient prefetching. 

Nevertheless, the gap between these kernels is smaller than the non-PDX counterparts:

```sh 
# Zen 4
Profiling `JACCARD_B512_VPOPCNTQ_PRECOMPUTED` over 4096 vectors and 1000 queries of 512d
- BOP/S: 411.9
- Elapsed (ms): 10.1829
Profiling `JACCARD_B512_VPSHUFB_SAD_PRECOMPUTED` over 4096 vectors and 1000 queries of 512d
- BOP/S: 229.62
- Elapsed (ms): 18.2659

# Sapphire Rapids
Profiling `JACCARD_B512_VPOPCNTQ_PRECOMPUTED` over 4096 vectors and 1000 queries of 512d
- BOP/S: 350.06
- Elapsed (ms): 11.9817
Profiling `JACCARD_B512_VPSHUFB_SAD_PRECOMPUTED` over 4096 vectors and 1000 queries of 512d
- BOP/S: 242.8
- Elapsed (ms): 17.2747
```


### 8 LUTs instead of 16
Let us define `LUT[i]` as the 16-byte LUT resulting from `XOR+POPCOUNT` the `i`th nibble with all the other nibbles. Interestingly, one does not need to explicitly store the 16 LUTs. For instance, `LUT[8]` to `LUT[15]` can be computed from `LUT[0]` to `LUT[7]` by doing the following: 
```py
LUT[8] = 4 - LUT[7]
LUT[9] = 4 - LUT[6]
...
LUT[15] = 4 - LUT[0]
```
Therefore, we can reduce the size of our global LUT to 128 bytes at the expense of an additional `ADD` in the outer loop of our kernels. Unfortunately, this was not enough to get substantial gains in the Hamming kernel. Although, I believe I did not optimize this code enough. 

It is pending to try this idea on the Jaccard kernel. For this, we would need to find similar properties for the `AND+POPCOUNT` LUTs and also for the `OR+POPCOUNT` LUTs if the popcounts are not precomputed.

### 4 LUTs instead of 16
Theoretically, one can reconstruct any of the 16 `XOR+POPCOUNT` LUTs from only 4 LUTs: `LUT[0]`, `LUT[1]`, `LUT[2]` and `LUT[4]` (nibbles `0000`, `0001`, `0010`, `0100`). Then: 
```py
# nibble 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
LUT[0] = [4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0]
LUT[1] = [3, 4, 2, 3, 2, 3, 1, 2, 2, 3, 1, 2, 1, 2, 0, 1]
LUT[2] = [3, 2, 4, 3, 2, 1, 3, 2, 2, 1, 3, 2, 1, 0, 2, 1]
LUT[4] = [3, 2, 2, 1, 4, 3, 3, 2, 2, 1, 1, 0, 3, 2, 2, 1]

LUT[3] = LUT[1] + LUT[2] − LUT[0]
LUT[5] = LUT[1] + LUT[4] − LUT[0]
LUT[6] = LUT[2] + LUT[4] − LUT[0]
LUT[7] = LUT[1] + LUT[2] + LUT[4] − 2*LUT[0]
LUT[8-15] = 4 - LUT[7-0]
```

Therefore, we can reduce the size of our LUT to just 64 bytes at the expense of several `ADD`s in the outer loop of our kernels. To fit it in 64 bytes, we will also need additional broadcasts from `_m128i` to `_m256i`/`_m512i`.

For the `AND+POPCOUNT` LUTs, the following formulas should work, only storing `LUT[1]` (`0001`), `LUT[2]` (`0010`), `LUT[4]` (`0100`), and `LUT[8]` (`1000`):

```py
# nibble 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
LUT[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # constant 
LUT[1] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
LUT[2] = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
LUT[4] = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
LUT[8] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

LUT[3] = LUT[1] + LUT[2] 
LUT[5] = LUT[1] + LUT[4]
LUT[6] = LUT[2] + LUT[4]
LUT[7] = LUT[1] + LUT[2] + LUT[4]
LUT[9] = LUT[1] + LUT[8]
LUT[10] = LUT[2] + LUT[8]
LUT[11] = LUT[1] + LUT[2] + LUT[8]
LUT[12] = LUT[4] + LUT[8]
LUT[13] = LUT[1] + LUT[4] + LUT[8]
LUT[14] = LUT[2] + LUT[4] + LUT[8]
LUT[15] = LUT[1] + LUT[2] + LUT[4] + LUT[8]
```

Combining this with precomputed popcounts to avoid the UNION step *could* result in the performance gains we are looking for.

We have not tried this approach yet.

### A bigger LUT with sequential access
One of the problems with the 256 bytes LUT is that the access patterns to it are not predictable. Forcing the kernels to use `LUT[i % 16]` as we are iterating over the `i`th dimension gives 1.4x speedup over fetching the correct LUT (note that this output wrong results). Therefore, we tried to precompute a bigger LUT which already positions every `LUT[i]` according to the nibbles in the incoming query. This is done *once* before the distances calculations starts. For 256d, this produces a 1KB LUT per operator (64 nibbles with 16 bytes LUT each). For 1024d, a 4KB LUT is needed.  

The latter did not improve the 256d kernels efficiency, in fact, it made them slightly worse. Most probably 1KB is too big for a LUT to be efficient, despite maximizing predictable access. 

## Other ideas
- Transposing the vector collection comes with data management challenges (updates/deletes become harder). Are the gains worth it? Probably yes, in static and medium collections. But what if we transpose the incoming queries? 
- We are using a transposed block size of 256 to not saturate the registers. What if we push it to 1024? That would require 16 registers per accumulator (16 for UNION and 16 for INTERSECTION). But, whatever we do to access the JUTs will then be amortized by 4x more distance calculations. If we precompute the popcounts, we could push the INTERSECTION accumulator to 24 registers (1536 vectors processed in the inner loop), or even 32 registers (2048 vectors). 
- Is there a way to use JUTs in the traditional 1-to-1 kernels? 
- Pruning: Hamming is a monotonically increasing metric. Depending on the data distribution, one could break-off computation at an earlier dimension and discard these vectors that are not able to make it into the KNN-heap of a query. An `if` conditional within the distance calculation kernel would be detrimental to performance. Not so much in the kernels that operate in the column-major layout. The latter would help us with our main bottleneck on big collections: data-access. Nevertheless, this is perhaps out of the scope of this project ;)

## Additional lectures

- 2019 study comparing the performance of ~20 POPCOUNT implementations on binary vector search scenarios. Includes benchmarks with the effect of inlining and manual unrolling. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0398-8
- An article on the expected performance of LUTs depending on their size and access patterns: https://specbranch.com/posts/lookup-tables/. Featured in: https://news.ycombinator.com/item?id=37823805







