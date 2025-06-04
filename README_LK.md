The code was changed to bypass USearch and always use a standalone implementation of KNN search. In the next hours I will attach some benchmarking results.

## Optimizations tried:

- Hybrid kernel: `VPOPCNTQ` + `VPSHUFB`
- Precomputed population counts
- Column-major layout
- Precomputed Jaccard JUTs

## Hybrid kernel: `VPOPCNTQ` + `VPSHUFB`
In Sapphire Rapids, both `VPOPCNTQ` and `VPSHUFB` are dispatched through port 5. However, in Zen4, port 0 is exclusive to `VPOPCNTQ`, and port 2 is exclusive to `VPSHUFB`, with both sharing port 1. Therefore, we implemented a LUT-based kernel that uses `VPSHUFB` for INTERSECTION and `VPOPCNTQ` for UNION.

...results...

While being better than the `VPSHUFB` kernel, it is never better than the `VPOPCNTQ` kernel. Perhaps we are overlooking that the `VPSHUFB` kernel comes at the expense of additional instructions: 1 `VPADDB` (to be executed after the 2 `VPSHUFB`), 2 `VPANDQ` (one of them waiting for 1 `PSRLW`), 1 `PSRLW`, and 1 `VPSADBW`.


## Precomputed population counts
Given the Jaccard metric as:

$$
\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

We can avoid the calculation of ${|A \cup B|}$ if the popcounts of the vector collection are stored as metadata. This saves 1 `VPORQ` and 1 `VPOPCNTQ`, reducing the Jaccard kernel to just 1 `VPANDQ` and 1 `VPOPCNTQ`. In the LUT-based kernel, it saves even more instructions: 2 `VPSHUFB`, 1 `VPADDQ`, 2 `VPSADBW`, 2 `VPANDQ`, and 1 `VPSRLQ`. 

The tradeoff is 1 POPCOUNT for every query vector, which is amortized by all the distance calculations. There is also a storage overhead to store the popcounts of every vector in the collection. The latter is 3% for 1024-bit vectors and 6% for 512-bit vectors if we store them in 32 bits. However, storing them on 16 bits should be possible.

This kernel is especially useful when ndim $\geq$ 512. However, once the kernels become data-access bound (> L3 size), the difference is little. It is pending to see whether these kernels would speed up an HNSW search. 


## Column-major layout
Transposing the vector collection has some interesting effects on the kernels. In these kernels, distances are computed between a single query dimension (a dimension being 1 byte or 1 nibble) and multiple vectors at a time. The latter lets us seamlessly accumulate the metric into different SIMD lanes. 

In the `VPOPCNTQ` kernel, this eliminates the `_mm256_reduce_add_epi64` step. In the LUT-based kernel, this also eliminates the `VPSADBW` steps.


This kernels deliver interesting performance improvements over their non-transposed counterpart:
- 3-10% faster when data does not fit in the cache.
- Up to 40% 
- Up to 2x...
- Up to 4x...

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

Transposing the vector collection comes with a data management overhead. It is yet to be seen if it is possible to use it on HNSW indexes search.

This data layout for KNN search is currently being researched for 32-bit floats and other quantized types in a side project called [PDX](https://github.com/cwida/PDX).



## Precomputed Jaccard JUTs.

Kernels in the column-major layout calculate distances dimensions-at-a-time (a dimension being 1 byte or 1 nibble). Knowing this, we can precompute INTERSECTION, UNION, and HAMMING LUTs that already have the result of the `AND+POPCNT`, `OR+POPCNT`, and `XOR+POPCNT`, respectively, of every nibble. 

This results in 16 lookup tables (one for every nibble). Each lookup table with a size of 16 bytes (1 byte for every pairing nibble), for a total of 256 bytes. Adding both INTERSECTION and UNION, we end up with 512 bytes in lookup tables for the Jaccard kernel. For Hamming, we only need 256 bytes. **The resolution of which lookup table to use is done in the outer loop of the kernels.** 

Unfortunately, this does not surpass the efficiency of the `VPOPCNTQ` kernels. We think this is due to:
- A LUT bigger than 64 bytes already affects performance and starts falling back in the cache hierarchy.
- Access to the LUTs is non-uniform and unpredictable. The latter also prevents efficient prefetching. 

### 8 LUTs instead of 16
Let us define `LUT[i]` as the 16-byte LUT resulting from `XOR+POPCOUNT` the `i`th nibble with all the other nibbles. However, one does not need to explicitly store the 16 LUTs. For instance, `LUT[8]` to `LUT[15]` can be computed as: 
```py
LUT[8] = 4 - LUT[7]
LUT[9] = 4 - LUT[6]
...
LUT[15] = 4 - LUT[0]
```
Therefore, we can reduce the size of our global LUT to 128 bytes at the expense of an additional `ADD` in the outer loop of our kernels. Unfortunately, this was not enough to get substantial gains in the Hamming kernel. Although, I believe I did not optimize this code enough. 

It is pending to try this idea on the Jaccard kernel. For this, we would need to find similar properties for the `AND+POPCOUNT` LUTs and also for the `OR+POPCOUNT` LUTs if the popcounts are not precomputed.

### 4 LUTs instead of 16
Theoretically, one can reconstruct any of the 16 `XOR+POPCOUNT` LUTs from only 4 LUTs (64 bytes): `LUT[0]`, `LUT[1]`, `LUT[2]` and `LUT[4]` (nibbles `0000`, `0001`, `0010`, `0100`). Then: 
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

The latter did not improve the 256d kernels efficiency. Probably 1KB is too big. 

## Other ideas
- Transposing the vector collection comes with data management challenges (updates/deletes become harder). Are the gains worth it? Probably yes, in static and medium collections. But what if we transpose the incoming queries? 
- We are using a transposed block size of 256 to not saturate the registers. What if we push it to 1024? That would require 16 registers per accumulator (16 for UNION and 16 for INTERSECTION). But, whatever we do to access the JUTs will then be amortized by 4x more distance calculations. 
- Is there a way to use JUTs in the traditional kernels? 
- Pruning: Hamming is a monotonically increasing metric. Depending on the data distribution, one could break-off computation at an earlier dimension and discard these vectors that are not able to make it into the KNN-heap of a query. An `if` conditional within the distance calculation kernel would be detrimental to performance. Not so much in the kernels that operate in the column-major layout. The latter would help us with our main bottleneck on big collections: data-access. Nevertheless, this is perhaps out of the scope of this project ;)

## Additional lectures

- 2019 study comparing the performance of ~20 POPCOUNT implementations on binary vector search scenarios. Includes benchmarks with the effect of inlining and manual unrolling. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0398-8
- An article on the expected performance of LUTs depending on their size and access patterns: https://specbranch.com/posts/lookup-tables/. Featured in: https://news.ycombinator.com/item?id=37823805







