The code was changed to bypass USearch and always use a standalone implementation of knn search.

## Optimizations:

- Hybrid kernel: VPOPCNTQ + VPSHUFB
- Precomputed Population Counts
- Using a column-major layout on the vectors collection.
- Precomputed Jaccard JUTs.

## Hybrid kernel: VPOPCNTQ + VPSHUFB
In Sapphire Rapids, both VPOPCNTQ and VPSHUFB are dispatched through port 5. However, in Zen4, port 0 is exclusive to VPOPCNTQ and port 2 is exclusive to VPSHUFB, with both sharing port 1. Therefore, we implemented a LUT-based kernel that used VPSHUFB for INTERSECTION and VPOPCNTQ for UNION.

...results...

While being better than the VPSHUFB kernel, it is almost never better than the VPOPCNTQ kernel. With the exception being when data fits in L1 (*verify this*).


## Precomputed Population Counts
Given the Jaccard metric as:

$$
\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

We can avoid the calculation of ${|A \cup B|}$ if the popcounts of the vector collection are stored as metadata. This saves 1 VPORQ and 1 VPOPCNTQ, reducing the Jaccard kernel to just 1 AND and 1 VPOPCNTQ. In the LUT-based kernel it saves even more instructions: 2 VPSHUFB, 1 VPADDQ, 2 VPSADBW, 2 VPANDQ, 1 VPSRLQ. 

The tradeoff is 1 POPCOUNT for every query vector, which is amortized by all the distance calculations. There is also a storage overhead for the POPCOUNTs of every vector in the collection (32-bit or 16-bit per vector). The latter is 3% for 1024-bit vectors, 6% for 512-bit vectors. 

This is especially useful when ndim $\geq$ 512. However, once the kernels become data-access bound (> L3 size), the difference is little. It is pending to see whether these kernels would speed up an HNSW search. 


## Column-major layout
Transposing the vector collection in blocks have some interesting effects on the kernels. Here, distances are computed between a query dimension and multiple vectors at a time. 

On these kernels, distances are already accumulated into different lanes. In the VPOPCNTQ kernel this eliminates the `_mm256_reduce_add_epi64` step. In the LUT-based kernel this also eliminates the VPSADBW step.


We measured the following performance improvements over their non-transposed counterpart:
- 3-10% faster when data does not fit in cache.
- Up to 40% 
- Up to 2x...
- Up to 4x...

P.s. Transposing the vector collection is not possible in HNSW indexes search, at least not trivially. 

P.s.2. This data layout for vectos is called PDX, and is currently being researched for 32-bit floats, and other quantized types.

## Precomputed Jaccard JUTs.

Kernels in the column-major layout calculate distances dimensions-at-a-time (a dimension being 1 byte, or 1 nibble). Knowing this, we can precompute INTERSECTION, UNION and HAMMING LUTs that already have the result of the AND + POPCNT, OR + POPCNT and XOR + POPCNT, respectively. 

This results in 16 lookup tables (one for every nibble). Each lookup table with 16 bytes (1 byte for every pairing nibble), for a total of 256 bytes. Adding both INTERSECTION and UNION we have 512 bytes in lookup tables for the Jaccard kernel. For Hamming, we need 256 bytes. 

Unfortunately, this does not surpass the efficiency of the PDX kernel that uses the VPOPCNTQ. We think this is due to many reasons:
- A LUT bigger than 64 bytes already affects performance and starts falling back in the cache hierarchy.
- Access to the LUTs is non-uniform and unpredictable. The latter also prevents efficient prefetching. 

### 8 LUTs instead of 16
The 16 XOR LUTs can be stored in only 8 LUTs (128 bytes). LUT9 to 16 can be computed on the fly: LUT9 = 4 - LUT8, LUT10 = 4 - LUT7, ... LUT16 = 4 - LUT1. Therefore, we tradeoff memory usage with an additional ADD, every N bytes (in our experiments N was set to 256). 

### 4 LUTs instead of 16
The 16 LUTs can actually be reconstructed from only 4 LUTs (64 bytes). We tradeoff memory usage with two additional ADDs, every N bytes. 

## Additional lectures

- 2019 study comparing the performance of ~20 POPCOUNT implementations on binary vector search scenarios. Includes benchmarks with the effect of inlining and manual unrolling. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0398-8








