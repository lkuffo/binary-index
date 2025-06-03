#!/bin/bash

outfile="$1"

echo "Compiling..."
clang++ -O3 -march=native -DNDEBUG -std=c++20 -shared -o ./include/avx512_jaccard.dylib ./include/avx512_jaccard.cpp

echo "Running Jaccard benchmarks..."
n_vectors=(256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 4194304) # 16777216)
n_queries=(1 10 100 1000)
dimensions=(128 256 512 1024)

#dimensions=(256 512 1024)
#n_vectors=(256 512 1024)
#n_queries=(1 10 100)

for d in "${dimensions[@]}"
do
    for n_vector in "${n_vectors[@]}"
    do
        for query_count in "${n_queries[@]}"
        do
            uv run --script kernels_jaccard_avx.py --count ${n_vector} --ndims ${d} --k 10 --query_count ${query_count} --output $outfile
        done
    done
done
