# Compile Libraries (wip)

NEON Hamming (MacOS):
```
clang++ -O3 -march=native -DNDEBUG -std=c++20 -dynamiclib -o ./include/neon_hamming.dylib ./include/neon_hamming.cpp
```

NEON Jaccard (MacOS):
```
clang++ -O3 -march=native -DNDEBUG -std=c++20 -dynamiclib -o ./include/neon_jaccard.dylib ./include/neon_jaccard.cpp
```
