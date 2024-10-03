#!/bin/sh

/usr/local/cuda/bin/nvcc -use_fast_math -maxrregcount=0 -Xptxas -O3 --ptxas-options=-v -ccbin g++ -m64 --compile --compiler-options -fPIC -O3 -arch=native -I/usr/local/cuda/include -o kernel.o kernel.cu
/usr/local/cuda/bin/nvcc -use_fast_math -maxrregcount=0 -Xptxas -O3 --ptxas-options=-v -ccbin g++ -m64 --compile --compiler-options -fPIC -O3 -arch=native -I/usr/local/cuda/include -o secp256k1.o secp256k1.cu
g++ -O3 -L/usr/local/cuda/lib64 kernel.o secp256k1.o -lcudart_static -o brute