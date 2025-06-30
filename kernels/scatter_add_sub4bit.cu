#include <cuda_fp16.h>

typedef unsigned char uint8_t;

extern "C"
__global__ void scatter_add_sub4bit(
    const half* __restrict__ src,       // nrows x ncols
    const uint8_t* __restrict__ codes,  // nrows x (ncols / 2)
    half* __restrict__ dst,             // nrows x 2^bits
    int bits, int ncols
) {
    extern __shared__ float cache[]; // 2^bits

    const int row_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int n_threads = blockDim.x;

    const int n_floats = 1 << bits;

#pragma unroll
    for (int i = thread_id; i < n_floats; i += n_threads) {
        cache[i] = 0;
    }
    __syncthreads();

    for (int i = thread_id; i < ncols / 2; i += n_threads) {
        uint8_t code = codes[row_id * ncols / 2 + i];
        atomicAdd(cache + (code >> 4), __half2float(src[row_id * ncols + i * 2]));
        atomicAdd(cache + (code & 0xf), __half2float(src[row_id * ncols + i * 2 + 1]));
    }
    __syncthreads();

#pragma unroll
    for (int i = thread_id; i < n_floats; i += n_threads) {
        dst[row_id * n_floats + i] = __float2half(cache[i]);
    }
}