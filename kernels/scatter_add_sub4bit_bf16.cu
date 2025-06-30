#include <cuda_bf16.h>

typedef unsigned char uint8_t;

extern "C"
__global__ void scatter_add_sub4bit_bf16(
    const __nv_bfloat16* __restrict__ src,      // [nrows x ncols]
    const uint8_t*       __restrict__ codes,    // [nrows x (ncols/2)]
    __nv_bfloat16*       __restrict__ dst,      // [nrows x 2^bits]
    int bits,
    int ncols
) {
    // One block per row:
    int row_id    = blockIdx.x;
    int tid       = threadIdx.x;
    int blockSize = blockDim.x;

    // Number of “buckets” is 2^bits (<= 16 if bits=4)
    const int n_buckets = 1 << bits;

    // We'll store partial sums in shared memory; total size: blockDim.x * n_buckets
    extern __shared__ float smem[];  
    // Each thread’s slice of shared memory:
    float* thread_sums = &smem[tid * n_buckets];

    // Step 1: Initialize thread-local partial sums to 0
    #pragma unroll
    for (int b = 0; b < n_buckets; b++) {
        thread_sums[b] = 0.0f;
    }

    // Step 2: Load data for this row and accumulate in registers or in thread_sums
    //         Each thread processes a “striped” subset of columns:
    //         ncols/2 is number of codes per row.  Each code encodes two BF16 values.
    for (int i = tid; i < (ncols / 2); i += blockSize) {
        uint8_t code = codes[row_id * (ncols/2) + i];
        // Extract the two BF16 values
        float v1 = __bfloat162float( src[row_id * ncols + 2*i    ] );
        float v2 = __bfloat162float( src[row_id * ncols + 2*i + 1] );
        // “High nibble” => code >> 4
        // “Low nibble”  => code & 0xF
        thread_sums[code >> 4] += v1;
        thread_sums[code & 0xF] += v2;
    }

    __syncthreads();

    // Step 3: Reduce all threads’ partial sums in shared memory
    //         We have blockDim.x “copies” of n_buckets each. Let’s do a simple
    //         parallel reduction over the threads for each bucket index.

    // For example, a simple “binary tree” reduction in shared memory.
    // (You can do warp shuffle if you want more advanced patterns.)
    // We'll do: stride = blockSize/2, blockSize/4, ... down to 1
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        __syncthreads();  // make sure all threads are done writing
        if (tid < stride) {
            float* sumsA = &smem[ tid         * n_buckets ];
            float* sumsB = &smem[ (tid+stride)* n_buckets ];
            #pragma unroll
            for (int b = 0; b < n_buckets; b++) {
                sumsA[b] += sumsB[b];
            }
        }
    }

    __syncthreads();

    // Step 4: Now thread 0 of the block has the final partial sums for the entire row
    if (tid == 0) {
        float* final_sums = &smem[0];
        // Store them out to global memory in BF16
        for (int b = 0; b < n_buckets; b++) {
            dst[row_id * n_buckets + b] = __float2bfloat16(final_sums[b]);
        }
    }
}