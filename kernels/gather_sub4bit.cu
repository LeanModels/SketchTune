#include <cuda_fp16.h>

typedef unsigned char uint8_t;

extern "C"
__global__ void gather_sub4bit(
    const half* __restrict__ src,       // nrows x 2^bits
    const uint8_t* __restrict__ codes,  // nrows x (ncols / 2)
    half* __restrict__ dst,             // nrows x ncols
    int bits, int ncols
) {
	extern __shared__ uint8_t shared_mem[];  // Raw shared memory block
	volatile half* cache = (volatile half*) shared_mem;  	// 2^bit * 2bytes, 32 bytes
	volatile half* buf = (volatile half*) &shared_mem[1<<bits * 2];	// 128 bits * n_threads

    const int row_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int n_threads = blockDim.x;
    const int n_floats = 1 << bits;	// 16

#pragma unroll
    for (int i = thread_id; i < n_floats; i += n_threads) { // Use 16 threads
        cache[i] = src[row_id * n_floats + i]; // cache now stores the 16 bf16 values from the LUT 
    }
    __syncthreads();

    for (int i = thread_id; i < ncols / 8; i += n_threads) {
		// each thread handles 8 bf16 (8*2byte * 8bits = 128Bits) 
		volatile half* t_buf = buf + thread_id * 8;

#pragma unroll
		for (int j = 0; j < 4; j++) {
			uint8_t code = codes[row_id * ncols / 2 + i * 4 + j];
			t_buf[j * 2] 	= cache[code >> 4];
			t_buf[j * 2 + 1]= cache[code & 0xf];
		}

		int offset = row_id * ncols + i * 8; 
		*((uint4*)&dst[offset]) = *((uint4*)t_buf);
    }
}