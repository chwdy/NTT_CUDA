#ifndef NTT_H
#define NTT_H

#include <cstdint> /* uint64_t */

/**
 * Perform an in-place iterative breadth-first decimation-in-time Cooley-Tukey
 *NTT on an input vector and return the result
 *
 * @param vec 	The input vector to be transformed
 * @param n	The size of the input vector
 * @param p	The prime to be used as the modulus of the transformation
 * @param r	The primitive root of the prime
 * @param rev	Whether to perform bit reversal on the input vector
 * @return 	The transformed vector
 **/

//__global__ void ntt_cuda_kernel(int * g_idata, uint64_t * n, uint64_t * p,
//uint64_t * r, bool * rev,int * g_odata);
__global__ void ntt_cuda_kernel(int *g_idata, uint64_t n, uint64_t p,
                                uint64_t r, bool rev, int *g_odata);

extern "C" {
uint64_t *inPlaceNTT_DIT(uint64_t *vec, uint64_t n, uint64_t p, uint64_t r,
                         bool rev = true);

uint64_t *inPlaceNTT_DIT_cuda(uint64_t *vec, uint64_t n, uint64_t p, uint64_t r,
                              bool rev = true);
};

#endif
