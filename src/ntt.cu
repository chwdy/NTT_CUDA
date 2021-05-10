#include <cstdint> /* uint64_t */
#include <cstdlib> /* malloc() */
#include <cuda_runtime.h>
#include <iostream>
#include <math.h> /* log2(), pow() */

#include "../include/cuda_device.cuh"
#include "../include/utils.h"
#include "../include/utils2.h"
#include "../include/utils_device.cuh"
/* bit_reverse(), modExp(), modulo() */
#include "../include/ntt.cuh" //INCLUDE HEADER FILE

/**
 * Perform an in-place iterative breadth-first decimation-in-time Cooley-Tukey
 * NTT on an input vector and return the result
 *
 * @param vec 	The input vector to be transformed
 * @param n	The size of the input vector
 * @param p	The prime to be used as the modulus of the transformation
 * @param r	The primitive root of the prime
 * @param rev	Whether to perform bit reversal on the input vector
 * @return 	The transformed vector
 */
using namespace std;

__global__ void ntt_cuda_kernel_rev(uint64_t *g_idata, int offset, int num_bits,
                                    uint64_t *n, bool rev, uint64_t *g_odata) {
  // uint64_t m, factor1, factor2;
  // set thread ID
  uint64_t tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  // boundary check
  if (tid >= *n || idx > *n)
    return;
  if (rev) {
    uint64_t reverse_num = 0;
    for (uint64_t j = 0; j < num_bits; j++) {
      reverse_num = reverse_num << 1;
      if (idx & (1 << j)) {
        reverse_num = reverse_num | 1;
      }
    }
    g_odata[offset * *n + reverse_num] = g_idata[offset * *n + idx];
  } else {
    g_odata[offset * *n + idx] = g_idata[offset * *n + idx];
  }
}
__global__ void ntt_cuda_kernel_fac_A(uint64_t *g_idata, int offset,
                                      uint64_t *table, uint64_t *n, uint64_t *p,
                                      uint64_t i, uint64_t *fac1_dev,
                                      uint64_t *fac2_dev, uint64_t *g_odata) {
  // set thread ID
  uint64_t tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  // boundary check
  if (tid >= *n || idx > *n)
    return;
  uint64_t m = pow_D(uint64_t(2), i);
  uint64_t k = idx % m;

  if (k < m / 2) {
    fac1_dev[offset * *n + idx] = g_odata[offset * *n + idx];
    fac2_dev[offset * *n + idx] =
        modulo_D(uint64_t(table[(i - 1) * 2048 + k]) *
                     uint64_t(g_odata[offset * *n + idx + m / 2]),
                 *p);
  } else {
    fac1_dev[offset * *n + idx] = g_odata[offset * *n + idx - m / 2];
    fac2_dev[offset * *n + idx] =
        modulo_D(uint64_t(table[(i - 1) * 2048 + k - m / 2]) *
                     uint64_t(g_odata[offset * *n + idx]),
                 *p);
  }
}
__global__ void ntt_cuda_kernel_fac_B(uint64_t *g_idata, int offset,
                                      uint64_t *table, uint64_t *n, uint64_t *p,
                                      uint64_t i, uint64_t *fac1_dev,
                                      uint64_t *fac2_dev, uint64_t *g_odata)

{
  // uint64_t factor1, factor2;
  // set thread ID
  uint64_t tid = threadIdx.x;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  // boundary check
  if (tid >= *n || idx > *n)
    return;
  uint64_t m = pow_D(uint64_t(2), i);
  uint64_t k = idx % m;
  if (k < m / 2) {
    g_odata[offset * *n + idx] =
        modulo_D(fac1_dev[offset * *n + idx] + fac2_dev[offset * *n + idx], *p);
  } else {
    g_odata[offset * *n + idx] =
        modulo_D(fac1_dev[offset * *n + idx] - fac2_dev[offset * *n + idx], *p);
  }
}

extern "C" uint64_t *inPlaceNTT_DIT(uint64_t **vec, uint64_t batch_size,
                                    uint64_t n, uint64_t p, uint64_t r,
                                    bool rev) {


  int blocksize = 1024;
  dim3 block(blocksize, 1);
  dim3 grid((n - 1) / block.x + 1, 1);

  // var init
  size_t bytes = n * batch_size * sizeof(uint64_t);
  uint64_t *vec_host = (uint64_t *)malloc(bytes);
  uint64_t *outVec_host =
      (uint64_t *)malloc(bytes); // grid.x * sizeof(uint64_t));

  for (int i = 0; i < batch_size; i++) {

    memcpy(&vec_host[i * n], vec[i], n * sizeof(uint64_t));
  }

  // device memory declare
  uint64_t *vec_dev = NULL;
  uint64_t *outVec_dev = NULL;

  // device memory allocate
  CHECK(cudaMalloc((void **)&vec_dev, bytes));
  CHECK(cudaMalloc((void **)&outVec_dev, bytes));

  // remove bitreversal
  uint64_t num_bits = log2(n);

  num_bits = log2(n);
  // pre-computed-modEXP
  uint64_t a_table[32];
  int i, j;
  for (i = 1; i <= 32; i++) {
    a_table[i - 1] = modExp(r, (p - 1) / pow(2, i), p);
  }
  uint64_t ak_table[65536];
  for (i = 0; i < 32; i++) {
    for (j = 0; j < 2048; j++) {
      ak_table[i * 2048 + j] = modExp(a_table[i], j, p);
    }
  }
  uint64_t *ak_table_dev = NULL;
  uint64_t *n_dev = NULL;
  uint64_t *p_dev = NULL;
  uint64_t *fac1_dev = NULL;
  uint64_t *fac2_dev = NULL;

  CHECK(cudaMalloc((void **)&ak_table_dev, sizeof(ak_table)));
  CHECK(cudaMalloc((void **)&n_dev, sizeof(n)));
  CHECK(cudaMalloc((void **)&p_dev, sizeof(p)));
  CHECK(cudaMalloc((void **)&fac1_dev, bytes));
  CHECK(cudaMalloc((void **)&fac2_dev, bytes));
  CHECK(cudaMemcpy(ak_table_dev, ak_table, sizeof(ak_table),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(n_dev, &n, sizeof(n), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(p_dev, &p, sizeof(p), cudaMemcpyHostToDevice));

  CHECK(cudaMemset(vec_dev, 0, bytes))
  CHECK(cudaMemset(outVec_dev, 0, bytes))

  CHECK(cudaMemcpy(vec_dev, vec_host, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());

  for (int offset = 0; offset < batch_size; offset++) {

    ntt_cuda_kernel_rev<<<grid, block>>>(vec_dev, offset, num_bits, n_dev, rev,
                                         outVec_dev);
    CHECK(cudaDeviceSynchronize());
    for (uint64_t i = 1; i <= num_bits; i++) {
      ntt_cuda_kernel_fac_A<<<grid, block>>>(vec_dev, offset, ak_table_dev,
                                             n_dev, p_dev, i, fac1_dev,
                                             fac2_dev, outVec_dev);
      CHECK(cudaDeviceSynchronize());
      ntt_cuda_kernel_fac_B<<<grid, block>>>(vec_dev, offset, ak_table_dev,
                                             n_dev, p_dev, i, fac1_dev,
                                             fac2_dev, outVec_dev);
      CHECK(cudaDeviceSynchronize());
    }
  }
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(outVec_host, outVec_dev, bytes, cudaMemcpyDeviceToHost));

  CHECK(cudaFree(ak_table_dev));
  CHECK(cudaFree(n_dev));
  CHECK(cudaFree(p_dev));
  CHECK(cudaFree(vec_dev));
  CHECK(cudaFree(outVec_dev));

  return outVec_host;
}
