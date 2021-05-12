#include <cuda_runtime.h>
#include <math.h>		/* log2(), pow() */
#include <cstdint>		/* uint64_t */
#include <cstdlib> 		/* malloc() */
#include <iostream>

#include "../include/utils2.h"	
#include "../include/utils.h"
/* bit_reverse(), modExp(), modulo() */
#include "../include/ntt.cuh" //INCLUDE HEADER FILE
#include "../include/utils_device.cuh"	
#include "../include/cuda_device.cuh"

/** Pure single thread implementation
 * Perform an in-place iterative breadth-first decimation-in-time Cooley-Tukey NTT on an input vector and return the result
 *
 * @param vec 	The input vector to be transformed
 * @param n	The size of the input vector
 * @param p	The prime to be used as the modulus of the transformation
 * @param r	The primitive root of the prime
 * @param rev	Whether to perform bit reversal on the input vector
 * @return 	The transformed vector
 */
using namespace std;


__global__ void ntt_kernel_cuda_asis(uint64_t *g_idata, uint64_t n, uint64_t p, uint64_t r, bool rev, uint64_t *g_odata)
{

	uint64_t m, k_, a, factor1, factor2;
	//set thread ID
	uint64_t tid = threadIdx.x;
	//boundary check
	if (tid >= n)return;
	if (rev)
	{
		if (blockIdx.x == 0 && tid == 0)
		{
			uint64_t *temp;
			temp = (uint64_t *) malloc(n*sizeof(uint64_t));
			temp = bit_reverse_D(g_idata, n);
			memcpy(g_odata, temp, n * sizeof(uint64_t));
		}
	}
	else
	{
		for (uint64_t i = 0; i < n; i++)
		{
			g_odata[i] = g_idata[i];
		}
	}
	
	if (blockIdx.x == 0 && tid == 0)
	{
		//printf("godata first two number is %d %d \n",  g_odata[55],g_odata[71]);
		for (uint64_t i = 1; i <= log2_D(n); i++)
		{
			m = pow_D(uint64_t(2), i);
			k_ = (p - 1) / m;
			a = modExp_D(r, k_, p);
			for (uint64_t j = 0; j < n; j += m)
			{
				for (uint64_t k = 0; k < m / 2; k++)
				{
					factor1 = g_odata[j + k];
					factor2 = modulo_D(modExp_D(a, k, p) * g_odata[j + k + m / 2], p);
					g_odata[j + k] = modulo_D(factor1 + factor2, p);
					g_odata[j + k + m / 2] = modulo_D(factor1 - factor2, p);
				}
			}
		}
	}
	__syncthreads();
}

extern "C" 
uint64_t *inPlaceNTT_DIT_cuda_asis(uint64_t *vec, uint64_t n, uint64_t p, uint64_t r, bool rev)
{

	double computestart, computeElaps,copystart,copyElaps;

	int blocksize = 1024;
	dim3 block(blocksize, 1);
	dim3 grid((n - 1) / block.x + 1, 1);

	//var init
	size_t bytes = n * sizeof(uint64_t);
	uint64_t *vec_host = (uint64_t *)malloc(bytes);
	uint64_t *outVec_host = (uint64_t *)malloc(bytes); //

	memcpy(vec_host, vec, bytes);

	// device memory declare
	uint64_t *vec_dev = NULL;
	uint64_t *outVec_dev = NULL;

	//device memory allocate
	CHECK(cudaMalloc((void **)&vec_dev, bytes));
	CHECK(cudaMalloc((void **)&outVec_dev, bytes));

	copystart= cpuSecond();
	//first task
	CHECK(cudaMemset(vec_dev,0,bytes))
	CHECK(cudaMemset(outVec_dev,0,bytes))
	
	CHECK(cudaMemcpy(vec_dev, vec_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	computestart= cpuSecond();
	ntt_kernel_cuda_asis<<<grid, block>>>(vec_dev, n, p, r, rev, outVec_dev);
	CHECK(cudaDeviceSynchronize());
	computeElaps = 1000 * (cpuSecond() - computestart);
	CHECK(cudaMemcpy(outVec_host, outVec_dev, bytes, cudaMemcpyDeviceToHost));
	copyElaps = 1000 * (cpuSecond() - copystart);
	printf("gpu 1 pure compute time: %lf compute+copy time: %lf for ### first task ### \n first two number is %lld %lld \n", computeElaps, copyElaps,outVec_host[0],outVec_host[1]);


	CHECK(cudaFree(vec_dev));
	CHECK(cudaFree(outVec_dev));


	return outVec_host;
}
