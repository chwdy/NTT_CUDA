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

/**
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

__global__ void ntt_cuda_kernel_stepB(uint64_t *g_idata, int num_bits,uint64_t *table ,uint64_t *n, uint64_t *p, bool rev, uint64_t *g_odata)
{

	uint64_t m, factor1, factor2;
	//set thread ID
	uint64_t tid = threadIdx.x;
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	//boundary check
	if (tid >= *n || idx >*n)return;
	if (rev)
	{
		uint64_t reverse_num= 0;
		for(uint64_t j = 0; j < num_bits; j++){
			reverse_num = reverse_num << 1;
			if(idx & (1 << j)){
				reverse_num = reverse_num | 1;
			}
		}
		g_odata[reverse_num] = g_idata[idx];
	}
	else
	{
		g_odata[idx] = g_idata[idx];
	}
	__syncthreads();
	if (idx == 0)
	{
		for (uint64_t i = 1; i <= num_bits; i++)
		{
			m = pow_D(uint64_t(2), i);
			for (uint64_t j = 0; j < *n; j += m)
			{
				for (uint64_t k = 0; k < m / 2; k++)
				{
					factor1 = g_odata[j + k];
					factor2 = modulo_D(uint64_t(table[(i-1)*2048+k])*uint64_t(g_odata[j + k + m / 2]), *p);
					g_odata[j + k] = modulo_D(factor1 + factor2, *p);
					g_odata[j + k + m / 2] = modulo_D(factor1 - factor2, *p);
				}
			}
		}	
		
	}

}
extern "C" 
uint64_t *inPlaceNTT_DIT_stepB(uint64_t *vec, uint64_t n, uint64_t p, uint64_t r, bool rev)
{

	double computestart, computeElaps,copystart,copyElaps;

	int blocksize = 1024;
	dim3 block(blocksize, 1);
	dim3 grid((n - 1) / block.x + 1, 1);

	//var init
	size_t bytes = n * sizeof(uint64_t);
	uint64_t *vec_host = (uint64_t *)malloc(bytes);
	uint64_t *outVec_host = (uint64_t *)malloc(bytes); //grid.x * sizeof(uint64_t));
	//printf("grid %d block %d \n", grid.x, block.x);

	memcpy(vec_host, vec, bytes);

	// device memory declare
	uint64_t *vec_dev = NULL;
	uint64_t *outVec_dev = NULL;

	//device memory allocate
	CHECK(cudaMalloc((void **)&vec_dev, bytes));
	CHECK(cudaMalloc((void **)&outVec_dev, bytes));

	//remove bitreversal
	uint64_t num_bits = log2(n);

	//modexp offline
	num_bits = log2(n);
	uint64_t a_table [32];
	int i,j;
	for (i=1;i<=32;i++){
		a_table[i-1] = modExp(r,(p-1)/pow(2,i),p);
		//printf("A: %llu i: %d \n",a_table[i-1],i);
	}
	uint64_t ak_table [65536] ;
	for (i=0;i<32;i++){
		for (j=0;j<2048;j++){
		ak_table[i*2048+j] = modExp(a_table[i],j,p);
		}
	}
	uint64_t *ak_table_dev =NULL;
	uint64_t *n_dev =NULL;
	uint64_t *p_dev =NULL;
	//uint64_t *r_dev =NULL;

	CHECK(cudaMalloc((void **)&ak_table_dev, sizeof(ak_table)));
	CHECK(cudaMalloc((void **)&n_dev, sizeof(n)));
	CHECK(cudaMalloc((void **)&p_dev, sizeof(p)));
	//CHECK(cudaMalloc((void **)&r_dev, sizeof(r)));
	CHECK(cudaMemcpy(ak_table_dev, ak_table, sizeof(ak_table), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(n_dev, &n, sizeof(n), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(p_dev, &p, sizeof(p), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(r_dev, &r, sizeof(r), cudaMemcpyHostToDevice));

	CHECK(cudaMemset(vec_dev,0,bytes))
	CHECK(cudaMemset(outVec_dev,0,bytes))
	copystart= cpuSecond();
	CHECK(cudaMemcpy(vec_dev, vec_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	computestart= cpuSecond();
	ntt_cuda_kernel_stepB<<<grid, block>>>(vec_dev,num_bits,ak_table_dev,n_dev, p_dev,rev, outVec_dev);
	CHECK(cudaDeviceSynchronize());	
	computeElaps = 1000 * (cpuSecond() - computestart);
	CHECK(cudaMemcpy(outVec_host, outVec_dev, bytes, cudaMemcpyDeviceToHost));
	copyElaps = 1000 * (cpuSecond() - copystart);
	printf("gpu 1 pure compute time: %lf compute+copy time: %lf for ### modexp offline### \n first two number is %lld %lld \n", computeElaps, copyElaps,outVec_host[0],outVec_host[1]);

	CHECK(cudaFree(ak_table_dev));
	CHECK(cudaFree(n_dev));
	CHECK(cudaFree(p_dev));
	//CHECK(cudaFree(r_dev));
	CHECK(cudaFree(vec_dev));
	CHECK(cudaFree(outVec_dev));

	return outVec_host;
}
