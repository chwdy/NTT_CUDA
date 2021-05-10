#include <cmath>   /* pow() */
#include <cstdint> /* uint64_t */
#include <ctime>   /* time() */

//#include <unistd.h>
#include <iostream>

#include "../include/utils.h" /* printVec() */
#include "../include/utils2.h"

//#include "../include/ntt.cuh" /* naiveNTT(), outOfPlaceNTT_DIT() */
#include "../include/ntt_cpu.h" 
#include "../include/ntt_as_is_cuda.cuh" 
#include "../include/ntt_step_a.cuh" 
#include "../include/ntt_step_b.cuh" 
#include "../include/ntt_step_b2.cuh" 
#include "../include/ntt_step_c.cuh" 
#include "../include/cuda_device.cuh" 


using namespace std;

int main(int argc, char *argv[])
{

	const uint64_t n = 4096;
	uint64_t p = 68719403009;
	uint64_t r = 36048964756;

	uint64_t vec[n];
	for (int i = 0; i < n; i++)
	{
		vec[i] = i;
	}

	
	double timeStart, timeElaps;
	uint64_t *outVec;

	//cpu
	timeStart = cpuSecond();
	outVec = inPlaceNTT_DIT_cpu(vec, n, p, r);
	timeElaps = 1000 * (cpuSecond() - timeStart);
	printf("cpu reduction elapsed %lf \n first tow number is %I64d %I64d \n", timeElaps, outVec[0],outVec[1]);
	printf("\n");

	//gpu init
	timeStart = cpuSecond();
	initDevice(0);
	timeElaps = 1000 * (cpuSecond() - timeStart);
	printf("GPU init elapsed %lf \n", timeElaps);
	printf("\n");

	// //gpu-as-is
	// timeStart = cpuSecond();
	// outVec = inPlaceNTT_DIT_cuda_asis(vec, n, p, r);
	// timeElaps = 1000 * (cpuSecond() - timeStart);
	// printf("gpu as-is total elapsed %lf \n", timeElaps);
	// printf("\n");

	// //gpu-step-a
	// timeStart = cpuSecond();
	// outVec = inPlaceNTT_DIT_stepA(vec, n, p, r);
	// timeElaps = 1000 * (cpuSecond() - timeStart);
	// printf("gpu stepA total elapsed %lf \n", timeElaps);
	// printf("\n");

	//gpu-step-b
	timeStart = cpuSecond();
	outVec = inPlaceNTT_DIT_stepB(vec, n, p, r);
	timeElaps = 1000 * (cpuSecond() - timeStart);
	printf("gpu stepB total elapsed %lf \n", timeElaps);
	printf("\n");

	//gpu-step-b2
	timeStart = cpuSecond();
	outVec = inPlaceNTT_DIT_stepB2(vec, n, p, r);
	timeElaps = 1000 * (cpuSecond() - timeStart);
	printf("gpu stepB2 total elapsed %lf \n", timeElaps);
	printf("\n");

	//gpu-step-c
	//make big input
	// const uint64_t input_size = 4096;
	// uint64_t **mat = new uint64_t*[input_size];
	// uint64_t **out_mat = new uint64_t*[input_size];
	// // mat[0][1]= new int(1);
	// for (int i=0;i<input_size;i++){
	// 	mat[i] = new uint64_t[n];
	// 	out_mat[i] = new uint64_t[n];
	// 	memcpy(mat[i],vec,n * sizeof(uint64_t));
	// }
	//
	// int batch_size[] = {1,16,64,256,512,1024};
	// for (int i =0;i<sizeof(batch_size)/sizeof(batch_size[0]);i++){
	// 	timeStart = cpuSecond();
	// 	for (int batch_count=0; batch_count< input_size/batch_size[i] ; batch_count++){
	// 		//printf("batch size : %d batch :%d index: %d first number %lld\n", batch_size[i],batch_count,batch_count*batch_size[i],mat[batch_count*batch_size[i]][0]);
	// 		uint64_t *result = inPlaceNTT_DIT_stepC(mat+batch_count*batch_size[i], batch_size[i],n, p, r);
	// 		for (int j=0;j<batch_size[i];j++){
	// 		//printf("batch: %lld index :%lld size:%llu /%zu\n", i,i*n,i*n * sizeof(uint64_t),bytes);
	// 		//printf("%llu ",vec_host[batch_size*n]);
	// 		// printf("%p %p %p",vec_host,&vec_host[i*n],&vec_host[batch_size*n]);
	// 			//printf("%d ",j+batch_count*batch_size[i]);
	// 			memcpy(*(out_mat+j+batch_count*batch_size[i]),result+j*n,n * sizeof(uint64_t));
	// 		}
	// 	}
	// 	timeElaps = 1000 * (cpuSecond() - timeStart);
	// 	printf("gpu stepC total elapsed %lf with batch size %d ,AVG: %f /vec\n", timeElaps,batch_size[i],timeElaps/n);
	// }

	// for (int i=0;i<input_size;i++){
	// 	delete[] mat[i];
	// 	delete[] out_mat[i];
	// }
	// delete[] mat;
	// printf("\n");

	return 0;
}
