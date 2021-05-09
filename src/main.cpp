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

	//gpu-as-is
	timeStart = cpuSecond();
	outVec = inPlaceNTT_DIT_cuda_asis(vec, n, p, r);
	timeElaps = 1000 * (cpuSecond() - timeStart);
	printf("gpu as-is total elapsed %lf \n", timeElaps);
	printf("\n");

	//gpu-step-a
	timeStart = cpuSecond();
	outVec = inPlaceNTT_DIT_stepA(vec, n, p, r);
	timeElaps = 1000 * (cpuSecond() - timeStart);
	printf("gpu stepA total elapsed %lf \n", timeElaps);
	printf("\n");

	//gpu-step-b
	timeStart = cpuSecond();
	outVec = inPlaceNTT_DIT_stepB(vec, n, p, r);
	timeElaps = 1000 * (cpuSecond() - timeStart);
	printf("gpu stepB total elapsed %lf \n", timeElaps);
	printf("\n");

	//gpu-step-b
	timeStart = cpuSecond();
	outVec = inPlaceNTT_DIT_stepC(vec, n, p, r);
	timeElaps = 1000 * (cpuSecond() - timeStart);
	printf("gpu stepB total elapsed %lf \n", timeElaps);
	printf("\n");

	return 0;
}
