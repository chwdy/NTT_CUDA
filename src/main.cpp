#include <cmath>   /* pow() */
#include <cstdint> /* uint64_t */
#include <ctime>   /* time() */

//#include <unistd.h>
#include <iostream>

#include "../include/ntt.cuh" /* naiveNTT(), outOfPlaceNTT_DIT() */
#include "../include/utils.h" /* printVec() */
#include "../include/utils2.h"



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
	timeStart = cpuSecond();
	outVec = inPlaceNTT_DIT(vec, n, p, r);
	timeElaps = 1000 * (cpuSecond() - timeStart);

	printf("cpu reduction elapsed %lf \n first tow number is %I64d %I64d \n", timeElaps, outVec[0],outVec[1]);
	printf("\n");

	timeStart = cpuSecond();
	outVec = inPlaceNTT_DIT_cuda(vec, n, p, r);
	timeElaps = 1000 * (cpuSecond() - timeStart);
	printf("gpu 1 total elapsed %lf", timeElaps);

	return 0;
}
