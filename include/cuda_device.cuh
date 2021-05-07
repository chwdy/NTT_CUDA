#ifndef CUDADEVICE_H
#define CUDADEVICE_H
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

void initDevice(int devNum);
// {
//   int dev = devNum;
//   cudaDeviceProp deviceProp;
//   CHECK(cudaGetDeviceProperties(&deviceProp,dev));
//   printf("Using device %d: %s\n",dev,deviceProp.name);
//   CHECK(cudaSetDevice(dev));

// }

void checkResult(float * hostRef,float * gpuRef,const int N);
// {
//   double epsilon=1.0E-8;
//   for(int i=0;i<N;i++)
//   {
//     if(abs(hostRef[i]-gpuRef[i])>epsilon)
//     {
//       printf("Results don\'t match!\n");
//       printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n",hostRef[i],i,gpuRef[i],i);
//       return;
//     }
//   }
//   printf("Check result success!\n");
// }

#endif

