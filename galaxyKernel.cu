//cuda inclusion
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Math inclusion
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <ctime>

//c++ and project inclusion
#include <stdio.h>
#include "galaxyKernel.h"

//define thread x and thread y
constexpr auto TX = 32; 
constexpr auto TY = 32; 
constexpr auto GRID = 100000; 
constexpr auto WINDOW = 1024; 

//Cuda error handling start here
inline void error_check(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		::fprintf(stderr, "\nCUDA ERROR at %s[%d] : %s\n", file, line, cudaGetErrorString(err));
		printf("\nGeneral error at %s[%d] : %s\n", file, line, cudaGetErrorString(err));
	}
}

#define CUDA_CHECK(err) do { error_check(err, __FILE__, __LINE__); } while(0)

//this function clamp the numbers ORI with float 
__device__ float ClampValue(float temp, float a, float b)
{
	return fmaxf(a, fminf(b, temp));
}

//this is the kernel
__global__
void LaunchTheGalaxy(float * deviceReal_ascension, float* deviceReal_declination, float * deviceFlat_ascension, float* deviceFlat_declination, unsigned long long int * deviceDD, unsigned long long int * deviceDR, unsigned long long int * deviceRR, size_t size)
{
	__shared__ float temp_real_ascension[1024]; 
	__shared__ float temp_real_declination[1024]; 
	__shared__ float temp_flat_ascension[1024]; 
	__shared__ float temp_flat_declination[1024]; 

	__shared__ float tempDD; 

	int tidX = threadIdx.x + blockIdx.x * blockDim.x;

	if ((threadIdx.x >= size) || (threadIdx.y >= size))
	{
		return; 
	}

	for (int i=tidX; i < WINDOW; i++)
	{
		if((i * WINDOW + tidX) <= GRID)
			temp_real_ascension[i] = deviceReal_ascension[i * WINDOW + tidX];
			temp_real_declination[i] = deviceReal_declination[i * WINDOW + tidX];


		//__syncthreads();

		//for (int row = 0; row < WINDOW; row += blockDim.y)
		//{
		//	for (int col = 0; col < WINDOW; col += blockDim.x)
		//	{
		//		tempDD = (acosf(ClampValue(sinf(temp_real_declination[threadIdx.x + col]) * sinf(temp_real_declination[threadIdx.y + row]) + cosf(temp_real_declination[threadIdx.x + col]) * cosf(temp_real_declination[threadIdx.y + row]) * cos(temp_real_ascension[threadIdx.x + col] - temp_real_ascension[threadIdx.y + row]), -1.0f, 1.0f)) * 720.0f / (float)M_PI);
		//		atomicAdd(deviceDD + (int)tempDD, 1);
		//	}
		//}
		//
	}
	__syncthreads();
}

//this is the kernel handler 
void KernelHandler(float * hostReal_ascension, float * hostReal_declination ,float * hostFlat_ascension, float * hostFlat_declination ,unsigned long long int * hostDD, unsigned long long int* hostDR, unsigned long long int* hostRR, size_t size, size_t histogram)
{
	//allocate memory 
	float * deviceReal_ascension = nullptr;
	float* deviceReal_declination = nullptr;
	float * deviceFlat_ascension = nullptr;
	float* deviceFlat_declination = nullptr;

	unsigned long long int* deviceDD = nullptr; 
	unsigned long long int* deviceDR = nullptr;
	unsigned long long int* deviceRR = nullptr;

	CUDA_CHECK(cudaMalloc(&deviceReal_ascension, size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&deviceReal_declination, size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&deviceFlat_ascension, size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&deviceFlat_declination, size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&deviceDD, size * sizeof(unsigned long long int)));
	CUDA_CHECK(cudaMalloc(&deviceDR, size * sizeof(unsigned long long int)));
	CUDA_CHECK(cudaMalloc(&deviceRR, size * sizeof(unsigned long long int)));

	//copy the array 
	CUDA_CHECK(cudaMemcpy(deviceReal_ascension, hostReal_ascension, size * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(deviceReal_declination, hostReal_declination, size * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(deviceFlat_ascension, hostFlat_ascension, size * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(deviceFlat_declination, hostFlat_declination, size * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(deviceDD, hostDD, histogram * sizeof(unsigned long long int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(deviceDR, hostDR, histogram * sizeof(unsigned long long int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(deviceRR, hostRR, histogram * sizeof(unsigned long long int), cudaMemcpyHostToDevice));

	//grid and block
	const dim3 blockSize(TX, TY);
	const int bx = ceil((size + blockSize.x - 1) / blockSize.x); 
	const int by = ceil((size + blockSize.y - 1) / blockSize.y);
	const dim3 gridSize(bx, by);

	//print the information on the screen 
	printf("\nSummary: TX is %d TY is %d\n", TX, TY); 

	//start timer
	clock_t kernelStart = clock();

	LaunchTheGalaxy <<<gridSize, blockSize >>> (deviceReal_ascension, deviceReal_declination, deviceFlat_ascension, deviceFlat_declination ,deviceDD, deviceDR, deviceRR, size);

	//check error in the kernel 
	cudaError_t errAsync = cudaDeviceSynchronize(); 
	cudaError_t errSync = cudaGetLastError(); 

	if (errSync != cudaSuccess)
	{
		printf("\nError in cuda kernel (sync side) %s\n", cudaGetErrorString(errSync));

	}
	//check error async
	if (errAsync != cudaSuccess)
	{
		printf("\nError in cuda kernel (async side) %s\n", cudaGetErrorString(errAsync));
	}

	//end timer 
	clock_t kernelEnd = clock();
	float elapsed = float(kernelEnd - kernelStart) / CLOCKS_PER_SEC;
	printf("\n\nThe kernel timer is %.2f\n\n", elapsed); 

	CUDA_CHECK(cudaMemcpy(hostDD, deviceDD, 720 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost)); 
	CUDA_CHECK(cudaMemcpy(hostDR, deviceDR, 720 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(hostRR, deviceRR, 720 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

	//free the cuda memory 
	CUDA_CHECK(cudaFree(deviceReal_ascension));
	CUDA_CHECK(cudaFree(deviceReal_declination));
	CUDA_CHECK(cudaFree(deviceFlat_ascension));
	CUDA_CHECK(cudaFree(deviceFlat_declination));
	CUDA_CHECK(cudaFree(deviceDD));
	CUDA_CHECK(cudaFree(deviceDR));
	CUDA_CHECK(cudaFree(deviceRR));
}