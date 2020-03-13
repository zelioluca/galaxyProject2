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
__device__ float Clamp(float temp, float a, float b)
{
	return fmaxf(a, fminf(b, temp));
}

__global__ void LaunchGalaxy(float * device_ascension, float * device_declination, unsigned long long int * histogram, size_t size)
{
	//ascension and declination shared 
	__shared__ float S_asc[1024];
	__shared__ float S_dec[1024];

	//shared result 
	__shared__ unsigned long long int S_result[1024]; 

	//threads x 
	int tid = threadIdx.x; 
	//init the array hist to 0
	for (int i = 0; i < 1024; i++)
	{
		S_result[threadIdx.x + i] = 0;
	}

	//go throw every block 
	for (int b = 0; b < 98; b++)
	{
		if (b * threadIdx.x + b * 1024 < size)
		{
			S_asc[threadIdx.x] = device_ascension[threadIdx.x + b * 1024]; 
			S_dec[threadIdx.x] = device_declination[threadIdx.x + b * 1024]; 

			__syncthreads();

			for (int col = 0; col < 1024; col++)
			{
				float temp = acosf(Clamp(__sinf(S_dec[threadIdx.x]) * __sinf(S_dec[threadIdx.x  + col]) +
					__cosf(S_dec[threadIdx.x]) * __cosf(S_dec[threadIdx.x + col]) * __cosf(S_asc[threadIdx.x] - S_asc[threadIdx.x + col]) , -1.f, 1.f)) * 180.0f / (float)M_PI * 4.0f;
				atomicAdd(&S_result[int(temp)], 1); 

				if (blockDim.x == b % gridDim.x)
				{
					S_asc[threadIdx.x] = device_ascension[threadIdx.x + b * 1024];
					S_dec[threadIdx.x] = device_declination[threadIdx.x + b * 1024];

					__syncthreads();
					

				}
			}


		}
	}

}

void Kernel_handler_single(float * host_ascension, float * host_declination, unsigned long long int * host_histogram, size_t size)
{
	//cuda malloc 
	float* device_ascension = nullptr;
	float* device_declination = nullptr; 

	//cuda histogram 
	unsigned long long int * device_histogram = nullptr; 

	//ascension and declination 
	CUDA_CHECK(cudaMalloc(&device_ascension, size * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&device_declination, size * sizeof(float)));

	//histogram 
	CUDA_CHECK(cudaMalloc(&device_histogram, 720 * sizeof(unsigned long long int))); 

	//copy memory
	CUDA_CHECK(cudaMemcpy(device_ascension, host_ascension, size * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(device_declination, host_declination, size * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(device_histogram, host_histogram, 720 * sizeof(float), cudaMemcpyHostToDevice));



	//take kernel time start 
	clock_t s = clock();

	//handle the kernel
	LaunchGalaxy <<< (1, 1, 1), (32, 1, 1) >>> (device_ascension, device_declination, device_histogram, size); 

	//device sync
	cudaError_t errAsync = cudaDeviceSynchronize();

	//check error sync
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("\nError in cuda kernel (sync side) %s\n", cudaGetErrorString(err));

	}
	//check error async
	if (errAsync != cudaSuccess)
	{
		printf("\nError in cuda kernel (async side) %s\n", cudaGetErrorString(errAsync));
	}

	clock_t e = clock();
	//calculate elapsed time
	double el = ((double)(e - s)) / CLOCKS_PER_SEC;
	printf("\n\n### The kernel timer took %f secs\n", el);

	CUDA_CHECK(cudaMemcpy(host_histogram, device_histogram, 720 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

	//cuda free 
	CUDA_CHECK(cudaFree(device_ascension));
	CUDA_CHECK(cudaFree(device_declination)); 
	CUDA_CHECK(cudaFree(device_histogram)); 
}