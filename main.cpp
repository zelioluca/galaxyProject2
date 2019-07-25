#include <iostream>
#include <stdio.h>
#include <ctime>

#include "loadFile.h"; 
#include "galaxyKernel.h"

constexpr auto HISTOGRAM = 720 ;

using namespace std; 

int main(int argc, char** argv)
{
	//get the size of the array 
	size_t size = VectorSize("Data/data_100k_arcmin.txt"); 
	std::cout << "The size of the array galaxy is " << size << std::endl; 

	//allocate memory for the two device real and flat
	float* hostReal_ascension = CreateDeviceVector("Data/data_100k_arcmin.txt", size, true);
	float* hostReal_descension = CreateDeviceVector("Data/data_100k_arcmin.txt", size, false);

	float* hostFlat_ascension = CreateDeviceVector("Data/flat_100k_arcmin.txt", size, true);
	float* hostFlat_descension = CreateDeviceVector("Data/flat_100k_arcmin.txt", size, false);

	//allocate memory for the histogram
	unsigned long long int* hostDD = (unsigned long long int*)malloc(HISTOGRAM * sizeof(unsigned long long int)); 
	unsigned long long int* hostRR = (unsigned long long int*)malloc(HISTOGRAM * sizeof(unsigned long long int));
	unsigned long long int* hostDR = (unsigned long long int*)malloc(HISTOGRAM * sizeof(unsigned long long int));
	float* scientificDifference = (float*)malloc(HISTOGRAM * sizeof(float));

	//initialize to 0
	for (int i = 0; i < HISTOGRAM; i++)
	{
		hostDD[i] = 0; 
		hostRR[i] = 0; 
		hostDR[i] = 0; 
		scientificDifference[i] = 0; 
	}

	//invoke kernel
	KernelHandler(hostReal_ascension, hostReal_descension ,hostFlat_ascension, hostFlat_descension, hostDD, hostDR, hostRR, size, HISTOGRAM);
	bool debug = true;

	if (debug)
	{
		unsigned long long int sumDD = 0; 

		for (int i = 0; i < HISTOGRAM; i++)
		{
			std::cout << "HistDD[" << i << "] => " << hostDD[i] << std::endl; 
			sumDD += hostDD[i]; 
		}

		std::cout << "The sum of histDD is " << sumDD << std::endl; 
	}

	//free the memory
	delete[] hostReal_ascension;
	delete[] hostReal_descension;
	delete[] hostFlat_ascension;
	delete[] hostFlat_descension;
	delete[] hostDD; 
	delete[] hostRR; 
	delete[] hostDR; 
	delete[] scientificDifference; 
}