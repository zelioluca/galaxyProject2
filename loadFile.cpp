//Author Luca Zelioli 
//Inclusion 
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <ctime>
#include <exception>

//local inclusion
#include "loadFile.h"

//Math inclusion
#define _USE_MATH_DEFINES
#include <string>
#include <math.h>

using namespace std; 

size_t VectorSize(string filename)
{
	size_t temp = NULL; 

	try
	{
		//Open the stream 
		ifstream MyFile(filename); 
 
		//check if it is open
		if (MyFile.is_open())
		{
			MyFile >> temp;
		}
		else
		{
			std::cout << "Impossible to open the file " + filename << std::endl; 
		}

		return temp; 
	}
	catch (exception& ex)
	{
		std::cout << "Exception thrown in the function VectorSize " << ex.what() << std::endl;
		return NULL;
	}
}

//this function is able to create local vector 
float* CreateDeviceVector(string filename, size_t size ,bool isAscension)
{
	float* vector = nullptr; 
	float temp = 0; 
	vector = new float[size]; 
	//insert the index 
	int idx = 0; 

	try
	{
		//start the timer
		clock_t startTimer = clock();
	
		//open the stream 
		ifstream MyFile(filename); 

		if (MyFile.is_open())
		{
			for (int i = 0; i < size; i++)
			{
				//jump the first element 
				if (i == 0)
				{
					continue;
				}
				//insert and transform the data 
				MyFile >> temp;
				temp = temp * 1 / 60 * (float)M_PI / 180.0f;
								
				if (isAscension == true)
				{
					if (i % 2 == 0)
					{
						vector[idx] = temp;
						idx++; 
					}
				}
				else
				{
					if (i % 2 != 0)
					{
						vector[idx] = temp;
						idx++; 
					}
				}
			}
		}
		else
		{
			std::cout << "Impossible to open the file stream" << std::endl; 
			return nullptr; 
		}

		//end the timer
		clock_t endTimer = clock();
		//calculate the elapsed time
		float elapsed = (float)(endTimer - startTimer) / CLOCKS_PER_SEC;
		//show 
		std::cout << "The file " << filename << " was loaded in " << elapsed << std::endl;

		return vector; 

	}
	catch (exception& ex)
	{
		std::cout << "Exception thrown in the function CreateDescensionVector " << ex.what() << std::endl;
		return nullptr;
	}

}


