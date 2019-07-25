#ifndef GALAXYKERNEL_H
#define GALAXYKERNEL_H

void KernelHandler(float* hostReal_ascension, float* hostReal_declination, float* hostFlat_ascension, float* hostFlat_declination, unsigned long long int* hostDD, unsigned long long int* hostDR, unsigned long long int* hostRR, size_t size, size_t histogram); 

#endif