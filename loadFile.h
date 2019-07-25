#ifndef LOADFILE_H
#define LOADFILE_H

#include <string>

size_t VectorSize(std::string filename); 
float* CreateDeviceVector(std::string filename, size_t size, bool isAscension); 


#endif
