//
// Created by andy on 8/17/22.
//

#ifndef CURVA_CUDAUTILS_H
#define CURVA_CUDAUTILS_H

#include "cuda_runtime_api.h"

dim3 get2DBlockSize (unsigned int blockSize);

dim3 get2DGridSize (
		dim3 blockSize,
		unsigned int m,
		unsigned int n
);

void launchKernel (
		void *kernelName,
		void **args,
		dim3 gridSize,
		dim3 blockSize
);

unsigned int getBlockSize (
		unsigned int n
);

#endif //CURVA_CUDAUTILS_H
