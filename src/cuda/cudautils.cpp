//
// Created by andy on 8/17/22.
//
#include "../../include/cuda/cudautils.h"
#include <cmath>
#include <algorithm>

dim3 get2DBlockSize (unsigned int blockSize) {
	return {
			static_cast<unsigned int>(sqrt(blockSize)),
			static_cast<unsigned int>(sqrt(blockSize))
	};
}

dim3 get2DGridSize (
		dim3 blockSize,
		unsigned int m,
		unsigned int n
) {
	unsigned int maxDim = std::max(
			m, n
	);
	return {
			(blockSize.x + maxDim - 1) / blockSize.x,
			(blockSize.y + maxDim - 1) / blockSize.y
	};
}

void launchKernel (
		void *kernelName,
		void **args,
		dim3 gridSize,
		dim3 blockSize
) {
	cudaLaunchKernel(
			kernelName,
			gridSize,
			blockSize,
			args, 0, nullptr
	);
}

unsigned int getBlockSize (
		unsigned int n
) {
	if (n < 1025 && n > 256) {
		return 1024;
	} else if (n < 257 && n > 64) {
		return 256;
	} else {
		return 64;
	}
}