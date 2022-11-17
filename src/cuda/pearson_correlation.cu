
#include <iostream>
#include "../../include/cuda/pearson_correlation.h"
#include "../../include/cuda/cudautils.h"


__global__ void normalizeKernel (
		const double *a,
		double *normalizedA,
		unsigned int n
) {
	__shared__ double meanAndMagnitudeSm[1024];
	__shared__ double normalizedASm[1024];

	unsigned int globalThreadIdx = (
			threadIdx.x + (blockIdx.x * blockDim.x)
	);
	unsigned int localThreadIdx  = threadIdx.x;

	unsigned int offset = blockDim.x - n;


	meanAndMagnitudeSm[localThreadIdx] = 0.0;
	normalizedASm[localThreadIdx]      = 0.0;
	__syncthreads();

	if (localThreadIdx < n) {
		meanAndMagnitudeSm[localThreadIdx] =
				a[globalThreadIdx - blockIdx.x * offset];
		normalizedASm[localThreadIdx]      =
				a[globalThreadIdx - blockIdx.x * offset];
	}
	__syncthreads();
	for (unsigned int s    = blockDim.x / 2;
	     s > 0;
	     s >>= 1) {
		if (localThreadIdx < s) {
			meanAndMagnitudeSm[
					localThreadIdx
			] += meanAndMagnitudeSm[
					s + localThreadIdx
			];
		}
		__syncthreads();
	}
	double            mean = meanAndMagnitudeSm[0] / n;
	__syncthreads();
	meanAndMagnitudeSm[localThreadIdx] = 0.0;
	__syncthreads();
	if (localThreadIdx < n) {
		double tmp        = normalizedASm[localThreadIdx] - mean;
		double tmpSquared = tmp * tmp;
		normalizedASm[localThreadIdx]      = tmp;
		meanAndMagnitudeSm[localThreadIdx] = tmpSquared;
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2;
	     s > 0;
	     s >>= 1) {
		if (localThreadIdx < s) {
			meanAndMagnitudeSm[
					localThreadIdx
			] += meanAndMagnitudeSm[
					s + localThreadIdx
			];
		}
		__syncthreads();
	}
	if (localThreadIdx < n) {
		normalizedA[globalThreadIdx - blockIdx.x * offset] = (
				normalizedASm[localThreadIdx]
				/ sqrtf(meanAndMagnitudeSm[0])
		);
	}
}

__global__ void pearsonCorrelationMatrixKernel (
		const double *a,
		const double *aT,
		const double *cutoffMatrix,
		double *p,
		unsigned int m,
		unsigned int n
) {
	unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
	if (row < m && col < m) {
		double            tmp = 0.0;
		for (unsigned int k   = 0;
		     k < n;
		     k++) {
			tmp += a[row * n + k] * aT[k * m + col];
		}
		p[row * m + col] += tmp * cutoffMatrix[row * m + col];
	}
}


void cudaPearsonCorrelation (
		CurvaMatrix<double> *deltaAverageNodePositionMatrix,
		CurvaMatrix<double> *pearsonCorrelationMatrix,
		CurvaMatrix<double> *cutoffMatrix,
		unsigned int numNodes,
		unsigned int numFrames
) {
	unsigned int blockSize =
			             getBlockSize(numFrames);
	deltaAverageNodePositionMatrix->toDevice();
	auto *normalizedDeltaAveragePositionMatrix = new
			CurvaMatrix<double>;
	normalizedDeltaAveragePositionMatrix->init(
			numNodes, numFrames
	);
	normalizedDeltaAveragePositionMatrix->allocate();
	normalizedDeltaAveragePositionMatrix->toDevice();
	auto *normalizedDeltaAveragePositionMatrixTranspose = new
			CurvaMatrix<double>;
	dim3 normalizeKernelBlockSize(blockSize);
	dim3 normalizeKernelGridSize(
			numNodes
	);
	pearsonCorrelationMatrix->toDevice();
	dim3 transposeAndPearsonCorrelationMatrixKernelBlockSize =
			     get2DBlockSize(blockSize);
	dim3 transposeAndPearsonCorrelationMatrixKernelGridSize  =
			     get2DGridSize(
					     transposeAndPearsonCorrelationMatrixKernelBlockSize,
					     numNodes, numFrames
			     );
	void *normalizeKernelArgs[]                              = {
			(void *) &deltaAverageNodePositionMatrix->device(),
			(void *) &normalizedDeltaAveragePositionMatrix->device(),
			(void *) &numFrames
	};
	normalizedDeltaAveragePositionMatrixTranspose->transposeHost();
	void *pearsonCorrelationMatrixKernelArgs[] = {
			(void *) &normalizedDeltaAveragePositionMatrix->device(),
			(void *)
					&normalizedDeltaAveragePositionMatrixTranspose
							->device(),
			(void *) &cutoffMatrix->device(),
			(void *) &pearsonCorrelationMatrix->device(),
			(void *) &numNodes,
			(void *) &numFrames
	};

	launchKernel(
			(void *) &normalizeKernel,
			normalizeKernelArgs,
			normalizeKernelGridSize,
			normalizeKernelBlockSize
	);
	normalizedDeltaAveragePositionMatrixTranspose->copyDevice(
			normalizedDeltaAveragePositionMatrix
	);
	normalizedDeltaAveragePositionMatrixTranspose->transposeDevice();
	launchKernel(
			(void *) &pearsonCorrelationMatrixKernel,
			pearsonCorrelationMatrixKernelArgs,
			transposeAndPearsonCorrelationMatrixKernelGridSize,
			transposeAndPearsonCorrelationMatrixKernelBlockSize
	);
	pearsonCorrelationMatrix->toHost();
	normalizedDeltaAveragePositionMatrix->deallocate();
	normalizedDeltaAveragePositionMatrixTranspose->deallocate();
	delete normalizedDeltaAveragePositionMatrix;
	delete normalizedDeltaAveragePositionMatrixTranspose;
}
