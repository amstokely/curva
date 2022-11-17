//
// Created by andy on 8/17/22.
//
#include "../../include/cuda/cutoff_matrix.h"
#include "../../include/cuda/cudautils.h"

__global__ void cutoffMatrixKernel (
		const double *averageNodePositions,
		double *cutoffMatrix,
		double cutoff,
		unsigned int numNodes
) {
	unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < numNodes && j < numNodes) {
		unsigned int xIdx1     = i;
		unsigned int xIdx2     = j;
		unsigned int yIdx1     = i + numNodes;
		unsigned int yIdx2     = j + numNodes;
		unsigned int zIdx1     = i + 2 * numNodes;
		unsigned int zIdx2     = j + 2 * numNodes;
		unsigned int cutoffIdx = i * numNodes + j;
		double       dx        = averageNodePositions[xIdx1] -
		                         averageNodePositions[xIdx2];
		double       dy        = averageNodePositions[yIdx1] -
		                         averageNodePositions[yIdx2];
		double       dz        = averageNodePositions[zIdx1] -
		                         averageNodePositions[zIdx2];
		double       dxdx      = dx * dx;
		double       dydy      = dy * dy;
		double       dzdz      = dz * dz;
		double       distance  = sqrtf(dxdx + dydy + dzdz);
		if (distance > cutoff) {
			cutoffMatrix[cutoffIdx] = 0.0;
		} else {
			cutoffMatrix[cutoffIdx] = 1.0;
		}
	}
}

void cudaCutoffMatrix (
		CurvaMatrix<double> *averageNodePositionMatrix,
		CurvaMatrix<double> *cutoffMatrix,
		double cutoff,
		unsigned int numNodes,
		unsigned int numFrames
) {
	unsigned int blockSize =
			             getBlockSize(numFrames);
	averageNodePositionMatrix->toDevice();
	cutoffMatrix->toDevice();
	dim3 cutoffMatrixKernelBlockSize                         =
			     get2DBlockSize(blockSize);
	dim3 cutoffMatrixKernelGridSize                          =
			     get2DGridSize(
					     cutoffMatrixKernelBlockSize,
					     numNodes, numNodes
			     );
	void *cutoffMatrixKernelArgs[]                           = {
			(void *) &averageNodePositionMatrix->device(),
			(void *) &cutoffMatrix->device(),
			(void *) &cutoff,
			(void *) &numNodes
	};
	launchKernel(
			(void *) &cutoffMatrixKernel,
			cutoffMatrixKernelArgs,
			cutoffMatrixKernelGridSize,
			cutoffMatrixKernelBlockSize
	);
	cutoffMatrix->toHost();
}

