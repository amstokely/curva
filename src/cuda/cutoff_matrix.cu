/*--------------------------------------------------------------------*
 *                         CuRva                                      *
 *--------------------------------------------------------------------*
 * This is part of the GPU-accelerated, random variable analysis      *
 * library CuRva.                                                     *
 * Copyright (C) 2022 Andy Stokely                                    *
 *                                                                    *
 * This program is free software: you can redistribute it             *
 * and/or modify it under the terms of the GNU General Public License *
 * as published by the Free Software Foundation, either version 3 of  *
 * the License, or (at your option) any later version.                *
 *                                                                    *
 * This program is distributed in the hope that it will be useful,    *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
 * GNU General Public License for more details.                       *
 *                                                                    *
 * You should have received a copy of the GNU General Public License  *
 * along with this program.                                           *
 * If not, see <https://www.gnu.org/licenses/>                        *
 * -------------------------------------------------------------------*/

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

