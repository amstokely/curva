//
// Created by andy on 8/12/22.
//
#include <iostream>
#include <map>
#include "../../include/cuda/mutual_information.h"

__device__ void maxDimDeltaKernel (
		double *X,
		double *delta,
		double *deltaCopy,
		int numFrames,
		int frame,
		int tid
) {
	double dx1 = abs(
			X[frame]
			- X[tid]
	);
	double dy1 = abs(
			X[frame + numFrames]
			- X[tid + numFrames]
	);
	double dz1 = abs(
			X[frame + 2 * numFrames]
			- X[tid + 2 * numFrames]
	);
	delta[tid]     = fmax(
			dx1, fmax(dy1, dz1)
	);
	deltaCopy[tid] = delta[tid];
}

__device__ void L1norm (
		double *X,
		double *delta,
		double *deltaCopy,
		int numFrames,
		int frame,
		int tid
) {
	double dx = abs(
			X[frame]
			- X[tid]
	);
	double dy = abs(
			X[frame + numFrames]
			- X[tid + numFrames]
	);
	double dz = abs(
			X[frame + 2 * numFrames]
			- X[tid + 2 * numFrames]
	);
	delta[tid]     = dx + dy + dz;
	deltaCopy[tid] = delta[tid];
}

__device__ void L2norm (
		double *X,
		double *delta,
		double *deltaCopy,
		int numFrames,
		int frame,
		int tid
) {
	double dx = abs(
			X[frame]
			- X[tid]
	);
	double dy = abs(
			X[frame + numFrames]
			- X[tid + numFrames]
	);
	double dz = abs(
			X[frame + 2 * numFrames]
			- X[tid + 2 * numFrames]
	);
	delta[tid]     = sqrt((dx * dx) + (dy * dy) + (dz * dz));
	deltaCopy[tid] = delta[tid];
}

__device__ void dSwap (
		double *a,
		int idx1,
		int idx2
) {
	double tmp1 = a[idx1];
	double tmp2 = a[idx2];
	a[idx1] = tmp2;
	a[idx2] = tmp1;
}

__device__ void iSwap (
		int *a,
		int idx1,
		int idx2
) {
	int tmp1 = a[idx1];
	int tmp2 = a[idx2];
	a[idx1] = tmp2;
	a[idx2] = tmp1;
}

__global__ void normalizeNodeCoordinatesKernel (
		double *nodeCoordinates,
		unsigned int totalNumFrames,
		unsigned int numFrames,
		unsigned int firstFrame
) {
	__shared__ double lx[1024];
	__shared__ double Lx[1];
	__shared__ double ly[1024];
	__shared__ double Ly[1];
	__shared__ double lz[1024];
	__shared__ double Lz[1];
	unsigned int      localThreadIdx = threadIdx.x;
	unsigned int      offset         = blockIdx.x * 3 *
	                                   totalNumFrames + firstFrame;
	unsigned int      xidx           = offset + localThreadIdx;
	unsigned int      yidx           =
			                  offset + localThreadIdx + totalNumFrames;
	unsigned int      zidx           =
			                  offset + localThreadIdx +
			                  2 * totalNumFrames;
	if (localThreadIdx < numFrames) {
		lx[localThreadIdx] =
				nodeCoordinates[xidx] * nodeCoordinates[xidx];
		ly[localThreadIdx] =
				nodeCoordinates[yidx] * nodeCoordinates[yidx];
		lz[localThreadIdx] =
				nodeCoordinates[zidx] * nodeCoordinates[zidx];
	} else {
		lx[localThreadIdx] = 0.0;
		ly[localThreadIdx] = 0.0;
		lz[localThreadIdx] = 0.0;
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2;
	     s > 0;
	     s >>= 1) {
		if (localThreadIdx < s) {
			lx[localThreadIdx] += lx[localThreadIdx + s];
			ly[localThreadIdx] += ly[localThreadIdx + s];
			lz[localThreadIdx] += lz[localThreadIdx + s];
		}
		__syncthreads();
	}
	if (localThreadIdx == 0) {
		Lx[localThreadIdx] = sqrt(lx[localThreadIdx]);
		Ly[localThreadIdx] = sqrt(ly[localThreadIdx]);
		Lz[localThreadIdx] = sqrt(lz[localThreadIdx]);
	}
	__syncthreads();
	if (localThreadIdx < numFrames) {
		nodeCoordinates[xidx] /= Lx[0];
		nodeCoordinates[yidx] /= Ly[0];
		nodeCoordinates[zidx] /= Lz[0];
	}
	__syncthreads();
}

__global__ void mutualInformationKernelL1Norm (
		double *XY,
		int *nNode1,
		int *nNode2,
		int numFrames,
		unsigned int totalNumFrames,
		unsigned int firstFrame,
		int k,
		unsigned int X1Index,
		unsigned int X2Index
) {
	__shared__ double delta1[1024];
	__shared__ double eps1[1];
	__shared__ double eps2[1];
	__shared__ double delta1Copy[1024];
	__shared__ double delta2[1024];
	__shared__ double delta2Copy[1024];
	__shared__ int    nodeIndices[1024];
	__shared__ int    kNearestNeighbors[1024];
	unsigned int      offset         = 3 * totalNumFrames * X2Index +
	                                   firstFrame;
	double            *X1            =
			                  &XY[3 * totalNumFrames *
			                      X1Index + firstFrame];
	double            *X2            = &XY[offset];
	int               localThreadIdx = threadIdx.x;
	int               frame          = blockIdx.x;
	delta1[localThreadIdx]      = RAND_MAX;
	delta1Copy[localThreadIdx]  = RAND_MAX;
	delta2[localThreadIdx]      = RAND_MAX;
	delta2Copy[localThreadIdx]  = RAND_MAX;
	nodeIndices[localThreadIdx] = localThreadIdx;
	__syncthreads();
	if (localThreadIdx < numFrames) {
		L1norm(
				X1,
				delta1,
				delta1Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
		L1norm(
				X2,
				delta2,
				delta2Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
	}
	__syncthreads();
	for (int i = 0;
	     i < k + 1;
	     i++) {
		for (int s                  = blockDim.x / 2;
		     s > 0;
		     s >>= 1) {
			if (localThreadIdx < s) {
				double d1tmp1 = delta1[localThreadIdx];
				double d2tmp1 = delta2[localThreadIdx];
				double d1tmp2 = delta1[localThreadIdx + s];
				double d2tmp2 = delta2[localThreadIdx + s];
				double tmp1   = fmax(
						d1tmp1,
						d2tmp1
				);
				double tmp2   = fmax(
						d1tmp2,
						d2tmp2
				);
				if (tmp2 < tmp1) {
					dSwap(delta1, localThreadIdx, localThreadIdx + s);
					dSwap(delta2, localThreadIdx, localThreadIdx + s);
					iSwap(
							nodeIndices,
							localThreadIdx, localThreadIdx + s
					);
				}
			}
			__syncthreads();
		}
		if (localThreadIdx == 0) {
			kNearestNeighbors[i]       = nodeIndices[
					0
			];
			delta1Copy[nodeIndices[0]] = RAND_MAX;
			delta2Copy[nodeIndices[0]] = RAND_MAX;
		}
		__syncthreads();
		if (localThreadIdx < numFrames) {
			if (localThreadIdx == nodeIndices[0]) {
				delta1[localThreadIdx] = RAND_MAX;
				delta2[localThreadIdx] = RAND_MAX;
			} else {
				delta1[localThreadIdx] = delta1Copy[localThreadIdx];
				delta2[localThreadIdx] = delta2Copy[localThreadIdx];
			}
		} else {
			delta1[localThreadIdx] = RAND_MAX;
			delta2[localThreadIdx] = RAND_MAX;
		}
		nodeIndices[localThreadIdx] = localThreadIdx;
		__syncthreads();
	}
	__syncthreads();
	delta1[localThreadIdx]      = RAND_MAX;
	delta1Copy[localThreadIdx]  = RAND_MAX;
	delta2[localThreadIdx]      = RAND_MAX;
	delta2Copy[localThreadIdx]  = RAND_MAX;
	nodeIndices[localThreadIdx] = localThreadIdx;
	__syncthreads();
	if (localThreadIdx < numFrames) {
		L1norm(
				X1,
				delta1,
				delta1Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
		L1norm(
				X2,
				delta2,
				delta2Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
	}
	__syncthreads();
	delta1Copy[localThreadIdx] = 0.0;
	delta2Copy[localThreadIdx] = 0.0;
	__syncthreads();
	if (localThreadIdx == 0) {
		eps1[localThreadIdx] = 0.0;
		eps2[localThreadIdx] = 0.0;
		for (unsigned int i = 1;
		     i < k + 1;
		     i++) {
			unsigned int kindex = kNearestNeighbors[i];
			double       dx1    = abs(
					X1[frame] -
					X1[kindex]
			);
			double       dy1    = abs(
					X1[frame + totalNumFrames] -
					X1[kindex + totalNumFrames]
			);
			double       dz1    = abs(
					X1[frame + 2 * totalNumFrames] -
					X1[kindex + 2 * totalNumFrames]
			);
			double       dX     = dx1 + dy1 + dz1;
			double       tmp    = eps1[0];
			eps1[0]             = fmax(tmp, dX);
			double dx2 = abs(
					X2[frame] -
					X2[kindex]
			);
			double dy2 = abs(
					X2[frame + totalNumFrames] -
					X2[kindex + totalNumFrames]
			);
			double dz2 = abs(
					X2[frame + 2 * totalNumFrames] -
					X2[kindex + 2 * totalNumFrames]
			);
			double dY  = dx2 + dy2 + dz2;
			tmp = eps2[0];
			eps2[0] = fmax(
					tmp, dY
			);
		}
	}
	__syncthreads();
	if (localThreadIdx < numFrames) {
		if (delta1[localThreadIdx] <= eps1[0]) {
			delta1Copy[localThreadIdx] = 1.0;
		}
		if (delta2[localThreadIdx] <= eps2[0]) {
			delta2Copy[localThreadIdx] = 1.0;
		}
	}
	__syncthreads();
	for (int s = blockDim.x / 2;
	     s > 0;
	     s >>= 1) {
		if (localThreadIdx < s) {
			delta1Copy[localThreadIdx]
					+= delta1Copy[localThreadIdx + s];
			delta2Copy[localThreadIdx]
					+= delta2Copy[localThreadIdx + s];
		}
		__syncthreads();
	}
	if (localThreadIdx == 0) {
		nNode1[frame + numFrames * X2Index] =
				(int) delta1Copy[0] - 1;
		nNode2[frame + numFrames * X2Index]
				= (int) delta2Copy[0] - 1;
	}
}

__global__ void mutualInformationKernelL2Norm (
		double *XY,
		int *nNode1,
		int *nNode2,
		int numFrames,
		unsigned int totalNumFrames,
		unsigned int firstFrame,
		int k,
		unsigned int X1Index,
		unsigned int X2Index
) {
	__shared__ double delta1[1024];
	__shared__ double eps1[1];
	__shared__ double eps2[1];
	__shared__ double delta1Copy[1024];
	__shared__ double delta2[1024];
	__shared__ double delta2Copy[1024];
	__shared__ int    nodeIndices[1024];
	__shared__ int    kNearestNeighbors[1024];
	unsigned int      offset         = 3 * totalNumFrames * X2Index +
	                                   firstFrame;
	double            *X1            =
			                  &XY[3 * totalNumFrames *
			                      X1Index + firstFrame];
	double            *X2            = &XY[offset];
	int               localThreadIdx = threadIdx.x;
	int               frame          = blockIdx.x;
	delta1[localThreadIdx]      = RAND_MAX;
	delta1Copy[localThreadIdx]  = RAND_MAX;
	delta2[localThreadIdx]      = RAND_MAX;
	delta2Copy[localThreadIdx]  = RAND_MAX;
	nodeIndices[localThreadIdx] = localThreadIdx;
	__syncthreads();
	if (localThreadIdx < numFrames) {
		L2norm(
				X1,
				delta1,
				delta1Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
		L2norm(
				X2,
				delta2,
				delta2Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
	}
	__syncthreads();
	for (int i = 0;
	     i < k + 1;
	     i++) {
		for (int s                  = blockDim.x / 2;
		     s > 0;
		     s >>= 1) {
			if (localThreadIdx < s) {
				double d1tmp1 = delta1[localThreadIdx];
				double d2tmp1 = delta2[localThreadIdx];
				double d1tmp2 = delta1[localThreadIdx + s];
				double d2tmp2 = delta2[localThreadIdx + s];
				double tmp1   = fmax(
						d1tmp1,
						d2tmp1
				);
				double tmp2   = fmax(
						d1tmp2,
						d2tmp2
				);
				if (tmp2 < tmp1) {
					dSwap(delta1, localThreadIdx, localThreadIdx + s);
					dSwap(delta2, localThreadIdx, localThreadIdx + s);
					iSwap(
							nodeIndices,
							localThreadIdx, localThreadIdx + s
					);
				}
			}
			__syncthreads();
		}
		if (localThreadIdx == 0) {
			kNearestNeighbors[i]       = nodeIndices[
					0
			];
			delta1Copy[nodeIndices[0]] = RAND_MAX;
			delta2Copy[nodeIndices[0]] = RAND_MAX;
		}
		__syncthreads();
		if (localThreadIdx < numFrames) {
			if (localThreadIdx == nodeIndices[0]) {
				delta1[localThreadIdx] = RAND_MAX;
				delta2[localThreadIdx] = RAND_MAX;
			} else {
				delta1[localThreadIdx] = delta1Copy[localThreadIdx];
				delta2[localThreadIdx] = delta2Copy[localThreadIdx];
			}
		} else {
			delta1[localThreadIdx] = RAND_MAX;
			delta2[localThreadIdx] = RAND_MAX;
		}
		nodeIndices[localThreadIdx] = localThreadIdx;
		__syncthreads();
	}
	__syncthreads();
	delta1[localThreadIdx]      = RAND_MAX;
	delta1Copy[localThreadIdx]  = RAND_MAX;
	delta2[localThreadIdx]      = RAND_MAX;
	delta2Copy[localThreadIdx]  = RAND_MAX;
	nodeIndices[localThreadIdx] = localThreadIdx;
	__syncthreads();
	if (localThreadIdx < numFrames) {
		L2norm(
				X1,
				delta1,
				delta1Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
		L2norm(
				X2,
				delta2,
				delta2Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
	}
	__syncthreads();
	delta1Copy[localThreadIdx] = 0.0;
	delta2Copy[localThreadIdx] = 0.0;
	__syncthreads();
	if (localThreadIdx == 0) {
		eps1[localThreadIdx] = 0.0;
		eps2[localThreadIdx] = 0.0;
		for (unsigned int i = 1;
		     i < k + 1;
		     i++) {
			unsigned int kindex = kNearestNeighbors[i];
			double       dx1    = abs(
					X1[frame] -
					X1[kindex]
			);
			double       dy1    = abs(
					X1[frame + totalNumFrames] -
					X1[kindex + totalNumFrames]
			);
			double       dz1    = abs(
					X1[frame + 2 * totalNumFrames] -
					X1[kindex + 2 * totalNumFrames]
			);
			double       dX     = sqrt(
					(dx1 * dx1)
					+ (dy1 * dy1)
					+ (dz1 * dz1)
			);
			double       tmp    = eps1[0];
			eps1[0]             = fmax(tmp, dX);
			double dx2 = abs(
					X2[frame] -
					X2[kindex]
			);
			double dy2 = abs(
					X2[frame + totalNumFrames] -
					X2[kindex + totalNumFrames]
			);
			double dz2 = abs(
					X2[frame + 2 * totalNumFrames] -
					X2[kindex + 2 * totalNumFrames]
			);
			double dY  = sqrt(
					(dx2 * dx2)
					+ (dy2 * dy2)
					+ (dz2 * dz2)
			);
			tmp = eps2[0];
			eps2[0] = fmax(
					tmp, dY
			);
		}
	}
	__syncthreads();
	if (localThreadIdx < numFrames) {
		if (delta1[localThreadIdx] <= eps1[0]) {
			delta1Copy[localThreadIdx] = 1.0;
		}
		if (delta2[localThreadIdx] <= eps2[0]) {
			delta2Copy[localThreadIdx] = 1.0;
		}
	}
	__syncthreads();
	for (int s = blockDim.x / 2;
	     s > 0;
	     s >>= 1) {
		if (localThreadIdx < s) {
			delta1Copy[localThreadIdx]
					+= delta1Copy[localThreadIdx + s];
			delta2Copy[localThreadIdx]
					+= delta2Copy[localThreadIdx + s];
		}
		__syncthreads();
	}
	if (localThreadIdx == 0) {
		nNode1[frame + numFrames * X2Index] =
				(int) delta1Copy[0] - 1;
		nNode2[frame + numFrames * X2Index]
				= (int) delta2Copy[0] - 1;
	}
}

__global__ void mutualInformationKernelMaxNorm (
		double *XY,
		int *nNode1,
		int *nNode2,
		int numFrames,
		unsigned int totalNumFrames,
		unsigned int firstFrame,
		int k,
		unsigned int X1Index,
		unsigned int X2Index
) {
	__shared__ double delta1[1024];
	__shared__ double eps1[1];
	__shared__ double eps2[1];
	__shared__ double delta1Copy[1024];
	__shared__ double delta2[1024];
	__shared__ double delta2Copy[1024];
	__shared__ int    nodeIndices[1024];
	__shared__ int    kNearestNeighbors[1024];
	unsigned int      offset         = 3 * totalNumFrames * X2Index +
	                                   firstFrame;
	double            *X1            =
			                  &XY[3 * totalNumFrames *
			                      X1Index + firstFrame];
	double            *X2            = &XY[offset];
	int               localThreadIdx = threadIdx.x;
	int               frame          = blockIdx.x;
	delta1[localThreadIdx]      = RAND_MAX;
	delta1Copy[localThreadIdx]  = RAND_MAX;
	delta2[localThreadIdx]      = RAND_MAX;
	delta2Copy[localThreadIdx]  = RAND_MAX;
	nodeIndices[localThreadIdx] = localThreadIdx;
	__syncthreads();
	if (localThreadIdx < numFrames) {
		maxDimDeltaKernel(
				X1,
				delta1,
				delta1Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
		maxDimDeltaKernel(
				X2,
				delta2,
				delta2Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
	}
	__syncthreads();
	for (int i = 0;
	     i < k + 1;
	     i++) {
		for (int s                  = blockDim.x / 2;
		     s > 0;
		     s >>= 1) {
			if (localThreadIdx < s) {
				double d1tmp1 = delta1[localThreadIdx];
				double d2tmp1 = delta2[localThreadIdx];
				double d1tmp2 = delta1[localThreadIdx + s];
				double d2tmp2 = delta2[localThreadIdx + s];
				double tmp1   = fmax(
						d1tmp1,
						d2tmp1
				);
				double tmp2   = fmax(
						d1tmp2,
						d2tmp2
				);
				if (tmp2 < tmp1) {
					dSwap(delta1, localThreadIdx, localThreadIdx + s);
					dSwap(delta2, localThreadIdx, localThreadIdx + s);
					iSwap(
							nodeIndices,
							localThreadIdx, localThreadIdx + s
					);
				}
			}
			__syncthreads();
		}
		if (localThreadIdx == 0) {
			kNearestNeighbors[i]       = nodeIndices[
					0
			];
			delta1Copy[nodeIndices[0]] = RAND_MAX;
			delta2Copy[nodeIndices[0]] = RAND_MAX;
		}
		__syncthreads();
		if (localThreadIdx < numFrames) {
			if (localThreadIdx == nodeIndices[0]) {
				delta1[localThreadIdx] = RAND_MAX;
				delta2[localThreadIdx] = RAND_MAX;
			} else {
				delta1[localThreadIdx] = delta1Copy[localThreadIdx];
				delta2[localThreadIdx] = delta2Copy[localThreadIdx];
			}
		} else {
			delta1[localThreadIdx] = RAND_MAX;
			delta2[localThreadIdx] = RAND_MAX;
		}
		nodeIndices[localThreadIdx] = localThreadIdx;
		__syncthreads();
	}
	__syncthreads();
	delta1[localThreadIdx]      = RAND_MAX;
	delta1Copy[localThreadIdx]  = RAND_MAX;
	delta2[localThreadIdx]      = RAND_MAX;
	delta2Copy[localThreadIdx]  = RAND_MAX;
	nodeIndices[localThreadIdx] = localThreadIdx;
	__syncthreads();
	if (localThreadIdx < numFrames) {
		maxDimDeltaKernel(
				X1,
				delta1,
				delta1Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
		maxDimDeltaKernel(
				X2,
				delta2,
				delta2Copy,
				totalNumFrames,
				frame,
				localThreadIdx
		);
	}
	__syncthreads();
	delta1Copy[localThreadIdx] = 0.0;
	delta2Copy[localThreadIdx] = 0.0;
	__syncthreads();
	if (localThreadIdx == 0) {
		eps1[localThreadIdx] = 0.0;
		eps2[localThreadIdx] = 0.0;
		for (unsigned int i = 1;
		     i < k + 1;
		     i++) {
			unsigned int kindex = kNearestNeighbors[i];
			double       dx1    = abs(
					X1[frame] -
					X1[kindex]
			);
			double       dy1    = abs(
					X1[frame + totalNumFrames] -
					X1[kindex + totalNumFrames]
			);
			double       dz1    = abs(
					X1[frame + 2 * totalNumFrames] -
					X1[kindex + 2 * totalNumFrames]
			);
			double       tmp    = eps1[0];
			eps1[0]             = fmax(
					tmp, fmax(
							dx1, fmax(dy1, dz1)
					));
			double dx2 = abs(
					X2[frame] -
					X2[kindex]
			);
			double dy2 = abs(
					X2[frame + totalNumFrames] -
					X2[kindex + totalNumFrames]
			);
			double dz2 = abs(
					X2[frame + 2 * totalNumFrames] -
					X2[kindex + 2 * totalNumFrames]
			);
			tmp = eps2[0];
			eps2[0] = fmax(
					tmp, fmax(
							dx2, fmax(dy2, dz2)
					));
		}
	}
	__syncthreads();
	if (localThreadIdx < numFrames) {
		if (delta1[localThreadIdx] <= eps1[0]) {
			delta1Copy[localThreadIdx] = 1.0;
		}
		if (delta2[localThreadIdx] <= eps2[0]) {
			delta2Copy[localThreadIdx] = 1.0;
		}
	}
	__syncthreads();
	for (int s = blockDim.x / 2;
	     s > 0;
	     s >>= 1) {
		if (localThreadIdx < s) {
			delta1Copy[localThreadIdx]
					+= delta1Copy[localThreadIdx + s];
			delta2Copy[localThreadIdx]
					+= delta2Copy[localThreadIdx + s];
		}
		__syncthreads();
	}
	if (localThreadIdx == 0) {
		nNode1[frame + numFrames * X2Index] =
				(int) delta1Copy[0] - 1;
		nNode2[frame + numFrames * X2Index]
				= (int) delta2Copy[0] - 1;
	}
}

void cudaNormalizeNodeCoordinates (
		CurvaMatrix<double> *nodeCoordinates,
		int numFrames,
		int numNodes,
		unsigned int totalNumFrames
) {
	nodeCoordinates->toDevice();
	for (unsigned int firstFrame = 0;
	     firstFrame < totalNumFrames;
	     firstFrame += numFrames) {
		normalizeNodeCoordinatesKernel<<<
		numNodes, 1024
		>>>(
				nodeCoordinates->device(),
				totalNumFrames,
				numFrames,
				firstFrame
		);
	}
}

void *getMutualInformationKernel (
		const std::string &norm
) {
	std::map<std::string, void *> kernels = {
			{"max", (void *) &mutualInformationKernelMaxNorm},
			{"l2",  (void *) &mutualInformationKernelL2Norm},
			{"l1",  (void *) &mutualInformationKernelL1Norm}
	};
	return kernels[norm];
}

void cudaMutualInformation (
		CurvaMatrix<double> *XY,
		CurvaMatrix<double> *mutualInformationMatrix,
		CurvaMatrix<double> *cutoffMatrix,
		int numNodes,
		int numFrames,
		unsigned int totalNumFrames,
		unsigned int firstFrame,
		unsigned int referenceIndex,
		int k,
		const std::string &norm
) {
	auto                *nNode1 = new CurvaMatrix<int>;
	auto                *nNode2 = new CurvaMatrix<int>;
	std::vector<double> psi(numFrames + 1, 0.0);
	std::vector<double> phi(k + 1, 0.0);
	psi.at(1) = -0.57721566490153;
	for (int i                = 0;
	     i < numFrames;
	     i++) {
		if (i > 0) {
			psi[i + 1] = psi[i] + (1.0 / (double) i);
		}
	}
	for (int i                = 1;
	     i < k + 1;
	     i++) {
		phi.at(i) = psi.at(i) - (1.0 / (double) i);
	}
	nNode1->init(numNodes * numFrames, 1);
	nNode2->init(numNodes * numFrames, 1);
	nNode1->allocate();
	nNode2->allocate();
	nNode1->toDevice();
	nNode2->toDevice();
	dim3              gridSize(numFrames);
	void              *kernel = getMutualInformationKernel(
			norm
	);
	for (unsigned int X2Index = 0;
	     X2Index < numNodes;
	     X2Index++) {
		void *mutualInformationKernelArgs[] = {
				(void *) &XY->device(),
				(void *) &nNode1->device(),
				(void *) &nNode2->device(),
				(void *) &numFrames,
				(void *) &totalNumFrames,
				(void *) &firstFrame,
				(void *) &k,
				(void *) &referenceIndex,
				(void *) &X2Index
		};
		if ((*(cutoffMatrix))(referenceIndex, X2Index) != 0.0) {
			cudaLaunchKernel(
					kernel,
					dim3(gridSize),
					dim3(1024),
					mutualInformationKernelArgs
			);
		}
	}
	nNode1->toHost();
	nNode2->toHost();
	for (int I = 0;
	     I < numNodes;
	     I++) {
		if ((*(cutoffMatrix))(referenceIndex, I) != 0.0) {
			double       mi     = 0.0;
			unsigned int offset = I * numFrames;
			for (int     i      = 0;
			     i < numFrames;
			     i++) {
				int idx1 = (*(nNode1))(i + offset, 0);
				int idx2 = (*(nNode2))(i + offset, 0);
				mi += psi[idx1] + psi[idx2];
			}
			double       tmp    =
					             psi[numFrames] + phi[k] -
					             (mi / numFrames);
			if (tmp > 0.0) {
				(*(mutualInformationMatrix))(referenceIndex, I) +=
						sqrt(
								1 - exp(
										-1 * tmp * (2.0 / 3.0)
								)
						);
			}
		}
	}
	delete nNode1;
	delete nNode2;
}