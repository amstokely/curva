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
