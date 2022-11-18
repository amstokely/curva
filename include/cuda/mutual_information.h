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

#ifndef CURVA_MUTUAL_INFORMATION_H
#define CURVA_MUTUAL_INFORMATION_H

#include "../matrix/curva_matrix.h"

void cudaGeneralizedCorrelation (
		CurvaMatrix<double> *XY,
		CurvaMatrix<double> *generalizedCorrelationMatrix,
		CurvaMatrix<double> *cutoffMatrix,
		int numNodes,
		int numFrames,
		unsigned int totalNumFrames,
		unsigned int firstFrame,
		unsigned int referenceIndex,
		int k
);

void cudaNormalizeNodeCoordinates (
		CurvaMatrix<double> *nodeCoordinates,
		int numFrames,
		int numNodes,
		unsigned int totalNumFrames
);

#endif //CURVA_MUTUAL_INFORMATION_H
