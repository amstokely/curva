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

#ifndef CURVA_CURVA_H
#define CURVA_CURVA_H

#include "calculation.h"

namespace curva {
	void parseDcd (
			const std::string &fname,
			Coordinates *coordinates,
			int *numAtoms,
			int *numFrames,
			int firstFrame = 0,
			int lastFrame = -1
	);

	void parsePdb (
			const std::string &fname,
			Atoms *atoms
	);

	void generateNodes (
			Nodes *nodes,
			Atoms *atoms,
			Coordinates *coordinates,
			unsigned int numWindows
	);

	void mutualInformationConstruct (
			Node *node,
			CurvaMatrix<double> *averagePositionMatrix,
			unsigned int windowIndex,
			unsigned int windowSize
	);

	namespace test {
		void generalizedCorrelationTest (
				CurvaMatrix<double> *mutualInformationMatrix,
				const std::string &xfname,
				const std::string &yfname,
				unsigned int referenceIndex
		);
	}

}

#endif //CURVA_CURVA_H
