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

/*!
@brief CuRva namespace
*/
namespace curva {
	/*!
	 * Extracts the atomic coordinates from a DCD file.
	 * @param fname DCD file name.
	 * @param coordinates Pointer to a Coordinates class object,
	 * which is where the parsed atomic coordinates are saved.
	 * @param numAtoms Number of atoms in a single trajectory frame.
	 * @param numFrames Total number of frames to be read from the
	 * dcd file.
	 * @param firstFrame Index of the first frame to be read from the
	 * dcd file
	 * @param lastFrame Index of the last frame to be read from the
	 * dcd file
	 */
	void parseDcd (
			const std::string &fname,
			Coordinates *coordinates,
			int *numAtoms,
			int *numFrames,
			int firstFrame = 0,
			int lastFrame = -1
	);

	/*!
	 * Extracts all atomic information from the first frame of a PDB
	 * file.
	 * @param fname PDB file name.
	 * @param atoms Pointer to an Atoms class object, which is where
	 * the parsed atomic information is saved.
	 */
	void parsePdb (
			const std::string &fname,
			Atoms *atoms
	);

	/*!
	 * @param nodes
	 * @param atoms
	 * @param coordinates
	 * @param numWindows
	 */
	void generateNodes (
			Nodes *nodes,
			Atoms *atoms,
			Coordinates *coordinates,
			unsigned int numWindows
	);

	/*!
	 *
	 * @param node
	 * @param averagePositionMatrix
	 * @param windowIndex
	 * @param windowSize
	 */
	void mutualInformationConstruct (
			Node *node,
			CurvaMatrix<double> *averagePositionMatrix,
			unsigned int windowIndex,
			unsigned int windowSize
	);

#ifndef DOXYGEN_IGNORE
	namespace test {
		void generalizedCorrelationTest (
				CurvaMatrix<double> *mutualInformationMatrix,
				const std::string &xfname,
				const std::string &yfname,
				unsigned int referenceIndex
		);
	}
#endif //DOXYGEN_IGNORE
}


#endif //CURVA_CURVA_H
