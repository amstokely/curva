
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

#ifndef CURVA_UTILS_H
#define CURVA_UTILS_H
#define HASH_A 54059 /* a prime */
#define HASH_B 76963 /* another prime */
#define HASH_FIRSTH 37 /* also prime */


#include <string>

namespace utils {
	void setLastFrame (
			int *lastFrame,
			int numFrames
	);

	void generateIndicesArray (
			unsigned int **indicesArray,
			int size
	);

	bool isRecordAtom (std::string &pdbLine);

	bool isEndOfFrame (std::string &pdbLine);

	double strToDouble (const std::string &str);

	int strToInt (const std::string &str);

	std::string removeWhiteSpace (std::string str);

	unsigned int hashString (const std::string &str);

}
#endif //CURVA_UTILS_H
