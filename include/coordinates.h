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

#ifndef CURVA_COORDINATES_H
#define CURVA_COORDINATES_H

#include "custom_iterators/random_access_iterator.h"
#include <vector>

class Coordinates {
public:
	Coordinates ();

	~Coordinates ();

	void init (
			unsigned int numAtoms_, unsigned int numFrames_);

	double x (int index, int frame);

	double y (int index, int frame);

	double z (int index, int frame);

	double &operator() (int i,
	                    int j, int k);

	unsigned int numAtoms() const;

	unsigned int numFrames() const;

	RandomAccessIterator<double> begin () {
		return RandomAccessIterator<double>(&coordinates[0]);
	}

	RandomAccessIterator<double> end () {
		return RandomAccessIterator<double>
		        (&coordinates[3*_numAtoms*_numFrames]);
	}

private:
	std::vector<double> coordinates;
	unsigned int _numAtoms;
	unsigned int _numFrames;
};

#endif //CURVA_COORDINATES_H
