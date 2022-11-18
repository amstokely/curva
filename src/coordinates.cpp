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

#include "../include/coordinates.h"

Coordinates::Coordinates () {
	this->_numAtoms  = 0;
	this->_numFrames = 0;
}

Coordinates::~Coordinates() = default;

void Coordinates::init (
		unsigned int numAtoms_,
		unsigned int numFrames_
) {
	coordinates.clear();
	this->_numAtoms  = numAtoms_;
	this->_numFrames = numFrames_;
	coordinates.resize(3 * _numAtoms * _numFrames, 0.0);
}

double Coordinates::x (
		int index,
		int frame
) {
	return coordinates[index + index*_numFrames + frame];
}

double Coordinates::y (
		int index,
		int frame
) {
	return coordinates[index + (_numAtoms * _numFrames) +
	index*_numFrames + frame];
}

double Coordinates::z (
		int index,
		int frame
) {
	return coordinates[index + 2 * (_numAtoms * _numFrames) +
	index*_numFrames + frame];
}

double &Coordinates::operator() (
		int i,
		int j,
		int k
) {
	return coordinates[i * (_numAtoms * _numFrames) + j*_numFrames + k];
}

unsigned int Coordinates::numAtoms () const {
	return _numAtoms;
}

unsigned int Coordinates::numFrames () const {
	return _numFrames;
}
