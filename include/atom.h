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

#ifndef CUDNA_PDB_H
#define CUDNA_PDB_H

#include <string>
#include "coordinates.h"
#include "json/include/nlohmann/json.hpp"

class Atom {
public:

	Atom ();

	explicit Atom (std::string &pdbLine);

	int index () const;

	std::string name ();

	std::string element ();

	std::string residueName ();

	int residueId () const;

	std::string chainId ();

	std::string segmentId ();

	double temperatureFactor () const;

	double occupancy () const;

	int serial () const;

	std::string tag ();

	RandomAccessIterator<double>
	xBegin (Coordinates *coordinates) const;

	RandomAccessIterator<double> xEnd (Coordinates *coordinates) const;

	RandomAccessIterator<double> yBegin (Coordinates *coordinates)
	const;

	RandomAccessIterator<double> yEnd (Coordinates *coordinates) const;

	RandomAccessIterator<double> zBegin (Coordinates *coordinates)
	const;

	RandomAccessIterator<double> zEnd (Coordinates *coordinates) const;

	double x(Coordinates *coordinates, int frameIndex) const;

	double y(Coordinates *coordinates, int frameIndex) const;

	double z(Coordinates *coordinates, int frameIndex) const;

	double mass() const {
		return _mass;
	}

	unsigned int hash() const;

private:
	friend nlohmann::adl_serializer<Atom>;
	friend nlohmann::adl_serializer<Atom*>;
	int         _index;
	std::string _name;
	std::string _element;
	std::string _residueName;
	int         _residueId;
	std::string _chainId;
	std::string _segmentId;
	double      _temperatureFactor;
	double      _occupancy;
	int         _serial;
	std::string _tag;
	double              _mass;
	unsigned int        _hash;
};

#endif //CUDNA_PDB_H
