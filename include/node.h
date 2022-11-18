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

#ifndef CURVA_NODE_H
#define CURVA_NODE_H

#include "atoms.h"
#include "json/include/nlohmann/json.hpp"

class Node {
public:
	Node ();

	Node (
			nlohmann::json j,
			const std::string &serializationDirectory
	);

	Node (
			unsigned int numFrames,
			unsigned int index_,
			unsigned int numWindows
	);

	void addAtom (
			Atom *atom,
			Coordinates *coordinates,
			unsigned int windowSize
	);

	std::vector<double> &centerOfMass ();

	std::vector<double> &averageCenterOfMass ();

	std::string tag ();

	unsigned int numAtoms () const;

	unsigned int index () const;

	double totalMass () const;

	ForwardIterator<Atom *> begin ();

	ForwardIterator<Atom *> end ();

	std::vector<Atom> TCL_list ();

	std::string serializationDirectory ();

	void serializationDirectory (std::string serializationDirectory);

	std::string calculationName ();

	void calculationName (std::string calculationName);

	unsigned int hash () const;

private:
	friend nlohmann::adl_serializer<Node *>;
	std::vector<double> _centerOfMass;
	std::vector<double> _averageCenterOfMass;
	unsigned int        _numAtoms;
	unsigned int        _index;
	double              _totalMass;
	std::string         _tag;
	std::vector<Atom *> atoms;
	unsigned int        _hash = 0;
	std::string         _serializationDirectory;
	std::string         _calculationName;
};

#endif //CURVA_NODE_H
