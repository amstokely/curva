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

#ifndef CURVA_NODES_H
#define CURVA_NODES_H

#include "custom_iterators/map_value_forward_iterator.h"
#include "node.h"
#include "json/include/nlohmann/json.hpp"


class Nodes {
public:
	Nodes ();

	~Nodes ();

	void init (
			Atoms *atoms,
			Coordinates *coordinates,
			unsigned int numFrames,
			unsigned int numWindows
	);

	MapValueForwardIterator<std::string, Node *> begin () {
		return MapValueForwardIterator<std::string, Node *>(
				nodes.begin()
		);
	}

	MapValueForwardIterator<std::string, Node *> end () {
		return MapValueForwardIterator<std::string, Node *>(
				nodes.end()
		);
	}

	unsigned int numNodes () const;

	Node *nodeFromAtomIndex (int atomIndex);

	Node *at (int nodeIndex);

	std::vector<Node> TCL_list ();

	nlohmann::json save (
			const std::string &serializationDirectory,
			const std::string& calculationName
	);

	void load (
			const nlohmann::json &j,
			const std::string &serializationDirectory
	);

private:
	std::map<std::string, Node *> nodes;
	std::map<int, Node *>         atomIndexNodeMap;
	std::map<int, Node *>         nodeIndexNodeMap;
	unsigned int                  _numNodes;

};

#endif //CURVA_NODES_H
