//
// Created by andy on 7/26/22.
//

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
