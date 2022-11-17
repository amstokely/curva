//
// Created by andy on 7/26/22.
//

#include "../include/nodes.h"
#include "../include/serializer.h"

Nodes::Nodes () = default;

Nodes::~Nodes () {
	for (auto &tagNode: this->nodes) {
		delete tagNode.second;
	}
}

void Nodes::init (
		Atoms *atoms,
		Coordinates *coordinates,
		unsigned int numFrames,
		unsigned int numWindows
) {
	auto atom = atoms->begin();
	while (atom != atoms->end()) {
		auto emplace_pair = nodes.emplace(
				atom->tag(),
				new Node(
						numFrames, nodes.size(),
						numWindows
				)
		);
		emplace_pair.first->second->addAtom(
				&(*atom), coordinates, numFrames / numWindows
		);
		this->atomIndexNodeMap[atom->index()] =
				emplace_pair.first->second;
		this->nodeIndexNodeMap[
				static_cast<int>(
						emplace_pair.first->second->index()
				)
		] = emplace_pair.first->second;
		atom++;
	}
	this->_numNodes = nodes.size();
}

unsigned int Nodes::numNodes () const {
	return this->_numNodes;
}

Node *Nodes::nodeFromAtomIndex (int atomIndex) {
	return this->atomIndexNodeMap[atomIndex];
}

Node *Nodes::at (int nodeIndex) {
	return this->nodeIndexNodeMap[nodeIndex];
}

std::vector<Node> Nodes::TCL_list () {
	std::vector<Node> TCL_nodes;
	TCL_nodes.reserve(this->_numNodes);
	for (int i = 0;
	     i < this->_numNodes;
	     i++) {
		TCL_nodes.push_back(*at(i));
	}
	return TCL_nodes;
}

nlohmann::json Nodes::save (
		const std::string &serializationDirectory,
		const std::string& calculationName
) {
	for (auto      node: nodes) {
		node.second->serializationDirectory(serializationDirectory);
		node.second->calculationName(calculationName);
	}
	nlohmann::json j = this->nodes;
	return j;
}

void Nodes::load (
		const nlohmann::json &j,
		const std::string &serializationDirectory
) {
	for (const auto &jsonNode: j) {
		auto *node = new Node(jsonNode, serializationDirectory);
		this->nodes[node->tag()] = node;
		this->nodeIndexNodeMap[(int)node->index()] = node;
		for (auto atom : *node) {
			this->atomIndexNodeMap[atom->index()] = node;
		}
	}
	this->_numNodes = this->nodes.size();
}



