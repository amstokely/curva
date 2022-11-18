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

#include <utility>
#include <valarray>
#include <sstream>
#include <iostream>
#include "../include/node.h"
#include "../include/utils.h"
#include "../include/serializer.h"
#include "../include/cnpy/cnpy.h"

Node::Node () = default;

Node::Node (
		unsigned int numFrames,
		unsigned int index_,
		unsigned int numWindows
) {
	this->_index = index_;
	this->_centerOfMass.resize(
			numFrames * 3, 0.0
	);
	this->_averageCenterOfMass.resize(
			numWindows * 3, 0.0
	);
	this->_totalMass = 0.0;
}

void Node::addAtom (
		Atom *atom,
		Coordinates *coordinates,
		unsigned int windowSize
) {
	atoms.push_back(atom);
	this->_tag      = atom->tag();
	this->_numAtoms = atoms.size();
	auto   xIterator = atom->xBegin(coordinates);
	auto   yIterator = atom->yBegin(coordinates);
	auto   zIterator = atom->zBegin(coordinates);
	double mass      = atom->mass();
	this->_totalMass += mass;
	unsigned int frame = 0;
	while (xIterator != atom->xEnd(coordinates)) {
		unsigned int windowIndex = frame / windowSize;
		this->_centerOfMass.at(3 * frame) += (*xIterator) * mass;
		this->_averageCenterOfMass.at(3 * windowIndex) +=
				(*xIterator) * mass;

		this->_centerOfMass.at(3 * frame + 1) += (*yIterator) * mass;
		this->_averageCenterOfMass.at(windowIndex * 3 + 1) +=
				(*yIterator) *
				mass;

		this->_centerOfMass.at(3 * frame + 2) += (*zIterator) *
		                                         mass;
		this->_averageCenterOfMass.at( 3 * windowIndex + 2) +=
				(*zIterator) * mass;
		frame++;
		xIterator++;
		yIterator++;
		zIterator++;
	}
	this->_hash = utils::hashString(
			this->_tag
	);
}

ForwardIterator<Atom *> Node::begin () {
	return ForwardIterator<Atom *>(&atoms[0]);
}

ForwardIterator<Atom *> Node::end () {
	return ForwardIterator<Atom *>(&atoms[atoms.size()]);
}

std::string Node::tag () {
	return _tag;
}

unsigned int Node::numAtoms () const {
	return _numAtoms;
}

unsigned int Node::index () const {
	return _index;
}

double Node::totalMass () const {
	return this->_totalMass;
}

std::vector<double> &Node::centerOfMass () {
	return this->_centerOfMass;
}

std::vector<double> &Node::averageCenterOfMass () {
	return this->_averageCenterOfMass;
}

std::vector<Atom> Node::TCL_list () {
	std::vector<Atom> TCL_atoms;
	for (auto         &atom: this->atoms) {
		TCL_atoms.push_back(*atom);
	}
	return TCL_atoms;
}

unsigned int Node::hash () const {
	return this->_hash;
}

/*
 *  * 	std::vector<std::vector<double>> _centerOfMass;
std::vector<std::vector<double>> _averageCenterOfMass;
unsigned int                     _numAtoms;
unsigned int                     _index;
double                           _totalMass;
std::string                      _tag;
std::vector<Atom *>              atoms;
unsigned int                     _hash = 0;
 */

Node::Node (
		nlohmann::json j,
		const std::string& serializationDirectory
) {
	this->_averageCenterOfMass = cnpy::npy_load(
			serializationDirectory + "/" +
			j.at("_averageCenterOfMass").get<std::string>()
			).as_vec<double>();
	this->_centerOfMass = cnpy::npy_load(
			serializationDirectory + "/" +
			j.at("_centerOfMass").get<std::string>()
	).as_vec<double>();
	this->_totalMass = j.at("_totalMass").get<double>();
	this->_numAtoms = j.at("_numAtoms").get<unsigned int>();
	this->_index = j.at("_index").get<unsigned int>();
	this->_tag = j.at("_tag").get<std::string>();
	this->_hash = j.at("_hash").get<unsigned int>();
	this->atoms = j.at("atoms").get<std::vector<Atom*>>();
}

std::string Node::serializationDirectory () {
	return this->_serializationDirectory;
}

void Node::serializationDirectory (
		std::string serializationDirectory
		) {
	this->_serializationDirectory = std::move(serializationDirectory);

}

void Node::calculationName (std::string calculationName) {
	this->_calculationName = std::move(calculationName);
}

std::string Node::calculationName () {
	return this->_calculationName;
}
