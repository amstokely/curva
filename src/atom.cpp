//
// Created by andy on 7/18/22.
//

#include <numeric>
#include <sstream>
#include <algorithm>
#include "../include/atom.h"
#include "../include/element_properties.h"
#include "../include/utils.h"


/*
 *COLUMNS        DATA TYPE       CONTENTS
--------------------------------------------------------------------------------
 1 -  6        Record name     "ATOM  "
 7 - 11        Integer         Atom serial number.
13 - 16        Atom            Atom name.
17             Character       Alternate location indicator.
18 - 20        Residue name    Residue name.
22             Character       Chain identifier.
23 - 26        Integer         Residue sequence number.
27             AChar           Code for insertion of residues.
31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.
55 - 60        Real(6.2)       Occupancy.
61 - 66        Real(6.2)       Temperature factor (Default = 0.0).
73 - 76        LString(4)      Segment identifier, left-justified.
77 - 78        LString(2)      Element symbol, right-justified.
 */


Atom::Atom () = default;

Atom::Atom (std::string &pdbLine) {
	ElementProperties elementProperties;
	_serial            = utils::strToInt(
			utils::removeWhiteSpace(
					pdbLine.substr(6, 5)
			)
	);
	_index             = _serial - 1;
	_name              = utils::removeWhiteSpace(
			pdbLine.substr(12, 3)
	);
	_residueName       = utils::removeWhiteSpace(
			pdbLine.substr(17, 3)
	);
	_chainId           = utils::removeWhiteSpace(pdbLine.substr(21, 1));
	_residueId         = utils::strToInt(
			utils::removeWhiteSpace(
					pdbLine.substr(22, 4)
			)
	);
	_occupancy         = utils::strToDouble(
			utils::removeWhiteSpace(
					pdbLine.substr(54, 6)
			)
	);
	_temperatureFactor = utils::strToDouble(
			utils::removeWhiteSpace(
					pdbLine.substr(54, 6)
			)
	);
	_segmentId         = utils::removeWhiteSpace(
			pdbLine.substr(72, 4)
	);
	utils::removeWhiteSpace(_segmentId);
	_element = utils::removeWhiteSpace(
			pdbLine.substr(76, 2)
	);
	std::transform(
			std::begin(this->_element),
			std::end(this->_element),
			std::begin(this->_element), []
					(char const &c) {
				return std::tolower(c);
			}
	);

	this->_element[0] = std::toupper(this->_element[0]);
	this->_tag  = (
			this->_residueName + "_" +
			utils::removeWhiteSpace(pdbLine.substr(22, 4)) + "_" +
			this->_chainId + "_" +
			this->_segmentId
	);
	this->_mass = 1000.0
	              * elementProperties.atomicWeight(this->_element);
	std::stringstream hashStringStream;
	hashStringStream << this->_index << "_" << this->_tag;
	this->_hash = utils::hashString(
			hashStringStream.str()
	);
}


int Atom::index () const {
	return _index;
}

std::string Atom::name () {
	return _name;
}

std::string Atom::element () {
	return _element;
}

std::string Atom::residueName () {
	return _residueName;
}

int Atom::residueId () const {
	return _residueId;
}

std::string Atom::chainId () {
	return _chainId;
}

std::string Atom::segmentId () {
	return _segmentId;
}

double Atom::temperatureFactor () const {
	return _temperatureFactor;
}

double Atom::occupancy () const {
	return _occupancy;
}

int Atom::serial () const {
	return _serial;
}


std::string Atom::tag () {
	return _tag;
}

RandomAccessIterator<double> Atom::xBegin (
		Coordinates *coordinates
) const {
	return (
			coordinates->begin() +
			(coordinates->numFrames() * this->_index)
	);
}

RandomAccessIterator<double> Atom::xEnd (
		Coordinates *coordinates
) const {
	return (
			coordinates->begin() +
			(coordinates->numFrames() * this->_serial)

	);
}

RandomAccessIterator<double> Atom::yBegin (
		Coordinates *coordinates
) const {
	return (
			coordinates->begin() +
			(coordinates->numFrames() * coordinates->numAtoms()) +
			+(this->_index) * coordinates->numFrames()
	);
}

RandomAccessIterator<double> Atom::yEnd (
		Coordinates *coordinates
) const {
	return (
			coordinates->begin() +
			(coordinates->numFrames() * coordinates->numAtoms()) +
			+(this->_serial) * coordinates->numFrames()
	);
}

RandomAccessIterator<double> Atom::zBegin (
		Coordinates *coordinates
) const {
	return (
			coordinates->begin() +
			2 * (coordinates->numFrames() * coordinates->numAtoms()) +
			+(this->_index) * coordinates->numFrames()
	);
}

RandomAccessIterator<double> Atom::zEnd (
		Coordinates *coordinates
) const {
	return (
			coordinates->begin() +
			2 * (coordinates->numFrames() * coordinates->numAtoms()) +
			+(this->_serial) * coordinates->numFrames()
	);
}

double Atom::x (
		Coordinates *coordinates,
		int frameIndex
) const {
	return *(xBegin(coordinates) + frameIndex);
}

double Atom::y (
		Coordinates *coordinates,
		int frameIndex
) const {
	return *(yBegin(coordinates) + frameIndex);
}

double Atom::z (
		Coordinates *coordinates,
		int frameIndex
) const {
	return *(zBegin(coordinates) + frameIndex);
}

unsigned int Atom::hash () const {
	return this->_hash;
}


