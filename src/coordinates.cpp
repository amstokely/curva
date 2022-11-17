//
// Created by andy on 7/21/22.
//
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
