//
// Created by andy on 7/25/22.
//
#include "../include/atoms.h"

Atoms::Atoms () = default;

void Atoms::addAtom(Atom atom) {
	atoms.push_back(std::move(atom));
}

int Atoms::numAtoms () const {
	return static_cast<int>(atoms.size());
}

Atom &Atoms::at(int atomIndex) {
	return atoms.at(atomIndex);
}

