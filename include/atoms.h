//
// Created by andy on 7/25/22.
//

#ifndef CURVA_ATOMS_H
#define CURVA_ATOMS_H

#include "atom.h"
#include "custom_iterators/forward_iterator.h"


class Atoms {
public:
	Atoms();
	void addAtom(Atom atom);
	RandomAccessIterator<Atom> begin() {
		return RandomAccessIterator<Atom>(&(*atoms.begin()));
	}
	RandomAccessIterator<Atom> end() {
		return RandomAccessIterator<Atom>(&(*atoms.end()));
	}
	int numAtoms() const;

	Atom &at (int atomIndex);

private:
	std::vector<Atom> atoms;
};

#endif //CURVA_ATOMS_H
