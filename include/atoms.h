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
