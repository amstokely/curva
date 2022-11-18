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

#ifndef CURVA_CURVA_VECTOR_H
#define CURVA_CURVA_VECTOR_H

#include "../custom_iterators/random_access_iterator.h"
#include "vector"

template<class Type>
class CurvaMatrix;


template<class Type>
class CurvaVector {
public:

	~CurvaVector ();

	void init (
			CurvaMatrix<Type> *curvaMatrix,
			int row
	);

	void init (
			std::vector<Type> *data_
	);

	RandomAccessIterator<Type> begin ();

	RandomAccessIterator<Type> end ();

	unsigned int size () const;

	void sortAscending ();

	void sortDescending ();

	std::vector<int> &argsortAscending ();

	std::vector<int> &argsortDescending ();

	void sortAscendingAbs ();

	void sortDescendingAbs ();

	size_t bytes() const;

	std::vector<int> &argsortAscendingAbs ();

	std::vector<int> &argsortDescendingAbs ();

	void load(const std::string& fname);

	void save(const std::string& fname);

	Type at(int index);

	Type operator[](int index);

private:
	RandomAccessIterator<Type> _begin;
	RandomAccessIterator<Type> _end;
	unsigned int               _size;
	std::vector<int>           indices;
	std::vector<Type> data;
	size_t _bytes;

};

#endif //CURVA_CURVA_VECTOR_H
