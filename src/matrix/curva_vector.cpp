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

#include "../../include/matrix/curva_vector.h"
#include "../../include/matrix/curva_matrix.h"
#include "../../include/cnpy/cnpy.h"
#include <algorithm>
#include <numeric>

template<class Type>
CurvaVector<Type>::~CurvaVector () = default;

template<class Type>
void CurvaVector<Type>::init (
		CurvaMatrix<Type> *curvaMatrix,
		int row
) {
	this->_begin = curvaMatrix->begin() + row * curvaMatrix->n();
	this->_end   = curvaMatrix->begin() + (row + 1) * curvaMatrix->n();
	this->_size  = curvaMatrix->n();
	this->_bytes = curvaMatrix->n() * sizeof(Type);
	this->indices.resize(this->_size, 0.0);
	std::iota(this->indices.begin(), this->indices.end(), 0);
}

template<class Type>
void CurvaVector<Type>::init (std::vector<Type> *data_) {
	this->_begin = RandomAccessIterator<Type>(&(*data_->begin()));
	this->_end   = RandomAccessIterator<Type>(&(*data_->end()));
	this->_size  = data_->size();
	this->_bytes  = data_->size() * sizeof(Type);
	this->indices.resize(this->_size, 0.0);
	std::iota(this->indices.begin(), this->indices.end(), 0);

}

template<class Type>
size_t CurvaVector<Type>::bytes () const {
	return this->_bytes;
}

template<class Type>
unsigned int CurvaVector<Type>::size () const {
	return this->_size;
}

template<class Type>
RandomAccessIterator<Type> CurvaVector<Type>::begin () {
	return RandomAccessIterator<Type>(&(*this->_begin));
}

template<class Type>
RandomAccessIterator<Type> CurvaVector<Type>::end () {
	return RandomAccessIterator<Type>(&(*this->_end));
}

template<class Type>
void CurvaVector<Type>::sortAscending () {
	std::sort(
			this->_begin, this->_end,
			[] (
					Type a,
					Type b
			) -> bool {
				return (
						a < b
				);
			}
	);
}

template<class Type>
void CurvaVector<Type>::sortDescending () {
	std::sort(
			this->_begin, this->_end,
			[] (
					Type a,
					Type b
			) -> bool {
				return (
						a > b
				);
			}
	);
}

template<class Type>
std::vector<int> &CurvaVector<Type>::argsortAscending () {
	RandomAccessIterator<Type> begin_ = this->_begin;
	std::sort(
			this->indices.begin(),
			this->indices.end(),
			[begin_] (
					unsigned int idx1,
					unsigned int idx2
			) -> bool {
				return (
						*(begin_ + idx1) <
						*(begin_ + idx2)
				);
			}
	);
	return this->indices;
}

template<class Type>
std::vector<int> &CurvaVector<Type>::argsortDescending () {
	RandomAccessIterator<Type> begin_ = this->_begin;
	std::sort(
			this->indices.begin(),
			this->indices.end(),
			[begin_] (
					unsigned int idx1,
					unsigned int idx2
			) -> bool {
				return (
						*(begin_ + idx1) >
						*(begin_ + idx2)
				);
			}
	);
	return this->indices;
}


template<class Type>
void CurvaVector<Type>::sortAscendingAbs () {
	std::sort(
			this->_begin, this->_end,
			[] (
					Type a,
					Type b
			) -> bool {
				return (
						std::abs(a) < std::abs(b)
				);
			}
	);
}

template<class Type>
void CurvaVector<Type>::sortDescendingAbs () {
	std::sort(
			this->_begin, this->_end,
			[] (
					Type a,
					Type b
			) -> bool {
				return (
						std::abs(a) > std::abs(b)
				);
			}
	);
}

template<class Type>
std::vector<int> &CurvaVector<Type>::argsortAscendingAbs () {
	RandomAccessIterator<Type> begin_ = this->_begin;
	std::sort(
			this->indices.begin(),
			this->indices.end(),
			[begin_] (
					unsigned int idx1,
					unsigned int idx2
			) -> bool {
				return (
						std::abs(*(begin_ + idx1)) <
						std::abs(*(begin_ + idx2))
				);
			}
	);
	return this->indices;
}

template<class Type>
std::vector<int> &CurvaVector<Type>::argsortDescendingAbs () {
	RandomAccessIterator<Type> begin_ = this->_begin;
	std::sort(
			this->indices.begin(),
			this->indices.end(),
			[begin_] (
					unsigned int idx1,
					unsigned int idx2
			) -> bool {
				return (
						std::abs(*(begin_ + idx1)) >
						std::abs(*(begin_ + idx2))
				);
			}
	);
	return this->indices;
}

template<class Type>
void CurvaVector<Type>::load (const std::string &fname) {
	cnpy::NpyArray    npyArray       = cnpy::npy_load(fname);
	std::vector<Type> npyArrayVector =
			                  cnpy::npy_load(fname).as_vec<Type>();
	this->_size  = npyArrayVector.size();
	this->data.resize(this->_size, 0.0);
	std::copy(
			npyArrayVector.begin(),
			npyArrayVector.end(),
			&this->data[0]
	);
	this->_begin = RandomAccessIterator<Type>(
			&(*this->data.begin())
	);
	this->_end   = RandomAccessIterator<Type>(
			&*(this->data.end())
	);
	this->indices.resize(this->_size, 0.0);
	std::iota(this->indices.begin(), this->indices.end(), 0);
}

template<class Type>
void CurvaVector<Type>::save (const std::string &fname) {
	std::vector<size_t> shape = {this->size()};
	cnpy::npy_save(
			fname,
			&(*this->begin()),
			shape
	);
}

template<class Type>
Type CurvaVector<Type>::at (int index) {
	return *(this->_begin + index);
}

template<class Type>
Type CurvaVector<Type>::operator[] (int index) {
	return this->at(index);
}


template class CurvaVector<double>;

template class CurvaVector<int>;


