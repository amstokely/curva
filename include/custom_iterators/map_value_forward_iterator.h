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

#ifndef CURVA_MAP_VALUE_FORWARD_ITERATOR_H
#define CURVA_MAP_VALUE_FORWARD_ITERATOR_H

#include <iterator>
#include <map>
#include <string>
#include <vector>
#include <iostream>


template<typename KeyType, typename ValueType>
class MapValueForwardIterator {
public:
	typedef ValueType                 value_type;
	typedef typename std::iterator<
			std::forward_iterator_tag, ValueType
	>::difference_type                difference_type;
	typedef ValueType                 *pointer;
	typedef ValueType                 &reference;
	typedef std::forward_iterator_tag iterator_category;

	explicit MapValueForwardIterator (
			typename std::map<KeyType, ValueType>::iterator it_
	)
			: it(it_) {}

	MapValueForwardIterator (
			const MapValueForwardIterator<KeyType, ValueType> &it_
	)
			: it(it_.it) {}

	inline bool operator== (
			const MapValueForwardIterator<KeyType, ValueType> &it_
	) const {
		return this->it == it_.it;
	}

	inline bool operator!= (
			const MapValueForwardIterator<KeyType, ValueType> &it_
	) const {
		return this->it != it_.it;
	}

	inline ValueType &operator* () const {
		return (it->second);
	}

	inline ValueType *operator-> () const {
		return &(it->second);
	}

	inline MapValueForwardIterator<KeyType, ValueType> &operator++ () {
		++it;
		return *this;
	}

	inline MapValueForwardIterator<KeyType, ValueType> operator++
			(int) {
		MapValueForwardIterator<KeyType, ValueType>
				result(*this);  // get a copy for return
		++(*this);
		return result;
	}

	inline MapValueForwardIterator incr () {
		it++;
		return MapValueForwardIterator(it);
	}

	inline bool eq (
			MapValueForwardIterator it_
	) {
		if (it_.it == this->it) {
			return true;
		} else {
			return false;
		}
	}

	inline ValueType at (unsigned int index) {
		while (index != 0) {
			this->it++;
			index -= 1;
		}
		return (it->second);
	}

	inline ValueType deref () {
		return (it->second);
	}

private:
	typename std::map<KeyType, ValueType>::iterator it;
};


#endif //CURVA_MAP_VALUE_FORWARD_ITERATOR_H
