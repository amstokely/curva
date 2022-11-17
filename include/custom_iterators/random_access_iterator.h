//
// Created by andy on 7/22/22.
//

#ifndef CURVA_RANDOM_ACCESS_ITERATOR_H
#define CURVA_RANDOM_ACCESS_ITERATOR_H

#include <iterator>

template<class Type>
class RandomAccessIterator {
public:

	typedef Type                         value_type;
	typedef typename std::iterator<
			std::random_access_iterator_tag, Type
	>::difference_type difference_type;
	typedef Type                         *pointer;
	typedef Type                         &reference;
	typedef std::random_access_iterator_tag iterator_category;

	RandomAccessIterator () : ptr(nullptr) {}

	explicit RandomAccessIterator (Type *rhs) : ptr(rhs) {}

	RandomAccessIterator (
			const RandomAccessIterator &rhs
	) : ptr(rhs.ptr) {}

	inline RandomAccessIterator &operator+= (
			difference_type rhs
	) {
		ptr += rhs;
		return *this;
	}

	inline RandomAccessIterator &operator-= (
			difference_type rhs
	) {
		ptr -= rhs;
		return *this;
	}

	inline Type &operator* () const {
		return *ptr;
	}

	inline Type *operator-> () const {
		return ptr;
	}

	inline Type &operator[] (
			difference_type rhs
	) const {
		return ptr[rhs];
	}

	inline RandomAccessIterator &operator++ () {
		++ptr;
		return *this;
	}

	inline RandomAccessIterator &operator-- () {
		--ptr;
		return *this;
	}

	inline RandomAccessIterator operator++ (int) {
		RandomAccessIterator tmp(*this);
		++ptr;
		return tmp;
	}

	inline RandomAccessIterator operator-- (int) {
		RandomAccessIterator tmp(*this);
		--ptr;
		return tmp;
	}

	inline difference_type
	operator- (const RandomAccessIterator &rhs) const {
		return ptr - rhs.ptr;
	}

	inline RandomAccessIterator operator+ (
			difference_type rhs
	) const {
		return RandomAccessIterator(
				ptr + rhs
		);
	}

	inline RandomAccessIterator operator- (
			difference_type rhs
	) const {
		return RandomAccessIterator(
				ptr - rhs
		);
	}

	friend inline RandomAccessIterator operator+ (
			difference_type lhs,
			const RandomAccessIterator &rhs
	) {
		return RandomAccessIterator(
				lhs + rhs.ptr
		);
	}

	inline bool operator== (
			const RandomAccessIterator &rhs
	) const {
		return ptr == rhs.ptr;
	}

	inline bool operator!= (
			const RandomAccessIterator &rhs
	) const {
		return ptr != rhs.ptr;
	}

	inline bool operator> (
			const RandomAccessIterator &rhs
	) const {
		return ptr > rhs.ptr;
	}

	inline bool operator< (
			const RandomAccessIterator &rhs
	) const {
		return ptr < rhs.ptr;
	}

	inline bool operator>= (
			const RandomAccessIterator &rhs
	) const {
		return ptr >= rhs.ptr;
	}

	inline bool operator<= (
			const RandomAccessIterator &rhs
	) const {
		return ptr <= rhs.ptr;
	}

	inline RandomAccessIterator incr (unsigned int incr_) {
		unsigned int i = 0;
		while (i < incr_) {
			ptr++;
			i++;
		}
		return RandomAccessIterator(ptr);
	}

	inline bool eq (RandomAccessIterator it) {
		if (it.ptr == this->ptr) {
			return true;
		} else {
			return false;
		}
	}

	inline Type deref() {
		return *ptr;
	}

private:
	Type *ptr;
};

#endif //CURVA_RANDOM_ACCESS_ITERATOR_H
