//
// Created by andy on 7/25/22.
//

#ifndef CURVA_FORWARD_ITERATOR_H
#define CURVA_FORWARD_ITERATOR_H
#include <iterator>

template<class Type>
class ForwardIterator {
public:
	typedef Type                         value_type;
	typedef typename std::iterator<
			std::forward_iterator_tag, Type
	>::difference_type difference_type;
	typedef Type                         *pointer;
	typedef Type                         &reference;
	typedef std::forward_iterator_tag iterator_category;

	explicit ForwardIterator (
			Type *ptr_
	) : ptr(ptr_) {}

	inline Type &operator* () const {
		return *ptr;
	}

	inline Type *operator-> () {
		return ptr;
	}

	inline ForwardIterator &operator+= (
			difference_type rhs
	) {
		ptr += rhs;
		return *this;
	}

	inline ForwardIterator &operator++ () {
		ptr++;
		return *this;
	}

	// Postfix increment
	inline ForwardIterator operator++ (int) {
		ForwardIterator tmp = *this;
		++(*this);
		return tmp;
	}

	inline friend bool operator== (
			const ForwardIterator &a,
			const ForwardIterator &b
	) {
		return a.ptr == b.ptr;
	};

	inline friend bool operator!= (
			const ForwardIterator &a,
			const ForwardIterator &b
	) {
		return a.ptr != b.ptr;
	};

	inline ForwardIterator incr () {
		ptr++;
		return ForwardIterator(ptr);
	}

	inline bool eq (
			ForwardIterator it
	) {
		if (it.ptr == this->ptr) {
			return true;
		} else {
			return false;
		}
	}

	inline Type deref() {
		return *ptr;
	}

	inline Type at(unsigned int index) {
		this->ptr = this->ptr + index;
		return *ptr;
	}

private:
	Type *ptr;
};

#endif //CURVA_FORWARD_ITERATOR_H
