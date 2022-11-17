//
// Created by andy on 8/1/22.
//

#ifndef CURVA_CURVA_MATRIX_H
#define CURVA_CURVA_MATRIX_H

#include <vector>
#include <string>
#include "../custom_iterators/random_access_iterator.h"

template<class Type>
class CurvaVector;

template<class Type>
class CurvaMatrix {
public:

	CurvaMatrix ();

	void init (
			Type *data_,
			int m_,
			int n_
	);

	~CurvaMatrix ();

	void transposeHost();

	void transpose();

	void transposeDevice();

	void init (
			unsigned int m_,
			unsigned int n_
	);

	void allocate ();

	void deallocate ();

	Type *&device ();

	Type *host ();

	void toDevice ();

	void toHost ();

	unsigned int m () const;

	unsigned int n () const;

	size_t bytes () const;

	RandomAccessIterator<Type> begin ();

	RandomAccessIterator<Type> end ();

	std::vector<int>::iterator rowIndicesBegin ();

	std::vector<int>::iterator rowIndicesEnd ();

	void save (const std::string &fname);

	void load (const std::string &fname);

	Type &operator() (
			unsigned int i,
			unsigned int j
	);

	void copy (CurvaMatrix *curvaMatrix);

	void copyDevice (CurvaMatrix *curvaMatrix);

	void init (CurvaVector<Type> *curvaVector);

private:
	unsigned int      _m{};
	unsigned int      _n{};
	std::vector<Type> _data;
	Type              *_deviceData = nullptr;
	size_t            _bytes;
	std::vector<int>  rowIndices;
};

#endif //CURVA_CURVA_MATRIX_H
