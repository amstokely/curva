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

#include "../../include/matrix/curva_matrix.h"
#include "../../include/matrix/curva_vector.h"
#include "../../include/cnpy/cnpy.h"
#include "../../include/cuda/cudautils.h"

template<class Type>
__global__ void transposeKernel (
		const Type *a,
		Type *aT,
		unsigned int m,
		unsigned int n

) {
	unsigned int i = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int j = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < m && j < n) {
		unsigned int idx  = i * n + j;
		unsigned int idxT = j * m + i;
		aT[idxT] = a[idx];
	}
}

template<class Type>
CurvaMatrix<Type>::CurvaMatrix () = default;

template<class Type>
CurvaMatrix<Type>::~CurvaMatrix () {
	this->deallocate();
}


template<class Type>
void CurvaMatrix<Type>::init (
		unsigned int m_,
		unsigned int n_
) {
	this->_data.clear();
	this->_m = m_;
	this->_n = n_;
	_data.resize(
			this->_m * this->_n,
			0.0
	);
	this->_bytes = sizeof(Type) * this->_m * this->_n;
	this->rowIndices.resize(this->_m, 0);
	std::iota(
			this->rowIndices.begin(),
			this->rowIndices.end(),
			0
	);
}

template<class Type>
void CurvaMatrix<Type>::copyDevice (
		CurvaMatrix *curvaMatrix
) {
	this->_m = curvaMatrix->m();
	this->_n = curvaMatrix->n();
	_data.clear();
	_data.resize(
			this->_m * this->_n,
			0
	);
	if (this->_bytes != curvaMatrix->bytes()) {
		this->_bytes = curvaMatrix->bytes();
		this->allocate();
	}
	if (this->_deviceData == nullptr) {
		this->allocate();
	}
	cudaMemcpy(
			this->device(),
			curvaMatrix->device(),
			this->bytes(),
			cudaMemcpyDeviceToDevice
	);
	this->rowIndices.resize(this->_m, 0.0);
	std::iota(
			this->rowIndices.begin(),
			this->rowIndices.end(),
			0
	);
}

template<class Type>
void CurvaMatrix<Type>::copy (
		CurvaMatrix *curvaMatrix
) {
	this->_m = curvaMatrix->m();
	this->_n = curvaMatrix->n();
	_data.resize(
			this->_m * this->_n,
			0.0
	);
	this->_bytes = curvaMatrix->bytes();
	std::copy(
			curvaMatrix->begin(),
			curvaMatrix->end(),
			&this->_data[0]
	);
	this->rowIndices.resize(this->_m, 0.0);
	std::iota(
			this->rowIndices.begin(),
			this->rowIndices.end(),
			0
	);
}

template<class Type>
size_t CurvaMatrix<Type>::bytes () const {
	return this->_bytes;
}

template<class Type>
Type *CurvaMatrix<Type>::host () {
	return _data.data();
}

template<class Type>
void CurvaMatrix<Type>::allocate () {
	this->deallocate();
	cudaMalloc(
			(void **) &_deviceData,
			this->_bytes
	);
}

template<class Type>
void CurvaMatrix<Type>::toDevice () {
	if (this->_deviceData != nullptr) {
		cudaMemcpy(
				this->_deviceData,
				this->_data.data(),
				this->bytes(),
				cudaMemcpyHostToDevice
		);
	}
}

template<class Type>
void CurvaMatrix<Type>::toHost () {
	if (this->_deviceData != nullptr) {
		cudaMemcpy(
				this->_data.data(),
				this->_deviceData,
				this->bytes(),
				cudaMemcpyDeviceToHost
		);
	}
}

template<class Type>
void CurvaMatrix<Type>::deallocate () {
	if (_deviceData != nullptr) {
		cudaFree(_deviceData);
		_deviceData = nullptr;
	}
}

template<class Type>
Type *&CurvaMatrix<Type>::device () {
	return _deviceData;
}

template<class Type>
unsigned int CurvaMatrix<Type>::m () const {
	return this->_m;
}

template<class Type>
unsigned int CurvaMatrix<Type>::n () const {
	return this->_n;
}

template<class Type>
void CurvaMatrix<Type>::save (const std::string &fname) {
	std::vector<size_t> shape = {
			(size_t) this->_m, this->_n
	};
	cnpy::npy_save(
			fname,
			&(*this->begin()),
			shape
	);
}

template<class Type>
void CurvaMatrix<Type>::load (const std::string &fname) {
	cnpy::NpyArray    npyArray       = cnpy::npy_load(fname);
	std::vector<Type> npyArrayVector =
			                  cnpy::npy_load(fname).as_vec<Type>();
	if (npyArray.shape.size() == 2) {
		this->init(
				npyArray.shape.at(0),
				npyArray.shape.at(1)
		);
	} else {
		this->init(
				1,
				npyArray.shape.at(0)
		);
	}
	std::copy(
			npyArrayVector.begin(),
			npyArrayVector.end(),
			&this->_data[0]
	);
}

template<class Type>
Type &CurvaMatrix<Type>::operator() (
		unsigned int i,
		unsigned int j
) {
	return _data.at(
			(i * this->_n) + j
	);
}

template<class Type>
void CurvaMatrix<Type>::init (
		Type *data_,
		int m_,
		int n_
) {
	this->_m     = m_;
	this->_n     = n_;
	this->_bytes = sizeof(Type) * this->_m * this->_n;
	this->_data.resize(this->_m * this->_n, 0.0);
	std::copy(
			data_,
			data_ + (this->_m * this->_n),
			&this->_data[0]
	);
	this->rowIndices.resize(this->_m, 0);
	std::iota(
			this->rowIndices.begin(),
			this->rowIndices.end(),
			0
	);
}

template<class Type>
void CurvaMatrix<Type>::init (
		CurvaVector<Type> *curvaVector
) {

	this->_m     = 1;
	this->_n     = curvaVector->size();
	this->_bytes = sizeof(Type) * this->_m * this->_n;
	this->_data.resize(this->_m * this->_n, 0.0);
	std::copy(
			curvaVector->begin(),
			curvaVector->end(),
			&this->_data[0]
	);
	this->rowIndices.resize(this->_m, 0);
	std::iota(
			this->rowIndices.begin(),
			this->rowIndices.end(),
			0
	);
}


template<class Type>
RandomAccessIterator<Type> CurvaMatrix<Type>::begin () {
	return RandomAccessIterator<Type>(
			&(*this->_data.begin())
	);
}

template<class Type>
RandomAccessIterator<Type> CurvaMatrix<Type>::end () {
	return RandomAccessIterator<Type>(
			&(*this->_data.end())
	);
}

template<class Type>
std::vector<int>::iterator CurvaMatrix<Type>::rowIndicesBegin () {
	return this->rowIndices.begin();
}

template<class Type>
std::vector<int>::iterator CurvaMatrix<Type>::rowIndicesEnd () {
	return this->rowIndices.end();
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "ArgumentSelectionDefects"

template<class Type>
void CurvaMatrix<Type>::transposeHost () {
	dim3         transposeKernelBlockSize = get2DBlockSize(1024);
	dim3         transposeKernelGridSize  = get2DGridSize(
			transposeKernelBlockSize, this->_m,
			this->_n
	);
	unsigned int m_                       = this->_m;
	unsigned int n_                       = this->_n;
	auto         *curvaMatrixTranspose    = new CurvaMatrix<Type>;
	curvaMatrixTranspose->init(n_, m_);
	curvaMatrixTranspose->allocate();
	curvaMatrixTranspose->toDevice();
	this->allocate();
	this->toDevice();
	void *transposeKernelArgs[] = {
			(void *) &this->device(),
			(void *) &curvaMatrixTranspose->device(),
			(void *) &m_,
			(void *) &n_
	};
	launchKernel(
			(void *) &transposeKernel<Type>,
			transposeKernelArgs,
			transposeKernelGridSize,
			transposeKernelBlockSize
	);
	curvaMatrixTranspose->toHost();
	this->copy(curvaMatrixTranspose);
	delete curvaMatrixTranspose;
}

template<class Type>
void CurvaMatrix<Type>::transposeDevice () {
	dim3         transposeKernelBlockSize = get2DBlockSize(1024);
	dim3         transposeKernelGridSize  = get2DGridSize(
			transposeKernelBlockSize, this->_m,
			this->_n
	);
	unsigned int m_                       = this->_m;
	unsigned int n_                       = this->_n;
	auto         *curvaMatrixTranspose    = new CurvaMatrix<Type>;
	curvaMatrixTranspose->init(n_, m_);
	curvaMatrixTranspose->allocate();
	curvaMatrixTranspose->toDevice();
	if (this->_deviceData == nullptr) {
		this->allocate();
		this->toDevice();
	}
	void *transposeKernelArgs[] = {
			(void *) &this->device(),
			(void *) &curvaMatrixTranspose->device(),
			(void *) &m_,
			(void *) &n_
	};
	launchKernel(
			(void *) &transposeKernel<Type>,
			transposeKernelArgs,
			transposeKernelGridSize,
			transposeKernelBlockSize
	);
	this->copyDevice(
			curvaMatrixTranspose
	);
	delete curvaMatrixTranspose;
}

template<class Type>
void CurvaMatrix<Type>::transpose () {
	this->transposeHost();
	this->toDevice();
}

#pragma clang diagnostic pop

template class CurvaMatrix<double>;

template class CurvaMatrix<int>;








