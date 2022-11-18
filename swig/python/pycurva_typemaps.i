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

%extend CurvaMatrix{
		void fromNumpy (
			Type *NUMPY_ARRAY,
			int NUMPY_ARRAY_DIM1,
			int NUMPY_ARRAY_DIM2
		) {
			$self->init(
				NUMPY_ARRAY,
				NUMPY_ARRAY_DIM1,
				NUMPY_ARRAY_DIM2);
		}

};


%extend CurvaMatrix {
	void toNumpy (
		Type **NUMPY_ARRAY,
		int **NUMPY_ARRAY_DIM1,
		int **NUMPY_ARRAY_DIM2
		) {
			*NUMPY_ARRAY_DIM1 = (int *) malloc(sizeof(int));
			*NUMPY_ARRAY_DIM2 = (int *) malloc(sizeof(int));
			(*(NUMPY_ARRAY_DIM1))[0] = static_cast<int>($self->m());
			(*(NUMPY_ARRAY_DIM2))[0] = static_cast<int>($self->n());
			int size = static_cast<int>(
					$self->m() * $self->n()
			);
			*NUMPY_ARRAY = (Type *) malloc($self->bytes());
			for (int i = 0;
			     i < size;
			     i++) {
				(*(NUMPY_ARRAY))[i] = $self->host()[i];
			}
		}
};

%extend CurvaVector {
		void toNumpy (
		Type **NUMPY_ARRAY,
		int **NUMPY_ARRAY_DIM1
		) {
			*NUMPY_ARRAY_DIM1 = (int *) malloc(sizeof(int));
			(*(NUMPY_ARRAY_DIM1))[0] = static_cast<int>($self->size());
			int size = static_cast<int>(
					$self->size()
			);
			*NUMPY_ARRAY = (Type *) malloc($self->bytes());
			int i = 0;
			for (auto val : *$self) {
				(*(NUMPY_ARRAY))[i] = val;
				i++;
			}
		}
};


%extend CurvaVector {
		Type __getitem__(int index) {
			auto val = *($self->begin() + index);
			return val;
		}
};

%extend CurvaVector {
		int __len__() {
			return $self->size();
		}
};

%extend CurvaVector {
	std::string __repr__() {
		std::stringstream ss;
		for (auto val : *$self) {
			ss << val << " ";
		}
		return ss.str();
		}
};

%extend CurvaMatrix {
		int __len__() {
			return $self->m();
		}
};

%extend CurvaMatrix {
		double at(int i, int j) {
			return (*($self))(i, j);
		}
};

%extend CurvaMatrix {
		CurvaVector<Type> __getitem__(unsigned int rowIndex) {
			CurvaVector<Type> curvaVector;
			curvaVector.init($self, rowIndex);
			return curvaVector;
		}
};

%extend Nodes {
		int __len__() {
			return $self->numNodes();
		}
};

%extend Nodes {
		Node* __getitem__(int nodeIndex) {
			return $self->at(nodeIndex);
		}
};
%extend Node {
		std::string __repr__() {
			std::stringstream ss;
			ss << "tag: " << $self->tag() << ", ";
			ss << "index: " << $self->index();
			return ss.str();
		}
};

%extend Node {
		Atom* __getitem__(int nodeAtomIndex) {
			auto it = $self->begin();
			int  i  = 0;
			while (i != nodeAtomIndex) {
				it++;
			}
			return *it;
		}
};

%extend Node {
		bool __eq__(
				const Node &node
				) {
			return $self == &node;
		}
		bool __ne__(
				const Node &node
				) {
			return $self != &node;
		}
		long __hash__() {
			return $self->hash();
		}
}

%extend Atoms {
		int __len__() {
			return $self->numAtoms();
		}
};

%extend Atoms {
		Atom &__getitem__(int atomIndex) {
			return $self->at(atomIndex);
		}
};

%extend Atom {
		std::string __repr__() {
			std::stringstream ss;
			ss << "name: " << $self->name() << ", ";
			ss << "element: " << $self->element() << ", ";
			ss << "index: " << $self->index() << ", ";
			ss << "residue name: " << $self->residueName() << ", ";
			ss << "residue id: " << $self->residueId() << ", ";
			ss << "mass: " << $self->mass();
			return ss.str();
		}
};
%extend Atom {
		bool __eq__(
		const Atom &atom
		) {
			return $self == &atom;
		}
		bool __ne__(
		const Atom &atom
		) {
			return $self != &atom;
		}
		long __hash__() {
			return $self->hash();
		}
}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(
	double* NUMPY_ARRAY,
	int NUMPY_ARRAY_DIM1,
	int NUMPY_ARRAY_DIM2
	)};

%fragment("FreeCap", "header") {
void FreeCap (PyObject *cap) {
	void *array = (void *) PyCapsule_GetPointer(cap, NULL);
	if (array != NULL) {
		free(array);
	}
}
}

%typemap(in, numinputs=0) (double **NUMPY_ARRAY,
int**NUMPY_ARRAY_DIM1)
(double *NUMPY_ARRAY_tmp, int *NUMPY_ARRAY_DIM1_tmp) {
$1 = &NUMPY_ARRAY_tmp;
$2 = &NUMPY_ARRAY_DIM1_tmp;
}

%typemap(argout, fragment="FreeCap") (
double **NUMPY_ARRAY, int**NUMPY_ARRAY_DIM1) {
npy_intp dims[1] = {(*($2))[0]};
PyObject* obj = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64,
                                          (void*)(*$1));
PyArrayObject* array = (PyArrayObject*) obj;
PyObject* cap = PyCapsule_New((void*)(*$1), NULL,
                              FreeCap);
PyArray_SetBaseObject(array, cap);
$result = SWIG_Python_AppendOutput($result,obj);
free(*$2);
}

%typemap(in, numinputs=0) (double **NUMPY_ARRAY,
int**NUMPY_ARRAY_DIM1, int **NUMPY_ARRAY_DIM2)
(double *NUMPY_ARRAY_tmp, int *NUMPY_ARRAY_DIM1_tmp,
int *NUMPY_ARRAY_DIM2_tmp) {
$1 = &NUMPY_ARRAY_tmp;
$2 = &NUMPY_ARRAY_DIM1_tmp;
$3 = &NUMPY_ARRAY_DIM2_tmp;
}

%typemap(argout, fragment="FreeCap") (
double **NUMPY_ARRAY, int**NUMPY_ARRAY_DIM1,
int**NUMPY_ARRAY_DIM2) {
npy_intp dims[2] = {(*($2))[0], (*($3))[0]};
PyObject* obj = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64,
                                          (void*)(*$1));
PyArrayObject* array = (PyArrayObject*) obj;
PyObject* cap = PyCapsule_New((void*)(*$1), NULL,
                              FreeCap);
PyArray_SetBaseObject(array, cap);
$result = SWIG_Python_AppendOutput($result,obj);
free(*$2);
free(*$3);
}

%typemap(out) double& {
$result = PyFloat_FromDouble(*$1);
}

%typemap(out) unsigned int& {
$result = PyInt_FromLong(*$1);
}


%typemap(in, numinputs=0) (int *numAtoms, int *numFrames) (int
		tmpNumAtoms, int tmpNumFrames){
$1 = &tmpNumAtoms;
$2 = &tmpNumFrames;
}