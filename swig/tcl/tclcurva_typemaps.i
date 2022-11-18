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

%include std_string.i
%include std_vector.i

%extend CurvaMatrix {
	std::vector<CurvaVector<Type>> list() {
		std::vector<CurvaVector<Type>> curvaVectorVector($self->m());
		for (int i = 0; i < $self->m(); i++) {
			CurvaVector<Type> curvaVector;
			curvaVector.init($self, i);
			curvaVectorVector.at(i) = curvaVector;
		}
		return curvaVectorVector;
	}
};

%extend CurvaVector {
		std::vector<Type> list() {
			std::vector<Type> curvaVectorValues($self->size());
			unsigned int i = 0;
			for (auto &val : *$self) {
				curvaVectorValues.at(i) = val;
				i++;
			}
			return curvaVectorValues;
		}
};

%typemap(out) std::vector<int>& {
for (auto &i : *$1) {
	Tcl_Obj *obj = Tcl_NewIntObj(i);
	Tcl_ListObjAppendElement(interp, $result, obj);
}
}

%typemap(out) std::vector<double> {
for (auto &i : $1) {
	Tcl_Obj *obj = Tcl_NewDoubleObj(i);
	Tcl_ListObjAppendElement(interp, $result, obj);
}
}

%typemap(out) std::vector<int> {
for (auto &i : $1) {
Tcl_Obj *obj = Tcl_NewIntObj(i);
Tcl_ListObjAppendElement(interp, $result, obj);
}
}

%typemap(argout) std::vector<Node> {
for (auto &node : $1) {
Tcl_Obj *obj = SWIG_NewPointerObj(
		SWIG_as_voidptr(&node),
		SWIGTYPE_p_Node, 0
);
Tcl_ListAppendElement($interp, $result, obj);
}
}

%typemap(argout) std::vector<Atom> {
for (auto &atom : $1) {
Tcl_Obj *obj = SWIG_NewPointerObj(
		SWIG_as_voidptr(&atom),
		SWIGTYPE_p_Atom, 0
);
Tcl_ListAppendElement($interp, $result, obj);
}
}

%typemap(argout) std::vector<CurvaVector<int>> {
for (auto curvaVector : $1) {
Tcl_Obj *obj = SWIG_NewPointerObj(
		SWIG_as_voidptr(&curvaVector),
		SWIGTYPE_p_CurvaVector, 0
);
Tcl_ListAppendElement($interp, $result, obj);
}
}

%typemap(argout) std::vector<CurvaVector<double>> {
for (auto curvaVector : $1) {
Tcl_Obj *obj = SWIG_NewPointerObj(
		SWIG_as_voidptr(&curvaVector),
		SWIGTYPE_p_CurvaVector, 0
);
Tcl_ListAppendElement($interp, $result, obj);
}
}

%typemap(in, numinputs=0) (int *numAtoms, int *numFrames) (int
		tmpNumAtoms, int tmpNumFrames){
$1 = &tmpNumAtoms;
$2 = &tmpNumFrames;
}
