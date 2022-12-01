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

%module pycurva

%include std_string.i
%include std_vector.i
%include std_map.i
%template(dVector) std::vector<double>;
%template(sdMap) std::map<std::string, double>;
%template(ssdMap) std::map<std::string, std::string>;
%template(ddVector) std::vector<std::vector<double>>;
%template(iVector) std::vector<int>;

%include "pycurva_headers1.i"
%include "swigerators.i"

%include "numpy.i"
%init %{
import_array();
%}

%include "pycurva_ignore.i"
%include "pycurva_typemaps.i"
%include "pycurva_headers2.i"

%template(dCurvaVector) CurvaVector<double>;
%template(dCurvaMatrix) CurvaMatrix<double>;
%template(iCurvaVector) CurvaVector<int>;
%template(iCurvaMatrix) CurvaMatrix<int>;

