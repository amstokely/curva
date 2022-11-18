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

%{
#define SWIG_FILE_WITH_INIT
#include <sstream>
#include "../include/curva.h"
#include "../include/calculation.h"
#include "../include/utils.h"
#include "../include/atom.h"
#include "../include/matrix/curva_matrix.h"
#include "../include/matrix/curva_vector.h"
#include "../include/node.h"
#include "../include/atoms.h"
#include "../include/custom_iterators/random_access_iterator.h"
#include "../include/custom_iterators/forward_iterator.h"
#include "../include/custom_iterators/map_value_forward_iterator.h"
#include "../include/nodes.h"
#include "../include/coordinates.h"
#include "../include/cnpy/cnpy.h"
%}