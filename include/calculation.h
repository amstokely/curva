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

#ifndef CURVA_CALCULATION_H
#define CURVA_CALCULATION_H

#include "matrix/curva_matrix.h"
#include "nodes.h"
#include "coordinates.h"
#include "json/include/nlohmann/json.hpp"

class MolecularDynamicsCalculation {
public:
	MolecularDynamicsCalculation ();

	~MolecularDynamicsCalculation ();

	void init (
			const std::string &dcd,
			const std::string &pdb,
			int firstFrame,
			int lastFrame,
			int windowSize,
			const std::string &name
	);

	Atoms *atoms ();

	Nodes *nodes ();

	int numFrames () const;

	int numAtoms () const;

	int numNodes () const;

	std::string dcd ();

	std::string pdb ();


	void mutualInformationAllocate (
			CurvaMatrix<double> *X
	);

	void generalizedCorrelation (
			int referenceIndex,
			double cutoff,
			int k
	);

	CurvaMatrix<double> *generalizedCorrelationMatrix () const;

	void save (
			const std::string &fname,
			int indent = -1
	);

	void load (const std::string &fname);

private:
	Nodes               *_nodes;
	Atoms               *_atoms;
	CurvaMatrix<double> *XY_;
	std::string         _dcd;
	std::string         _pdb;
	int                 _numAtoms;
	int                 _numFrames;
	int                 _windowSize;
	CurvaMatrix<double> *_generalizedCorrelationMatrix;
	int                 _numNodes;
	std::string         _name;
};

#endif //CURVA_CALCULATION_H
