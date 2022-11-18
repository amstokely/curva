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

#include "../include/curva.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include "../include/dcd/dcd.h"
#include "../include/utils.h"
#include "../include/cuda/mutual_information.h"

void curva::parsePdb (
		const std::string &fname,
		Atoms *atoms
) {
	std::string   line;
	std::ifstream pdb(fname);
	while (std::getline(pdb, line)) {
		if (utils::isRecordAtom(line)) {
			atoms->addAtom(Atom(line));
		}
	}
}

void curva::parseDcd (
		const std::string &fname,
		Coordinates *coordinates,
		int *numAtoms,
		int *numFrames,
		int firstFrame,
		int lastFrame
) {
	int       totalNumFrames             = 0;
	dcdhandle *dcd                       = open_dcd_read(
			&fname[0], numAtoms, &totalNumFrames
	);
	utils::setLastFrame(&lastFrame, totalNumFrames);
	molfile_timestep_t ts;
	unsigned int       *atomIndicesArray = nullptr;
	utils::generateIndicesArray(&atomIndicesArray, *numAtoms);
	*numFrames = (lastFrame - firstFrame) + 1;
	coordinates->init(*numAtoms, *numFrames);
	auto getAtomCoordinatesFromDcdLambda = [] (
			Coordinates *coordinates,
			int numFrames,
			const float *x,
			const float *y,
			const float *z,
			int atomIndex,
			int frame
	) {
		(*(coordinates))(0, atomIndex, frame) = static_cast<double>(
				x[atomIndex]
		);
		(*(coordinates))(1, atomIndex, frame) = static_cast<double>(
				y[atomIndex]
		);
		(*(coordinates))(2, atomIndex, frame) = static_cast<double>(
				z[atomIndex]
		);
	};
	while (dcd->setsread <= lastFrame) {
		read_next_timestep(dcd, *numAtoms, &ts);
		if (dcd->setsread > firstFrame) {
			std::for_each(
					atomIndicesArray,
					atomIndicesArray + *numAtoms,
					[
							getAtomCoordinatesFromDcdLambda,
							coordinates,
							numFrames,
							capture0 = dcd->x,
							capture1 = dcd->y,
							capture2 = dcd->z,
							capture3 = (dcd->setsread - 1 - firstFrame)
					] (
							auto &&PH1
					) {
						return getAtomCoordinatesFromDcdLambda(
								coordinates,
								*numFrames,
								capture0,
								capture1,
								capture2,
								std::forward<decltype(PH1)>(PH1),
								capture3
						);
					}
			);
		}
	}
	close_file_read(dcd);
	delete[] atomIndicesArray;
}

void curva::generateNodes (
		Nodes *nodes,
		Atoms *atoms,
		Coordinates *coordinates,
		unsigned int numWindows
) {
	unsigned int numFrames = coordinates->numFrames();
	nodes->init(atoms, coordinates, numFrames, numWindows);
	auto nodesIterator = nodes->begin();
	while (nodesIterator != nodes->end()) {
		for (
			auto  &averageCom: (*(nodesIterator))
				->averageCenterOfMass()) {
			averageCom /= ((*(nodesIterator))->totalMass() *
			               numFrames / numWindows);
		}
		for (auto &com: (*(nodesIterator))->centerOfMass()) {
			com /= (*(nodesIterator))->totalMass();
		}
		nodesIterator++;
	}
}

void curva::mutualInformationConstruct (
		Node *node,
		CurvaMatrix<double> *averagePositionMatrix,
		unsigned int windowIndex,
		unsigned int windowSize
) {
	double avgX = node->averageCenterOfMass().at(
			3 * windowIndex
	);
	double avgY = node->averageCenterOfMass().at(
			3 * windowIndex + 1
	);
	double avgZ = node->averageCenterOfMass().at(
			3 * windowIndex + 2
	);
	(*(averagePositionMatrix))(0, node->index()) = avgX;
	(*(averagePositionMatrix))(1, node->index()) = avgY;
	(*(averagePositionMatrix))(2, node->index()) = avgZ;
}

void curva::test::generalizedCorrelationTest (
		CurvaMatrix<double> *mutualInformationMatrix,
		const std::string &xfname,
		const std::string &yfname,
		unsigned int referenceIndex
) {
	int  k             = 8;
	int  numNodes      = 2;
	auto *cutoffMatrix = new CurvaMatrix<double>;
	auto *X            = new CurvaMatrix<double>;
	auto *Y            = new CurvaMatrix<double>;
	auto *XY           = new CurvaMatrix<double>;
	mutualInformationMatrix->init(
			numNodes,
			numNodes
	);
	cutoffMatrix->init(numNodes, numNodes);
	for (auto &val: *cutoffMatrix) {
		val = 1.0;
	}
	X->load(xfname);
	Y->load(yfname);

	int       numFrames = X->n() / 3;
	XY->init(3 * numFrames * numNodes, 1);
	for (unsigned int i = 0;
	     i < numFrames;
	     i++) {
		(*(XY))(i, 0) = (*(X))(0, i);
	}

	for (unsigned int i = 0;
	     i < numFrames;
	     i++) {
		(*(XY))(i + numFrames, 0) = (*(X))(0, i + numFrames);
	}

	for (unsigned int i = 0;
	     i < numFrames;
	     i++) {
		(*(XY))(i + 2 * numFrames, 0) = (*(X))(0, i + 2 * numFrames);
	}

	for (unsigned int i = 0;
	     i < numFrames;
	     i++) {
		(*(XY))(i + 3 * numFrames, 0) = (*(Y))(0, i);
	}

	for (unsigned int i = 0;
	     i < numFrames;
	     i++) {
		(*(XY))(i + 4 * numFrames, 0) = (*(Y))(0, i + numFrames);
	}

	for (unsigned int i = 0;
	     i < numFrames;
	     i++) {
		(*(XY))(i + 5 * numFrames, 0) = (*(Y))(0, i + 2 * numFrames);
	}

	XY->allocate();
	XY->toDevice();
	cudaGeneralizedCorrelation(
			XY,
			mutualInformationMatrix, cutoffMatrix,
			numNodes,
			1024, numFrames,
			0,
			referenceIndex, k
	);
	XY->deallocate();

	delete XY;
	delete X;
	delete Y;
	delete cutoffMatrix;
}
