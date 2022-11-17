//
// Created by andy on 8/12/22.
//

#ifndef CURVA_MUTUAL_INFORMATION_H
#define CURVA_MUTUAL_INFORMATION_H

#include "../matrix/curva_matrix.h"

void cudaMutualInformation (
		CurvaMatrix<double> *XY,
		CurvaMatrix<double> *mutualInformationMatrix,
		CurvaMatrix<double> *cutoffMatrix,
		int numNodes,
		int numFrames,
		unsigned int totalNumFrames,
		unsigned int firstFrame,
		unsigned int referenceIndex,
		int k,
		const std::string& norm
);

void cudaNormalizeNodeCoordinates (
		CurvaMatrix<double> *nodeCoordinates,
		int numFrames,
		int numNodes,
		unsigned int totalNumFrames
);

#endif //CURVA_MUTUAL_INFORMATION_H
