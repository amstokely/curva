//
// Created by andy on 8/17/22.
//

#ifndef CURVA_CUTOFF_MATRIX_H
#define CURVA_CUTOFF_MATRIX_H
#include "../matrix/curva_matrix.h"

void cudaCutoffMatrix (
		CurvaMatrix<double> *averageNodePositionMatrix,
		CurvaMatrix<double> *cutoffMatrix,
		double cutoff,
		unsigned int numNodes,
		unsigned int numFrames
);

#endif //CURVA_CUTOFF_MATRIX_H
