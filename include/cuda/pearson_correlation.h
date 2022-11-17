//
// Created by andy on 7/29/22.
//

#ifndef CURVA_PEARSON_CORRELATION_H
#define CURVA_PEARSON_CORRELATION_H

#include <vector>
#include "../matrix/curva_matrix.h"


void cudaPearsonCorrelation (
		CurvaMatrix<double> *deltaAverageNodePositionMatrix,
		CurvaMatrix<double> *pearsonCorrelationMatrix,
		CurvaMatrix<double> *cutoffMatrix,
		unsigned int numNodes,
		unsigned int numFrames
);

#endif //CURVA_PEARSON_CORRELATION_H
