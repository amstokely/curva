//
// Created by andy on 7/18/22.
//

#ifndef PARSER_CUDNA_H
#define PARSER_CUDNA_H

#include "calculation.h"

namespace curva {
	void parseDcd (
			const std::string &fname,
			Coordinates *coordinates,
			int *numAtoms,
			int *numFrames,
			int firstFrame = 0,
			int lastFrame = -1
	);

	void parsePdb (
			const std::string &fname,
			Atoms *atoms
	);

	void generateNodes (
			Nodes *nodes,
			Atoms *atoms,
			Coordinates *coordinates,
			unsigned int numWindows
	);

	void pearsonCorrelationConstruct (
			Node *node,
			CurvaMatrix<double> *deltaAveragePositionMatrix,
			CurvaMatrix<double> *averagePositionMatrix,
			unsigned int windowIndex,
			unsigned int windowSize
	);

	void mutualInformationConstruct (
			Node *node,
			CurvaMatrix<double> *averagePositionMatrix,
			unsigned int windowIndex,
			unsigned int windowSize
	);

	namespace test {
		void mutualInformationTest (
				CurvaMatrix<double> *mutualInformationMatrix,
				const std::string &xfname,
				const std::string &yfname,
				unsigned int referenceIndex
		);
	}

}

#endif //PARSER_CUDNA_H
