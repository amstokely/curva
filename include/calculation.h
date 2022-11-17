//
// Created by andy on 8/18/22.
//

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

	void mutualInformation (
			int referenceIndex,
			double cutoff,
			int k,
			const std::string& norm
	);

	void pearsonCorrelation (double cutoff);

	CurvaMatrix<double> *mutualInformationMatrix () const;

	CurvaMatrix<double> *pearsonCorrelationMatrix () const;

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
	CurvaMatrix<double> *_mutualInformationMatrix;
	CurvaMatrix<double> *_pearsonCorrelationMatrix;
	int                 _numNodes;
	std::string         _name;
};

#endif //CURVA_CALCULATION_H
